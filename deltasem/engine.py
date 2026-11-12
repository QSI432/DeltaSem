"""Single-writer maintenance of nonnegative, binary-oracle aggregates.

An oracle identity (model, prompt, decoding policy) must remain fixed for an
engine's lifetime. Semantic keys must canonically identify that oracle's input.
The caller supplies occurrences after relational candidate construction.
"""
from dataclasses import asdict
import json
from math import log, sqrt
from pathlib import Path
from random import Random

from .anchors import Anchor, LabelLedger, Occurrence, Snapshot, composed_interval
from .certificates import AuditBudgetLedger, safe_midpoint, separated_topk


class DeltaSem:
    def __init__(self, oracle, occurrences=(), *, delta=0.05, seed=0):
        self.oracle = oracle
        self.snapshot = Snapshot(occurrences)
        self.ledger = LabelLedger()
        self.anchors = {}
        self.budget = AuditBudgetLedger(delta)
        self.rng = Random(seed)
        self.oracle_calls = 0

    def insert(self, occurrence):
        self.snapshot.insert(occurrence)

    def delete(self, identity):
        self.snapshot.delete(identity)

    def replace(self, identity, occurrence):
        if occurrence.identity in self.snapshot.known_occurrences:
            raise ValueError("replacement requires a fresh occurrence identity")
        if identity not in self.snapshot.active:
            raise KeyError(identity)
        self.delete(identity)
        self.insert(occurrence)

    def resolve(self, identity):
        """Retained deleted inputs can tighten an old anchor retrospectively."""
        item = self.snapshot.known_occurrences[identity]
        if item.semantic_key not in self.ledger.labels:
            value = self.oracle(item.semantic_key, item.preimage)
            self.ledger.observe(item.semantic_key, value)
            self.oracle_calls += 1
        return self.ledger.labels[item.semantic_key]

    def intervals(self):
        groups = sorted({x.group for x in self.snapshot.active.values()})
        return {g: composed_interval(self.snapshot, self.ledger,
                                    self.anchors.get(g, ()), g) for g in groups}

    def audit(self, group, count):
        """Uniform WOR audit of a frozen residual frame with bounded weights.

        Hoeffding's WOR bound is conservative for arbitrary nonnegative weights.
        A fresh audit spends a fresh global identity; census needs no probability
        bound. The exact hypergeometric primitive is also exposed separately.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be positive")
        frame = tuple(self.snapshot.residual(self.ledger, group).values())
        if not frame:
            return self.intervals().get(group, (0.0, 0.0))
        index, _ = self.budget.create_audit()
        _, alpha = self.budget.next_stage(index)
        sample = self.rng.sample(frame, min(count, len(frame)))
        observed = sum(x.weight * self.resolve(x.identity) for x in sample)
        mass = sum(x.weight for x in frame)
        if len(sample) == len(frame):
            lower = upper = observed
        else:
            estimate = len(frame) * observed / len(sample)
            radius = len(frame) * max(x.weight for x in frame) * sqrt(
                log(2 / alpha) / (2 * len(sample)))
            lower, upper = max(0.0, estimate-radius), min(mass, estimate+radius)
        self.anchors.setdefault(group, []).append(
            Anchor(f"audit-{index}", frame, lower, upper, group))
        return self.intervals()[group]

    def publish(self, *, epsilon=0.05, floor=1.0, k=None, batch=32):
        if not 0 < epsilon < 1 or floor <= 0 or batch < 1:
            raise ValueError("invalid publication contract")
        if k is not None and not 1 <= k <= len(self.intervals()):
            raise ValueError("bad k")
        while True:
            intervals = self.intervals()
            estimates = {g: safe_midpoint(*v, epsilon, floor)
                         for g, v in intervals.items()}
            top = separated_topk(intervals, k) if k is not None else None
            if all(v is not None for v in estimates.values()) and (k is None or top is not None):
                return {"estimates": estimates, "intervals": intervals,
                        "topk": top, "oracle_calls": self.oracle_calls}
            # Output-sensitive scheduling: renew a widest blocking group first.
            blocking = [g for g in intervals if estimates[g] is None]
            if not blocking:
                blocking = list(intervals)
            group = max(blocking, key=lambda g: intervals[g][1]-intervals[g][0])
            self.audit(group, batch)

    def save(self, path):
        """Save between operations; do not resume two writers from one checkpoint.

        This is a sequential checkpoint, not a crash-safe audit reservation WAL.
        After an interrupted audit, start a new independent risk budget/run.
        """
        state = {"schema": 1, "budget": self.budget.to_state(),
                 "known": [asdict(x) for x in self.snapshot.known_occurrences.values()],
                 "active": list(self.snapshot.active), "labels": self.ledger.labels,
                 "calls": self.oracle_calls, "rng": self.rng.getstate(),
                 "anchors": {g: [asdict(a) for a in values]
                             for g, values in self.anchors.items()}}
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf8")
        temporary.replace(path)

    @classmethod
    def load(cls, path, oracle):
        state = json.loads(Path(path).read_text(encoding="utf8"))
        if state["schema"] != 1:
            raise ValueError("unsupported checkpoint")
        engine = cls(oracle)
        engine.snapshot = Snapshot(Occurrence(**x) for x in state["known"])
        engine.snapshot.active = {i: engine.snapshot.known_occurrences[i] for i in state["active"]}
        for key, label in state["labels"].items():
            engine.ledger.observe(key, label)
        engine.budget = AuditBudgetLedger.from_state(state["budget"])
        engine.oracle_calls = state["calls"]
        def tuples(value):
            return tuple(map(tuples, value)) if isinstance(value, list) else value
        engine.rng.setstate(tuples(state["rng"]))
        engine.anchors = {g: [Anchor(a["identity"], tuple(Occurrence(**x) for x in a["population"]),
                                   a["lower"], a["upper"], a["group"]) for a in values]
                          for g, values in state["anchors"].items()}
        return engine
