"""Check anchored replay on fixed oracle tapes and adaptive version updates.

All values below are test fixtures. This program measures no system speedups.
"""
import json
from pathlib import Path
import random
from deltasem.anchors import Anchor, LabelLedger, Occurrence, SignedSlice, Snapshot, composed_interval


def contains(interval, truth):
    assert interval[0] - 1e-8 <= truth <= interval[1] + 1e-8, (interval, truth)


# Retrospective refinement must replay the anchor rather than mutate clipped bounds.
base = (
    Occurrence("d", "old-d", 5.0, preimage="retained deleted input"),
    Occurrence("a", "a", 6.0),
    Occurrence("b", "b", 6.0),
)
anchor = Anchor("example", base, 8.0, 10.0)
snapshot = Snapshot(base)
snapshot.delete("d")
ledger = LabelLedger()
assert anchor.project(snapshot.residual(ledger, "group"), ledger) == (3.0, 10.0)
ledger.observe("old-d", 0)
assert anchor.project(snapshot.residual(ledger, "group"), ledger) == (8.0, 10.0)
ledger_positive = LabelLedger()
ledger_positive.observe("old-d", 1)
assert anchor.project(snapshot.residual(ledger_positive, "group"), ledger_positive) == (3.0, 5.0)

# Exact promotion preserves the group interval until a physical cap tightens it.
small = tuple(Occurrence(str(i), str(i), 1.0) for i in range(20))
snapshot = Snapshot(small)
ledger = LabelLedger()
anchor = Anchor("promotion", small, 8.0, 12.0)
before = composed_interval(snapshot, ledger, [anchor])
ledger.observe("0", 1)
after = composed_interval(snapshot, ledger, [anchor])
assert before == after == (8.0, 12.0)

# An occurrence reinserted with the same immutable identity cancels its deletion.
snapshot = Snapshot(base)
ledger = LabelLedger()
anchor = Anchor("cancel", base, 8.0, 10.0)
snapshot.delete("d")
snapshot.insert(base[0])
assert anchor.project(snapshot.residual(ledger, "group"), ledger) == (8.0, 10.0)

# A certified removed slice substitutes one disjoint signed node.
snapshot = Snapshot(small)
ledger = LabelLedger()
anchor = Anchor("retired-slice", small, 8.0, 12.0)
for i in range(5):
    snapshot.delete(str(i))
raw = anchor.project(snapshot.residual(ledger, "group"), ledger)
certificate = SignedSlice(frozenset(str(i) for i in range(5)), -1, 1.0, 2.0)
refined = anchor.project(snapshot.residual(ledger, "group"), ledger, [certificate])
assert raw == (3.0, 12.0)
assert refined == (6.0, 11.0)

rng = random.Random(731983)
checks = 0
for trial in range(120):
    labels = {}
    next_identity = 0

    def new_occurrence(group="group", same_semantic_key=None):
        global next_identity
        identity = f"{trial}-v{next_identity}"
        next_identity += 1
        semantic_key = same_semantic_key or f"input-{identity}"
        labels.setdefault(semantic_key, rng.randrange(2))
        return Occurrence(identity, semantic_key, float(rng.randrange(1, 8)), group)

    initial = tuple(new_occurrence() for _ in range(16))
    initial_total = sum(item.weight * labels[item.semantic_key] for item in initial)
    initial_mass = sum(item.weight for item in initial)
    initial_anchor = Anchor(
        f"trial-{trial}", initial,
        max(0.0, initial_total - rng.randrange(0, 6)),
        min(initial_mass, initial_total + rng.randrange(0, 6)),
    )
    snapshot = Snapshot(initial)
    ledger = LabelLedger()
    anchors = [initial_anchor]
    deleted = []
    for epoch in range(80):
        # Some deletion choices depend on labels already revealed to the algorithm.
        action = rng.randrange(7)
        active = list(snapshot.active.values())
        if action == 0 or not active:
            snapshot.insert(new_occurrence())
        elif action == 1:
            candidates = [item for item in active
                          if ledger.labels.get(item.semantic_key) == 1] or active
            item = rng.choice(candidates)
            snapshot.delete(item.identity)
            deleted.append(item)
        elif action == 2:
            item = rng.choice(active)
            snapshot.delete(item.identity)
            deleted.append(item)
            snapshot.insert(new_occurrence())  # distinct semantic-input version
        elif action == 3:
            item = rng.choice(active)
            ledger.observe(item.semantic_key, labels[item.semantic_key])
        elif action == 4 and deleted:
            item = rng.choice(deleted)
            ledger.observe(item.semantic_key, labels[item.semantic_key])
        elif action == 5:
            item = rng.choice(active)
            snapshot.delete(item.identity)
            deleted.append(item)
            snapshot.insert(new_occurrence(same_semantic_key=item.semantic_key))
        else:
            residual = tuple(snapshot.residual(ledger, "group").values())
            true_residual = sum(item.weight * labels[item.semantic_key] for item in residual)
            mass = sum(item.weight for item in residual)
            # A valid fresh anchor fixture tests graph intersection algebra only.
            anchors.append(Anchor(
                f"refresh-{trial}-{epoch}", residual,
                max(0.0, true_residual - rng.randrange(0, 4)),
                min(mass, true_residual + rng.randrange(0, 4)),
            ))
        truth = sum(item.weight * labels[item.semantic_key]
                    for item in snapshot.active.values())
        contains(composed_interval(snapshot, ledger, anchors), truth)
        checks += 1

# Reusing one occurrence identifier for edited contents is rejected.
snapshot = Snapshot([Occurrence("x", "old-input", 1.0)])
snapshot.delete("x")
try:
    snapshot.insert(Occurrence("x", "new-input", 1.0))
except ValueError:
    pass
else:
    raise AssertionError("changed occurrence identity was accepted")

report = {
    "scope": "Anchored interval-algebra tests on fixed oracle tapes. No empirical LLM or serving measurements.",
    "adaptive_version_transition_checks": checks,
    "retrospective_deletion_refinement": "passed",
    "exact_promotion_nonshrinking_case": "passed",
    "cancelling_occurrence_deltas": "passed",
    "certified_disjoint_retraction_slice": "passed",
    "immutable_occurrence_guard": "passed",
    "status": "passed",
}
Path(__file__).with_name("anchored_validation_results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
