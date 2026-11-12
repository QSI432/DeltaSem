"""Executable reference for finite-population certificates and delta transport.

These functions validate mathematical building blocks. They are not an LLM
serving implementation and do not contain measured system performance data.
"""
from dataclasses import dataclass, field
from math import isfinite
from scipy.stats import hypergeom


def hypergeom_count_interval(N: int, n: int, x: int, alpha: float):
    """Two-sided exact integer CI for population positives under uniform WOR."""
    if not (0 <= x <= n <= N and 0 < alpha < 1):
        raise ValueError("invalid finite-population parameters")
    if n == 0:
        return 0, N
    if n == N:
        return x, x
    lo, hi = x, N - n + x
    # SF monotonically increases in population-positive count K.
    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if hypergeom.sf(x - 1, N, mid, n) >= alpha / 2:
            right = mid
        else:
            left = mid + 1
    L = left
    # CDF monotonically decreases in population-positive count K.
    left, right = lo, hi
    while left < right:
        mid = (left + right + 1) // 2
        if hypergeom.cdf(x, N, mid, n) >= alpha / 2:
            left = mid
        else:
            right = mid - 1
    return L, left


def transport(L, U, removed_unknown_mass, added_unknown_mass,
              removed_resolved_value, new_unresolved_mass):
    """Transport a residual interval through disjoint membership deltas.

    removed_resolved_value is a sum of labels newly obtained for items removed
    from the previous residual domain, including items promoted into the exact
    ledger. It is not the sum of values already in the exact ledger.
    """
    if min(removed_unknown_mass, added_unknown_mass,
           removed_resolved_value, new_unresolved_mass) < 0:
        raise ValueError("nonnegative aggregate required")
    lower = max(0.0, L - removed_resolved_value - removed_unknown_mass)
    upper = min(new_unresolved_mass, U - removed_resolved_value + added_unknown_mass)
    if lower > upper + 1e-9:
        raise ValueError("empty certificate: audit failure or state corruption")
    return lower, upper


def safe_midpoint(L, U, epsilon, floor):
    """Mixed relative/absolute contract: |estimate-A|<=eps*max(A,floor)."""
    if U - L <= 2 * epsilon * max(L, floor):
        return (L + U) / 2
    return None


def separated_topk(intervals, k):
    """Certify membership by strict separation or exactly resolved boundary ties.

    Group keys use a fixed ascending order, which is lexicographic for strings.
    Every selected/outsider comparison must be strictly separated or consist of
    two equal point intervals, with the selected key winning the tie. Merely
    touching unresolved intervals is insufficient. Nonboundary intervals can
    remain approximate. The returned list certifies membership, not a total
    ordering among selected groups whose own intervals overlap.
    """
    if not 1 <= k <= len(intervals):
        raise ValueError("bad k")
    for lower, upper in intervals.values():
        if not (isfinite(lower) and isfinite(upper) and 0 <= lower <= upper):
            raise ValueError("invalid nonnegative group interval")
    selected = sorted(
        intervals,
        key=lambda g: (-(intervals[g][0] + (intervals[g][1] - intervals[g][0]) / 2), g),
    )[:k]
    others = set(intervals).difference(selected)
    for group in selected:
        lower, upper = intervals[group]
        for outsider in others:
            other_lower, other_upper = intervals[outsider]
            if lower > other_upper:
                continue
            if lower == upper == other_lower == other_upper and group < outsider:
                continue
            return None
    return selected


def audit_budget(delta, creation_index):
    """Summable global budget over fresh certificates, indexed from one."""
    if creation_index < 1:
        raise ValueError("indices start at one")
    return delta / (creation_index * (creation_index + 1))


def stage_budget(alpha_audit, stage_index):
    if stage_index < 1:
        raise ValueError("indices start at one")
    return alpha_audit / (stage_index * (stage_index + 1))


@dataclass
class AuditBudgetLedger:
    """Persistent global audit/stage identity allocation for one risk budget.

    Reserve an identity/stage and durably save ``to_state()`` before drawing its
    random sample. Snapshot result commits do not reset this ledger. Restore the
    saved ledger after a restart, rather than starting creation_index at one.
    The reference class is single-writer. A concurrent engine must serialize
    reservations and their persistence in its own metadata transaction.
    """
    risk_budget: float
    last_creation_index: int = 0
    stages: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        if not isfinite(self.risk_budget) or not 0 < self.risk_budget < 1:
            raise ValueError("risk budget must be between zero and one")
        if not isinstance(self.last_creation_index, int) or self.last_creation_index < 0:
            raise ValueError("invalid global audit index")
        if len(self.stages) != self.last_creation_index:
            raise ValueError("stored audit history is incomplete")
        if any(not isinstance(j, int) or not 1 <= j <= self.last_creation_index
               or not isinstance(r, int) or r < 0 for j, r in self.stages.items()):
            raise ValueError("invalid stored audit-stage identity")
        self.stages = dict(self.stages)

    def create_audit(self):
        self.last_creation_index += 1
        index = self.last_creation_index
        self.stages[index] = 0
        return index, audit_budget(self.risk_budget, index)

    def next_stage(self, creation_index):
        if creation_index not in self.stages:
            raise ValueError("unknown audit identity")
        self.stages[creation_index] += 1
        stage = self.stages[creation_index]
        risk = stage_budget(audit_budget(self.risk_budget, creation_index), stage)
        return stage, risk

    def to_state(self):
        return {
            "schema_version": 1,
            "risk_budget": self.risk_budget,
            "last_creation_index": self.last_creation_index,
            "stages": {str(j): r for j, r in sorted(self.stages.items())},
        }

    @classmethod
    def from_state(cls, state):
        if state.get("schema_version") != 1:
            raise ValueError("unsupported budget-ledger schema")
        return cls(
            risk_budget=state["risk_budget"],
            last_creation_index=state["last_creation_index"],
            stages={int(j): r for j, r in state["stages"].items()},
        )
