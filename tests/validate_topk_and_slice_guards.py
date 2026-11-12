"""Focused publication/frame/persistence guards, with no performance benchmarks."""
import hashlib
import json
from math import isclose
from pathlib import Path
from deltasem.anchors import (
    Anchor, IncompatibleSliceFrame, LabelLedger, Occurrence, SignedSlice, Snapshot,
)
from deltasem.certificates import AuditBudgetLedger, separated_topk

root = Path(__file__).resolve().parent
unchanged_reports = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root / "validation_results.json", root / "anchored_validation_results.json")
}

cases = [
    ("all_exact_ties", {"z": (5, 5), "a": (5, 5), "m": (5, 5)}, 2, ["a", "m"]),
    ("one_exact_tie", {"z": (5, 5), "a": (5, 5)}, 1, ["a"]),
    ("unresolved_equal_intervals", {"a": (4, 6), "b": (4, 6)}, 1, None),
    ("unresolved_outsider_touches", {"a": (5, 5), "b": (4, 5)}, 1, None),
    ("unresolved_selected_touches", {"a": (5, 7), "b": (5, 5)}, 1, None),
    ("wrong_lexicographic_boundary", {"z": (5, 5), "a": (4, 5)}, 1, None),
    ("mixed_strict_and_exact_tie", {
        "top": (10, 14), "a": (7, 7), "b": (7, 7), "low": (0, 6)
    }, 2, ["top", "a"]),
    ("approximate_nonboundary", {
        "top": (10, 14), "a": (7, 7), "b": (7, 7), "low": (0, 6)
    }, 3, ["top", "a", "b"]),
    ("strict_approximate_intervals", {"top": (10, 11), "low": (1, 9)}, 1, ["top"]),
    ("all_groups_selected", {"a": (1, 9), "b": (2, 6)}, 2, ["a", "b"]),
    ("distinct_exact_values", {"a": (2, 2), "b": (8, 8), "c": (3, 3)}, 2, ["b", "c"]),
    ("near_values_are_not_ties", {"a": (1, 1), "b": (1 + 1e-12, 1 + 1e-12)}, 1, ["b"]),
]
for name, intervals, k, expected in cases:
    assert separated_topk(intervals, k) == expected, name

for invalid_k in (0, 3):
    try:
        separated_topk({"a": (1, 1), "b": (2, 2)}, invalid_k)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid k accepted")

# Frozen signed-slice membership is invalid after partial reinsertion.
items = tuple(Occurrence(str(i), str(i), 1.0) for i in range(20))
truth = {str(i): int(i == 0 or 5 <= i <= 13) for i in range(20)}
anchor = Anchor("fixed-frame", items, 8.0, 12.0)
snapshot = Snapshot(items)
ledger = LabelLedger()
for i in range(5):
    snapshot.delete(str(i))
retired = SignedSlice(frozenset(str(i) for i in range(5)), -1, 1.0, 2.0)
assert anchor.project(snapshot.residual(ledger, "group"), ledger, [retired]) == (6.0, 11.0)
snapshot.insert(items[0])
try:
    anchor.project(snapshot.residual(ledger, "group"), ledger, [retired])
except IncompatibleSliceFrame:
    # Expected caller policy: discard this shortcut, retaining the anchor.
    interval = anchor.project(snapshot.residual(ledger, "group"), ledger)
else:
    raise AssertionError("a frozen deletion slice was silently reused after reinsertion")
current_truth = sum(item.weight * truth[item.semantic_key] for item in snapshot.active.values())
assert interval == (4.0, 12.0)
assert interval[0] <= current_truth <= interval[1]

# Commit/restart round trips preserve globally monotonic creation and stage IDs.
budget = AuditBudgetLedger(0.05)
j1, a1 = budget.create_audit()
r11, e11 = budget.next_stage(j1)
assert (j1, r11) == (1, 1)
assert isclose(a1, 0.025) and isclose(e11, 0.0125)
checkpoint = json.loads(json.dumps(budget.to_state()))
budget = AuditBudgetLedger.from_state(checkpoint)
j2, a2 = budget.create_audit()
r21, e21 = budget.next_stage(j2)
r12, e12 = budget.next_stage(j1)
assert (j2, r21, r12) == (2, 1, 2)
assert isclose(a2, 0.05 / 6)
assert isclose(e21, 0.05 / 12) and isclose(e12, 0.05 / 12)
checkpoint = json.loads(json.dumps(budget.to_state()))
budget = AuditBudgetLedger.from_state(checkpoint)
j3, _ = budget.create_audit()
r13, _ = budget.next_stage(j1)
assert (j3, r13) == (3, 3)
assert e11 + e21 + e12 < budget.risk_budget

for name, digest in unchanged_reports.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest

report = {
    "scope": "Focused deterministic guard and persistence checks only. No LLM or system-performance measurements.",
    "topk_boundary_cases": len(cases),
    "invalid_topk_sizes": "rejected",
    "partial_reinsertion_slice": "rejected with IncompatibleSliceFrame",
    "bounded_leaf_fallback": "valid",
    "global_budget_commit_restart_persistence": "passed",
    "original_validation_reports": "unchanged",
    "status": "passed",
}
(root / "topk_and_slice_guard_results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
