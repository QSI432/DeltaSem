# Implementation notes

| Component | Source | Contract |
|---|---|---|
| Certificate transport, exact count CI, risk allocation, ranking | `deltasem/certificates.py` | Mathematical primitives |
| Versioned occurrences, retained inputs, signed slices, anchor replay | `deltasem/anchors.py` | Immutable identity and nonnegative bounded weights |
| Incremental runner and repair | `deltasem/engine.py` | Fixed binary oracle, sequential operations |
| JSONL entry point | `deltasem/__main__.py` | Fixture oracle; replace callback for real inference |

The maintenance runner constructs a fresh uniformly sampled residual frame on each audit. Its weighted Hoeffding interval spends one audit and stage allocation before sampling. Exact observed labels are then promoted into the ledger, while replay subtracts their contributions from the immutable anchor frame. Multiple anchor projections are intersected. Deleted occurrences retain their preimages; `resolve(id)` can evaluate them later.

The scheduler selects a widest blocking group and uses fixed-size audit batches. It is a reference heuristic, not the paper's full cost-aware relational planner. Occurrences must be supplied by the caller's join/domain construction. The oracle must map each semantic key to an immutable binary value. For stochastic models, freeze a response tape or define first-write semantics independently of audit randomness.

`save` atomically replaces a JSON checkpoint after an operation. It preserves labels, retained occurrences, anchors, RNG state, and budget history for sequential continuation. It does not persist audit reservations before model calls, offer interprocess locking, or support a forked run sharing a risk budget. An interrupted audit is not safely resumable from an earlier checkpoint under the same lifetime claim.

The release was assembled as an executable reference after manuscript preparation. Its correctness checks do not establish provenance or reproduction of the manuscript's original experiments.
