# DeltaSem: Certified Incremental Maintenance for LLM-Powered Semantic Join Analytics

This repository contains an executable reference implementation accompanying the manuscript:

> DeltaSem: Certified Incremental Maintenance for LLM-Powered Semantic Join Analytics.

## Overview

DeltaSem maintains bounded, nonnegative aggregates over changing semantic-join candidate populations. A versioned label ledger reuses fixed-oracle decisions. Immutable certificate anchors transport bounds through insertions and deletions, including retrospective evaluation of retained deleted inputs. Output-sensitive repair evaluates additional inputs until aggregate error and optional top-k membership contracts are satisfied.

Implemented components include weighted occurrences, exact-label reuse, anchored interval replay, signed-slice algebra, uniform-without-replacement audits, a lifetime audit-budget ledger, strict top-k boundary checks, incremental updates, and sequential checkpoints. Weighted audits in the maintenance runner use a conservative Hoeffding bound; the separate certificate module also provides exact hypergeometric count intervals.

**Release scope:** this is a sequential reference implementation with functional validation. It is not the original distributed experimental system, and it does not reproduce the manuscript's reported performance measurements. Relational candidate construction, production model serving, cost-calibrated scheduling, and crash-safe audit reservations are outside this release. No planning CSVs or manuscript performance numbers are bundled as measured results.

---

## Requirements

- Python >= 3.10
- NumPy >= 1.26
- SciPy >= 1.11

The included examples and tests run on CPU. A real semantic predicate is supplied through the Python oracle callback.

---

## Usage

### 1. Installation

```bash
git clone https://github.com/QSI432/DeltaSem.git
cd DeltaSem
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -e .
```

### 2. Replay the Example

```bash
python -m deltasem examples/updates.jsonl
```

The example uses explicit binary test labels in JSON preimages. These are functional fixtures, not model outputs or performance experiments. Each publication prints estimates, certified intervals, optional top-k membership, and actual oracle-call counters.

### 3. Integrate a Semantic Oracle

```python
from deltasem import DeltaSem, Occurrence

def oracle(semantic_key, preimage):
    # Call your fixed model/prompt/decoding pipeline here.
    # Return the parsed binary predicate result, 0 or 1.
    return int(preimage == "positive")

engine = DeltaSem(oracle, delta=0.05, seed=42)
engine.insert(Occurrence("review-1-v1", "canonical-input-v1", 1.0,
                         group="product-a", preimage="positive"))
result = engine.publish(epsilon=0.05, floor=1.0, batch=32)
engine.delete("review-1-v1")
updated = engine.publish(epsilon=0.05, floor=1.0)
engine.save("checkpoint.json")
restored = DeltaSem.load("checkpoint.json", oracle)
```

Semantic keys must include the oracle identity and complete canonical input. Occurrence IDs are immutable; replacement uses a fresh ID. Weights must be finite and nonnegative. `publish(k=...)` certifies membership, not the ordering of overlapping selected intervals. Use one engine and one lifetime risk budget for an adaptive run. Checkpoints are valid only between completed operations; they are not a crash-recovery WAL. See [implementation notes](docs/IMPLEMENTATION.md).

### 4. Run Correctness Validation

```bash
python -m unittest discover -s tests -v
python -m tests.validate_certificates
python -m tests.validate_anchored_replay
python -m tests.validate_topk_and_slice_guards
```

The final guard script expects the two preceding validation reports. Tests cover exact finite-population coverage, pathwise transport, retrospective deletion refinement, adaptive updates, strict ranking ties, label reuse, and checkpoint restoration. Validation reports describe correctness checks only. [Release validation](docs/VALIDATION.md) records the checks run for this release.

---

## Reference

- [SciPy hypergeometric distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)
- W. Hoeffding. Probability Inequalities for Sums of Bounded Random Variables. JASA, 1963. [DOI](https://doi.org/10.1080/01621459.1963.10500830).

The repository accompanies a manuscript; no acceptance or archival publication status is asserted here.
