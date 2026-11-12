# Release validation

Executed on 2026-11-12 with CPython 3.12, NumPy 2.5.4, and SciPy 1.18.1 on Windows.

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -v` | 3 tests passed |
| Incremental runner | 180 randomized update epochs; exact truth inside published intervals; top-k membership checked |
| Sequential checkpoint continuation | Passed; intervals and budget history preserved |
| Retrospective evaluation of deleted input | Passed |
| Semantic-key label reuse and invalid replacement | Passed |
| Exact hypergeometric coverage | 5,200 cases; worst failure probability 0.0497737556561086 at nominal alpha 0.05 |
| Pathwise certificate transport | 5,100 cases passed |
| Mixed relative/absolute publication contract | 10,000 cases passed |
| Adaptive anchored version transitions | 9,600 checks passed |
| Top-k boundary guards | 12 cases passed |
| Signed-slice/frame compatibility and risk-ledger restoration | Passed |
| `python -m deltasem examples/updates.jsonl` | Passed; 2 publications |

Generated numerical reports are in `tests/*results.json`. These report
correctness-validation fixtures only, not empirical LLM or system speedups.
