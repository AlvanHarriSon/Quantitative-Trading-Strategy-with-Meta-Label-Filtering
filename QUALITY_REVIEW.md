# Engineering quality review

## 1. Project Assessment

Reviewed baseline commit `2a983f61332cd1254c0aa025d32673d9a6b59e98`. All tracked first-party source, documentation,
dependency/configuration files and entrypoints were inspected. Original source
compiled; no tests or CI existed. No database, server, Docker or build system was
present. Original PixelScribe and trading imports required missing environment
dependencies; this was not caused by the changes. Baseline evidence is retained
in the delivered review archive. Validation environment: Windows, Python 3.14.6.

## 2. Major Issues Found

P1: unresolved future labels leaked into walk-forward training; RSI zero-loss case and ADX ties were wrong; trailing stops used same-bar future close; daily return metrics omitted costs; initial-capital drawdown, cooldown and final liquidation were inconsistent.

## 3. Changes Implemented

Purged unresolved labels and excluded censored outcomes; corrected RSI/ADX and prior-bar stop timing; reconciled labels, leveraged daily returns, equity and net trade costs; separated reported/risk drawdown; fixed cooldown/final liquidation; validated config/prices; removed global warning suppression; made notebook dependencies optional. Added behavioral regression tests, a focused CI matrix, and updated
installation/run instructions. No Git history, user data or model resources were deleted.

## 4. Bugs Fixed

The concrete defects listed above are covered by focused regression tests;
external-service behavior is distinguished below from local evidence.

## 5. Architecture Improvements

Kept Config, public function signatures and script entrypoint. Added explicit validation and resolved-training-window helpers; retained close-price execution and documented its limitations.

## 6. Performance Improvements

No throughput or latency improvement is claimed: no performance benchmark was run.
Resource-lifetime and correctness changes are not reported as measured speedups.

## 7. Testing & Validation

PASS: `14` tests using `python -m unittest discover -s tests -v`.
PASS: `python -m compileall -q .`.
PASS: `python -m pip check` in the shared isolated review environment.
PASS: `python -m ruff check . --select E9,F63,F7,F82` (focused correctness rules).
PASS: `python -m mypy --follow-imports=skip --ignore-missing-imports "Quantitative Trading Strategy V1.py"`
(basic checking, not strict coverage of all unannotated bodies or dependencies).
PASS: `git diff --check`.

PASS: 7 focused regression cases all failed on the original commit and passed after fixes. PASS: synthetic end-to-end pipeline, multiclass walk-forward and meta backtest. FAIL (external): real Yahoo historical download returned YFRateLimitError.

NOT TESTED locally: Python 3.10/3.12 and Linux/macOS. GitHub CI defines Linux
Python 3.10/3.12/3.14 checks; adding a workflow is not evidence that it has run.
No separate packaging/build target exists; bytecode compilation is the build check.

## 8. Remaining Risks

Real-market end-to-end run remains blocked by Yahoo rate limiting. Signal-close entries and close-price stop exits remain research assumptions; liquidity, financing and margin liquidation are not modeled. Historical README performance was not revalidated and is no longer presented as current evidence.

## 9. How to Run

From the repository root, use Python 3.10+ and a virtual environment.
Install runtime dependencies with `python -m pip install -r requirements.txt`.

```bash
python "Quantitative Trading Strategy V1.py"
python -m unittest discover -s tests -v
```

## 10. Recommended Next Steps

Retry after Yahoo rate limiting clears; freeze a dated market dataset and independently rerun full out-of-sample evaluation.
