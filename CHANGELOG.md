# Changelog

## 0.3.0
### Added
- **Live lab** tab: a self-refreshing crash simulator on virtual credits (97% RTP model) with auto cash-out, balance chart, history and a session loss limit.
- **Fixtures** tab: clickable upcoming matches that open a probability and scoreline preview; optional fixtures CSV upload.
- Sidebar **profile**: display name and session loss limit.
- `site/index.html`: a standalone, shareable static dashboard (no install needed), with a GitHub Pages deploy workflow.
- `sample_crash_points()` and `load_fixtures()` in the library; demo `data/sample_fixtures.csv` with relative dates.
- Root-level CI (lint, tests, dashboard smoke test on Python 3.10 and 3.12).


## 0.2.0
### Added
- AviatorLab: Wilson intervals, fair-game (RTP/m) comparison, cash-out RTP and house-edge estimate, permutation independence test, risk-of-ruin metric.
- MatchWise: shrinkage, `predict_proba`, unseen-team flag, log loss, calibration table, frequency baseline, walk-forward backtest with bootstrap Brier skill.
- `loaders` module with validated CSV ingestion (size cap, tolerant of `2.35x`, rejects self-play and negative goals).
- Dashboard: backtest and calibration view, equity curve, Monte Carlo ruin, session-limit planner, cached computation.
- Reproducible synthetic datasets (500 rounds, 60 matches) and generator script.
- Ruff configuration, dev requirements, CI workflow, changelog.

### Changed
- `simulate_random_bankroll` is vectorised (orders of magnitude faster).
- `CashoutSimulation` gains `bankroll_path` and `stopped_early`.
- `pytest` moved out of runtime requirements; Streamlit requirement raised to >=1.50 (`width="stretch"`).

### Fixed
- CSV uploads that fail to parse no longer crash the dashboard.
- Chronological sorting is stable for matches on the same date.
