# Changelog

## 0.4.0
### Added
- **Match centre** on the site: in-play (delayed), upcoming and recent matches for the Premier League, Champions League, La Liga, Serie A and Bundesliga, filterable by competition.
- **Deep analysis** for every upcoming match: model win/draw/loss probabilities, expected goals, most likely score, last-5 form (overall and at home/away), head-to-head, and plain-language key factors, with a "where to watch" link to the official competition site.
- `matchcentre.py` (pure, tested logic), `scripts/fetch_matches.py` (football-data.org free plan, rate-limit aware) and `scripts/build_demo_matches.py`.
- Scheduled workflow refreshes the data every 20 minutes and redeploys Pages; the API key lives in the `FOOTBALL_DATA_API_KEY` repository secret and never reaches the page.
- Demo match data (fictional teams) ships by default so the site works before a key is added.

### Changed
- The old demo-only Fixtures tab is now the Match centre. Workflow runner pinned to ubuntu-24.04.


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
