# AviatorLab + MatchWise

Open-source **educational** toolkit that shows what statistics can and cannot tell you about crash-style betting games and football outcomes, alongside responsible-play guidance.

> **Important:** Aviator history cannot reliably predict the next crash point. MatchWise probabilities are uncertain estimates, not guaranteed outcomes or betting advice. This project does not place bets, connect to betting accounts, or promise profits.

## What it does

**AviatorLab**
- Descriptive statistics with 95% Wilson intervals on every threshold rate.
- **Fair-game check:** compares your sample with the memoryless model `P(crash ≥ m) = RTP / m`.
- **House-edge estimate:** shows every fixed cash-out target has the same expected return, so no target is "smarter".
- **Independence test:** lag-1 autocorrelation with a permutation p-value. Past rounds carrying no information is the point.
- Hypothetical cash-out replay with bankroll path and drawdown.
- Vectorised risk-of-ruin Monte Carlo under an explicitly assumed win rate.

**MatchWise**
- Transparent independent-Poisson baseline with **shrinkage** so small samples do not produce extreme ratings.
- **Walk-forward backtest** (fit on the past, predict the next match) against a naive frequency baseline.
- Brier score, log loss, Brier skill score with bootstrap interval, and a calibration table.
- Flags teams the model has never seen.

**Dashboard (Streamlit):** Live lab (self-refreshing crash simulator on virtual credits), clickable fixtures with match previews, profile and loss limit, safe CSV uploads, and a session-limit planner.

**Shareable site:** `site/index.html` is a self-contained web dashboard (no install, no server). Enable it with GitHub Pages (below) and share the link.

## Quick start

```bash
git clone https://github.com/<your-username>/aviatorlab-matchwise.git
cd aviatorlab-matchwise

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q
streamlit run app.py
```

Requires Python 3.10+. To use the library without the dashboard: `pip install -e .`

## Share it as a website
1. Push the repo to GitHub.
2. In the repo go to **Settings → Pages** and set **Source** to **GitHub Actions**.
3. The **Deploy site** workflow publishes `site/` at `https://<your-username>.github.io/aviatorlab-matchwise/`.

To run the Streamlit app online instead, deploy the repo on Streamlit Community Cloud with `app.py` as the entry point.

## Use as a library

```python
from aviatorlab_matchwise.loaders import load_multipliers, load_matches
from aviatorlab_matchwise.aviator import estimate_cashout_rtp, independence_test
from aviatorlab_matchwise.matchwise import walk_forward_evaluation

values = load_multipliers("data/sample_aviator_multipliers.csv")
print(estimate_cashout_rtp(values, cashout=2.0))   # RTP estimate with interval
print(independence_test(values))                   # is there any serial structure?

result = walk_forward_evaluation(load_matches("data/sample_matches.csv"), min_train=20)
print(result["brier_skill"], result["brier_skill_interval"])
```

## Data formats

- Aviator CSV: one column named `multiplier` (values like `2.35x` are accepted).
- Football CSV: `date,home_team,away_team,home_goals,away_goals` (`date` is needed for backtesting).
- Fixtures CSV: `home_team,away_team` plus `date` (or `day_offset`), optional `kickoff` (HH:MM).

Only use data you are authorized to access. Record its source, licence, collection date, coverage and limitations. Do not publish API keys or personal/account data. The included samples are **synthetic** (see `scripts/generate_sample_data.py`; seed 2025): 500 rounds from a 97%-RTP model and 60 matches between 6 fictional teams. They are not live Betika or current sports data.

## Reading the results honestly

- A sample of a few hundred rounds is **noisy**. A 2x strategy can look break-even or profitable in-sample while the true expected return is the house's edge. Check the interval, not just the point estimate.
- A Brier skill score whose interval spans zero means the sample cannot separate model skill from luck.
- Sort chronologically, fit only on earlier matches, never tune on the test set. A good historical fit does not prove future predictive value.

## Responsible-play principles

Do not chase losses or borrow to gamble. Set limits in advance and stop when reached. A winning streak does not guarantee another win. If gambling is affecting finances, work, relationships, or wellbeing, take a break and seek appropriate support.

## Repository structure

```text
aviatorlab-matchwise/
├── app.py                      # Streamlit dashboard
├── site/index.html             # standalone shareable web dashboard
├── data/                       # synthetic sample CSVs (matches, multipliers, fixtures)
├── .github/workflows/          # CI and GitHub Pages deploy
├── scripts/generate_sample_data.py
├── src/aviatorlab_matchwise/
│   ├── aviator.py              # stats, fair-game checks, simulations
│   ├── matchwise.py            # Poisson model, metrics, backtesting
│   └── loaders.py              # validated CSV loading
├── tests/
├── CHANGELOG.md
├── requirements.txt / requirements-dev.txt
└── pyproject.toml
```

## License

MIT; see `LICENSE`.
