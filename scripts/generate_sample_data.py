"""Regenerate the synthetic sample datasets (fully reproducible, seed=2025).

Aviator: 500 rounds drawn from a memoryless crash model with 97% RTP.
Matches: 6 fictional teams, two double round-robins (60 matches) from known
attack/defence strengths. Neither file contains real or live data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
rng = np.random.default_rng(2025)

u = 1 - rng.random(500)  # (0, 1]
mult = np.maximum(1.0, np.floor(0.97 / u * 100) / 100)
pd.DataFrame({"multiplier": np.round(mult, 2)}).to_csv(DATA / "sample_aviator_multipliers.csv", index=False)

teams = {"Harbor FC": (1.25, 0.85), "Valley United": (1.00, 1.00), "Metro City": (1.30, 0.80),
         "River Rovers": (0.85, 1.15), "Lakeside Town": (0.95, 1.05), "Hill Athletic": (0.75, 1.25)}
pairs = [(h, a) for h in teams for a in teams if h != a]
rows, day = [], pd.Timestamp("2025-08-02")
for _ in range(2):
    order = rng.permutation(len(pairs))
    for p in order:
        h, a = pairs[p]
        hg = rng.poisson(1.45 * teams[h][0] * teams[a][1])
        ag = rng.poisson(1.15 * teams[a][0] * teams[h][1])
        rows.append({"date": (day + pd.Timedelta(days=int(len(rows) // 3) * 7 + len(rows) % 3)).date(),
                     "home_team": h, "away_team": a, "home_goals": hg, "away_goals": ag})
pd.DataFrame(rows).to_csv(DATA / "sample_matches.csv", index=False)
print("Wrote", DATA)
