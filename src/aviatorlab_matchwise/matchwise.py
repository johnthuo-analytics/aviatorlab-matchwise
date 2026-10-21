"""Transparent Poisson football baseline and out-of-sample evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

REQUIRED_COLUMNS = {"home_team", "away_team", "home_goals", "away_goals"}


def validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = matches.copy()
    if frame.empty:
        raise ValueError("At least one match is required.")
    for col in ("home_goals", "away_goals"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[col].isna().any() or (frame[col] < 0).any():
            raise ValueError(f"{col} must contain non-negative goal counts.")
    return frame


class PoissonMatchModel:
    """Independent-Poisson baseline from team attack/defence averages.

    Team rates are *shrunk* toward the league average: a team with few games is
    pulled toward 1.0 (average) as if it had ``shrinkage`` extra average games.
    This stops one lucky 5-0 from producing extreme forecasts. Set
    ``shrinkage=0`` to recover raw averages.
    """

    def __init__(self, max_goals: int = 8, shrinkage: float = 4.0):
        if max_goals < 1:
            raise ValueError("max_goals must be at least 1.")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative.")
        self.max_goals = max_goals
        self.shrinkage = shrinkage
        self.team_rates: dict[str, tuple[float, float]] | None = None
        self.league_home_mean = 1.45
        self.league_away_mean = 1.15

    @property
    def known_teams(self) -> set[str]:
        return set(self.team_rates or {})

    def fit(self, matches: pd.DataFrame) -> PoissonMatchModel:
        frame = validate_matches(matches)
        frame = frame.assign(home_team=frame.home_team.astype(str), away_team=frame.away_team.astype(str))
        self.league_home_mean = max(float(frame.home_goals.mean()), 0.05)
        self.league_away_mean = max(float(frame.away_goals.mean()), 0.05)
        league_avg = max((self.league_home_mean + self.league_away_mean) / 2, 0.05)
        k = self.shrinkage
        rates = {}
        for team in set(frame.home_team) | set(frame.away_team):
            home = frame[frame.home_team == team]
            away = frame[frame.away_team == team]
            games = len(home) + len(away)
            scored = float(home.home_goals.sum() + away.away_goals.sum())
            conceded = float(home.away_goals.sum() + away.home_goals.sum())
            # (scored + k * league_avg) / (games + k) is the shrunk goals/game.
            attack = (scored + k * league_avg) / (games + k) / league_avg
            defence = (conceded + k * league_avg) / (games + k) / league_avg
            rates[team] = (max(attack, 0.05), max(defence, 0.05))
        self.team_rates = rates
        return self

    def predict_match(self, home_team: str, away_team: str) -> dict:
        if self.team_rates is None:
            raise RuntimeError("Call fit() before predict_match().")
        ha, hd = self.team_rates.get(home_team, (1.0, 1.0))
        aa, ad = self.team_rates.get(away_team, (1.0, 1.0))
        h_lambda = max(0.05, self.league_home_mean * ha * ad)
        a_lambda = max(0.05, self.league_away_mean * aa * hd)
        goals = np.arange(self.max_goals + 1)
        matrix = np.outer(poisson.pmf(goals, h_lambda), poisson.pmf(goals, a_lambda))
        matrix /= matrix.sum()
        probs = np.array([np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()])
        probs /= probs.sum()
        return {"home_team": home_team, "away_team": away_team,
                "home_win": float(probs[0]), "draw": float(probs[1]), "away_win": float(probs[2]),
                "home_goal_mean": float(h_lambda), "away_goal_mean": float(a_lambda),
                "unseen_teams": [t for t in (home_team, away_team) if t not in self.team_rates],
                "score_matrix": matrix}

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """n×3 array of [home, draw, away] probabilities for each fixture row."""
        rows = [self.predict_match(str(h), str(a)) for h, a in zip(fixtures.home_team, fixtures.away_team, strict=True)]
        return np.array([[r["home_win"], r["draw"], r["away_win"]] for r in rows])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def outcome_label(home_goals: int, away_goals: int) -> int:
    """Outcome labels: 0 home win, 1 draw, 2 away win."""
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def _check_probs(probabilities, outcomes) -> tuple[np.ndarray, np.ndarray]:
    probs, y = np.asarray(probabilities, float), np.asarray(outcomes, int)
    if probs.ndim != 2 or probs.shape[1] != 3 or len(probs) != len(y) or len(y) == 0:
        raise ValueError("Expected n×3 probabilities and n outcome labels.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("Probabilities must be finite and non-negative.")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1.")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("Outcome labels must be 0, 1, or 2.")
    return probs, y


def multiclass_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean multiclass Brier score (0 is perfect; 2 is worst)."""
    probs, y = _check_probs(probabilities, outcomes)
    return float(np.mean(np.sum((probs - np.eye(3)[y]) ** 2, axis=1)))


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the observed outcomes."""
    probs, y = _check_probs(probabilities, outcomes)
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], eps, 1.0))))


def calibration_table(probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 5) -> pd.DataFrame:
    """Pooled reliability table: predicted probability bins vs observed frequency.

    Every (match, outcome) pair contributes one predicted probability and one
    0/1 result. A well-calibrated model has ``observed`` close to ``predicted``.
    """
    probs, y = _check_probs(probabilities, outcomes)
    if bins < 2:
        raise ValueError("bins must be at least 2.")
    p = probs.ravel()
    hit = np.eye(3)[y].ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": f"{edges[b]:.0%}–{edges[b + 1]:.0%}", "predicted": float(p[mask].mean()),
                         "observed": float(hit[mask].mean()), "count": int(mask.sum())})
    return pd.DataFrame(rows)


def frequency_baseline(train_outcomes: np.ndarray, n: int) -> np.ndarray:
    """Naive benchmark: always predict the training-set outcome frequencies."""
    counts = np.bincount(np.asarray(train_outcomes, int), minlength=3).astype(float)
    freq = (counts + 1) / (counts.sum() + 3)  # Laplace smoothing
    return np.tile(freq, (n, 1))


# --------------------------------------------------------------------------- #
# Splitting and backtesting
# --------------------------------------------------------------------------- #
def chronological_split(matches: pd.DataFrame, test_fraction: float = 0.2):
    """Return earlier training matches and later test matches, sorted by date."""
    if "date" not in matches.columns:
        raise ValueError("A 'date' column is required for chronological splitting.")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    frame = validate_matches(matches)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    cut = int(len(frame) * (1 - test_fraction))
    if cut < 1 or cut >= len(frame):
        raise ValueError("Not enough rows for the requested split.")
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def walk_forward_evaluation(matches: pd.DataFrame, min_train: int = 20, step: int = 1,
                            shrinkage: float = 4.0) -> dict:
    """Expanding-window backtest: fit on the past, predict the next ``step`` matches.

    Compares the Poisson model with a frequency baseline using Brier score and
    log loss. ``brier_skill`` > 0 means the model beat the baseline; it is
    reported with a bootstrap interval so small samples are not over-read.
    """
    if "date" not in matches.columns:
        raise ValueError("A 'date' column is required for walk-forward evaluation.")
    if min_train < 1 or step < 1:
        raise ValueError("min_train and step must be positive.")
    frame = validate_matches(matches)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    if len(frame) <= min_train:
        raise ValueError(f"Need more than min_train={min_train} matches; got {len(frame)}.")
    labels = np.array([outcome_label(h, a) for h, a in zip(frame.home_goals, frame.away_goals, strict=True)])
    model_p, base_p, idx = [], [], []
    for start in range(min_train, len(frame), step):
        stop = min(start + step, len(frame))
        model = PoissonMatchModel(shrinkage=shrinkage).fit(frame.iloc[:start])
        model_p.append(model.predict_proba(frame.iloc[start:stop]))
        base_p.append(frequency_baseline(labels[:start], stop - start))
        idx.extend(range(start, stop))
    model_p, base_p = np.vstack(model_p), np.vstack(base_p)
    y = labels[idx]
    brier_m, brier_b = multiclass_brier_score(model_p, y), multiclass_brier_score(base_p, y)
    per_m = np.sum((model_p - np.eye(3)[y]) ** 2, axis=1)
    per_b = np.sum((base_p - np.eye(3)[y]) ** 2, axis=1)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        s = rng.integers(0, len(y), len(y))
        boots.append(1 - per_m[s].mean() / per_b[s].mean())
    return {
        "n_test_matches": int(len(y)), "brier_model": brier_m, "brier_baseline": brier_b,
        "brier_skill": float(1 - brier_m / brier_b),
        "brier_skill_interval": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "log_loss_model": log_loss(model_p, y), "log_loss_baseline": log_loss(base_p, y),
        "calibration": calibration_table(model_p, y), "probabilities": model_p, "outcomes": y,
    }
