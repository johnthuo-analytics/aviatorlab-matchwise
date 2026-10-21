"""Descriptive crash-multiplier analysis and hypothetical risk simulations.

This module deliberately does **not** predict the next round. Instead it offers
tools that show *why* that is not possible:

* :func:`fair_game_comparison` checks a sample against the survival curve of a
  memoryless crash game with a fixed house edge, ``P(crash >= m) = RTP / m``.
* :func:`independence_test` checks whether consecutive rounds are correlated.
* :func:`estimate_cashout_rtp` shows that every fixed cash-out target has the
  same expected return, so no target is "smarter" than another.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

DEFAULT_THRESHOLDS = (1.5, 2.0, 5.0, 10.0)
DEFAULT_RTP = 0.97


# --------------------------------------------------------------------------- #
# Validation and basic statistics
# --------------------------------------------------------------------------- #
def validate_multipliers(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Provide at least one multiplier.")
    if not np.all(np.isfinite(arr)) or np.any(arr < 1.0):
        raise ValueError("Multipliers must be finite numbers >= 1.0.")
    return arr


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default ~95%)."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n.")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def summarize_multipliers(values: Iterable[float], thresholds: Sequence[float] = DEFAULT_THRESHOLDS) -> dict:
    """Descriptive statistics, with Wilson intervals on each threshold rate."""
    arr = validate_multipliers(values)
    n = int(arr.size)
    rates, intervals = {}, {}
    for t in thresholds:
        hits = int(np.sum(arr >= t))
        rates[float(t)] = hits / n
        intervals[float(t)] = wilson_interval(hits, n)
    return {
        "n_rounds": n,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "threshold_rates": rates,
        "threshold_intervals": intervals,
    }


# --------------------------------------------------------------------------- #
# Fair-game and independence checks
# --------------------------------------------------------------------------- #
def theoretical_survival(multiplier: float, rtp: float = DEFAULT_RTP) -> float:
    """P(crash >= m) for a memoryless crash game with return-to-player ``rtp``."""
    if not 0 < rtp <= 1:
        raise ValueError("rtp must be in (0, 1].")
    if multiplier < 1:
        raise ValueError("multiplier must be >= 1.")
    return float(min(1.0, rtp / multiplier))


def estimate_cashout_rtp(values: Iterable[float], cashout: float) -> dict:
    """Empirical return-to-player of always cashing out at ``cashout``.

    Expected return per unit staked is ``cashout * P(multiplier >= cashout)``.
    Returns the point estimate, a ~95% interval, and the implied house edge.
    """
    arr = validate_multipliers(values)
    if cashout <= 1:
        raise ValueError("cashout must be > 1.")
    n = int(arr.size)
    hits = int(np.sum(arr >= cashout))
    lo, hi = wilson_interval(hits, n)
    rtp = cashout * hits / n
    return {
        "cashout": float(cashout),
        "n_rounds": n,
        "win_rate": hits / n,
        "estimated_rtp": float(rtp),
        "rtp_interval": (float(cashout * lo), float(cashout * hi)),
        "implied_house_edge": float(1 - rtp),
    }


def fair_game_comparison(values: Iterable[float], thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
                         rtp: float = DEFAULT_RTP) -> list[dict]:
    """Compare observed threshold rates with a memoryless ``rtp / m`` model.

    ``consistent`` is True when the model value lies inside the Wilson interval.
    Consistency is *not* proof of fairness; it only means the sample cannot
    distinguish the data from this model.
    """
    arr = validate_multipliers(values)
    n = int(arr.size)
    rows = []
    for t in thresholds:
        hits = int(np.sum(arr >= t))
        lo, hi = wilson_interval(hits, n)
        expected = theoretical_survival(t, rtp)
        rows.append({"threshold": float(t), "observed": hits / n, "expected": expected,
                     "ci_low": lo, "ci_high": hi, "consistent": lo <= expected <= hi})
    return rows


def independence_test(values: Iterable[float], n_permutations: int = 2000, seed: int = 42) -> dict:
    """Lag-1 autocorrelation of log-multipliers with a permutation p-value.

    A large p-value means there is no evidence that one round carries
    information about the next. A small p-value would be unusual and worth
    investigating (data errors, duplicated rows, sorted data).
    """
    arr = validate_multipliers(values)
    if arr.size < 10:
        raise ValueError("At least 10 rounds are needed for an independence test.")
    if n_permutations < 100:
        raise ValueError("n_permutations must be at least 100.")
    x = np.log(arr)

    def lag1(v: np.ndarray) -> float:
        a, b = v[:-1] - v.mean(), v[1:] - v.mean()
        denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
        return 0.0 if denom == 0 else float(np.sum(a * b) / denom)

    observed = lag1(x)
    rng = np.random.default_rng(seed)
    perm = np.array([lag1(rng.permutation(x)) for _ in range(n_permutations)])
    p_value = float((1 + np.sum(np.abs(perm) >= abs(observed))) / (n_permutations + 1))
    return {"lag1_autocorrelation": observed, "p_value": p_value, "n_rounds": int(arr.size),
            "n_permutations": int(n_permutations)}


# --------------------------------------------------------------------------- #
# Simulations
# --------------------------------------------------------------------------- #
def sample_crash_points(n: int, rtp: float = DEFAULT_RTP, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw ``n`` crash multipliers from a memoryless model with return-to-player ``rtp``.

    ``P(crash >= m) = rtp / m``, truncated to 2 decimals and floored at 1.00x.
    Used by the live simulator; rounds are independent by construction.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 < rtp <= 1:
        raise ValueError("rtp must be in (0, 1].")
    rng = rng if rng is not None else np.random.default_rng()
    u = 1 - rng.random(n)  # (0, 1]
    return np.maximum(1.0, np.floor(rtp / u * 100) / 100)



@dataclass(frozen=True)
class CashoutSimulation:
    rounds: int
    stake_per_round: float
    cashout_multiplier: float
    starting_bankroll: float
    ending_bankroll: float
    net_profit_loss: float
    winning_rounds: int
    losing_rounds: int
    max_drawdown: float
    bankroll_path: tuple = field(default_factory=tuple, repr=False)
    stopped_early: bool = False


def simulate_historical_cashout(values: Iterable[float], cashout_multiplier: float = 2.0,
                                stake_per_round: float = 10.0,
                                starting_bankroll: float = 1000.0) -> CashoutSimulation:
    """Replay a fixed cash-out rule on supplied history; not a forecast.

    Simplified rule: multiplier >= target earns stake*(target-1), otherwise
    the stake is lost. Real game rules may differ. Play stops when the bankroll
    can no longer cover a stake (``stopped_early``).
    """
    arr = validate_multipliers(values)
    if cashout_multiplier <= 1:
        raise ValueError("cashout_multiplier must be > 1.")
    if stake_per_round <= 0 or starting_bankroll < 0:
        raise ValueError("Stake must be positive and bankroll non-negative.")
    bankroll = float(starting_bankroll)
    peak = bankroll
    max_drawdown = 0.0
    wins = losses = 0
    path = [bankroll]
    stopped_early = False
    for multiplier in arr:
        if bankroll < stake_per_round:
            stopped_early = True
            break
        if multiplier >= cashout_multiplier:
            bankroll += stake_per_round * (cashout_multiplier - 1)
            wins += 1
        else:
            bankroll -= stake_per_round
            losses += 1
        path.append(bankroll)
        peak = max(peak, bankroll)
        if peak > 0:
            max_drawdown = max(max_drawdown, (peak - bankroll) / peak)
    return CashoutSimulation(wins + losses, float(stake_per_round), float(cashout_multiplier),
                             float(starting_bankroll), float(bankroll),
                             float(bankroll - starting_bankroll), wins, losses,
                             float(max_drawdown), tuple(path), stopped_early)


def simulate_random_bankroll(n_rounds: int, n_simulations: int, win_probability: float,
                             cashout_multiplier: float, stake: float = 10.0,
                             starting_bankroll: float = 1000.0, seed: int = 42) -> dict:
    """Vectorised bankroll simulation under an *assumed* win probability.

    Every path stops once its bankroll cannot cover another stake ("ruin").
    """
    if n_rounds <= 0 or n_simulations <= 0:
        raise ValueError("n_rounds and n_simulations must be positive.")
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be between 0 and 1.")
    if cashout_multiplier <= 1 or stake <= 0 or starting_bankroll < 0:
        raise ValueError("Invalid multiplier, stake, or bankroll.")
    rng = np.random.default_rng(seed)
    wins = rng.random((n_simulations, n_rounds)) < win_probability
    bankroll = np.full(n_simulations, float(starting_bankroll))
    ruined = np.zeros(n_simulations, dtype=bool)
    gain = stake * (cashout_multiplier - 1)
    for t in range(n_rounds):
        ruined |= bankroll < stake
        active = ~ruined
        bankroll = np.where(active, bankroll + np.where(wins[:, t], gain, -stake), bankroll)
    ruined |= bankroll < stake
    return {
        "median_ending_bankroll": float(np.median(bankroll)),
        "mean_ending_bankroll": float(np.mean(bankroll)),
        "probability_of_finishing_below_start": float(np.mean(bankroll < starting_bankroll)),
        "probability_of_losing_at_least_half": float(np.mean(bankroll <= starting_bankroll * 0.5)),
        "probability_of_ruin": float(np.mean(ruined)),
        "ending_bankrolls": bankroll,
    }
