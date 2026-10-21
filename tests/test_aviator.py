import numpy as np
import pytest

from aviatorlab_matchwise.aviator import (
    estimate_cashout_rtp,
    fair_game_comparison,
    independence_test,
    simulate_historical_cashout,
    simulate_random_bankroll,
    summarize_multipliers,
    theoretical_survival,
    validate_multipliers,
    wilson_interval,
)


def fair_sample(n=20000, rtp=0.97, seed=1):
    u = 1 - np.random.default_rng(seed).random(n)
    return np.maximum(1.0, np.floor(rtp / u * 100) / 100)


def test_summary_empirical_rates():
    result = summarize_multipliers([1.0, 2.0, 5.0, 10.0])
    assert result["n_rounds"] == 4
    assert result["threshold_rates"][2.0] == 0.75
    assert result["median"] == 3.5
    lo, hi = result["threshold_intervals"][2.0]
    assert lo < 0.75 < hi


def test_invalid_multipliers_rejected():
    with pytest.raises(ValueError):
        validate_multipliers([1.0, 0.9])
    with pytest.raises(ValueError):
        validate_multipliers([])
    with pytest.raises(ValueError):
        validate_multipliers([1.0, float("nan")])


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0 and 0 < lo < 1
    with pytest.raises(ValueError):
        wilson_interval(5, 0)


def test_historical_simulation_math():
    result = simulate_historical_cashout([2.0, 1.0, 3.0], 2.0, 10.0, 100.0)
    assert result.winning_rounds == 2 and result.losing_rounds == 1
    assert result.ending_bankroll == 110.0
    assert result.net_profit_loss == 10.0
    assert result.bankroll_path == (100.0, 110.0, 100.0, 110.0)
    assert not result.stopped_early


def test_historical_simulation_stops_when_broke():
    result = simulate_historical_cashout([1.0] * 5, 2.0, 10.0, 25.0)
    assert result.losing_rounds == 2 and result.stopped_early
    assert result.ending_bankroll == 5.0
    assert result.max_drawdown == pytest.approx(0.8)


def test_simulation_reproducible_and_vectorised_semantics():
    a = simulate_random_bankroll(20, 50, 0.4, 2.0, seed=7)
    b = simulate_random_bankroll(20, 50, 0.4, 2.0, seed=7)
    assert a["median_ending_bankroll"] == b["median_ending_bankroll"]
    assert len(a["ending_bankrolls"]) == 50


def test_random_simulation_extremes():
    never = simulate_random_bankroll(50, 20, 0.0, 2.0, stake=10, starting_bankroll=100)
    assert never["probability_of_ruin"] == 1.0
    assert np.all(never["ending_bankrolls"] == 0.0)
    always = simulate_random_bankroll(10, 20, 1.0, 2.0, stake=10, starting_bankroll=100)
    assert np.all(always["ending_bankrolls"] == 200.0)
    assert always["probability_of_ruin"] == 0.0


def test_theoretical_survival():
    assert theoretical_survival(2.0, 0.97) == pytest.approx(0.485)
    assert theoretical_survival(1.0, 0.97) == 0.97
    with pytest.raises(ValueError):
        theoretical_survival(0.5)


def test_fair_sample_matches_model_and_rtp():
    values = fair_sample()
    rows = fair_game_comparison(values)
    assert sum(r["consistent"] for r in rows) >= 3
    for cashout in (1.5, 2.0, 5.0):
        est = estimate_cashout_rtp(values, cashout)
        assert est["estimated_rtp"] == pytest.approx(0.97, abs=0.05)
        assert est["implied_house_edge"] == pytest.approx(0.03, abs=0.05)


def test_independence_test_on_iid_and_trending_data():
    assert independence_test(fair_sample(500), n_permutations=500)["p_value"] > 0.01
    trending = np.linspace(1.0, 5.0, 200)
    res = independence_test(trending, n_permutations=500)
    assert res["lag1_autocorrelation"] > 0.9 and res["p_value"] < 0.01
    with pytest.raises(ValueError):
        independence_test([1.0, 2.0])


def test_sample_crash_points_distribution_and_validation():
    from aviatorlab_matchwise.aviator import sample_crash_points

    pts = sample_crash_points(50000, rng=np.random.default_rng(5))
    assert pts.min() >= 1.0
    assert (pts >= 2.0).mean() == pytest.approx(0.485, abs=0.01)
    assert len(sample_crash_points(1)) == 1
    with pytest.raises(ValueError):
        sample_crash_points(0)
    with pytest.raises(ValueError):
        sample_crash_points(5, rtp=1.5)
