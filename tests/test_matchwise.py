import numpy as np
import pandas as pd
import pytest

from aviatorlab_matchwise.matchwise import (
    PoissonMatchModel,
    calibration_table,
    chronological_split,
    frequency_baseline,
    log_loss,
    multiclass_brier_score,
    outcome_label,
    walk_forward_evaluation,
)


def sample_data():
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=8, freq="7D"),
        "home_team": ["A", "B", "A", "C", "B", "C", "A", "B"],
        "away_team": ["B", "C", "C", "A", "A", "B", "B", "C"],
        "home_goals": [2, 1, 0, 1, 3, 0, 2, 1], "away_goals": [1, 1, 2, 0, 0, 1, 0, 2]})


def league(n_rounds=6, seed=3):
    rng = np.random.default_rng(seed)
    strength = {"S": (1.6, 0.6), "M": (1.0, 1.0), "W": (0.6, 1.5)}
    rows = []
    for _ in range(n_rounds):
        for h in strength:
            for a in strength:
                if h != a:
                    rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=len(rows)),
                                 "home_team": h, "away_team": a,
                                 "home_goals": rng.poisson(1.4 * strength[h][0] * strength[a][1]),
                                 "away_goals": rng.poisson(1.1 * strength[a][0] * strength[h][1])})
    return pd.DataFrame(rows)


def test_probabilities_sum_to_one():
    result = PoissonMatchModel().fit(sample_data()).predict_match("A", "B")
    assert np.isclose(result["home_win"] + result["draw"] + result["away_win"], 1.0)
    assert result["score_matrix"].shape == (9, 9)


def test_unseen_team_flagged():
    result = PoissonMatchModel().fit(sample_data()).predict_match("A", "Z")
    assert result["unseen_teams"] == ["Z"]


def test_shrinkage_pulls_rates_toward_average():
    df = pd.DataFrame({"home_team": ["A"], "away_team": ["B"], "home_goals": [6], "away_goals": [0]})
    raw = PoissonMatchModel(shrinkage=0).fit(df).team_rates["A"][0]
    shrunk = PoissonMatchModel(shrinkage=5).fit(df).team_rates["A"][0]
    assert abs(shrunk - 1.0) < abs(raw - 1.0)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        PoissonMatchModel(shrinkage=-1)
    with pytest.raises(ValueError):
        PoissonMatchModel(max_goals=0)


def test_strong_team_favoured():
    model = PoissonMatchModel().fit(league())
    p = model.predict_match("S", "W")
    assert p["home_win"] > p["away_win"]
    assert model.predict_proba(pd.DataFrame({"home_team": ["S", "W"], "away_team": ["W", "S"]})).shape == (2, 3)


def test_perfect_brier_score():
    assert multiclass_brier_score(np.eye(3), np.array([0, 1, 2])) == 0.0


def test_brier_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        multiclass_brier_score(np.array([[.2, .2, .2]]), np.array([0]))


def test_log_loss_and_baseline():
    assert log_loss(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0, abs=1e-9)
    base = frequency_baseline(np.array([0, 0, 1, 2]), 3)
    assert base.shape == (3, 3) and np.allclose(base.sum(axis=1), 1)


def test_calibration_table_perfect_forecasts():
    cal = calibration_table(np.eye(3), np.array([0, 1, 2]))
    assert (cal.predicted - cal.observed).abs().max() < 1e-9
    assert cal["count"].sum() == 9


def test_chronological_split():
    train, test = chronological_split(sample_data(), .25)
    assert train.date.max() < test.date.min()
    assert len(train) == 6 and len(test) == 2


def test_walk_forward_no_lookahead_and_beats_baseline():
    df = league()
    res = walk_forward_evaluation(df, min_train=12)
    assert res["n_test_matches"] == len(df) - 12
    assert res["brier_model"] < res["brier_baseline"]
    assert res["brier_skill"] > 0
    lo, hi = res["brier_skill_interval"]
    assert lo <= res["brier_skill"] <= hi
    with pytest.raises(ValueError):
        walk_forward_evaluation(df, min_train=len(df))


def test_outcome_labels():
    assert outcome_label(2, 1) == 0
    assert outcome_label(1, 1) == 1
    assert outcome_label(0, 2) == 2
