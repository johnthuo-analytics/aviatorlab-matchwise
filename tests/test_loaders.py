import io

import pytest

from aviatorlab_matchwise.loaders import load_matches, load_multipliers


def test_load_multipliers_ok_and_tolerant():
    assert load_multipliers(io.StringIO("Multiplier\n1.5\n2.35x\n\n")) == [1.5, 2.35]


@pytest.mark.parametrize("text", ["foo\n1.5\n", "multiplier\nabc\n", "multiplier\n0.5\n", "multiplier\n", ""])
def test_load_multipliers_rejects_bad_input(text):
    with pytest.raises(ValueError):
        load_multipliers(io.StringIO(text))


def test_load_matches_ok():
    df = load_matches(io.StringIO("date,home_team,away_team,home_goals,away_goals\n2025-01-01, A ,B,1,0\n"))
    assert df.home_team.iloc[0] == "A"


def test_load_matches_rejects_self_play_and_negative_goals():
    with pytest.raises(ValueError):
        load_matches(io.StringIO("home_team,away_team,home_goals,away_goals\nA,A,1,0\n"))
    with pytest.raises(ValueError):
        load_matches(io.StringIO("home_team,away_team,home_goals,away_goals\nA,B,-1,0\n"))


def test_sample_files_load():
    from pathlib import Path
    data = Path(__file__).parents[1] / "data"
    assert len(load_multipliers(data / "sample_aviator_multipliers.csv")) == 500
    assert len(load_matches(data / "sample_matches.csv")) == 60


def test_load_fixtures_relative_dates_and_validation():
    import pandas as pd

    from aviatorlab_matchwise.loaders import load_fixtures

    today = pd.Timestamp("2026-01-10")
    df = load_fixtures(io.StringIO("day_offset,kickoff,home_team,away_team\n2,16:00, A ,B\n1,12:00,C,D\n"), today)
    assert list(df.home_team) == ["C", "A"] and df.date.iloc[0] == pd.Timestamp("2026-01-11")
    with pytest.raises(ValueError):
        load_fixtures(io.StringIO("home_team,away_team\nA,B\n"), today)
    with pytest.raises(ValueError):
        load_fixtures(io.StringIO("day_offset,home_team,away_team\n1,A,A\n"), today)


def test_sample_fixtures_file_loads():
    from pathlib import Path

    from aviatorlab_matchwise.loaders import load_fixtures

    assert len(load_fixtures(Path(__file__).parents[1] / "data" / "sample_fixtures.csv")) == 9
