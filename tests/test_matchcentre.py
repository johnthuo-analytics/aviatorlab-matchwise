import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aviatorlab_matchwise.matchcentre import (
    build_competition,
    build_payload,
    head_to_head,
    normalise_match,
    team_form,
)

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)
TEAMS = ["Alpha", "Bravo", "Cobra", "Delta"]


def raw(i, home, away, when, status="FINISHED", hg=None, ag=None):
    return {"id": i, "utcDate": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": status,
            "homeTeam": {"name": home}, "awayTeam": {"name": away},
            "score": {"fullTime": {"home": hg, "away": ag}}}


def season():
    matches, i = [], 0
    for _ in range(3):  # 3 round-robins = 36 finished matches; Alpha strong, Delta weak
        for h in TEAMS:
            for a in TEAMS:
                if h != a:
                    i += 1
                    hg = 3 if h == "Alpha" else 0 if h == "Delta" else 1
                    ag = 2 if a == "Alpha" else 0 if a == "Delta" else 1
                    matches.append(raw(i, h, a, NOW - timedelta(days=60 - i), hg=hg, ag=ag))
    matches.append(raw(900, "Alpha", "Delta", NOW + timedelta(days=2), "TIMED"))
    matches.append(raw(901, "Bravo", "Cobra", NOW + timedelta(days=30), "TIMED"))  # beyond horizon
    matches.append(raw(902, "Cobra", "Delta", NOW - timedelta(minutes=30), "IN_PLAY", 1, 0))
    matches.append({"id": 903, "utcDate": "2026-10-25T12:00:00Z", "status": "TIMED",
                    "homeTeam": {"name": None}, "awayTeam": {"name": None}})  # TBD knockout tie
    return matches


def test_normalise_skips_tbd_and_reads_score():
    assert normalise_match({"id": 1, "utcDate": "2026-01-01T00:00:00Z", "homeTeam": {}, "awayTeam": {}}, "PL") is None
    m = normalise_match(raw(1, "A", "B", NOW, hg=2, ag=1), "PL")
    assert (m["hg"], m["ag"], m["comp"]) == (2, 1, "PL")


def test_form_and_head_to_head():
    done = [normalise_match(r, "X") for r in season() if r["status"] == "FINISHED"]
    assert team_form(done, "Alpha") == ["W"] * 5
    assert set(team_form(done, "Delta", venue="away")) <= {"W", "D", "L"}
    h2h = head_to_head(done, "Alpha", "Delta")
    assert h2h["n"] == 5 and h2h["home_w"] + h2h["draws"] + h2h["away_w"] == 5


def test_build_competition_groups_and_analyses():
    view = build_competition("T", season(), NOW)
    assert [m["id"] for m in view["live"]] == [902]
    assert [m["id"] for m in view["upcoming"]] == [900]  # 901 too far ahead, 903 has no teams
    a = view["upcoming"][0]["analysis"]
    assert sum(a["p"]) == pytest.approx(1, abs=0.01) and a["p"][0] > a["p"][2]
    assert a["factors"] and not a["low_sample"] and len(view["recent"]) == 10


def test_no_model_when_too_few_finished_matches():
    few = [raw(1, "A", "B", NOW - timedelta(days=3), hg=1, ag=0), raw(2, "A", "B", NOW + timedelta(days=1), "TIMED")]
    assert build_competition("T", few, NOW)["upcoming"][0]["analysis"] is None


def test_unseen_team_gets_no_analysis():
    data = season() + [raw(950, "Alpha", "Newcomer", NOW + timedelta(days=1), "TIMED")]
    up = {m["id"]: m for m in build_competition("T", data, NOW)["upcoming"]}
    assert up[950]["analysis"] is None and up[900]["analysis"] is not None


def test_payload_is_json_serialisable_and_labelled():
    payload = build_payload({"PL": season()}, NOW, ["CL: HTTP 429"])
    text = json.dumps(payload)
    assert payload["delayed"] is True and payload["competitions"]["PL"]["url"].startswith("https://")
    assert payload["errors"] == ["CL: HTTP 429"] and "generated_at" in json.loads(text)


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script_without_key_keeps_demo(monkeypatch, tmp_path):
    mod = load_script("fetch_matches")
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    monkeypatch.setattr(mod, "OUT", tmp_path / "m.json")
    assert mod.main() == 0 and not (tmp_path / "m.json").exists()


def test_fetch_script_partial_and_total_failure(monkeypatch, tmp_path):
    import urllib.error

    mod = load_script("fetch_matches")
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "x")
    monkeypatch.setattr(mod, "OUT", tmp_path / "m.json")
    monkeypatch.setattr(mod, "PAUSE_SECONDS", 0)

    def partial(code, key):
        if code != "PL":
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return season()

    monkeypatch.setattr(mod, "fetch", partial)
    assert mod.main() == 0
    out = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert list(out["competitions"]) == ["PL"] and len(out["errors"]) == 4

    def broken(code, key):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(mod, "fetch", broken)
    assert mod.main() == 1  # job fails, so the previous deployment stays live
