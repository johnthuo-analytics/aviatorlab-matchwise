"""Match-centre builder: turns football-data.org v4 responses into analysed JSON.

Everything here is a pure function (no network), so it is fully unit-testable.
The scheduled workflow fetches raw matches and calls :func:`build_payload`.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from .matchwise import PoissonMatchModel

LIVE = {"IN_PLAY", "PAUSED", "LIVE"}
UPCOMING = {"SCHEDULED", "TIMED"}
DONE = {"FINISHED", "AWARDED"}

# code -> (display name, official competition site used for "where to watch")
COMPETITIONS = {
    "PL": ("Premier League", "https://www.premierleague.com/"),
    "CL": ("UEFA Champions League", "https://www.uefa.com/uefachampionsleague/"),
    "PD": ("La Liga", "https://www.laliga.com/en-GB"),
    "SA": ("Serie A", "https://www.legaseriea.it/en"),
    "BL1": ("Bundesliga", "https://www.bundesliga.com/en/bundesliga"),
}


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def normalise_match(raw: dict, comp: str) -> dict | None:
    """Reduce an API match to the fields we use; None when teams are still TBD."""
    home = (raw.get("homeTeam") or {}).get("name")
    away = (raw.get("awayTeam") or {}).get("name")
    if not home or not away or "id" not in raw or "utcDate" not in raw:
        return None
    full = (raw.get("score") or {}).get("fullTime") or {}
    return {"id": raw["id"], "comp": comp, "utc": raw["utcDate"], "status": raw.get("status", "SCHEDULED"),
            "home": home, "away": away, "hg": full.get("home"), "ag": full.get("away")}


def team_form(done: list[dict], team: str, n: int = 5, venue: str | None = None) -> list[str]:
    """Last ``n`` results (oldest to newest) as W/D/L, optionally home or away only."""
    out = []
    for m in sorted(done, key=lambda x: x["utc"]):
        if team not in (m["home"], m["away"]):
            continue
        if (venue == "home" and m["home"] != team) or (venue == "away" and m["away"] != team):
            continue
        gf, ga = (m["hg"], m["ag"]) if m["home"] == team else (m["ag"], m["hg"])
        out.append("W" if gf > ga else "D" if gf == ga else "L")
    return out[-n:]


def head_to_head(done: list[dict], home: str, away: str, n: int = 5) -> dict:
    """Recent meetings (available data only), from the upcoming home side's view."""
    games = [m for m in sorted(done, key=lambda x: x["utc"]) if {m["home"], m["away"]} == {home, away}][-n:]
    w = d = lost = 0
    for m in games:
        gf, ga = (m["hg"], m["ag"]) if m["home"] == home else (m["ag"], m["hg"])
        if gf > ga:
            w += 1
        elif gf == ga:
            d += 1
        else:
            lost += 1
    return {"n": len(games), "home_w": w, "draws": d, "away_w": lost}


def key_factors(home: str, away: str, pred: dict, model: PoissonMatchModel, form_h: list, form_a: list,
                h2h: dict) -> list[str]:
    """Plain-language reasons derived only from the numbers (no hidden inputs)."""
    ph, pa = pred["home_win"], pred["away_win"]
    out = []
    if abs(ph - pa) < 0.08:
        out.append("Evenly matched on the model: no clear favourite.")
    else:
        fav, p = (home, ph) if ph > pa else (away, pa)
        out.append(f"{fav} are favourites at {p:.0%} to win.")
    for team, form in ((home, form_h), (away, form_a)):
        if len(form) >= 3:
            out.append(f"{team}: {form.count('W')}W {form.count('D')}D {form.count('L')}L in their last {len(form)}.")
    rates = model.team_rates or {}
    if home in rates and away in rates:
        (ha, hd), (aa, ad) = rates[home], rates[away]
        out.append(f"Attack v defence: {home} score at {ha:.2f}x the league rate against a defence conceding "
                   f"{ad:.2f}x; {away} {aa:.2f}x against {hd:.2f}x.")
    total = pred["home_goal_mean"] + pred["away_goal_mean"]
    style = "High-scoring" if total >= 3 else "Low-scoring" if total <= 2.2 else "Moderate-scoring"
    out.append(f"{style} outlook: {total:.1f} expected goals.")
    if h2h["n"]:
        out.append(f"Head-to-head this season: {home} {h2h['home_w']}W {h2h['draws']}D {h2h['away_w']}L "
                   f"in {h2h['n']} meeting(s).")
    return out


def analyse_fixture(model: PoissonMatchModel, done: list[dict], home: str, away: str) -> dict:
    pred = model.predict_match(home, away)
    mat = pred["score_matrix"]
    i, j = divmod(int(mat.argmax()), mat.shape[1])
    fh, fa = team_form(done, home), team_form(done, away)
    h2h = head_to_head(done, home, away)
    games = {t: sum(1 for m in done if t in (m["home"], m["away"])) for t in (home, away)}
    return {
        "p": [round(pred["home_win"], 3), round(pred["draw"], 3), round(pred["away_win"], 3)],
        "xg": [round(pred["home_goal_mean"], 2), round(pred["away_goal_mean"], 2)],
        "likely": [i, j], "likely_p": round(float(mat[i, j]), 3),
        "form": {"home": fh, "away": fa, "home_at_home": team_form(done, home, venue="home"),
                 "away_away": team_form(done, away, venue="away")},
        "h2h": h2h, "factors": key_factors(home, away, pred, model, fh, fa, h2h),
        "low_sample": min(games.values()) < 5,
    }


def build_competition(code: str, raw_matches: list[dict], now: datetime, days_ahead: int = 7,
                      min_finished: int = 10) -> dict:
    norm = [m for m in (normalise_match(r, code) for r in raw_matches) if m]
    done = [m for m in norm if m["status"] in DONE and m["hg"] is not None and m["ag"] is not None]
    live = [m for m in norm if m["status"] in LIVE]
    horizon = now + timedelta(days=days_ahead)
    upcoming = sorted((m for m in norm if m["status"] in UPCOMING
                       and now - timedelta(hours=3) <= parse_utc(m["utc"]) <= horizon), key=lambda m: m["utc"])
    model = None
    if len(done) >= min_finished:
        frame = pd.DataFrame({"home_team": [m["home"] for m in done], "away_team": [m["away"] for m in done],
                              "home_goals": [m["hg"] for m in done], "away_goals": [m["ag"] for m in done]})
        model = PoissonMatchModel().fit(frame)
    for m in upcoming:
        ok = model is not None and {m["home"], m["away"]} <= model.known_teams
        m["analysis"] = analyse_fixture(model, done, m["home"], m["away"]) if ok else None
    recent = sorted(done, key=lambda m: m["utc"], reverse=True)[:10]
    return {"live": live, "upcoming": upcoming, "recent": recent}


def build_payload(per_comp: dict[str, list[dict]], now: datetime | None = None, errors: list[str] | None = None,
                  demo: bool = False, competitions: dict | None = None) -> dict:
    """Assemble the JSON the website reads. ``per_comp`` maps code -> raw API matches."""
    now = now or datetime.now(timezone.utc)
    meta = competitions or COMPETITIONS
    live, upcoming, recent = [], [], []
    for code, raw in per_comp.items():
        view = build_competition(code, raw, now)
        live += view["live"]
        upcoming += view["upcoming"]
        recent += view["recent"]
    return {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "demo": demo, "delayed": True,
        "source": "football-data.org", "errors": errors or [],
        "competitions": {c: {"name": meta[c][0], "url": meta[c][1]} for c in per_comp if c in meta},
        "live": sorted(live, key=lambda m: m["utc"]),
        "upcoming": sorted(upcoming, key=lambda m: m["utc"])[:60],
        "recent": sorted(recent, key=lambda m: m["utc"], reverse=True)[:30],
    }
