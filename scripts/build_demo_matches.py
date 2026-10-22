"""Build site/data/matches.json demo data (fictional teams) from the sample CSVs.

The site shifts demo dates forward so the demo always looks current.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aviatorlab_matchwise.matchcentre import build_payload  # noqa: E402

now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
team = lambda n: {"name": n}  # noqa: E731
raw, nid = [], 1
for r in pd.read_csv(ROOT / "data" / "sample_matches.csv").itertuples():
    score = {"fullTime": {"home": int(r.home_goals), "away": int(r.away_goals)}}
    raw.append({"id": nid, "utcDate": f"{r.date}T15:00:00Z", "status": "FINISHED", "homeTeam": team(r.home_team),
                "awayTeam": team(r.away_team), "score": score})
    nid += 1
for r in pd.read_csv(ROOT / "data" / "sample_fixtures.csv").itertuples():
    when = (now + timedelta(days=int(r.day_offset))).replace(hour=int(r.kickoff[:2]), minute=int(r.kickoff[3:]))
    raw.append({"id": nid, "utcDate": iso(when), "status": "TIMED", "homeTeam": team(r.home_team),
                "awayTeam": team(r.away_team), "score": {"fullTime": {"home": None, "away": None}}})
    nid += 1
raw.append({"id": nid, "utcDate": iso(now - timedelta(minutes=40)), "status": "IN_PLAY", "homeTeam": team("Metro City"),
            "awayTeam": team("Harbor FC"), "score": {"fullTime": {"home": 1, "away": 0}}})
payload = build_payload({"DEMO": raw}, now, demo=True, competitions={"DEMO": ("Demo League (fictional)", "#")})
out = ROOT / "site" / "data" / "matches.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("Wrote", out)
