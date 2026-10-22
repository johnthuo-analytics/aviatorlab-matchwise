#!/usr/bin/env python3
"""Fetch matches from football-data.org (free plan) and write site/data/matches.json.

Needs the FOOTBALL_DATA_API_KEY environment variable (a GitHub secret in CI).
Without a key it exits cleanly and leaves the bundled demo data in place.
Free plan: 10 requests/minute, so competitions are fetched 7 seconds apart.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aviatorlab_matchwise.matchcentre import COMPETITIONS, build_payload  # noqa: E402

API = "https://api.football-data.org/v4/competitions/{code}/matches"
OUT = ROOT / "site" / "data" / "matches.json"
PAUSE_SECONDS = 7


def fetch(code: str, key: str) -> list[dict]:
    req = urllib.request.Request(API.format(code=code),
                                 headers={"X-Auth-Token": key, "User-Agent": "aviatorlab-matchwise"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (fixed https URL)
        return json.load(resp).get("matches", [])


def main() -> int:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        print("FOOTBALL_DATA_API_KEY not set; keeping existing site/data/matches.json (demo data).")
        return 0
    per_comp, errors = {}, []
    for n, code in enumerate(COMPETITIONS):
        if n:
            time.sleep(PAUSE_SECONDS)
        try:
            per_comp[code] = fetch(code, key)
        except urllib.error.HTTPError as exc:
            errors.append(f"{code}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{code}: {type(exc).__name__}")
    if not per_comp:
        print("No competition could be fetched:", errors)
        return 1  # fail the job so the previous good deployment stays online
    payload = build_payload(per_comp, datetime.now(timezone.utc), errors)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUT}: {len(payload['upcoming'])} upcoming, {len(payload['live'])} in play, "
          f"{len(payload['recent'])} recent. Errors: {errors or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
