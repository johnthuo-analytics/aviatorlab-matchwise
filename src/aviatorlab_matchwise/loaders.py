"""Safe CSV loaders with clear validation errors for user-supplied data."""
from __future__ import annotations

from typing import IO, Union

import pandas as pd

from .matchwise import validate_matches

Source = Union[str, "os.PathLike[str]", IO]  # noqa: F821
MAX_ROWS = 200_000


def _read(source: Source) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source, nrows=MAX_ROWS + 1)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError(f"File has more than {MAX_ROWS:,} rows; please supply a smaller sample.")
    return frame


def load_multipliers(source: Source) -> list[float]:
    """Load a CSV with a ``multiplier`` column; drops blanks, rejects bad values."""
    frame = _read(source)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "multiplier" not in frame.columns:
        raise ValueError("CSV must include a 'multiplier' column.")
    series = frame["multiplier"]
    if not pd.api.types.is_numeric_dtype(series):  # tolerate values such as "2.35x"
        series = series.astype("string").str.strip().str.rstrip("xX")
    values = pd.to_numeric(series, errors="coerce")
    bad = int(values.isna().sum() - frame["multiplier"].isna().sum())
    if bad:
        raise ValueError(f"{bad} value(s) in 'multiplier' are not numeric.")
    values = values.dropna()
    if values.empty:
        raise ValueError("No multipliers found in the file.")
    if (values < 1).any():
        raise ValueError("Multipliers must be >= 1.0.")
    return values.astype(float).tolist()


def load_matches(source: Source) -> pd.DataFrame:
    """Load football results (date optional); validates goals and team columns."""
    frame = _read(source)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    frame = validate_matches(frame)
    for col in ("home_team", "away_team"):
        frame[col] = frame[col].astype(str).str.strip()
    if (frame.home_team == frame.away_team).any():
        raise ValueError("A team cannot play itself (home_team == away_team).")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    return frame


def load_fixtures(source: Source, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load upcoming fixtures: ``home_team, away_team`` plus ``date`` or ``day_offset``.

    ``day_offset`` (days from today) keeps bundled demo fixtures always "upcoming".
    ``kickoff`` (HH:MM) is optional and defaults to 15:00.
    """
    frame = _read(source)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = {"home_team", "away_team"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("No fixtures found in the file.")
    for col in ("home_team", "away_team"):
        frame[col] = frame[col].astype(str).str.strip()
    if (frame.home_team == frame.away_team).any():
        raise ValueError("A team cannot play itself (home_team == away_team).")
    base = (today if today is not None else pd.Timestamp.today()).normalize()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    elif "day_offset" in frame.columns:
        frame["date"] = base + pd.to_timedelta(pd.to_numeric(frame["day_offset"], errors="raise"), unit="D")
    else:
        raise ValueError("Fixtures need a 'date' or 'day_offset' column.")
    frame["kickoff"] = frame["kickoff"].astype(str).str.strip() if "kickoff" in frame.columns else "15:00"
    return frame.sort_values(["date", "kickoff"], kind="stable").reset_index(drop=True)
