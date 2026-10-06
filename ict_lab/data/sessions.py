"""Time helpers. Stored data is UTC; session logic uses US/Eastern wall-clock time via the tz database.

No UTC offset is ever hardcoded: DST is handled by converting with `America/New_York`.
Bar timestamps are treated as bar-OPEN times, and windows are half-open: start <= t < end.
"""
from __future__ import annotations

import pandas as pd

ET = "America/New_York"

# name -> (start, end) in ET wall-clock minutes-of-day
WINDOWS: dict[str, tuple[str, str]] = {
    "rth": ("09:30", "16:00"),
    "killzone_london": ("03:00", "04:00"),
    "killzone_ny_am": ("10:00", "11:00"),
    "killzone_ny_pm": ("14:00", "15:00"),
}


def to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        raise ValueError("index must be tz-aware (UTC)")
    return index.tz_convert(ET)


def _minute_of_day(et: pd.DatetimeIndex) -> pd.Index:
    return pd.Index(et.hour * 60 + et.minute)


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def session_date(index: pd.DatetimeIndex) -> pd.Series:
    """CME trading date. A session opens 18:00 ET the prior calendar day, so bars from 18:00 ET
    belong to the NEXT date. Computed on ET wall-clock time, so DST days need no special case."""
    wall = to_et(index).tz_localize(None)
    shifted = wall.where(_minute_of_day(to_et(index)) < _hhmm("18:00"), wall + pd.Timedelta(days=1))
    return pd.Series(shifted.normalize(), index=index, name="session_date")


def in_session_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True when CME equity-index futures are expected to trade.

    Closed: daily break 17:00-18:00 ET, Friday 17:00 -> Sunday 18:00 ET.
    Holiday closures are not modelled; they appear as missing days in the quality report.
    """
    et = to_et(index)
    minute = _minute_of_day(et)
    dow = et.dayofweek  # Mon=0
    closed = (
        ((minute >= _hhmm("17:00")) & (minute < _hhmm("18:00")))
        | ((dow == 4) & (minute >= _hhmm("17:00")))
        | (dow == 5)
        | ((dow == 6) & (minute < _hhmm("18:00")))
    )
    return pd.Series(~closed, index=index)


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add session_date plus the rth / killzone boolean columns (ET wall-clock, half-open)."""
    out = df.copy()
    out["session_date"] = session_date(df.index).to_numpy()
    minute = _minute_of_day(to_et(df.index)).to_numpy()
    for name, (start, end) in WINDOWS.items():
        out[name] = (minute >= _hhmm(start)) & (minute < _hhmm(end))
    return out
