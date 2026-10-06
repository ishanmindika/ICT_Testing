"""Time helpers. All stored data is UTC; strategy logic reads New York (ET) wall-clock time."""
from __future__ import annotations

import pandas as pd

ET = "America/New_York"


def to_et(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(ET)


def session_date(index: pd.DatetimeIndex) -> pd.Series:
    """CME trading date: the Globex session opens 18:00 ET and belongs to the next calendar day."""
    et = to_et(index)
    return pd.Series((et + pd.Timedelta(hours=6)).normalize().tz_localize(None), index=index, name="session_date")


def in_session_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True when CME equity-index futures are expected to trade.

    Closed: daily break 17:00-18:00 ET, Friday 17:00 -> Sunday 18:00 ET.
    (Holiday closures are not modelled; they show up as reported gaps.)
    """
    et = to_et(index)
    minute = et.hour * 60 + et.minute
    dow = et.dayofweek  # Mon=0
    closed = (
        ((minute >= 17 * 60) & (minute < 18 * 60))
        | ((dow == 4) & (minute >= 17 * 60))
        | (dow == 5)
        | ((dow == 6) & (minute < 18 * 60))
    )
    return pd.Series(~closed, index=index)
