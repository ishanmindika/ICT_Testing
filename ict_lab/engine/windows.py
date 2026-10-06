"""Killzone window bounds on the ET calendar, DST-correct. Window names: london, ny_am, ny_pm."""
from __future__ import annotations

import pandas as pd

from ..data.sessions import ET, WINDOWS

RTH_END = "16:00"


def _hhmm(s: str) -> pd.Timedelta:
    return pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:]))


def to_utc(dates: pd.DatetimeIndex, hhmm: str) -> pd.DatetimeIndex:
    """ET wall-clock `hhmm` on each (naive) date -> UTC timestamps."""
    return (pd.DatetimeIndex(dates).normalize() + _hhmm(hhmm)).tz_localize(ET).tz_convert("UTC")


def window_bounds(dates: pd.DatetimeIndex, window: str) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    start, end = WINDOWS[f"killzone_{window}"]
    return to_utc(dates, start), to_utc(dates, end)


def window_order(windows: tuple[str, ...]) -> list[str]:
    return sorted(windows, key=lambda w: WINDOWS[f"killzone_{w}"][0])
