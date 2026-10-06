"""Synthetic CME-style 1-minute bars for tests (winter, so ET = UTC-5; no DST)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ict_lab.data.sessions import add_session_columns


def make_sessions(n_sessions: int = 6, first_day: str = "2024-01-09", seed: int = 0,
                  path: Callable[[int, np.ndarray], np.ndarray] | None = None, tick: float = 0.25) -> pd.DataFrame:
    """Mon-Fri sessions, each 18:00 ET (previous evening) to 17:00 ET, with the maintenance break absent.

    `path(day_i, minute_of_session)` may return the close path for a session; default is a random walk
    with occasional jumps so every detector has something to find.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(first_day, periods=n_sessions)
    frames = []
    last = 5000.0
    for i, d in enumerate(days):
        start_et = (d - pd.Timedelta(days=3 if d.dayofweek == 0 else 1)).tz_localize("America/New_York") + pd.Timedelta(hours=18)
        idx = pd.date_range(start_et, periods=23 * 60, freq="1min").tz_convert("UTC")
        k = np.arange(len(idx))
        if path is not None:
            close = np.asarray(path(i, k), float)
        else:
            step = rng.normal(0, 0.35, len(idx)) + np.where(rng.random(len(idx)) < 0.01, rng.normal(0, 3.0, len(idx)), 0)
            close = last + np.cumsum(step)
        close = np.round(close / tick) * tick
        open_ = np.concatenate([[last], close[:-1]])
        high = np.maximum(open_, close) + np.round(rng.uniform(0, 0.75, len(idx)) / tick) * tick
        low = np.minimum(open_, close) - np.round(rng.uniform(0, 0.75, len(idx)) / tick) * tick
        frames.append(pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 100}, index=idx))
        last = close[-1]
    df = pd.concat(frames)
    df.index.name = "ts"
    return add_session_columns(df)
