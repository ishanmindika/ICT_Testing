"""Shared helpers for feature detectors.

Conventions
- Detectors take the BACK-ADJUSTED 1-minute frame from `ict_lab.data.store.load_bars` (UTC index,
  open/high/low/close/volume/session_date/...), plus a frozen params dataclass, and return a DataFrame.
- Every threshold is in points (or a multiple of ATR, itself in points). Never a percentage of price.
- No look-ahead: each record carries `available_at`, the close time of the bar that confirms it.
  Anything that happens later (mitigation, sweeps) is recorded separately with its own timestamp.
- Timestamps in `*_ts` columns are bar OPEN times; `available_at` is a bar CLOSE time.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import replace
from typing import Any, Iterator, TypeVar

import numpy as np
import pandas as pd

EPS = 1e-9  # float noise from back-adjustment must not create or kill zero-size gaps
T = TypeVar("T")


def tf_minutes(tf: str) -> int:
    """'1m' -> 1, '5m' -> 5, '15m' -> 15. Must divide 60 so bars align to the clock."""
    m = re.fullmatch(r"(\d+)m", tf)
    if not m or int(m.group(1)) < 1 or 60 % int(m.group(1)):
        raise ValueError(f"timeframe must look like '1m', '5m', '15m' (minutes dividing 60), got {tf!r}")
    return int(m.group(1))


def resample_bars(bars: pd.DataFrame, tf: str, complete_only: bool = True) -> pd.DataFrame:
    """1-minute bars -> `tf` bars labelled by open time. With complete_only, a higher-timeframe bar
    missing any of its 1-minute bars is dropped rather than built from partial data."""
    n = tf_minutes(tf)
    cols = ["open", "high", "low", "close", "volume", "session_date"]
    if n == 1:
        out = bars[cols].copy()
        out["n_1m"] = 1
        return out
    out = bars.resample(f"{n}min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), session_date=("session_date", "last"), n_1m=("close", "size"))
    out = out[out["n_1m"] > 0]
    if complete_only:
        out = out[out["n_1m"] == n]
    return out


def contiguous_mask(index: pd.DatetimeIndex, minutes: int, k: int) -> np.ndarray:
    """ok[i] is True when bars i-k+1..i are consecutive (no time gaps, daily break or weekend)."""
    ok = np.zeros(len(index), dtype=bool)
    if len(index) >= k:
        span = index[k - 1:] - index[: len(index) - k + 1]
        ok[k - 1:] = span == pd.Timedelta(minutes=(k - 1) * minutes)
    return ok


def atr(bars: pd.DataFrame, period: int = 14, method: str = "wilder") -> pd.Series:
    """Average true range in points. 'wilder' = EMA with alpha 1/period (seeded by the first TR);
    'sma' = simple mean. NaN until `period` bars exist."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - prev_close).abs(),
                    (bars["low"] - prev_close).abs()], axis=1).max(axis=1, skipna=True)
    if method == "wilder":
        return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    if method == "sma":
        return tr.rolling(period).mean()
    raise ValueError(f"atr method must be 'wilder' or 'sma', got {method!r}")


def first_index(arr: np.ndarray, start: int, stop: int, threshold: float, op: str) -> int:
    """First i in [start, stop) where arr[i] <op> threshold (within EPS), else -1.
    op: 'le' <=, 'ge' >=, 'lt' <, 'gt' >. Scans in growing chunks so near hits stay cheap."""
    n, i, width = min(stop, len(arr)), start, 256
    while i < n:
        seg = arr[i: min(n, i + width)]
        if op == "le":
            hit = seg <= threshold + EPS
        elif op == "ge":
            hit = seg >= threshold - EPS
        elif op == "lt":
            hit = seg < threshold - EPS
        elif op == "gt":
            hit = seg > threshold + EPS
        else:
            raise ValueError(op)
        pos = np.flatnonzero(hit)
        if pos.size:
            return i + int(pos[0])
        i += len(seg)
        width *= 4
    return -1


def at(index: pd.DatetimeIndex, idx: np.ndarray) -> pd.DatetimeIndex:
    """Timestamps for integer positions, NaT where idx < 0."""
    idx = np.asarray(idx)
    out = pd.DatetimeIndex(index.take(np.where(idx >= 0, idx, 0)))
    return out.where(idx >= 0)


def param_grid(base: T, **axes: list[Any]) -> Iterator[T]:
    """Cartesian product of parameter values over a frozen params dataclass:
    param_grid(FVGParams(), timeframe=["1m", "5m"], min_size_points=[0, 1, 2])."""
    keys = list(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        yield replace(base, **dict(zip(keys, combo)))
