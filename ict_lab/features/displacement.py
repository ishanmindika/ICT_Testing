"""Displacement: a bar (or run of same-direction bars) whose range is large relative to recent activity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import EPS, atr, contiguous_mask, resample_bars, tf_minutes


@dataclass(frozen=True)
class DisplacementParams:
    timeframe: str = "1m"
    method: str = "atr"            # 'atr' | 'percentile'
    atr_mult: float = 1.5          # atr: window range >= atr_mult * ATR (test 1.0 / 1.5 / 2.0 / 3.0)
    atr_period: int = 14
    atr_method: str = "wilder"
    top_pct: float = 5.0           # percentile: window range in the top X% of window ranges over `lookback`
    lookback: int = 200
    run_length: int = 1            # 1 = single bar; N = N consecutive bars, all in one direction
    min_body_atr_mult: float = 0.0 # net open->close move of the run must be >= this many ATRs (0 = off)
    require_contiguous: bool = True


def detect_displacement(bars: pd.DataFrame, p: DisplacementParams = DisplacementParams()) -> pd.DataFrame:
    """One row per qualifying window, stamped at its LAST bar (so it is knowable at that bar's close).

    Window range = highest high - lowest low over the run. ATR is taken as of the bar BEFORE the run
    starts ('recent activity', not diluted by the move itself); the percentile threshold uses only
    the `lookback` windows before the current one.
    Columns: direction (+1/-1), start_idx, idx, start_ts, ts, available_at, range_points, atr,
    range_atr_mult, threshold_points, session_date.
    """
    if p.method not in ("atr", "percentile"):
        raise ValueError("method must be 'atr' or 'percentile'")
    if p.run_length < 1:
        raise ValueError("run_length must be >= 1")
    m, N = tf_minutes(p.timeframe), p.run_length
    tf = resample_bars(bars, p.timeframe)
    cols = ["direction", "start_idx", "idx", "start_ts", "ts", "available_at", "range_points", "atr",
            "range_atr_mult", "threshold_points", "session_date"]
    if len(tf) < N + 1:
        return pd.DataFrame(columns=cols)

    a = atr(tf, p.atr_period, p.atr_method)
    a_ref = a.shift(N).to_numpy()                              # ATR as of the bar before the run
    wr = (tf["high"].rolling(N).max() - tf["low"].rolling(N).min())
    up, down = (tf["close"] > tf["open"]), (tf["close"] < tf["open"])
    same_up = up.rolling(N).sum() == N
    same_down = down.rolling(N).sum() == N
    net = (tf["close"] - tf["open"].shift(N - 1)).abs().to_numpy()

    if p.method == "atr":
        thr = p.atr_mult * a_ref
    else:
        thr = wr.shift(1).rolling(p.lookback, min_periods=p.lookback).quantile(1 - p.top_pct / 100).to_numpy()
    wr_arr = wr.to_numpy()
    with np.errstate(invalid="ignore"):
        ok = (wr_arr >= thr - EPS) & ~np.isnan(thr) & (same_up | same_down).to_numpy()
        if p.min_body_atr_mult > 0:
            ok &= net >= p.min_body_atr_mult * a_ref - EPS
    if p.require_contiguous:
        ok &= contiguous_mask(tf.index, m, N)
    i = np.flatnonzero(ok)
    t = tf.index
    with np.errstate(invalid="ignore", divide="ignore"):
        mult = wr_arr[i] / a_ref[i]
    return pd.DataFrame({
        "direction": np.where(same_up.to_numpy()[i], 1, -1),
        "start_idx": i - N + 1, "idx": i, "start_ts": t[i - N + 1], "ts": t[i],
        "available_at": t[i] + pd.Timedelta(minutes=m),
        "range_points": wr_arr[i], "atr": a_ref[i], "range_atr_mult": mult, "threshold_points": thr[i],
        "session_date": tf["session_date"].to_numpy()[i],
    })


def displacement_mask(events: pd.DataFrame, index: pd.DatetimeIndex, minutes: int = 1) -> pd.Series:
    """Boolean per bar of `index`: True inside any displacement window."""
    mask = np.zeros(len(index), dtype=bool)
    for s, e in zip(events["start_ts"], events["ts"]):
        mask[index.searchsorted(s): index.searchsorted(e + pd.Timedelta(minutes=minutes), side="left")] = True
    return pd.Series(mask, index=index, name="displacement")
