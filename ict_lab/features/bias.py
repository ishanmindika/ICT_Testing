"""Higher-timeframe bias, several switchable definitions.

Output is a per-bar frame: `bias` (+1 bullish, -1 bearish, 0 none/unknown), `bias_method`,
`bias_lookahead`. Every method except 'perfect' uses only information available at the bar's open.
'perfect' peeks at the future by design (an upper bound for testing): its method name reads
'perfect(LOOKAHEAD)', `bias_lookahead` is True, `attrs['lookahead']` is set, and a warning is emitted.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.sessions import WINDOWS, add_session_columns, to_et
from .liquidity import SwingParams, detect_swings

METHODS = ("none", "prior_day_oc", "trend_swing", "daily_ma_slope", "perfect")


class LookaheadWarning(UserWarning):
    pass


@dataclass(frozen=True)
class BiasParams:
    method: str = "none"
    daily_basis: str = "session"     # 'session' (18:00-17:00 ET) | 'rth' (09:30-16:00 ET) for daily open/close
    min_move_points: float = 0.0     # prior_day_oc / daily_ma_slope: |move| must exceed this to count
    trend_timeframe: str = "1h"      # trend_swing: '15m' | '1h'
    trend_swing_n: int = 2           # fractal size on that timeframe
    ma_period: int = 20              # daily_ma_slope
    perfect_window: str = "killzone_ny_am"  # perfect: direction from this window's start to the session close


def _sign(x: pd.Series | np.ndarray, thr: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > thr, 1, np.where(x < -thr, -1, 0)).astype(np.int8)


def _daily(bars: pd.DataFrame, basis: str) -> pd.DataFrame:
    """Per session_date: open (first bar's open) and close (last bar's close) on the chosen basis."""
    flags = bars if "rth" in bars.columns else add_session_columns(bars[["high"]])
    src = bars if basis == "session" else bars[flags["rth"].to_numpy()]
    if basis not in ("session", "rth"):
        raise ValueError("daily_basis must be 'session' or 'rth'")
    g = src.groupby("session_date")
    return pd.DataFrame({"open": g["open"].first(), "close": g["close"].last()})


def _prior_day_value(bars: pd.DataFrame, per_day: pd.Series) -> np.ndarray:
    """Map a per-session series onto bars using the PREVIOUS session's value."""
    sd = bars["session_date"]
    days = pd.DatetimeIndex(sorted(sd.unique()))
    prev = pd.Series(per_day.reindex(days).shift(1).to_numpy(), index=days)
    return sd.map(prev).fillna(0).to_numpy()


def htf_bias(bars: pd.DataFrame, p: BiasParams = BiasParams()) -> pd.DataFrame:
    if p.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    idx = bars.index
    if p.method == "none":
        val = np.zeros(len(bars), dtype=np.int8)
    elif p.method == "prior_day_oc":
        d = _daily(bars, p.daily_basis)
        val = _prior_day_value(bars, pd.Series(_sign(d["close"] - d["open"], p.min_move_points), index=d.index))
    elif p.method == "daily_ma_slope":
        d = _daily(bars, p.daily_basis)
        ma = d["close"].rolling(p.ma_period).mean()
        slope = ma - ma.shift(1)                          # MA through day d minus MA through d-1
        val = _prior_day_value(bars, pd.Series(_sign(slope.fillna(0), p.min_move_points), index=d.index))
    elif p.method == "trend_swing":
        sw = detect_swings(bars, SwingParams(timeframe=p.trend_timeframe, n=p.trend_swing_n))
        val = _trend_from_swings(sw, idx)
    else:  # perfect
        warnings.warn("bias method 'perfect' uses FUTURE data (window start -> session close). "
                      "Results are an upper bound, not tradable.", LookaheadWarning, stacklevel=2)
        val = _perfect(bars, p.perfect_window)
    out = pd.DataFrame({"bias": val.astype(np.int8)}, index=idx)
    out["bias_method"] = "perfect(LOOKAHEAD)" if p.method == "perfect" else p.method
    out["bias_lookahead"] = p.method == "perfect"
    out.attrs["lookahead"] = p.method == "perfect"
    return out


def _trend_from_swings(sw: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    """+1 after higher-high AND higher-low (last two confirmed of each), -1 after lower-high AND lower-low."""
    if sw.empty:
        return np.zeros(len(index), dtype=np.int8)
    sw = sw.sort_values("available_at", kind="stable")
    highs: list[float] = []
    lows: list[float] = []
    times, states = [], []
    for kind, price, avail in zip(sw["kind"], sw["price"], sw["available_at"]):
        (highs if kind == "high" else lows).append(price)
        state = 0
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                state = 1
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                state = -1
        times.append(avail)
        states.append(state)
    s = pd.Series(states, index=pd.DatetimeIndex(times))
    s = s[~s.index.duplicated(keep="last")]                      # several swings can confirm on one bar
    return s.reindex(index, method="ffill").fillna(0).to_numpy().astype(np.int8)


def _perfect(bars: pd.DataFrame, window: str) -> np.ndarray:
    start_min = int(WINDOWS[window][0][:2]) * 60 + int(WINDOWS[window][0][3:])
    et = to_et(bars.index)
    rel = (np.asarray(et.hour * 60 + et.minute) - 18 * 60) % 1440      # minutes since the 18:00 ET session open
    in_or_after = rel >= (start_min - 18 * 60) % 1440
    day_close = bars["close"].groupby(bars["session_date"]).last()
    starts = bars["open"].where(in_or_after).groupby(bars["session_date"]).first()
    direction = pd.Series(_sign(day_close - starts, 0.0), index=day_close.index)
    return bars["session_date"].map(direction).fillna(0).to_numpy()
