"""Market structure shift: after a sweep, price breaks the most recent swing in the opposite direction."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import EPS, first_index, resample_bars, tf_minutes
from .liquidity import SwingParams, detect_swings


@dataclass(frozen=True)
class MSSParams:
    timeframe: str = "1m"           # bars on which the break is checked
    break_mode: str = "close"       # 'close' (close-through) | 'wick' (wick-through)
    reference: str = "at_sweep"     # 'at_sweep': swing fixed when the sweep happened
                                    # 'latest': most recent confirmed swing at each bar
    max_bars: int = 30              # give up this many bars after the sweep is confirmed
    min_break_points: float = 0.0   # the break must clear the swing by at least this many points
    swing: SwingParams = field(default_factory=SwingParams)


def detect_mss(bars: pd.DataFrame, sweeps: pd.DataFrame, p: MSSParams = MSSParams(),
               swings: pd.DataFrame | None = None) -> pd.DataFrame:
    """A swept LOW (sweep_direction +1) is followed by a bullish MSS: a break above the most recent swing
    HIGH. A swept HIGH is followed by a bearish MSS: a break below the most recent swing LOW.

    Only swings already CONFIRMED are eligible (their `available_at` is at or before the open of the
    reference bar). The search starts at the sweep's confirming bar (inclusive: a bar can confirm the
    sweep and break structure at once) and runs `max_bars` bars.
    Columns: sweep level_id, direction (+1 bullish / -1 bearish), mss_idx, mss_ts, available_at,
    broken_price, swing_ts, bars_after_confirm, break_mode, reference, session_date.
    """
    if p.break_mode not in ("close", "wick"):
        raise ValueError("break_mode must be 'close' or 'wick'")
    if p.reference not in ("at_sweep", "latest"):
        raise ValueError("reference must be 'at_sweep' or 'latest'")
    m = tf_minutes(p.timeframe)
    tf = resample_bars(bars, p.timeframe)
    n = len(tf)
    swings = detect_swings(bars, p.swing) if swings is None else swings
    cols = ["level_id", "direction", "mss_idx", "mss_ts", "available_at", "broken_price", "swing_ts",
            "bars_after_confirm", "break_mode", "reference", "session_date"]
    if sweeps.empty or swings.empty or n == 0:
        return pd.DataFrame(columns=cols)

    by_kind = {}
    for kind in ("high", "low"):
        s = swings[swings["kind"] == kind].sort_values("available_at", kind="stable")
        by_kind[kind] = (pd.DatetimeIndex(s["available_at"]), s["price"].to_numpy(), pd.DatetimeIndex(s["ts"]))
    hi, lo, cl = tf["high"].to_numpy(), tf["low"].to_numpy(), tf["close"].to_numpy()
    t = tf.index

    def latest(kind: str, as_of: pd.Timestamp) -> int:
        avail = by_kind[kind][0]
        return int(avail.searchsorted(as_of, side="right")) - 1   # last swing confirmed at or before as_of

    rows = []
    for sw in sweeps.itertuples(index=False):
        bull = sw.sweep_direction == 1
        kind = "high" if bull else "low"
        start = int(t.searchsorted(sw.confirm_ts, side="left"))
        stop = min(n, start + p.max_bars)
        if start >= n:
            continue
        found = None
        if p.reference == "at_sweep":
            ref = latest(kind, sw.sweep_ts)
            if ref >= 0:
                price = by_kind[kind][1][ref]
                arr = (hi if p.break_mode == "wick" else cl) if bull else (lo if p.break_mode == "wick" else cl)
                if bull:
                    j = first_index(arr, start, stop, price + p.min_break_points, "gt")
                else:
                    j = first_index(arr, start, stop, price - p.min_break_points, "lt")
                if j >= 0:
                    found = (j, price, by_kind[kind][2][ref])
        else:
            for j in range(start, stop):
                ref = latest(kind, t[j])
                if ref < 0:
                    continue
                price = by_kind[kind][1][ref]
                if bull:
                    x = hi[j] if p.break_mode == "wick" else cl[j]
                    hit = x > price + p.min_break_points + EPS
                else:
                    x = lo[j] if p.break_mode == "wick" else cl[j]
                    hit = x < price - p.min_break_points - EPS
                if hit:
                    found = (j, price, by_kind[kind][2][ref])
                    break
        if found:
            j, price, sts = found
            rows.append((sw.level_id, 1 if bull else -1, j, price, sts, j - start, tf["session_date"].to_numpy()[j]))
    if not rows:
        return pd.DataFrame(columns=cols)
    lid, d, j, price, sts, after, sd = zip(*rows)
    j = np.array(j, dtype=np.int64)
    out = pd.DataFrame({
        "level_id": np.array(lid, dtype=np.int64), "direction": np.array(d, dtype=np.int64), "mss_idx": j,
        "mss_ts": t[j], "available_at": t[j] + pd.Timedelta(minutes=m), "broken_price": np.array(price, dtype=float),
        "swing_ts": pd.DatetimeIndex(sts), "bars_after_confirm": np.array(after, dtype=np.int64),
        "break_mode": p.break_mode, "reference": p.reference, "session_date": pd.DatetimeIndex(sd),
    })
    return out.sort_values(["available_at", "level_id"], kind="stable").reset_index(drop=True)
