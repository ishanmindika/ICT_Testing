"""Fair value gap detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import EPS, at, atr, contiguous_mask, first_index, resample_bars, tf_minutes


@dataclass(frozen=True)
class FVGParams:
    timeframe: str = "1m"          # '1m' | '5m' | '15m' (any minutes dividing 60)
    min_size_points: float = 0.0   # gap must be at least this many points
    min_size_atr_mult: float = 0.0 # ...and at least this multiple of ATR (0 disables; NaN ATR fails)
    atr_period: int = 14
    atr_method: str = "wilder"     # 'wilder' | 'sma'
    atr_ref: str = "bar3"          # ATR as of the close of bar 3 (confirming bar) or bar 2
    require_contiguous: bool = True  # the three bars must be back-to-back in time
    expire_bars: int | None = None   # stop tracking mitigation this many bars after formation


def detect_fvgs(bars: pd.DataFrame, p: FVGParams = FVGParams()) -> pd.DataFrame:
    """Three consecutive bars (1, 2, 3).
      bullish: high[1] < low[3]  -> gap = [high1, low3]
      bearish: low[1]  > high[3] -> gap = [high3, low1]

    One row per FVG, in order of formation. Columns:
      direction (+1/-1), bar_idx (position of bar 3 in the `timeframe` frame), ts_bar1, ts_bar3 (open
      times), available_at (close of bar 3: when the FVG becomes knowable), session_date,
      top, bottom, mid, size_points, atr, size_atr,
      touch_idx/ts  first later bar that trades back into the gap (bull: low <= top; bear: high >= bottom)
      ce_idx/ts     first later bar that reaches the 50% midpoint
      fill_idx/ts   first later bar that trades through the whole gap (bull: low <= bottom; bear: high >= top)
    Indices count bars of the detection timeframe; NA means it has not happened (or expired).
    Mitigation is checked from the bar AFTER bar 3; touch and fill can occur on the same bar.
    """
    if p.atr_ref not in ("bar2", "bar3"):
        raise ValueError("atr_ref must be 'bar2' or 'bar3'")
    m = tf_minutes(p.timeframe)
    tf = resample_bars(bars, p.timeframe)
    n = len(tf)
    cols = ["direction", "bar_idx", "ts_bar1", "ts_bar3", "available_at", "session_date", "top", "bottom", "mid",
            "size_points", "atr", "size_atr", "touch_idx", "touch_ts", "ce_idx", "ce_ts", "fill_idx", "fill_ts"]
    if n < 3:
        return pd.DataFrame(columns=cols)

    h, l = tf["high"].to_numpy(), tf["low"].to_numpy()
    bull_size, bear_size = l[2:] - h[:-2], l[:-2] - h[2:]   # entry j <-> bar 3 at i = j + 2
    is_bull, is_bear = bull_size > EPS, bear_size > EPS
    ok = contiguous_mask(tf.index, m, 3)[2:] if p.require_contiguous else np.ones(n - 2, dtype=bool)
    i3 = np.flatnonzero((is_bull | is_bear) & ok) + 2
    j = i3 - 2
    bull = is_bull[j]
    size = np.where(bull, bull_size[j], bear_size[j])
    top = np.where(bull, l[i3], l[i3 - 2])
    bottom = np.where(bull, h[i3 - 2], h[i3])

    a = atr(tf, p.atr_period, p.atr_method).to_numpy()
    a = a[i3] if p.atr_ref == "bar3" else a[i3 - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        size_atr = size / a

    keep = size + EPS >= p.min_size_points
    if p.min_size_atr_mult > 0:
        keep &= np.nan_to_num(size_atr, nan=-1.0) + EPS >= p.min_size_atr_mult
    i3, bull, size, top, bottom, a, size_atr = (x[keep] for x in (i3, bull, size, top, bottom, a, size_atr))

    mid = (top + bottom) / 2
    touch, ce, fill = (np.full(len(i3), -1, dtype=np.int64) for _ in range(3))
    lo_arr, hi_arr = l, h
    for k in range(len(i3)):
        start = i3[k] + 1
        stop = n if p.expire_bars is None else i3[k] + 1 + p.expire_bars
        if bull[k]:
            t = first_index(lo_arr, start, stop, top[k], "le")
            if t >= 0:
                touch[k] = t
                ce[k] = first_index(lo_arr, t, stop, mid[k], "le")
                fill[k] = first_index(lo_arr, t, stop, bottom[k], "le")
        else:
            t = first_index(hi_arr, start, stop, bottom[k], "ge")
            if t >= 0:
                touch[k] = t
                ce[k] = first_index(hi_arr, t, stop, mid[k], "ge")
                fill[k] = first_index(hi_arr, t, stop, top[k], "ge")

    idx = tf.index
    out = pd.DataFrame({
        "direction": np.where(bull, 1, -1),
        "bar_idx": i3,
        "ts_bar1": idx[i3 - 2],
        "ts_bar3": idx[i3],
        "available_at": idx[i3] + pd.Timedelta(minutes=m),
        "session_date": tf["session_date"].to_numpy()[i3],
        "top": top, "bottom": bottom, "mid": mid, "size_points": size, "atr": a, "size_atr": size_atr,
    })
    for name, arr in (("touch", touch), ("ce", ce), ("fill", fill)):
        out[f"{name}_idx"] = pd.array(np.where(arr >= 0, arr, pd.NA), dtype="Int64")
        out[f"{name}_ts"] = at(idx, arr)
    return out.reset_index(drop=True)
