"""Liquidity sweep: price trades through a level, then closes back on the original side within K bars."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import EPS, first_index, resample_bars, tf_minutes
from .levels import LEVEL_TYPES


@dataclass(frozen=True)
class SweepParams:
    level_types: tuple[str, ...] = LEVEL_TYPES  # which level types count
    k: int = 3                                  # bars allowed to close back
    min_penetration_ticks: float = 1.0          # how far through the level the bar must trade
    tick_size: float = 0.25                     # points per tick (ES/NQ 0.25)
    allow_same_bar: bool = True                 # the breach bar itself may close back (a pure wick)
    timeframe: str = "1m"


def detect_sweeps(bars: pd.DataFrame, levels: pd.DataFrame, p: SweepParams = SweepParams()) -> pd.DataFrame:
    """For each eligible level, find the first bar j (while the level is active) that trades at least
    `min_penetration_ticks` through it. If a bar in j..j+K (j+1..j+K when allow_same_bar is False)
    closes back on the original side, that is a sweep, confirmed at that bar's close. If none does,
    the level was broken, not swept: no event. A level yields at most one event.

    sweep_direction: -1 for a swept HIGH (buy-side liquidity taken, bearish expectation), +1 for a swept LOW.
    Columns: level_id, level_type, level_side, level_price, sweep_direction, sweep_idx/ts (breach bar),
    confirm_idx/ts (closing-back bar), available_at (its close), extreme_price, penetration_points/ticks,
    session_date. Everything is known at `available_at`.
    """
    unknown = set(p.level_types) - set(LEVEL_TYPES)
    if unknown:
        raise ValueError(f"unknown level types {unknown}")
    m = tf_minutes(p.timeframe)
    tf = resample_bars(bars, p.timeframe)
    n = len(tf)
    pen = p.min_penetration_ticks * p.tick_size
    h, l, c = tf["high"].to_numpy(), tf["low"].to_numpy(), tf["close"].to_numpy()
    cols = ["level_id", "level_type", "level_side", "level_price", "sweep_direction", "sweep_idx", "sweep_ts",
            "confirm_idx", "confirm_ts", "available_at", "extreme_price", "penetration_points",
            "penetration_ticks", "session_date"]
    lv = levels[levels["type"].isin(p.level_types)]
    if lv.empty or n == 0:
        return pd.DataFrame(columns=cols)

    start = tf.index.searchsorted(pd.DatetimeIndex(lv["active_from"]), side="left")
    stop = np.where(lv["expires"].isna(), n, tf.index.searchsorted(pd.DatetimeIndex(lv["expires"].fillna(tf.index[-1])), side="left"))
    first_after = 0 if p.allow_same_bar else 1
    rows = []
    for (lid, typ, side, price), s, e in zip(lv[["level_id", "type", "side", "price"]].itertuples(index=False), start, stop):
        if side == "high":
            j = first_index(h, s, e, price + pen, "ge") if pen > 0 else first_index(h, s, e, price, "gt")
            if j < 0:
                continue
            k = first_index(c, j + first_after, min(n, j + p.k + 1), price, "lt")
            if k < 0:
                continue
            ext = h[j: k + 1].max()
            rows.append((lid, typ, side, price, -1, j, k, ext, ext - price))
        else:
            j = first_index(l, s, e, price - pen, "le") if pen > 0 else first_index(l, s, e, price, "lt")
            if j < 0:
                continue
            k = first_index(c, j + first_after, min(n, j + p.k + 1), price, "gt")
            if k < 0:
                continue
            ext = l[j: k + 1].min()
            rows.append((lid, typ, side, price, 1, j, k, ext, price - ext))
    if not rows:
        return pd.DataFrame(columns=cols)
    lid, typ, side, price, direction, j, k, ext, pts = (np.array(x) for x in zip(*rows))
    j, k = j.astype(np.int64), k.astype(np.int64)
    out = pd.DataFrame({
        "level_id": lid.astype(np.int64), "level_type": typ, "level_side": side, "level_price": price.astype(float),
        "sweep_direction": direction.astype(np.int64), "sweep_idx": j, "sweep_ts": tf.index[j],
        "confirm_idx": k, "confirm_ts": tf.index[k], "available_at": tf.index[k] + pd.Timedelta(minutes=m),
        "extreme_price": ext.astype(float), "penetration_points": pts.astype(float),
        "penetration_ticks": (pts / p.tick_size).astype(float),
        "session_date": tf["session_date"].to_numpy()[k],
    })
    return out.sort_values(["available_at", "level_id"], kind="stable").reset_index(drop=True)
