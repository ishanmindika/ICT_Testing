"""Level registry: every liquidity level as one row with its life span. Levels stay active until swept."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import first_index
from .liquidity import LevelParams, SwingParams, detect_swings, session_levels

LEVEL_TYPES = ("prior_session", "prior_rth", "pre_london", "pre_ny_am", "pre_ny_pm", "swing")


def level_table(bars: pd.DataFrame, level_p: LevelParams = LevelParams(), swing_p: SwingParams = SwingParams(),
                per_bar: pd.DataFrame | None = None, swings: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per level instance:
      level_id, type, side ('high' = buy-side liquidity above price, 'low' = sell-side below), price,
      active_from (first bar open at which the level is usable), expires (end of session for session-based
      levels, NaT for swings), swept_ts (open of the first bar that trades through it, NaT if none),
      session_date.
    Active at bar-open time t:  active_from <= t < expires  and  (swept_ts is NaT or swept_ts >= t).
    Session-based levels cover only their own session; a swing lives until it is swept.
    `swept_ts` is ANY trade-through; liquidity sweeps (with a minimum penetration and close-back) are in sweep.py.
    """
    per_bar = session_levels(bars, level_p) if per_bar is None else per_bar
    swings = detect_swings(bars, swing_p) if swings is None else swings
    sd = bars["session_date"]
    session_end = bars.index.to_series().groupby(sd).max() + pd.Timedelta(minutes=1)
    frames = []
    for col in per_bar.columns:
        if col.startswith("swept_") or not col.endswith(("_high", "_low")):
            continue
        sub = per_bar[col].dropna()
        if sub.empty:
            continue
        g = pd.DataFrame({"ts": sub.index, "price": sub.to_numpy(), "session_date": sd.loc[sub.index].to_numpy()})
        g = g.groupby("session_date", as_index=False).agg(active_from=("ts", "first"), price=("price", "first"))
        base, side = col.rsplit("_", 1)
        g["type"] = base  # prior_session | prior_rth | pre_london | pre_ny_am | pre_ny_pm
        g["side"] = side
        g["expires"] = g["session_date"].map(session_end)
        frames.append(g)
    if len(swings):
        w = pd.DataFrame({"type": "swing", "side": swings["kind"], "price": swings["price"],
                          "active_from": swings["available_at"], "expires": pd.NaT,
                          "session_date": swings["session_date"]})
        frames.append(w)
    cols = ["level_id", "type", "side", "price", "active_from", "expires", "swept_ts", "session_date"]
    if not frames:
        return pd.DataFrame(columns=cols)
    t = pd.concat(frames, ignore_index=True).sort_values(["active_from", "type", "side"], kind="stable").reset_index(drop=True)
    t["level_id"] = np.arange(len(t))

    idx = bars.index
    hi, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    start = idx.searchsorted(pd.DatetimeIndex(t["active_from"]), side="left")
    stop = np.where(t["expires"].isna(), len(idx), idx.searchsorted(pd.DatetimeIndex(t["expires"].fillna(idx[-1])), side="left"))
    swept = np.full(len(t), -1, dtype=np.int64)
    for i, (s, e, price, side) in enumerate(zip(start, stop, t["price"].to_numpy(), t["side"].to_numpy())):
        swept[i] = first_index(hi, s, e, price, "gt") if side == "high" else first_index(lo, s, e, price, "lt")
    t["swept_ts"] = pd.DatetimeIndex(idx.take(np.where(swept >= 0, swept, 0))).where(swept >= 0)
    return t[cols]


def active_levels(table: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Levels usable at the open of the bar at `ts`."""
    live = (table["active_from"] <= ts) & (table["expires"].isna() | (table["expires"] > ts)) \
        & (table["swept_ts"].isna() | (table["swept_ts"] >= ts))
    return table[live]
