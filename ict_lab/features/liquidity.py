"""Liquidity levels: prior session / prior RTH / pre-window highs and lows, and N-bar fractal swings."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.sessions import WINDOWS, add_session_columns, to_et
from .common import EPS, at, contiguous_mask, first_index, resample_bars, tf_minutes


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


@dataclass(frozen=True)
class LevelParams:
    killzones: tuple[str, ...] = ("killzone_london", "killzone_ny_am", "killzone_ny_pm")
    pre_window_start: str = "18:00"   # ET; "session open to killzone start"


def session_levels(bars: pd.DataFrame, p: LevelParams = LevelParams()) -> pd.DataFrame:
    """Per-bar liquidity levels, aligned to `bars.index`. All are known at the time of the bar.

      prior_session_high/low   previous session_date's full-session extremes (known from the open)
      prior_rth_high/low       previous session_date's 09:30-16:00 ET extremes
      pre_<kz>_high/low        extremes from `pre_window_start` up to the killzone start; NaN before the
                               killzone starts, frozen from then on (never uses bars inside the killzone)
      swept_<level>            True from the first bar of the session whose high (for highs) / low (for
                               lows) trades through the level, inclusive, until the session ends
      prior_session_date       which session the prior_* levels came from (gaps in data show up here)
    """
    flags = bars if "rth" in bars.columns else add_session_columns(bars[["high"]])
    sd = bars["session_date"] if "session_date" in bars.columns else flags["session_date"]
    out = pd.DataFrame(index=bars.index)

    def per_session(mask: pd.Series | np.ndarray | None, how: str, col: str) -> pd.Series:
        s = bars[col] if mask is None else bars[col].where(mask)
        return s.groupby(sd).agg(how)

    hi, lo = per_session(None, "max", "high"), per_session(None, "min", "low")
    rth = flags["rth"].to_numpy()
    rhi, rlo = per_session(rth, "max", "high"), per_session(rth, "min", "low")
    days = hi.index  # sorted unique session dates
    prior = pd.Series(days[:-1], index=days[1:]).reindex(days)
    for name, series in (("prior_session_high", hi), ("prior_session_low", lo),
                         ("prior_rth_high", rhi), ("prior_rth_low", rlo)):
        shifted = pd.Series(series.reindex(prior.to_numpy()).to_numpy(), index=days)
        out[name] = sd.map(shifted).to_numpy()
    out["prior_session_date"] = sd.map(pd.Series(prior.to_numpy(), index=days)).to_numpy()

    et = to_et(bars.index)
    rel = (np.asarray(et.hour * 60 + et.minute) - _hhmm(p.pre_window_start)) % 1440  # minutes since window start
    for kz in p.killzones:
        kz_rel = (_hhmm(WINDOWS[kz][0]) - _hhmm(p.pre_window_start)) % 1440
        pre = rel < kz_rel
        short = kz.removeprefix("killzone_")
        for side, col, how in (("high", "high", "max"), ("low", "low", "min")):
            frozen = per_session(pre, how, col)
            out[f"pre_{short}_{side}"] = np.where(rel >= kz_rel, sd.map(frozen).to_numpy(), np.nan)

    for name in [c for c in out.columns if c.endswith(("_high", "_low"))]:
        beyond = bars["high"] > out[name] + EPS if name.endswith("_high") else bars["low"] < out[name] - EPS
        out[f"swept_{name}"] = beyond.groupby(sd).cummax().astype(bool).to_numpy()
    return out


@dataclass(frozen=True)
class SwingParams:
    timeframe: str = "1m"
    n: int = 3                       # fractal: bar must beat the N bars on each side
    strict: bool = True              # strict: > both sides; False: ties allowed (>=)
    require_contiguous: bool = True  # the 2N+1 bars must be back-to-back in time


def detect_swings(bars: pd.DataFrame, p: SwingParams = SwingParams()) -> pd.DataFrame:
    """N-bar fractal swing highs/lows, one row each, in order of confirmation.

    A swing high at bar i needs high[i] > max(high[i-N..i-1]) and > max(high[i+1..i+N]) (>= if not strict).
    It is only knowable once bar i+N closes: `available_at`. Columns:
      kind ('high'/'low'), idx, ts (open time of the swing bar), price, session_date, confirmed_idx,
      available_at,
      swept_idx/ts          first bar after confirmation that trades THROUGH the level (wick counts)
      close_through_idx/ts  first bar after confirmation that CLOSES beyond the level
    NA means it has not happened yet in the data.
    """
    m, n = tf_minutes(p.timeframe), p.n
    if n < 1:
        raise ValueError("n must be >= 1")
    tf = resample_bars(bars, p.timeframe)
    size = len(tf)
    cols = ["kind", "idx", "ts", "price", "session_date", "confirmed_idx", "available_at",
            "swept_idx", "swept_ts", "close_through_idx", "close_through_ts"]
    if size < 2 * n + 1:
        return pd.DataFrame(columns=cols)

    h, l, c = tf["high"].to_numpy(), tf["low"].to_numpy(), tf["close"].to_numpy()
    win = np.lib.stride_tricks.sliding_window_view
    ok = contiguous_mask(tf.index, m, 2 * n + 1)[2 * n:] if p.require_contiguous else np.ones(size - 2 * n, dtype=bool)
    i = np.arange(n, size - n)  # candidate swing bars; ok[i - n] says the 2N+1 window is contiguous

    rows = []
    for kind, arr in (("high", h), ("low", l)):
        wmax = win(arr, n).max(axis=1) if kind == "high" else win(arr, n).min(axis=1)
        left, right = wmax[i - n], wmax[i + 1]
        if kind == "high":
            hit = (arr[i] > np.maximum(left, right) + EPS) if p.strict else (arr[i] >= np.maximum(left, right) - EPS)
        else:
            hit = (arr[i] < np.minimum(left, right) - EPS) if p.strict else (arr[i] <= np.minimum(left, right) + EPS)
        sel = i[hit & ok[i - n]]
        for s in sel:
            conf = s + n
            if kind == "high":
                sw = first_index(h, conf + 1, size, arr[s], "gt")
                ct = first_index(c, conf + 1, size, arr[s], "gt")
            else:
                sw = first_index(l, conf + 1, size, arr[s], "lt")
                ct = first_index(c, conf + 1, size, arr[s], "lt")
            rows.append((kind, s, arr[s], conf, sw, ct))

    if not rows:
        return pd.DataFrame(columns=cols)
    kind, idx, price, conf, sw, ct = (np.array(x) for x in zip(*rows))
    t = tf.index
    out = pd.DataFrame({
        "kind": kind, "idx": idx.astype(np.int64), "ts": t[idx.astype(np.int64)], "price": price.astype(float),
        "session_date": tf["session_date"].to_numpy()[idx.astype(np.int64)],
        "confirmed_idx": conf.astype(np.int64),
        "available_at": t[conf.astype(np.int64)] + pd.Timedelta(minutes=m),
    })
    for name, arr in (("swept", sw.astype(np.int64)), ("close_through", ct.astype(np.int64))):
        out[f"{name}_idx"] = pd.array(np.where(arr >= 0, arr, pd.NA), dtype="Int64")
        out[f"{name}_ts"] = at(t, arr)
    return out.sort_values(["confirmed_idx", "kind"], kind="stable").reset_index(drop=True)
