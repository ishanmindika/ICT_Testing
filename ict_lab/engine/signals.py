"""Signal builder: canonical setup sequence per (session, window), from the FeatureStore.

Sequence, evaluated inside each killzone window (first failing stage is recorded):
  eligibility -> bias gate -> sweep -> [MSS] -> [displacement] -> direction-matching FVG
Every valid setup in the window is emitted: each direction-matching FVG that becomes knowable after a
qualifying chain, once (a later chain does not re-emit an FVG an earlier one already produced).

Time rules (all causal):
  * a sweep counts when its breach bar and its confirming close are both inside the window;
  * MSS / displacement must be knowable (available_at) before the window ends and, for displacement,
    must start at or after the breach bar and match the sweep's direction;
  * chain_ready = latest available_at among sweep, MSS, displacement;
  * an FVG qualifies when its available_at >= chain_ready and < window end, in the sweep's direction
    (+1 = bullish FVG after a swept low). `setup_ts` = FVG.available_at: the order is active from then.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import DisplacementParams, LevelParams
from ..features.common import tf_minutes
from .config import Instrument, SignalConfig
from .store import FeatureStore
from .timeutil import ns
from .windows import window_bounds, window_order

REASONS = ("bias_gate", "no_sweep", "no_mss", "no_displacement", "no_fvg")
SIGNAL_COLS = ["session_date", "symbol", "window", "direction", "setup_ts", "fvg_row", "fvg_top", "fvg_bottom",
               "fvg_mid", "fvg_ts_bar1", "fvg_ts_bar3", "fvg_minutes", "sweep_level_id", "sweep_level_type",
               "sweep_level_price", "sweep_ts", "sweep_available_at", "mss_available_at", "disp_start_ts",
               "disp_ts", "disp_available_at", "swing_extreme", "bias", "window_start", "window_end"]


def universe(cfg: SignalConfig, window: str):
    """(swing params or None, level types allowed for this window)."""
    pre = f"pre_{window}"
    u = cfg.sweep_universe
    if u == "session_refs":
        return None, {"prior_session", pre}
    if u == "session_refs_plus_swings":
        return cfg.swing_1m, {"prior_session", pre, "swing"}
    if u == "bsl_ssl_15m":
        return cfg.swing_15m, {"prior_session", pre, "swing"}
    return cfg.swing_1m, {"swing"}                                   # swings_only


def displacement_params(cfg: SignalConfig) -> DisplacementParams | None:
    if cfg.displacement_atr_mult is None:
        return None
    tf = cfg.fvg.timeframe if cfg.displacement_timeframe == "match_fvg" else cfg.displacement_timeframe
    return DisplacementParams(timeframe=tf, method="atr", atr_mult=cfg.displacement_atr_mult,
                              atr_period=cfg.displacement_atr_period)


@dataclass
class Signals:
    signals: pd.DataFrame      # one row per setup
    windows: pd.DataFrame      # one row per session x window (eligible or not)


def _extreme(low: np.ndarray, high: np.ndarray, i0: int, i1: int, direction: int) -> float:
    return float(low[i0:i1].min()) if direction == 1 else float(high[i0:i1].max())


def build_signals(store: FeatureStore, cfg: SignalConfig, instrument: Instrument,
                  sessions: pd.DatetimeIndex | None = None) -> Signals:
    bars, idx = store.bars, store.bars.index
    low, high = bars["low"].to_numpy(), bars["high"].to_numpy()
    all_sessions = pd.DatetimeIndex(sorted(bars["session_date"].unique()))
    sessions = all_sessions if sessions is None else all_sessions.intersection(pd.DatetimeIndex(sessions))

    fvgs = store.fvgs(cfg.fvg).sort_values("available_at", kind="stable")
    fvg_row = fvgs.index.to_numpy()
    f_avail = ns(fvgs["available_at"])
    f_dir = fvgs["direction"].to_numpy()
    f_minutes = tf_minutes(cfg.fvg.timeframe)

    dparams = displacement_params(cfg)
    disp = store.displacement(dparams) if dparams else None
    d_minutes = tf_minutes(dparams.timeframe) if dparams else 0
    bias_arr = store.bias(cfg.bias)["bias"].to_numpy()

    wrows, srows = [], []
    for window in window_order(cfg.windows):
        swing_p, types = universe(cfg, window)
        sweeps = pd.DataFrame()
        if cfg.sweep_required:
            sw = store.sweeps(swing_p, cfg.sweep_k, cfg.sweep_min_penetration_ticks, instrument.tick_size)
            sweeps = sw[sw["level_type"].isin(types)].sort_values("sweep_ts", kind="stable").reset_index(drop=True)
            s_ts, s_av = ns(sweeps["sweep_ts"]), ns(sweeps["available_at"])
        mss = store.mss(swing_p, cfg.sweep_k, cfg.sweep_min_penetration_ticks, instrument.tick_size, cfg.mss) \
            if cfg.mss_required else None
        mss_by_level = {} if mss is None else {int(r.level_id): r.available_at for r in mss.itertuples()}

        w0s, w1s = window_bounds(sessions, window)
        i0s, i1s = idx.searchsorted(w0s, side="left"), idx.searchsorted(w1s, side="left")
        for sd, w0, w1, i0, i1 in zip(sessions, w0s, w1s, i0s, i1s):
            base = {"session_date": sd, "symbol": instrument.symbol, "window": window, "window_start": w0, "window_end": w1}
            if i1 <= i0 or (idx[i0] - w0) > pd.Timedelta(minutes=cfg.max_window_start_delay_min) \
                    or bars["session_date"].iloc[i0] != sd:
                wrows.append({**base, "status": "ineligible", "bias": 0, "n_sweeps": 0, "n_chains": 0,
                              "n_setups": 0, "reason": ""})
                continue
            bias = int(bias_arr[i0])
            if cfg.bias.method != "none" and bias == 0:
                wrows.append({**base, "status": "ok", "bias": 0, "n_sweeps": 0, "n_chains": 0, "n_setups": 0,
                              "reason": "bias_gate"})
                continue
            allowed = (1, -1) if cfg.bias.method == "none" else (bias,)

            # ---- chains: (direction, ready_ns, sweep row or None, mss_avail, [displacement rows]) ----
            chains, stage_fail, n_sweeps, blocked_by_bias = [], [], 0, 0
            w0n, w1n = w0.value, w1.value
            if cfg.sweep_required:
                lo_i, hi_i = np.searchsorted(s_ts, w0n, "left"), np.searchsorted(s_ts, w1n, "left")
                cand = sweeps.iloc[lo_i:hi_i]
                cand = cand[(ns(cand["available_at"]) <= w1n)]
                for sw in cand.itertuples():
                    if sw.sweep_direction not in allowed:
                        blocked_by_bias += 1
                        continue
                    n_sweeps += 1
                    ready, mss_av = pd.Timestamp(sw.available_at), pd.NaT
                    if cfg.mss_required:
                        mav = mss_by_level.get(int(sw.level_id))
                        if mav is None or mav > w1:
                            stage_fail.append("no_mss")
                            continue
                        mss_av, ready = mav, max(ready, mav)
                    ds = []
                    if disp is not None:
                        d = disp[(disp["direction"] == sw.sweep_direction)
                                 & (disp["start_ts"] + pd.Timedelta(minutes=d_minutes) > sw.sweep_ts)
                                 & (disp["available_at"] <= w1)]
                        if d.empty:
                            stage_fail.append("no_displacement")
                            continue
                        ds = list(d.itertuples())
                        ready = max(ready, d["available_at"].min())
                    chains.append((int(sw.sweep_direction), ready, sw, mss_av, ds))
            else:
                for dr in allowed:
                    ds = []
                    if disp is not None:
                        d = disp[(disp["direction"] == dr) & (disp["start_ts"] >= w0) & (disp["available_at"] <= w1)]
                        if d.empty:
                            stage_fail.append("no_displacement")
                            continue
                        ds = list(d.itertuples())
                        chains.append((dr, d["available_at"].min(), None, pd.NaT, ds))
                    else:
                        chains.append((dr, w0, None, pd.NaT, ds))

            # ---- FVGs after each chain; each FVG emitted once, by the earliest chain ----
            used: set[int] = set()
            setups = []
            for dr, ready, sw, mss_av, ds in sorted(chains, key=lambda c: c[1]):
                lo_i, hi_i = np.searchsorted(f_avail, ready.value, "left"), np.searchsorted(f_avail, w1n, "left")
                for j in range(lo_i, hi_i):
                    if f_dir[j] != dr or j in used:
                        continue
                    used.add(j)
                    f = fvgs.iloc[j]
                    d_use = None
                    for d in ds:                                    # latest displacement knowable before this FVG
                        if d.available_at <= f["available_at"]:
                            d_use = d
                    t0 = f["ts_bar1"] if d_use is None else min(f["ts_bar1"], d_use.start_ts)
                    t1 = f["ts_bar3"] + pd.Timedelta(minutes=f_minutes)
                    a, b = int(idx.searchsorted(t0)), int(idx.searchsorted(t1))
                    setups.append({
                        **base, "direction": dr, "setup_ts": f["available_at"], "fvg_row": int(fvg_row[j]),
                        "fvg_top": f["top"], "fvg_bottom": f["bottom"], "fvg_mid": f["mid"],
                        "fvg_ts_bar1": f["ts_bar1"], "fvg_ts_bar3": f["ts_bar3"], "fvg_minutes": f_minutes,
                        "sweep_level_id": -1 if sw is None else int(sw.level_id),
                        "sweep_level_type": "" if sw is None else sw.level_type,
                        "sweep_level_price": np.nan if sw is None else sw.level_price,
                        "sweep_ts": pd.NaT if sw is None else sw.sweep_ts,
                        "sweep_available_at": pd.NaT if sw is None else sw.available_at,
                        "mss_available_at": mss_av,
                        "disp_start_ts": pd.NaT if d_use is None else d_use.start_ts,
                        "disp_ts": pd.NaT if d_use is None else d_use.ts,
                        "disp_available_at": pd.NaT if d_use is None else d_use.available_at,
                        "swing_extreme": _extreme(low, high, a, max(b, a + 1), dr), "bias": bias,
                    })
            srows += sorted(setups, key=lambda r: r["setup_ts"])

            reason = ""
            if not setups:
                if not cfg.sweep_required:
                    reason = stage_fail[0] if stage_fail else "no_fvg"
                elif n_sweeps == 0:
                    reason = "bias_gate" if blocked_by_bias else "no_sweep"
                elif not chains:
                    # furthest stage reached by any chain: displacement beats mss
                    reason = "no_displacement" if "no_displacement" in stage_fail else "no_mss"
                else:
                    reason = "no_fvg"
            wrows.append({**base, "status": "ok", "bias": bias, "n_sweeps": n_sweeps, "n_chains": len(chains),
                          "n_setups": len(setups), "reason": reason})

    sig = pd.DataFrame(srows, columns=SIGNAL_COLS)
    wl = pd.DataFrame(wrows)
    if not sig.empty:
        sig = sig.sort_values(["session_date", "setup_ts", "window"], kind="stable").reset_index(drop=True)
    wl = wl.sort_values(["session_date", "window_start"], kind="stable").reset_index(drop=True)
    return Signals(sig, wl)
