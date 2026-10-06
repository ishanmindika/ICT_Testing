"""Execution engine: turns signal logs into trades.

All price math is done in INTEGER TICKS on the back-adjusted bars, so every fill is on the tick grid and
every comparison is exact. Shorts are simulated by negating prices, so there is a single long-side code
path; rounding in that space is always conservative (worse for the trader).

Rules
  * Entry: limit at the configured FVG level, active from the bar after the setup is confirmed until the
    window ends. It fills only when price trades THROUGH the level (low <= level - limit_through_ticks);
    the fill price is the level (no price improvement, no slippage).
  * One working order / open position at a time. Setups are taken sequentially: setups that arrive while a
    position is open are skipped; after an exit the search resumes from the next bar. An order that never
    fills is cancelled at window end (the window is then over).
  * Stops are market orders: triggered when price trades at the stop, filled at the worse of stop and the
    bar's open (gaps), plus slippage. Targets are limits: they fill only when price trades `target_through_ticks`
    beyond the target, at the target price. Hard-exit / time-target exits are market orders at the OPEN of the
    exit bar, with `time_exit_slippage_ticks`.
  * Inside one bar the order of events is unknowable. If a bar reaches both the stop and the target, the
    STOP fills and `ambiguous_bar` is True. On the entry bar only the stop is checked (a target hit there
    could have happened before the fill); if that bar also reached the target, it is flagged ambiguous.
  * MAE includes the entry bar; MFE excludes it (same reason). Timestamps are bar-open times.
  * Positions never overlap, across windows too (relevant only when hard_exit extends past a window).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.sessions import ET
from .config import ExecConfig, Instrument, StrategyConfig
from .signals import Signals, build_signals, universe
from .store import FeatureStore
from .timeutil import ns
from .windows import RTH_END, to_utc, window_bounds, window_order

EPS = 1e-9
TWO_DAYS_NS = 2 * 24 * 3600 * 10**9
INT_MAX = np.iinfo(np.int64).max

TRADE_COLS = ["session_date", "symbol", "window", "direction", "trade_no", "setup_ts", "entry_ts", "entry_price",
              "stop_price", "target_price", "target_source", "exit_ts", "exit_price", "exit_reason", "bars_held",
              "risk_points", "gross_pnl", "commission", "net_pnl", "r_multiple", "r_net", "mae_points",
              "mfe_points", "ambiguous_bar", "is_roll_day", "bias_lookahead", "sweep_level_type", "fvg_top",
              "fvg_bottom", "entry_i", "exit_i", "config_hash", "config"]


@dataclass
class Market:
    idx: pd.DatetimeIndex
    ns: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    roll: np.ndarray
    tick: float

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, tick: float) -> "Market":
        t = lambda col: np.rint(bars[col].to_numpy() / tick).astype(np.int64)
        return cls(bars.index, ns(bars.index), t("open"), t("high"), t("low"), t("close"),
                   bars["is_roll_day"].to_numpy() if "is_roll_day" in bars else np.zeros(len(bars), bool), tick)


class LevelIndex:
    """Alive-at-time queries over a level table. A level is alive at t when active_from <= t < expires and
    t <= swept_ts (swept during a bar is still alive at that bar's open)."""

    def __init__(self, table: pd.DataFrame, types: set[str]):
        t = table[table["type"].isin(types)].sort_values("active_from", kind="stable")
        self.start = ns(t["active_from"])
        exp = np.where(t["expires"].isna(), INT_MAX, ns(t["expires"])) if len(t) else np.array([], np.int64)
        swp = np.where(t["swept_ts"].isna(), INT_MAX, ns(t["swept_ts"])) if len(t) else np.array([], np.int64)
        self.end = np.minimum(exp, np.where(swp == INT_MAX, INT_MAX, swp + 1))
        self.price = t["price"].to_numpy(float)
        self.high = (t["side"].to_numpy() == "high")
        self.long_lived = np.flatnonzero(self.end - self.start > TWO_DAYS_NS)

    def alive(self, t_ns: int, want_high: bool) -> np.ndarray:
        lo = np.searchsorted(self.start, t_ns - TWO_DAYS_NS, "left")
        hi = np.searchsorted(self.start, t_ns, "right")
        cand = np.union1d(np.arange(lo, hi), self.long_lived[self.long_lived < hi])
        ok = (self.start[cand] <= t_ns) & (t_ns < self.end[cand]) & (self.high[cand] == want_high)
        return self.price[cand[ok]]


@dataclass
class Backtest:
    trades: pd.DataFrame
    windows: pd.DataFrame       # one row per session x window, with outcome
    no_trades: pd.DataFrame     # eligible windows without a trade, with the first failing condition
    signals: pd.DataFrame


def _hard_exit_ns(cfg: ExecConfig, dates: pd.DatetimeIndex, w1: pd.DatetimeIndex) -> np.ndarray:
    if cfg.hard_exit == "window_end":
        return ns(w1)
    return ns(to_utc(dates, RTH_END if cfg.hard_exit == "rth_end" else cfg.hard_exit))


def _snap(x: float, tick: float) -> int:
    return int(np.rint(x / tick))


def run_backtest(store: FeatureStore, cfg: StrategyConfig, instrument: Instrument,
                 sessions: pd.DatetimeIndex | None = None, signals: Signals | None = None) -> Backtest:
    """`signals` may be supplied (tests, replays); by default they are built from the FeatureStore."""
    sig_cfg, ex = cfg.signal, cfg.exec
    mk = Market.from_bars(store.bars, instrument.tick_size)
    n, tick, tv = len(mk.idx), instrument.tick_size, instrument.tick_value
    sg = signals if signals is not None else build_signals(store, sig_cfg, instrument, sessions)
    cfg_json, cfg_hash = cfg.canonical_json(instrument), cfg.hash(instrument)
    lookahead = cfg.signal.bias.method == "perfect"

    # level indexes for liquidity targets: same preset as the sweep, per window
    lvl: dict[str, LevelIndex] = {}
    if ex.target == "liquidity":
        for w in sig_cfg.windows:
            swing_p, types = universe(sig_cfg, w)
            lvl[w] = LevelIndex(store.level_table(swing_p), types)

    by_win = {k: v.reset_index(drop=True) for k, v in sg.signals.groupby(["session_date", "window"], sort=False)} \
        if not sg.signals.empty else {}
    wl = sg.windows.copy().reset_index(drop=True)
    res_trades = np.zeros(len(wl), np.int64)
    res_busy, res_invalid = np.zeros(len(wl), np.int64), np.zeros(len(wl), np.int64)
    res_unfilled = np.zeros(len(wl), bool)
    final_reason = wl["reason"].to_numpy(object).copy()
    h_all = {}
    for w in sig_cfg.windows:
        rows = wl[wl["window"] == w]
        h_all[w] = dict(zip(rows["session_date"], _hard_exit_ns(ex, pd.DatetimeIndex(rows["session_date"]), pd.DatetimeIndex(rows["window_end"]))))

    out, busy = [], 0
    trade_rows = []
    cap = ex.trade_cap
    for wi, row in enumerate(wl.itertuples()):
        setups = by_win.get((row.session_date, row.window))
        if row.status != "ok" or setups is None:
            continue
        i0, i1 = int(mk.idx.searchsorted(row.window_start)), int(mk.idx.searchsorted(row.window_end))
        h = int(np.searchsorted(mk.ns, h_all[row.window][row.session_date], "left"))
        n_trades = skipped = invalid = 0
        unfilled = False
        cursor = max(i0, busy)
        for s in setups.itertuples():
            if n_trades >= cap:
                break
            b0 = int(np.searchsorted(mk.ns, s.setup_ts.value, "left"))
            if b0 >= i1:
                continue
            if b0 < cursor:
                skipped += 1
                continue
            d = int(s.direction)
            # ---- plan, in long-space ticks ----
            if d == 1:
                top, bot = _snap(s.fvg_top, tick), _snap(s.fvg_bottom, tick)
                extreme = _snap(s.swing_extreme, tick)
            else:
                top, bot = -_snap(s.fvg_bottom, tick), -_snap(s.fvg_top, tick)
                extreme = -_snap(s.swing_extreme, tick)
            L = {"proximal": top, "mid": (top + bot) // 2, "distal": bot}[ex.entry]
            if ex.stop == "swing":
                S = extreme - ex.stop_buffer_ticks
            elif ex.stop == "distal":
                S = bot - ex.stop_buffer_ticks
            else:
                S = L - int(np.ceil(ex.stop_points / tick - EPS))
            risk = L - S
            if risk < 1:
                invalid += 1
                continue
            # ---- fill scan: window bars only ----
            sl = slice(b0, i1)
            lo_ls = mk.l[sl] if d == 1 else -mk.h[sl]
            hit = np.flatnonzero(lo_ls <= L - ex.limit_through_ticks)
            if hit.size == 0:
                unfilled = True
                break                                            # order cancelled at window end
            f = b0 + int(hit[0])
            # ---- target ----
            T, src = None, ""
            if ex.target == "r":
                T, src = L + max(1, int(np.floor(ex.target_r * risk + EPS))), "r"
            elif ex.target == "liquidity":
                prices = lvl[row.window].alive(int(mk.ns[f]), want_high=(d == 1))
                if prices.size:
                    lt = np.floor(d * prices / tick + EPS).astype(np.int64)
                    lt = lt[lt >= L + 1]
                    if lt.size:
                        T, src = int(lt.min()), "liquidity"
                if T is None:
                    T, src = L + max(1, int(np.floor(ex.fallback_r * risk + EPS))), "fallback_r"
            else:
                src = "time"
            # ---- exit simulation on bars f..h (long-space views) ----
            X = min(h + 1, n)
            sgn = d
            H = mk.h[f:X] if sgn == 1 else -mk.l[f:X]
            Lw = mk.l[f:X] if sgn == 1 else -mk.h[f:X]
            O = mk.o[f:X] * sgn
            C = mk.c[f:X] * sgn
            cut = len(H)                                          # bars strictly before an open-exit
            reason_open = "data_end"
            if h < n:
                cut, reason_open = h - f, "hard_exit"
            if ex.target == "time":
                tr_ = int(np.searchsorted(mk.ns, mk.ns[f] + ex.target_time_minutes * 60 * 10**9, "left")) - f
                if tr_ < cut:
                    cut, reason_open = max(tr_, 1), "target_time"
            stops = np.flatnonzero(Lw[:cut] <= S)
            s_i = int(stops[0]) if stops.size else None
            t_i = None
            if T is not None and cut > 1:
                tg = np.flatnonzero(H[1:cut] >= T + ex.target_through_ticks)
                t_i = int(tg[0]) + 1 if tg.size else None
            ambiguous = False
            if s_i is not None and (t_i is None or s_i <= t_i):
                px = S if s_i == 0 else min(S, int(O[s_i]))
                exit_px, reason, x = px - ex.stop_slippage_ticks, "stop", s_i
                ambiguous = T is not None and H[s_i] >= T + ex.target_through_ticks
                last = s_i
            elif t_i is not None:
                exit_px, reason, x, last = T, "target_" + ("liquidity" if src == "liquidity" else "r"), t_i, t_i
            elif reason_open == "data_end":
                exit_px, reason, x, last = int(C[-1]), "data_end", len(H) - 1, len(H) - 1
            else:
                exit_px = int(O[cut]) - ex.time_exit_slippage_ticks
                reason, x, last = reason_open, cut, cut - 1
            x_abs = f + x
            seg_lo, seg_hi = Lw[: last + 1], H[1: last + 1]
            mae = max(0, L - int(seg_lo.min()))
            mfe = max(0, int(seg_hi.max()) - L) if seg_hi.size else 0
            if reason in ("hard_exit", "target_time"):            # the exit open itself is a price point
                mae, mfe = max(mae, L - int(O[cut])), max(mfe, int(O[cut]) - L)

            ticks = exit_px - L
            gross = ticks * tv * ex.contracts
            comm = ex.commission_rt * ex.contracts
            risk_usd = risk * tv * ex.contracts
            trade_rows.append((
                row.session_date, instrument.symbol, row.window, d, n_trades + 1, s.setup_ts, mk.idx[f],
                sgn * L * tick, sgn * S * tick, np.nan if T is None else sgn * T * tick, src, mk.idx[min(f + x, n - 1)],
                sgn * exit_px * tick, reason, x_abs - f, risk * tick, gross, comm, gross - comm, ticks / risk,
                (gross - comm) / risk_usd, mae * tick, mfe * tick, ambiguous, bool(mk.roll[f]), lookahead,
                s.sweep_level_type, s.fvg_top, s.fvg_bottom, f, f + x, cfg_hash, cfg_json))
            n_trades += 1
            cursor = busy = f + x + 1
        res_trades[wi], res_busy[wi], res_invalid[wi], res_unfilled[wi] = n_trades, skipped, invalid, unfilled
        if n_trades == 0:
            final_reason[wi] = ("limit_unfilled" if unfilled else "invalid_risk" if invalid
                                else "position_open" if skipped else row.reason)
    wl["n_trades"], wl["n_skipped_busy"], wl["n_invalid_risk"], wl["unfilled"] = res_trades, res_busy, res_invalid, res_unfilled
    wl["final_reason"] = final_reason
    trades = pd.DataFrame(trade_rows, columns=TRADE_COLS)
    ok = wl[(wl["status"] == "ok") & (wl["n_trades"] == 0)]
    no_trades = ok[["session_date", "symbol", "window", "final_reason"]].rename(columns={"final_reason": "reason"}).reset_index(drop=True)
    return Backtest(trades, wl, no_trades, sg.signals)
