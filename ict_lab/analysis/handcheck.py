"""Hand-check printout: every selected trade with its surrounding bars, adjusted AND unadjusted, so entries and
exits can be checked by eye against a chart.

    python -m ict_lab.analysis.handcheck NQ 2024-03-01 2024-03-31 --configs as_traded as_taught_5m

Trades are chosen to cover each exit type (stop, target, liquidity-level exit, time exit, ambiguous bar) first.
Trade prices are BACK-ADJUSTED (what the strategy used); the unadjusted equivalents (adjusted + offset) are
printed next to them. Timestamps are ET bar-open times.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data.sessions import ET
from ..data.store import CLEAN_DIR, load_bars
from ..engine.config import load_named_configs
from ..engine.execute import Backtest, run_backtest
from ..engine.runner import FEATURE_CACHE, prepare

CATEGORIES = {
    "stop fill": lambda t: t.exit_reason == "stop",
    "target fill (R)": lambda t: t.exit_reason == "target_r",
    "liquidity-level exit": lambda t: t.exit_reason == "target_liquidity",
    "time exit": lambda t: t.exit_reason.isin(["hard_exit", "target_time"]),
    "ambiguous bar": lambda t: t.ambiguous_bar,
}


def select(trades: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Cover each category once (first match), then fill with the earliest remaining trades."""
    picked: list[int] = []
    for pred in CATEGORIES.values():
        hit = trades.index[pred(trades)]
        if len(hit) and hit[0] not in picked:
            picked.append(hit[0])
    for i in trades.index:
        if len(picked) >= limit:
            break
        if i not in picked:
            picked.append(i)
    return trades.loc[sorted(picked)]


def coverage(trades: pd.DataFrame) -> dict[str, int]:
    return {name: int(pred(trades).sum()) for name, pred in CATEGORIES.items()}


def trade_block(t, adj: pd.DataFrame, unadj: pd.DataFrame, idx: int, total: int, pad: int = 3) -> str:
    e = lambda ts: ts.tz_convert(ET).strftime("%H:%M")
    off = float((unadj["close"] - adj["close"]).loc[t.entry_ts])
    u = lambda p: "" if pd.isna(p) else f" (unadj {p + off:,.2f})"
    side = "LONG" if t.direction == 1 else "SHORT"
    lines = [
        f"=== TRADE {idx}/{total}  {t.symbol} {t.session_date:%Y-%m-%d} {t.window}  {side}  exit={t.exit_reason}"
        f"  ambiguous_bar={t.ambiguous_bar}  roll_day={t.is_roll_day}",
        f"setup {e(t.setup_ts)}  entry {e(t.entry_ts)} @ {t.entry_price:,.2f}{u(t.entry_price)}   stop {t.stop_price:,.2f}{u(t.stop_price)}"
        f"   target {t.target_price:,.2f}{u(t.target_price)} [{t.target_source}]",
        f"exit {e(t.exit_ts)} @ {t.exit_price:,.2f}{u(t.exit_price)}   held {t.bars_held} bars   risk {t.risk_points:.2f} pts   "
        f"R {t.r_multiple:+.2f}   gross ${t.gross_pnl:,.2f}  net ${t.net_pnl:,.2f}   MAE {t.mae_points:.2f}  MFE {t.mfe_points:.2f}",
        f"FVG [{t.fvg_bottom:,.2f}, {t.fvg_top:,.2f}]   sweep level: {t.sweep_level_type}   adj->unadj offset {off:+,.2f}",
        f"{'ET':>6} | {'adj O':>10} {'H':>10} {'L':>10} {'C':>10} | {'unadj O':>10} {'H':>10} {'L':>10} {'C':>10} | note",
    ]
    lo = max(adj.index.get_loc(min(t.setup_ts, t.entry_ts)) - pad, 0)
    hi = min(adj.index.get_loc(t.exit_ts) + pad, len(adj) - 1)
    for i in range(lo, hi + 1):
        ts, a, b = adj.index[i], adj.iloc[i], unadj.iloc[i]
        note = " ".join(x for x, c in (("SETUP", ts == t.setup_ts), ("ENTRY", ts == t.entry_ts), ("EXIT", ts == t.exit_ts)) if c)
        lines.append(f"{e(ts):>6} | {a.open:>10,.2f} {a.high:>10,.2f} {a.low:>10,.2f} {a.close:>10,.2f} | "
                     f"{b.open:>10,.2f} {b.high:>10,.2f} {b.low:>10,.2f} {b.close:>10,.2f} | {note}")
    return "\n".join(lines)


def handcheck(bt: Backtest, adj: pd.DataFrame, unadj: pd.DataFrame, label: str, limit: int = 8) -> tuple[str, pd.DataFrame]:
    chosen = select(bt.trades.reset_index(drop=True), limit)
    parts = [f"########## {label}: {len(bt.trades)} trades, printing {len(chosen)} ##########"]
    for k, t in enumerate(chosen.itertuples(), 1):
        parts.append(trade_block(t, adj, unadj, k, len(chosen)))
    return "\n\n".join(parts), chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--configs", nargs="+", default=["as_traded", "as_taught_5m"])
    ap.add_argument("--per-run", type=int, default=8)
    ap.add_argument("--cache", default=str(FEATURE_CACHE))
    a = ap.parse_args()
    named = load_named_configs()
    store, inst, days = prepare(a.symbol, a.start, a.end, cache_dir=a.cache)
    unadj = load_bars(a.symbol, "unadjusted").reindex(store.bars.index)
    shown = []
    for name in a.configs:
        bt = run_backtest(store, named[name], inst, sessions=days)
        text, chosen = handcheck(bt, store.bars, unadj, f"{name} {a.symbol} {a.start}..{a.end}", a.per_run)
        print(text + "\n")
        shown.append(chosen)
    allc = pd.concat(shown)
    cov = coverage(allc)
    print("COVERAGE across printed trades:", cov, f"| total printed: {len(allc)} (need 10+)")
    missing = [k for k, v in cov.items() if v == 0]
    if missing or len(allc) < 10:
        print("NOT YET COVERED:", missing or "-", "-> extend to another month (--start/--end) if needed")


if __name__ == "__main__":
    main()
