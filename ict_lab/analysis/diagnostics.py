"""Full-span FREQUENCY diagnostic: counts and reasons only. No PnL, win rate or R statistics, by design:
frequency tuning must stay blind to performance. (The engine produces PnL columns; this module never reads them.)

    python -m ict_lab.analysis.diagnostics NQ [--configs as_taught_5m as_taught_1m as_traded]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..engine.config import load_named_configs
from ..engine.execute import run_backtest
from ..engine.runner import FEATURE_CACHE, prepare

LAB = Path(__file__).resolve().parents[1]
REASON_ORDER = ["traded", "bias_gate", "no_sweep", "no_mss", "no_displacement", "no_fvg", "limit_unfilled",
                "invalid_risk", "position_open"]
_ALLOWED_TRADE_COLS = ["session_date", "window"]          # the ONLY trade columns this module touches


def diagnose(bt, trading_days: pd.DatetimeIndex) -> dict[str, pd.DataFrame | pd.Series]:
    wl = bt.windows[bt.windows["status"] == "ok"].copy()
    wl["year"] = wl["session_date"].dt.year
    wl["has_setup"] = wl["n_setups"] > 0
    wl["traded"] = wl["n_trades"] > 0
    trades = bt.trades[_ALLOWED_TRADE_COLS].copy()
    trades["year"] = trades["session_date"].dt.year

    def pct(x: pd.Series) -> float:
        return round(100.0 * x.mean(), 1)

    out: dict = {}
    out["windows_with_setup_pct_by_window"] = wl.groupby("window")["has_setup"].agg(["sum", "count", pct]).rename(
        columns={"sum": "with_setup", "count": "eligible", "pct": "pct"})
    out["windows_with_setup_pct_by_year_window"] = wl.pivot_table(index="year", columns="window", values="has_setup", aggfunc=pct)
    traded_days = pd.DatetimeIndex(trades["session_date"].unique())
    days = pd.DataFrame({"session_date": trading_days})
    days["year"] = days["session_date"].dt.year
    days["n_trades"] = days["session_date"].map(trades.groupby("session_date").size()).fillna(0).astype(int)
    out["days_with_trade_pct_overall"] = pd.Series({"trading_days": len(days), "days_with_trade": int((days["n_trades"] > 0).sum()),
                                                   "pct": round(100.0 * (days["n_trades"] > 0).mean(), 1)})
    out["days_with_trade_pct_by_year"] = days.groupby("year")["n_trades"].agg(
        trading_days="count", days_with_trade=lambda s: int((s > 0).sum()), pct=lambda s: round(100.0 * (s > 0).mean(), 1))
    out["trades_per_year"] = trades.groupby("year").size().rename("trades")
    out["trades_per_day_distribution"] = days["n_trades"].value_counts().sort_index().rename("days")
    reason = wl.assign(reason_final=wl["final_reason"].where(~wl["traded"], "traded"))
    cols = [c for c in REASON_ORDER if c in set(reason["reason_final"])] + \
        sorted(set(reason["reason_final"]) - set(REASON_ORDER))
    out["reason_mix_by_year"] = pd.crosstab(reason["year"], reason["reason_final"]).reindex(columns=cols, fill_value=0)
    out["reason_mix_by_window"] = pd.crosstab(reason["window"], reason["reason_final"]).reindex(columns=cols, fill_value=0)
    out["ineligible_windows"] = pd.Series({"ineligible": int((bt.windows["status"] != "ok").sum()),
                                           "total": len(bt.windows)})
    return out


def format_diagnostic(name: str, d: dict) -> str:
    lines = [f"===== {name} ====="]
    for k, v in d.items():
        lines += [f"-- {k}", v.to_string() if hasattr(v, "to_string") else str(v), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--configs", nargs="+", default=["as_taught_5m", "as_taught_1m", "as_traded"])
    ap.add_argument("--cache", default=str(FEATURE_CACHE))
    ap.add_argument("--out", default=str(LAB / "analysis" / "diagnostics"))
    a = ap.parse_args()
    named = load_named_configs()
    store, inst, days = prepare(a.symbol, cache_dir=a.cache)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    text = []
    for name in a.configs:
        bt = run_backtest(store, named[name], inst, sessions=days)
        text.append(format_diagnostic(name, diagnose(bt, days)))
    report = f"{a.symbol} frequency diagnostic ({days.min():%Y-%m-%d}..{days.max():%Y-%m-%d}, non-holdout only)\n\n" + "\n".join(text)
    print(report)
    (Path(a.out) / f"frequency_{a.symbol}.txt").write_text(report)


if __name__ == "__main__":
    main()
