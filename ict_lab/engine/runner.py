"""Run a named config on a symbol/date range.

    python -m ict_lab.engine.runner as_traded NQ 2024-03-01 2024-03-31 [--cache DIR]

Detectors see `context_days` of history before `start` (warm-up for ATR, swings, prior-session levels); only
sessions inside [start, end] are traded. The holdout is excluded by the data loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..data.store import CLEAN_DIR, load_bars
from .config import Instrument, StrategyConfig, load_named_configs
from .execute import Backtest, run_backtest
from .store import FeatureStore

LAB = Path(__file__).resolve().parents[1]
FEATURE_CACHE = CLEAN_DIR / "features"


def prepare(symbol: str, start: str | None = None, end: str | None = None, context_days: int = 45,
            cache_dir: Path | str | None = None, bars: pd.DataFrame | None = None, **load_kw):
    """-> (FeatureStore, Instrument, sessions). `bars` may be passed instead of loading from the cache."""
    bars = load_bars(symbol, "backadjusted", **load_kw) if bars is None else bars
    s0 = None if start is None else pd.Timestamp(start)
    s1 = None if end is None else pd.Timestamp(end)
    if s0 is not None:
        bars = bars[bars["session_date"] >= s0 - pd.Timedelta(days=context_days)]
    if s1 is not None:
        bars = bars[bars["session_date"] <= s1]
    days = pd.DatetimeIndex(sorted(bars["session_date"].unique()))
    if s0 is not None:
        days = days[days >= s0]
    return FeatureStore(bars, symbol, cache_dir), Instrument.from_config(symbol), days


def run_named(name_or_cfg: str | StrategyConfig, symbol: str, start: str | None = None, end: str | None = None,
              **kw) -> Backtest:
    cfg = load_named_configs()[name_or_cfg] if isinstance(name_or_cfg, str) else name_or_cfg
    store, inst, days = prepare(symbol, start, end, **kw)
    return run_backtest(store, cfg, inst, sessions=days)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("symbol")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--cache", default=str(FEATURE_CACHE))
    a = ap.parse_args()
    bt = run_named(a.config, a.symbol, a.start, a.end, cache_dir=a.cache)
    print(f"{a.config} {a.symbol} {a.start}..{a.end}: {len(bt.trades)} trades, {len(bt.no_trades)} windows without a trade")
    print(bt.no_trades["reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
