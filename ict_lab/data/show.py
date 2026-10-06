"""Print raw OHLCV for a window so it can be eyeballed against a chart.

  python -m ict_lab.data.show ES "2024-03-08 09:30" "2024-03-08 10:00" --series unadjusted

Window times are US/Eastern by default (--tz UTC to override). Reads the RAW parquet file
(not the cache). Holdout dates are refused unless --include-holdout is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .continuous import file_paths
from .loader import read_canonical
from .sessions import ET, session_date
from .store import CLEAN_DIR, RAW_DIR, holdout_cutoff, load_config


def window(root: str, start: str, end: str, series: str = "unadjusted", tz: str = ET,
           include_holdout: bool = False, raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR,
           config: dict | None = None) -> pd.DataFrame:
    config = load_config() if config is None else config
    df = read_canonical(file_paths(root, raw_dir)[series], config["source"]).dropna(subset=["ts"]).set_index("ts").sort_index()
    t0, t1 = (pd.Timestamp(t).tz_localize(tz) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t) for t in (start, end))
    if not include_holdout:
        cutoff = holdout_cutoff(clean_dir, config)
        if session_date(pd.DatetimeIndex([t1.tz_convert("UTC")])).iloc[0] >= cutoff:
            raise PermissionError(f"window reaches the holdout (>= {cutoff:%Y-%m-%d}); pass include_holdout=True / --include-holdout")
    out = df[(df.index >= t0) & (df.index <= t1)]
    out = out.set_index(out.index.tz_convert(tz).rename(f"time_{tz}"))
    return out[["open", "high", "low", "close", "volume", "contract"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--series", choices=["backadjusted", "unadjusted"], default="unadjusted")
    ap.add_argument("--tz", default=ET)
    ap.add_argument("--include-holdout", action="store_true")
    a = ap.parse_args()
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(window(a.root, a.start, a.end, a.series, a.tz, a.include_holdout))


if __name__ == "__main__":
    main()
