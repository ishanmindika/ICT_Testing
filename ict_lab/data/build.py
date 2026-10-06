"""Build cleaned parquet from raw vendor files:  python -m ict_lab.data.build [--config PATH]"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .cleaner import clean
from .loader import read_raw_dir
from .store import CLEAN_DIR, LAB, RAW_DIR, save_clean


def build(config: dict, raw_dir: Path = RAW_DIR, out_dir: Path = CLEAN_DIR) -> dict[str, dict]:
    raw, file_stats = read_raw_dir(raw_dir, config["source"])
    roots = sorted(config["instruments"], key=len, reverse=True)  # MES before ES
    if "symbol" in raw.columns:
        sym = raw["symbol"].astype(str)
        claimed = sym.map(lambda x: next((r for r in roots if x.startswith(r)), None))
        parts = {r: raw[claimed == r] for r in roots if (claimed == r).any()}
    else:
        if len(roots) != 1:
            raise ValueError("raw files have no symbol column; configure exactly one instrument")
        parts = {roots[0]: raw}

    reports = {}
    for root, frame in parts.items():
        c = config["clean"]
        df, report, gaps, rolls = clean(frame, root, c["max_gap_minutes"], c["roll"])
        report["files"] = file_stats
        save_clean(df, root, report, gaps, rolls, out_dir)
        reports[root] = report
        print(f"{root}: {report['rows_out']:,} bars {report['start'][:10]}..{report['end'][:10]} "
              f"| dropped {report['rows_in'] - report['rows_out']:,} | gaps {report['gap_count']} "
              f"({report['gap_minutes']:,} min) | rolls {report.get('rolls', 0)}")
    return reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(LAB / "configs" / "data.yaml"))
    args = ap.parse_args()
    build(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
