"""Build cleaned parquet from raw vendor files:  python -m ict_lab.data.build [--config PATH]"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from . import continuous
from .cleaner import clean, find_gaps
from .loader import read_raw_dir
from .sessions import in_session_mask
from .store import CLEAN_DIR, LAB, RAW_DIR, save_clean


def build(config: dict, raw_dir: Path = RAW_DIR, out_dir: Path = CLEAN_DIR) -> dict[str, dict]:
    triplets = {r: continuous.file_paths(r, raw_dir) for r in config["instruments"]}
    triplets = [r for r, p in triplets.items() if all(f.exists() for f in p.values())]
    if triplets:
        return build_continuous(config, triplets, raw_dir, out_dir)
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


def build_continuous(config: dict, roots: list[str], raw_dir: Path, out_dir: Path) -> dict[str, dict]:
    reports = {}
    for root in roots:
        df, report = continuous.load_triplet(root, raw_dir, config["source"])
        rolls = report.pop("rolls")
        gaps = find_gaps(df, config["clean"]["max_gap_minutes"])
        report.update(gap_count=len(gaps), gap_minutes=int(gaps["missing_minutes"].sum()) if len(gaps) else 0,
                      out_of_session_bars=int((~in_session_mask(df.index)).sum()))
        save_clean(df, root, report, gaps, rolls, out_dir)
        reports[root] = report
        print(f"{root}: {report['rows_out']:,} bars {report['start'][:10]}..{report['end'][:10]} "
              f"| dropped {report['rows_unadjusted'] - report['rows_out']:,} | gaps {report['gap_count']} "
              f"| rolls {report['roll_count']}")
        for f in report["roll_findings"]:
            print(f"  WARN {f}")
    return reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(LAB / "configs" / "data.yaml"))
    args = ap.parse_args()
    build(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
