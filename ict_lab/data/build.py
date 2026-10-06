"""Build the cache from raw triplets:  python -m ict_lab.data.build [--config PATH]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import continuous
from .quality import format_report, quality_report, save_report
from .sessions import add_session_columns
from .store import CLEAN_DIR, CONFIG_PATH, RAW_DIR, load_config, split_holdout, write_cache, write_meta, write_rolls


def build(config: dict, raw_dir: Path = RAW_DIR, out_dir: Path = CLEAN_DIR) -> dict[str, dict]:
    loaded = {}
    for root in config["instruments"]:
        if not all(p.exists() for p in continuous.file_paths(root, raw_dir).values()):
            continue
        merged, rolls = continuous.read_triplet(root, raw_dir, config["source"])
        loaded[root] = (add_session_columns(merged), rolls)
    if not loaded:
        raise FileNotFoundError(f"no complete {{ROOT}}_1m_unadjusted/_1m_backadjusted/_rolls set in {raw_dir}")

    pinned = (config.get("holdout") or {}).get("cutoff_date")
    if pinned:
        cutoff, source = pd.Timestamp(pinned).normalize(), "config"
    else:
        last = max(m["session_date"].max() for m, _ in loaded.values())
        cutoff = (last - pd.DateOffset(years=config["holdout"]["years"])).normalize()
        source = "auto"
        print(f"holdout cutoff not pinned; using {cutoff:%Y-%m-%d} (last session - {config['holdout']['years']}y). "
              f"Pin it with holdout.cutoff_date in configs/data.yaml.")
    write_meta(out_dir, cutoff, source)

    reports = {}
    for root, (raw, rolls) in loaded.items():
        clean = continuous.clean_merged(raw)
        clean = add_session_columns(clean.drop(columns="session_date", errors="ignore"))
        # Full-history cache (the holdout is written but excluded by every loader by default).
        for series, frame in continuous.finalize(clean, rolls).items():
            write_cache(root, series, frame, out_dir)
        write_rolls(root, rolls, out_dir)

        # Everything below inspects development data only.
        raw_dev, _ = split_holdout(raw, cutoff)
        clean_dev, clean_hold = split_holdout(clean, cutoff)
        if clean_dev.empty:
            raise ValueError(f"{root}: no development data before holdout cutoff {cutoff:%Y-%m-%d}")
        rolls_dev = rolls[rolls["date"] < cutoff]
        report = quality_report(root, raw_dev, clean_dev, rolls_dev, n_holdout_rows=len(clean_hold))
        report["roll_findings"] = continuous.check_rolls(clean_dev, rolls_dev)
        save_report(report, out_dir / "quality")
        reports[root] = report
        print(format_report(report) + "\n")
    return reports


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG_PATH))
    args = ap.parse_args()
    build(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
