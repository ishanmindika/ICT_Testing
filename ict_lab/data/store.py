"""Parquet persistence for cleaned bars plus a JSON manifest describing how each file was built."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LAB = Path(__file__).resolve().parents[1]
RAW_DIR = LAB / "data" / "raw"
CLEAN_DIR = LAB / "data" / "clean"


def save_clean(df: pd.DataFrame, root: str, report: dict, gaps: pd.DataFrame,
               rolls: pd.DataFrame | None = None, out_dir: Path = CLEAN_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{root}_1m.parquet"
    df.to_parquet(path)
    gaps.to_parquet(out_dir / f"{root}_1m.gaps.parquet")
    if rolls is not None and len(rolls):
        rolls.to_parquet(out_dir / f"{root}_1m.rolls.parquet")
    (out_dir / f"{root}_1m.manifest.json").write_text(json.dumps(report, indent=2, default=str))
    return path


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_bars(root: str, start=None, end=None, clean_dir: Path = CLEAN_DIR) -> pd.DataFrame:
    """Cleaned 1-minute bars, UTC index. `start`/`end` are inclusive and may be tz-naive (read as UTC)."""
    df = pd.read_parquet(Path(clean_dir) / f"{root}_1m.parquet")
    if start is not None:
        df = df[df.index >= _utc(start)]
    if end is not None:
        df = df[df.index <= _utc(end)]
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 1-minute bars to a higher timeframe (e.g. '5min', '1h'), labelled by bar open time."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for extra in ("symbol", "session_date"):
        if extra in df.columns:
            agg[extra] = "last"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
