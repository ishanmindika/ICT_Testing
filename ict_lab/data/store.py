"""Cached parquet store (partitioned by symbol / series / year) and the holdout-aware loaders."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

LAB = Path(__file__).resolve().parents[1]
RAW_DIR = LAB / "data" / "raw"
CLEAN_DIR = LAB / "data" / "clean"
CONFIG_PATH = LAB / "configs" / "data.yaml"
BARS = "bars"
SERIES = ("backadjusted", "unadjusted")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _utc(t: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---- holdout -------------------------------------------------------------------------------

def holdout_cutoff(clean_dir: Path = CLEAN_DIR, config: dict | None = None) -> pd.Timestamp:
    """First session_date that belongs to the holdout. Config wins; otherwise the value the last
    build recorded. Raises if neither exists, so the holdout can never be silently exposed."""
    config = load_config() if config is None else config
    pinned = (config.get("holdout") or {}).get("cutoff_date")
    if pinned:
        return pd.Timestamp(pinned).normalize()
    meta = Path(clean_dir) / "meta.json"
    if meta.exists():
        return pd.Timestamp(json.loads(meta.read_text())["holdout_cutoff"]).normalize()
    raise RuntimeError("holdout cutoff unknown: set holdout.cutoff_date in configs/data.yaml or run the build")


def split_holdout(df: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(development, holdout) by session_date, so a session is never split."""
    is_dev = df["session_date"] < cutoff
    return df[is_dev], df[~is_dev]


def write_meta(clean_dir: Path, cutoff: pd.Timestamp, source: str) -> None:
    Path(clean_dir).mkdir(parents=True, exist_ok=True)
    (Path(clean_dir) / "meta.json").write_text(
        json.dumps({"holdout_cutoff": f"{cutoff:%Y-%m-%d}", "cutoff_source": source}, indent=2))


# ---- cache ---------------------------------------------------------------------------------

def write_cache(root: str, series: str, df: pd.DataFrame, clean_dir: Path = CLEAN_DIR) -> None:
    """Write `bars/symbol=ES/series=backadjusted/year=2021/*.parquet` (year = session_date year)."""
    base = Path(clean_dir) / BARS
    target = base / f"symbol={root}" / f"series={series}"
    shutil.rmtree(target, ignore_errors=True)
    part = df.assign(symbol=root, series=series, year=df["session_date"].dt.year.astype("int32"))
    pq.write_to_dataset(pa.Table.from_pandas(part), base, partition_cols=["symbol", "series", "year"])


def write_rolls(root: str, rolls: pd.DataFrame, clean_dir: Path = CLEAN_DIR) -> None:
    d = Path(clean_dir) / "rolls"
    d.mkdir(parents=True, exist_ok=True)
    rolls.to_parquet(d / f"{root}.parquet", index=False)


def _read_cache(root: str, series: str, clean_dir: Path) -> pd.DataFrame:
    if series not in SERIES:
        raise ValueError(f"price_series must be one of {SERIES}, got {series!r}")
    path = Path(clean_dir) / BARS
    df = pd.read_parquet(path, filters=[("symbol", "=", root), ("series", "=", series)])
    return df.drop(columns=["symbol", "series", "year"]).sort_index(kind="stable")


# ---- loaders: every one of these excludes the holdout unless told otherwise -------------------

def load_bars(root: str, price_series: str = "backadjusted", start: str | pd.Timestamp | None = None,
              end: str | pd.Timestamp | None = None, *, include_holdout: bool = False,
              clean_dir: Path = CLEAN_DIR, config: dict | None = None) -> pd.DataFrame:
    """Cleaned 1-minute bars, UTC index. Default is the BACK-ADJUSTED series (strategy logic);
    pass price_series="unadjusted" for charting/verification. `start`/`end` inclusive; naive = UTC.
    The holdout is excluded unless include_holdout=True."""
    df = _read_cache(root, price_series, clean_dir)
    if not include_holdout:
        df, _ = split_holdout(df, holdout_cutoff(clean_dir, config))
    if start is not None:
        df = df[df.index >= _utc(start)]
    if end is not None:
        df = df[df.index <= _utc(end)]
    return df


def load_rolls(root: str, *, include_holdout: bool = False, clean_dir: Path = CLEAN_DIR,
               config: dict | None = None) -> pd.DataFrame:
    """Roll log: date, old_contract, new_contract, adjustment. Use with is_roll_day to reconcile series."""
    rolls = pd.read_parquet(Path(clean_dir) / "rolls" / f"{root}.parquet")
    if not include_holdout:
        rolls = rolls[rolls["date"] < holdout_cutoff(clean_dir, config)]
    return rolls.reset_index(drop=True)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate 1-minute bars to a higher timeframe, labelled by bar open time."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for extra in ("contract", "session_date"):
        if extra in df.columns:
            agg[extra] = "last"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])
