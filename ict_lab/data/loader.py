"""Read vendor 1-minute bar files into a canonical frame: UTC DatetimeIndex + OHLCV (+ symbol)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANON = ["open", "high", "low", "close", "volume"]


def _resolve(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)  # compression inferred from .gz/.zip/.bz2


def _timestamps(df: pd.DataFrame, src: dict) -> tuple[pd.Series, int]:
    """Return UTC timestamps (NaT where unparseable/ambiguous) and the count of failures."""
    cols = src["columns"]
    if src.get("date_column") and src.get("time_column"):
        raw = df[src["date_column"]].astype(str) + " " + df[src["time_column"]].astype(str)
    else:
        col = _resolve(df, cols["timestamp"])
        if col is None:
            raise ValueError(f"no timestamp column found; have {list(df.columns)}")
        raw = df[col]

    if pd.api.types.is_numeric_dtype(raw):
        ts = pd.to_datetime(raw, unit=src.get("epoch_unit", "ns"), utc=True)
    else:
        ts = pd.to_datetime(raw, errors="coerce", format="mixed")
        if ts.dt.tz is None:  # naive -> interpret in the vendor's zone
            ts = ts.dt.tz_localize(src.get("tz", "UTC"), ambiguous="NaT", nonexistent="NaT")
        ts = ts.dt.tz_convert("UTC")
    return ts, int(ts.isna().sum())


def read_raw(path: str | Path, src: dict) -> tuple[pd.DataFrame, dict]:
    """Load one raw file. Returns (frame indexed by UTC `ts`, per-file stats)."""
    path = Path(path)
    df = _read_file(path)
    ts, bad_ts = _timestamps(df, src)

    out = pd.DataFrame({"ts": ts})
    for name in CANON + ["symbol"]:
        col = _resolve(df, src["columns"][name])
        if col is None:
            if name == "symbol":
                continue
            raise ValueError(f"{path.name}: missing column for '{name}'; have {list(df.columns)}")
        out[name] = df[col].to_numpy()
    for name in CANON:
        out[name] = pd.to_numeric(out[name], errors="coerce")

    out = out.dropna(subset=["ts"]).set_index("ts")
    out.index.name = "ts"
    return out, {"file": path.name, "rows_read": len(df), "bad_timestamps": bad_ts}


def read_raw_dir(raw_dir: str | Path, src: dict) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(Path(raw_dir).glob(src.get("glob", "*.csv*")))
    if not files:
        raise FileNotFoundError(f"no files matching {src.get('glob')} in {raw_dir}")
    frames, stats = zip(*(read_raw(f, src) for f in files))
    return pd.concat(frames), list(stats)
