"""Low-level helpers for reading vendor files into a canonical frame (UTC `ts` + OHLCV + symbol)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CANON = ["open", "high", "low", "close", "volume"]


def resolve(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_timestamps(df: pd.DataFrame, src: dict) -> pd.Series:
    """UTC timestamps; NaT where unparseable or DST-ambiguous. Naive values are read in `src['tz']`."""
    if src.get("date_column") and src.get("time_column"):
        raw = df[src["date_column"]].astype(str) + " " + df[src["time_column"]].astype(str)
    else:
        col = resolve(df, src["columns"]["timestamp"])
        if col is None:
            raise ValueError(f"no timestamp column found; have {list(df.columns)}")
        raw = df[col]
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_datetime(raw, unit=src.get("epoch_unit", "ns"), utc=True)
    ts = pd.to_datetime(raw, errors="coerce", format="mixed")
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(src.get("tz", "UTC"), ambiguous="NaT", nonexistent="NaT")
    return ts.dt.tz_convert("UTC")


def read_canonical(path: Path, src: dict) -> pd.DataFrame:
    """One vendor file -> columns ts, open..volume, contract (positional order preserved, NaT kept)."""
    df = read_frame(path)
    if isinstance(df.index, pd.DatetimeIndex):  # timestamp stored as the index
        df = df.rename_axis("timestamp").reset_index()
    out = pd.DataFrame({"ts": parse_timestamps(df, src)})
    for name in CANON + ["symbol"]:
        col = resolve(df, src["columns"][name])
        if col is None:
            raise ValueError(f"{path.name}: missing '{name}' column; have {list(df.columns)}")
        out[name] = df[col].to_numpy()
    for name in CANON:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    return out.rename(columns={"symbol": "contract"})
