"""Load, align, validate and clean the {ROOT}_1m_unadjusted / _1m_backadjusted / _rolls triplet.

The back-adjusted series drives strategy logic; unadjusted is for charts and manual verification.
Both are returned with identical index, volume, contract and session/flag columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .loader import read_canonical
from .sessions import add_session_columns

PRICE = ["open", "high", "low", "close"]
SERIES = ("backadjusted", "unadjusted")
ROLL_COLS = {
    "date": ["date", "roll_date", "timestamp", "datetime"],
    "old_contract": ["old_contract", "from_contract", "old", "from"],
    "new_contract": ["new_contract", "to_contract", "new", "to"],
    "adjustment": ["adjustment", "price_adjustment", "adjustment_applied", "adj", "price_adjustment_applied"],
}


def file_paths(root: str, raw_dir: Path) -> dict[str, Path]:
    d = Path(raw_dir)
    return {"unadjusted": d / f"{root}_1m_unadjusted.parquet",
            "backadjusted": d / f"{root}_1m_backadjusted.parquet",
            "rolls": d / f"{root}_rolls.csv"}


def read_rolls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in df.columns}
    out = {}
    for name, aliases in ROLL_COLS.items():
        col = next((lower[a] for a in aliases if a in lower), None)
        if col is None:
            raise ValueError(f"{path.name}: no column for '{name}'; have {list(df.columns)}")
        out[name] = df[col]
    rolls = pd.DataFrame(out)
    rolls["date"] = pd.to_datetime(rolls["date"]).dt.tz_localize(None).dt.normalize()
    rolls["adjustment"] = pd.to_numeric(rolls["adjustment"])
    return rolls.sort_values("date").reset_index(drop=True)


def read_triplet(root: str, raw_dir: Path, src: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (merged, rolls). `merged` is sorted, indexed by UTC ts, with real prices in
    open..close and back-adjusted prices in adj_*. Raises unless the two parquets align exactly."""
    p = file_paths(root, raw_dir)
    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"{root}: missing {missing}")
    un, ba = read_canonical(p["unadjusted"], src), read_canonical(p["backadjusted"], src)

    # Alignment is asserted on the raw files, positionally, before anything is dropped or sorted.
    if len(un) != len(ba):
        raise AssertionError(f"{root}: row counts differ (unadjusted {len(un)}, backadjusted {len(ba)})")
    if not un["ts"].equals(ba["ts"]):
        n = int((un["ts"].ne(ba["ts"]) & ~(un["ts"].isna() & ba["ts"].isna())).sum())
        raise AssertionError(f"{root}: timestamp index differs between unadjusted and backadjusted ({n} rows)")
    for c in ("volume", "contract"):
        if not un[c].equals(ba[c]):
            raise AssertionError(f"{root}: '{c}' differs between unadjusted and backadjusted")

    merged = un.copy()
    for c in PRICE:
        merged[f"adj_{c}"] = ba[c].to_numpy()
    merged = merged.dropna(subset=["ts"]).sort_values("ts", kind="stable").set_index("ts")
    return merged, read_rolls(p["rolls"])


def ohlc_problems(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Boolean columns: high_lt_low, open_outside_range, close_outside_range."""
    o, h, l, c = (df[f"{prefix}{x}"] for x in PRICE)
    return pd.DataFrame({
        "high_lt_low": h < l,
        "open_outside_range": (o > h) | (o < l),
        "close_outside_range": (c > h) | (c < l),
    })


def clean_merged(merged: pd.DataFrame) -> pd.DataFrame:
    """Drop rows invalid in EITHER series (so the two stay aligned) and de-duplicate timestamps.
    Zero-volume bars are kept (they are reported, not removed)."""
    bad = merged[PRICE + [f"adj_{c}" for c in PRICE] + ["volume"]].isna().any(axis=1)
    bad |= ohlc_problems(merged).any(axis=1) | ohlc_problems(merged, "adj_").any(axis=1)
    bad |= merged["volume"] < 0
    bad |= (merged[PRICE] <= 0).any(axis=1)  # real prices are positive; adjusted ones may legitimately not be
    df = merged[~bad]
    df = df.sort_values("volume", kind="stable")  # stable sort, so ties keep file order
    df = df[~df.index.duplicated(keep="last")].sort_index(kind="stable")
    return df


def finalize(df: pd.DataFrame, rolls: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Add session/killzone columns and is_roll_day, then split into the two series frames."""
    df = add_session_columns(df)
    df["is_roll_day"] = df["session_date"].isin(set(rolls["date"]))
    extra = ["volume", "contract", "session_date", "rth", "killzone_london", "killzone_ny_am",
             "killzone_ny_pm", "is_roll_day"]
    back = df[[f"adj_{c}" for c in PRICE]].rename(columns=lambda c: c.removeprefix("adj_")).join(df[extra])
    real = df[PRICE].join(df[extra])
    return {"backadjusted": back, "unadjusted": real}


def check_rolls(df: pd.DataFrame, rolls: pd.DataFrame, tol: float = 1e-4) -> list[str]:
    """Cross-check rolls.csv against what the two price series imply. Findings are returned, not raised.
    `df` is the merged frame (needs close, adj_close, contract, session_date)."""
    findings: list[str] = []
    offset = df["close"] - df["adj_close"]
    run = (df["contract"] != df["contract"].shift()).cumsum()
    runs = pd.DataFrame({"contract": df["contract"], "session_date": df["session_date"], "offset": offset}) \
        .groupby(run).agg(contract=("contract", "first"), first_session=("session_date", "first"),
                          off_min=("offset", "min"), off_max=("offset", "max"), off=("offset", "first"))
    for _, r in runs[(runs["off_max"] - runs["off_min"]).abs() > tol].iterrows():
        findings.append(f"offset not constant within {r['contract']} run starting {r['first_session']:%Y-%m-%d} "
                        "(adjustment is not additive?)")
    bounds = runs.iloc[1:].assign(prev_contract=runs["contract"].shift().iloc[1:],
                                  implied=runs["off"].diff().iloc[1:])
    if len(bounds) != len(rolls):
        findings.append(f"{len(bounds)} contract changes in bars vs {len(rolls)} rows in rolls.csv")
    for (_, b), (_, r) in zip(bounds.iterrows(), rolls.iterrows()):
        day = f"{r['date']:%Y-%m-%d}"
        if b["contract"] != r["new_contract"] or b["prev_contract"] != r["old_contract"]:
            findings.append(f"contracts differ at roll {day}: bars {b['prev_contract']}->{b['contract']}, "
                            f"csv {r['old_contract']}->{r['new_contract']}")
        elif not np.isclose(abs(b["implied"]), abs(r["adjustment"]), atol=1e-6):
            findings.append(f"adjustment at {day}: bars imply {b['implied']:.4f}, csv says {r['adjustment']:.4f}")
        elif abs((b["first_session"] - r["date"]).days) > 1:
            findings.append(f"csv roll date {day} but new contract first trades on {b['first_session']:%Y-%m-%d}")
    return findings
