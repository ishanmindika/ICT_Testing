"""Load the unadjusted / back-adjusted / rolls triplet for one root and merge it into one aligned frame.

Convention in the merged frame:
  open/high/low/close      real traded prices (unadjusted) - use for fills, stops, price levels
  adj_open..adj_close      back-adjusted continuous series - use for structure/indicators across rolls
  offset                   close - adj_close (what the back-adjustment added at that bar)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import loader
from .loader import CANON, _read_file
from .sessions import session_date

PRICE = ["open", "high", "low", "close"]
ROLL_COLS = {
    "date": ["date", "roll_date", "timestamp", "datetime"],
    "old_contract": ["old_contract", "from_contract", "old", "from"],
    "new_contract": ["new_contract", "to_contract", "new", "to"],
    "adjustment": ["adjustment", "price_adjustment", "adjustment_applied", "adj", "price_adjustment_applied"],
}


def file_paths(root: str, raw_dir: Path) -> dict[str, Path]:
    return {k: Path(raw_dir) / f"{root}_{k}" for k in ("1m_unadjusted.parquet", "1m_backadjusted.parquet", "rolls.csv")}


def read_rolls(path: Path, tz: str = "UTC") -> pd.DataFrame:
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


def _load(path: Path, src: dict) -> pd.DataFrame:
    df = _read_file(path)
    if isinstance(df.index, pd.DatetimeIndex):  # timestamp stored as the index
        df = df.rename_axis("timestamp").reset_index()
    ts, bad = loader._timestamps(df, src)
    out = pd.DataFrame({"ts": ts})
    for name in CANON + ["symbol"]:
        col = loader._resolve(df, src["columns"][name])
        if col is None:
            raise ValueError(f"{path.name}: missing '{name}' column; have {list(df.columns)}")
        out[name] = df[col].to_numpy()
    for name in CANON:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.rename(columns={"symbol": "contract"})
    out["_bad_ts"] = ts.isna().to_numpy()
    return out


def load_triplet(root: str, raw_dir: Path, src: dict) -> tuple[pd.DataFrame, dict]:
    """Merge the three files. Raises if the parquets are not aligned; collects softer findings in the report."""
    p = file_paths(root, raw_dir)
    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"missing: {missing}")
    un, adj = _load(p["1m_unadjusted.parquet"], src), _load(p["1m_backadjusted.parquet"], src)
    rolls = read_rolls(p["rolls.csv"])
    report: dict = {"root": root, "rows_unadjusted": len(un), "rows_backadjusted": len(adj)}

    if len(un) != len(adj):
        raise ValueError(f"{root}: row counts differ ({len(un)} vs {len(adj)})")
    # Rows are compared positionally, so check alignment BEFORE dropping or sorting anything.
    if not un["ts"].equals(adj["ts"]):
        raise ValueError(f"{root}: timestamps differ between unadjusted and back-adjusted files")
    for c in ("volume", "contract"):
        if not un[c].reset_index(drop=True).equals(adj[c].reset_index(drop=True)):
            raise ValueError(f"{root}: '{c}' differs between unadjusted and back-adjusted files")

    df = un.drop(columns="_bad_ts")
    for c in PRICE:
        df[f"adj_{c}"] = adj[c].to_numpy()
    report["bad_timestamps"] = int(un["_bad_ts"].sum())
    df = df.dropna(subset=["ts"])

    # A bar is kept only if it is valid in BOTH series so the two stay aligned.
    bad = {}
    for pre in ("", "adj_"):
        o, h, l, c = (df[f"{pre}{x}"] for x in PRICE)
        bad[pre] = o.isna() | h.isna() | l.isna() | c.isna() | (h < pd.concat([o, c, l], axis=1).max(axis=1)) | (l > pd.concat([o, c, h], axis=1).min(axis=1))
    bad_any = bad[""] | bad["adj_"] | df["volume"].isna() | (df["volume"] < 0) | (df[PRICE] <= 0).any(axis=1)  # adjusted prices may be <=0; unadjusted may not
    report["dropped_invalid"] = int(bad_any.sum())
    df = df[~bad_any]

    dup = df.duplicated(subset="ts", keep=False)
    report["duplicate_timestamps"] = int(dup.sum())
    df = df.sort_values(["ts", "volume"], kind="stable").drop_duplicates("ts", keep="last").set_index("ts")
    df["session_date"] = session_date(df.index).to_numpy()
    df["offset"] = df["close"] - df["adj_close"]

    report.update(_check_rolls(df, rolls))
    report.update(rows_out=len(df), start=str(df.index.min()), end=str(df.index.max()))
    return df, {**report, "rolls": rolls}


def _check_rolls(df: pd.DataFrame, rolls: pd.DataFrame, tol: float = 1e-6) -> dict:
    """Cross-check rolls.csv against what the two price files imply. Findings are reported, not raised."""
    findings: list[str] = []
    run = (df["contract"] != df["contract"].shift()).cumsum()
    runs = df.groupby(run).agg(contract=("contract", "first"), first_session=("session_date", "first"),
                               off_min=("offset", "min"), off_max=("offset", "max"), off=("offset", "first"))
    unstable = runs[(runs["off_max"] - runs["off_min"]).abs() > max(tol, 1e-4)]
    for _, r in unstable.iterrows():
        findings.append(f"offset not constant within {r['contract']} run starting {r['first_session']:%Y-%m-%d}")

    # Additive adjustment <=> constant offset per run; ratio adjustment would fail the check above.
    boundaries = runs.iloc[1:].assign(prev_contract=runs["contract"].shift().iloc[1:], prev_off=runs["off"].shift().iloc[1:])
    boundaries["implied"] = boundaries["off"] - boundaries["prev_off"]
    if len(boundaries) != len(rolls):
        findings.append(f"{len(boundaries)} contract changes in bars vs {len(rolls)} rows in rolls.csv")
    n = min(len(boundaries), len(rolls))
    for (_, b), (_, r) in zip(boundaries.iloc[:n].iterrows(), rolls.iloc[:n].iterrows()):
        if b["contract"] != r["new_contract"] or b["prev_contract"] != r["old_contract"]:
            findings.append(f"contracts differ at roll {r['date']:%Y-%m-%d}: bars {b['prev_contract']}->{b['contract']}, csv {r['old_contract']}->{r['new_contract']}")
        elif not (np.isclose(abs(b["implied"]), abs(r["adjustment"]), atol=1e-6)):
            findings.append(f"adjustment at {r['date']:%Y-%m-%d}: bars imply {b['implied']:.4f}, csv says {r['adjustment']:.4f}")
        elif abs((b["first_session"] - r["date"]).days) > 1:
            findings.append(f"roll date {r['date']:%Y-%m-%d} but new contract first trades in file on {b['first_session']:%Y-%m-%d}")
    return {"roll_findings": findings, "roll_count": len(rolls)}
