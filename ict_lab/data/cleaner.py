"""Clean raw bars: validate, dedupe, resolve contract rolls, and report (never silently fill) gaps."""
from __future__ import annotations

import pandas as pd

from .sessions import in_session_mask, session_date


def drop_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove rows that cannot be real bars. Counts are returned so nothing disappears silently."""
    n0 = len(df)
    nan = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    nonpos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    inconsistent = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
    )
    negvol = df["volume"] < 0
    bad = nan | nonpos | inconsistent | negvol
    report = {
        "rows_in": n0,
        "dropped_nan": int(nan.sum()),
        "dropped_nonpositive_price": int((nonpos & ~nan).sum()),
        "dropped_ohlc_inconsistent": int((inconsistent & ~nan & ~nonpos).sum()),
        "dropped_negative_volume": int((negvol & ~nan).sum()),
    }
    return df[~bad], report


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Sort by time; for duplicate (ts, symbol) keep the row with the most volume."""
    keys = ["symbol"] if "symbol" in df.columns else []
    df = df.sort_values("volume", kind="stable").reset_index()
    before = len(df)
    df = df.drop_duplicates(subset=["ts"] + keys, keep="last").set_index("ts").sort_index(kind="stable")
    return df, before - len(df)


def front_month(df: pd.DataFrame, root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one contract per session: the highest-volume contract of the PREVIOUS session.

    Using the prior session avoids look-ahead (today's volume isn't known at today's open).
    Prices are left UNADJUSTED so levels match what was actually traded; roll dates are returned
    so downstream code can avoid treating the roll jump as a real move.
    """
    df = df[df["symbol"].str.startswith(root)]
    df = df.assign(session_date=session_date(df.index).to_numpy())
    vol = df.groupby(["session_date", "symbol"])["volume"].sum().unstack(fill_value=0)
    leader = vol.idxmax(axis=1)
    # Sticky: switch only when another contract out-trades the current one, so a contract
    # never flips back and forth around the roll.
    chosen, current = {}, None
    for day in vol.index:
        cand = leader[day]
        if current is None or (cand != current and vol.loc[day, cand] > vol.loc[day, current]):
            current = cand
        chosen[day] = current
    leaders = pd.Series(chosen)
    front = leaders.shift(1).fillna(leaders)  # contract held on day d = leader as of d-1
    df["front"] = df["session_date"].map(front)
    out = df[df["symbol"] == df["front"]].drop(columns="front")
    changed = front != front.shift()
    rolls = pd.DataFrame(
        {"from_contract": front.shift()[changed], "to_contract": front[changed]}
    ).iloc[1:]
    rolls.index.name = "first_session"
    return out, rolls


def find_gaps(df: pd.DataFrame, max_gap_minutes: int) -> pd.DataFrame:
    """In-session intervals with no bars longer than the threshold (halts, holidays, data holes)."""
    idx = df.index
    delta = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    gap = delta > pd.Timedelta(minutes=max_gap_minutes)
    starts = idx[:-1][gap.to_numpy()]
    ends = idx[1:][gap.to_numpy()]
    rows = []
    for s, e in zip(starts, ends):
        # Expected in-session minutes missing between the two bars (excludes daily break / weekend).
        minutes = pd.date_range(s + pd.Timedelta(minutes=1), e - pd.Timedelta(minutes=1), freq="1min")
        missing = int(in_session_mask(minutes).sum()) if len(minutes) else 0
        if missing > max_gap_minutes:
            rows.append({"start": s, "end": e, "missing_minutes": missing})
    return pd.DataFrame(rows, columns=["start", "end", "missing_minutes"])


def clean(raw: pd.DataFrame, root: str, max_gap_minutes: int = 5, roll: str = "volume") -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Full pipeline for one instrument. Returns (clean bars, quality report, gaps, rolls)."""
    report: dict = {"root": root}
    df, inv = drop_invalid(raw)
    report.update(inv)
    df, report["dropped_duplicates"] = dedupe(df)

    rolls = pd.DataFrame()
    if "symbol" in df.columns:
        if roll != "volume":
            raise ValueError(f"unknown roll method {roll!r}")
        df, rolls = front_month(df, root)
        report["rolls"] = len(rolls)
    df = df.assign(session_date=session_date(df.index).to_numpy())

    gaps = find_gaps(df, max_gap_minutes)
    report.update(
        rows_out=len(df),
        start=str(df.index.min()),
        end=str(df.index.max()),
        gap_count=len(gaps),
        gap_minutes=int(gaps["missing_minutes"].sum()) if len(gaps) else 0,
        out_of_session_bars=int((~in_session_mask(df.index)).sum()),
    )
    return df, report, gaps, rolls
