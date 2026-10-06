"""Data quality report. Works on development data only (holdout rows are never inspected)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .continuous import PRICE, ohlc_problems
from .sessions import add_session_columns, in_session_mask

NY_AM_EXPECTED_BARS = 60  # 10:00-11:00 ET on 1-minute bars


def _year_counts(dates: pd.Series | pd.DatetimeIndex) -> dict[str, int]:
    s = pd.Series(1, index=pd.DatetimeIndex(dates)).groupby(pd.DatetimeIndex(dates).year).sum()
    return {str(k): int(v) for k, v in s.items()}


def missing_minutes_by_year(index: pd.DatetimeIndex, sessions: pd.Series) -> dict[str, int]:
    """In-session minutes with no bar, counted only on sessions that have at least one bar
    (fully absent days are listed separately). Early-close days count their closed hours as missing."""
    expected = pd.date_range(index.min().floor("min"), index.max().ceil("min"), freq="1min", tz="UTC")
    expected = expected[in_session_mask(expected).to_numpy()]
    missing = expected.difference(index)
    if len(missing) == 0:
        return {}
    msd = add_session_columns(pd.DataFrame(index=missing))["session_date"]
    present = set(sessions.unique())
    msd = msd[msd.isin(present)]
    return _year_counts(msd.to_numpy())


def quality_report(root: str, raw: pd.DataFrame, clean: pd.DataFrame, rolls: pd.DataFrame,
                   n_holdout_rows: int = 0) -> dict:
    """raw: merged un-dropped frame (open..close, adj_*, volume); clean: merged cleaned frame with
    session columns (open..close, adj_*, volume, session_date, killzone_ny_am); rolls: roll log.
    All inputs must already exclude the holdout."""
    r: dict = {"root": root, "holdout_rows_excluded": n_holdout_rows}
    r["date_range"] = {"start": str(clean.index.min()), "end": str(clean.index.max()),
                       "sessions": int(clean["session_date"].nunique())}
    r["rows"] = {"raw": len(raw), "clean": len(clean)}
    r["duplicate_timestamps"] = int(raw.index.duplicated().sum())
    r["zero_volume_bars"] = int((raw["volume"] == 0).sum())
    r["bad_ohlc"] = {
        label: {k: int(v.sum()) for k, v in ohlc_problems(raw, prefix).items()}
        for label, prefix in (("unadjusted", ""), ("backadjusted", "adj_"))
    }
    r["missing_minutes_by_year"] = missing_minutes_by_year(clean.index, clean["session_date"])

    am = clean[clean["killzone_ny_am"]].groupby("session_date").size()
    am = am.reindex(clean["session_date"].unique(), fill_value=0)
    short = am[am < NY_AM_EXPECTED_BARS]
    r["short_ny_am_days"] = {f"{d:%Y-%m-%d}": int(n) for d, n in short.items()}

    days = pd.DatetimeIndex(clean["session_date"].unique())
    weekdays = pd.bdate_range(days.min(), days.max())
    r["missing_trading_days"] = [f"{d:%Y-%m-%d}" for d in weekdays.difference(days)]  # includes market holidays
    r["roll_days_by_year"] = _year_counts(rolls["date"]) if len(rolls) else {}

    year = clean.index.tz_convert("America/New_York").year
    g = clean.assign(offset=clean["close"] - clean["adj_close"]).groupby(year)
    r["backadjusted_close_by_year"] = {
        str(y): {"min": float(x["adj_close"].min()), "max": float(x["adj_close"].max()),
                 "cumulative_adjustment_max": float(x["offset"].max())}
        for y, x in g
    }
    return r


def format_report(r: dict) -> str:
    L = [f"=== {r['root']} data quality (development data only) ==="]
    d = r["date_range"]
    L.append(f"range: {d['start']} .. {d['end']}  ({d['sessions']} sessions)")
    if r["holdout_rows_excluded"]:
        L.append(f"holdout: {r['holdout_rows_excluded']:,} rows excluded and NOT inspected")
    L.append(f"rows: raw {r['rows']['raw']:,} -> clean {r['rows']['clean']:,}")
    L.append(f"duplicate timestamps: {r['duplicate_timestamps']}   zero-volume bars: {r['zero_volume_bars']}")
    for series, b in r["bad_ohlc"].items():
        L.append(f"bad OHLC [{series}]: high<low {b['high_lt_low']}, open outside {b['open_outside_range']}, "
                 f"close outside {b['close_outside_range']}")
    L.append(f"missing in-session minutes by year: {r['missing_minutes_by_year'] or 'none'}")
    short = r["short_ny_am_days"]
    L.append(f"days with < {NY_AM_EXPECTED_BARS} bars in 10-11 ET: {len(short)}"
             + (f"  e.g. {dict(list(short.items())[:5])}" if short else ""))
    miss = r["missing_trading_days"]
    L.append(f"weekdays with no bars (includes holidays): {len(miss)}" + (f"  {miss[:10]}{' ...' if len(miss) > 10 else ''}" if miss else ""))
    L.append(f"roll days by year: {r['roll_days_by_year']}")
    L.append("back-adjusted close by year (ET):")
    L.append(f"  {'year':<6}{'min':>12}{'max':>12}{'cum.adj(max)':>14}")
    for y, v in r["backadjusted_close_by_year"].items():
        L.append(f"  {y:<6}{v['min']:>12.2f}{v['max']:>12.2f}{v['cumulative_adjustment_max']:>14.2f}")
    for f in r.get("roll_findings", []):
        L.append(f"WARN roll check: {f}")
    return "\n".join(L)


def save_report(r: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{r['root']}.json"
    path.write_text(json.dumps(r, indent=2, default=str))
    (out_dir / f"{r['root']}.txt").write_text(format_report(r) + "\n")
    return path
