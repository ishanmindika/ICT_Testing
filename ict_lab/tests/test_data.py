import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from ict_lab.data import build as build_mod
from ict_lab.data.cleaner import clean, drop_invalid, find_gaps
from ict_lab.data.loader import read_raw
from ict_lab.data.sessions import in_session_mask, session_date
from ict_lab.data.store import load_bars, resample

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())


def bars(start, periods, symbol=None, base=5000.0, vol=100):
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC", name="ts")
    close = base + np.arange(periods) * 0.25
    df = pd.DataFrame({"open": close - 0.25, "high": close + 0.5, "low": close - 0.5,
                       "close": close, "volume": vol}, index=idx)
    if symbol:
        df["symbol"] = symbol
    return df


def test_session_date_rolls_at_1800_et():
    # Sun 2024-01-07 23:00 UTC = 18:00 ET -> Monday's session; 22:59 UTC = 17:59 ET -> Sunday (closed)
    idx = pd.DatetimeIndex(["2024-01-07 23:00", "2024-01-08 15:00"], tz="UTC")
    assert list(session_date(idx).dt.strftime("%Y-%m-%d")) == ["2024-01-08", "2024-01-08"]


def test_session_mask_handles_dst():
    # 17:30 ET is the daily break in both winter (22:30 UTC) and summer (21:30 UTC)
    idx = pd.DatetimeIndex(["2024-01-09 22:30", "2024-07-09 21:30", "2024-07-09 22:30"], tz="UTC")
    assert list(in_session_mask(idx)) == [False, False, True]


def test_naive_timestamps_localized_and_dst_ambiguity_dropped(tmp_path):
    # 2024-11-03 01:30 occurs twice in New York; naive values can't be resolved -> dropped and counted
    f = tmp_path / "x.csv"
    f.write_text("timestamp,open,high,low,close,volume\n"
                 "2024-11-02 09:30:00,1,2,1,2,10\n2024-11-03 01:30:00,1,2,1,2,10\n")
    df, stats = read_raw(f, {**CFG["source"], "tz": "America/New_York"})
    assert len(df) == 1 and df.index[0] == pd.Timestamp("2024-11-02 13:30", tz="UTC")
    assert stats["bad_timestamps"] == 1


def test_epoch_ns_and_alt_column_names(tmp_path):
    f = tmp_path / "x.csv"
    ns = pd.Timestamp("2024-03-01 15:00", tz="UTC").value
    f.write_text(f"ts_event,Open,High,Low,Close,Vol\n{ns},1,2,1,2,10\n")
    df, _ = read_raw(f, CFG["source"])
    assert df.index[0] == pd.Timestamp("2024-03-01 15:00", tz="UTC") and df["volume"].iloc[0] == 10


def test_invalid_rows_removed_and_counted():
    df = bars("2024-03-04 15:00", 4)
    df.iloc[0, df.columns.get_loc("high")] = df.iloc[0]["low"] - 1  # high < low
    df.iloc[1, df.columns.get_loc("close")] = np.nan
    df.iloc[2, df.columns.get_loc("volume")] = -5
    out, rep = drop_invalid(df)
    assert len(out) == 1 and rep["dropped_nan"] == 1 and rep["dropped_ohlc_inconsistent"] == 1


def test_duplicates_keep_highest_volume():
    df = bars("2024-03-04 15:00", 3)
    dup = df.iloc[[1]].copy(); dup["volume"] = 999
    out, rep, _, _ = clean(pd.concat([df, dup]).sort_index(ascending=False), "ES")
    assert rep["dropped_duplicates"] == 1 and out.loc["2024-03-04 15:01", "volume"] == 999
    assert out.index.is_monotonic_increasing


def test_gaps_ignore_daily_break_and_weekend_but_flag_real_holes():
    a = bars("2024-03-04 20:00", 30)          # Mon 15:00 ET
    b = bars("2024-03-04 21:30", 10)          # +30min hole in-session
    gaps = find_gaps(pd.concat([a, b]), 5)
    assert len(gaps) == 1 and gaps["missing_minutes"].iloc[0] == 60
    brk = pd.concat([bars("2024-03-04 21:55", 5), bars("2024-03-04 23:00", 5)])  # 16:55 ET .. 18:00 ET
    assert len(find_gaps(brk, 5)) == 0
    wk = pd.concat([bars("2024-03-08 21:55", 5), bars("2024-03-10 22:00", 5)])   # Fri close -> Sun 18:00 EDT open (DST day)
    assert len(find_gaps(wk, 5)) == 0


def test_front_month_uses_prior_session_volume_no_lookahead():
    days = pd.date_range("2024-03-04 15:00", periods=4, freq="1D", tz="UTC")
    parts = []
    for i, d in enumerate(days):
        old_vol, new_vol = (1000, 10) if i < 2 else (10, 1000)   # new contract out-trades from day 3 (idx 2)
        parts += [bars(d, 3, "ESH4", 5000, old_vol), bars(d, 3, "ESM4", 5050, new_vol)]
    out, rep, _, rolls = clean(pd.concat(parts), "ES")
    held = out.groupby("session_date")["symbol"].first().tolist()
    # volume flips on day idx 2, but we only know that after the day -> switch on idx 3
    assert held == ["ESH4", "ESH4", "ESH4", "ESM4"]
    assert len(rolls) == 1 and rolls["to_contract"].iloc[0] == "ESM4"


def test_end_to_end_build_and_load(tmp_path):
    raw_dir, out_dir = tmp_path / "raw", tmp_path / "clean"
    raw_dir.mkdir()
    es = bars("2024-03-04 14:30", 120, "ESH4")
    nq = bars("2024-03-04 14:30", 120, "NQH4", base=18000)
    pd.concat([es, nq]).reset_index().rename(columns={"ts": "timestamp"}).to_csv(raw_dir / "a.csv", index=False)
    reports = build_mod.build(CFG, raw_dir, out_dir)
    assert set(reports) == {"ES", "NQ"}
    df = load_bars("ES", "2024-03-04 15:00", clean_dir=out_dir)
    assert df.index.tz is not None and df.index.min() == pd.Timestamp("2024-03-04 15:00", tz="UTC")
    r5 = resample(df, "5min")
    assert len(r5) == 18 and r5["volume"].iloc[0] == 500
    assert r5["high"].iloc[0] == df["high"].iloc[:5].max()
