import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ict_lab.data import build as build_mod
from ict_lab.data.quality import missing_minutes_by_year
from ict_lab.data.sessions import add_session_columns, in_session_mask, session_date
from ict_lab.data.show import window
from ict_lab.data.store import holdout_cutoff, load_bars, load_rolls, resample

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())


def utc(*times):
    return pd.DatetimeIndex(times, tz="UTC")


def flags(*times):
    df = add_session_columns(pd.DataFrame(index=utc(*times)))
    return df


# ---------------- sessions / DST ----------------

def test_session_date_rolls_at_1800_et():
    idx = utc("2024-01-07 22:59", "2024-01-07 23:00", "2024-01-08 21:59", "2024-01-08 22:00")
    # 17:59 ET Sun (closed, same date), 18:00 ET Sun -> Mon, 16:59 ET Mon, 17:00 ET Mon (break, still Mon)
    assert list(session_date(idx).dt.strftime("%m-%d")) == ["01-07", "01-08", "01-08", "01-08"]


def test_spring_forward_2024_03_10():
    # Sunday open is 18:00 EDT = 22:00 UTC (not 23:00): session belongs to Monday 03-11
    idx = utc("2024-03-10 21:59", "2024-03-10 22:00")
    assert list(session_date(idx).dt.strftime("%m-%d")) == ["03-10", "03-11"]
    # 09:30 ET is 14:30 UTC on Fri 03-08 (EST) but 13:30 UTC on Mon 03-11 (EDT)
    df = flags("2024-03-08 14:29", "2024-03-08 14:30", "2024-03-11 13:29", "2024-03-11 13:30", "2024-03-11 14:30")
    assert list(df["rth"]) == [False, True, False, True, True]
    # NY AM killzone 10:00 ET
    k = flags("2024-03-08 14:59", "2024-03-08 15:00", "2024-03-11 13:59", "2024-03-11 14:00", "2024-03-11 15:00")
    assert list(k["killzone_ny_am"]) == [False, True, False, True, False]


def test_fall_back_2024_11_03():
    # Sunday open is 18:00 EST = 23:00 UTC: Monday 11-04 session
    idx = utc("2024-11-03 22:59", "2024-11-03 23:00")
    assert list(session_date(idx).dt.strftime("%m-%d")) == ["11-03", "11-04"]
    # 09:30 ET is 13:30 UTC on Fri 11-01 (EDT) but 14:30 UTC on Mon 11-04 (EST)
    df = flags("2024-11-01 13:29", "2024-11-01 13:30", "2024-11-04 14:29", "2024-11-04 14:30", "2024-11-04 13:30")
    assert list(df["rth"]) == [False, True, False, True, False]
    # London killzone 03:00-04:00 ET: 07:00 UTC in EDT, 08:00 UTC in EST
    lk = flags("2024-11-01 06:59", "2024-11-01 07:00", "2024-11-04 07:00", "2024-11-04 08:00", "2024-11-04 09:00")
    assert list(lk["killzone_london"]) == [False, True, False, True, False]
    # NY PM 14:00-15:00 ET
    pm = flags("2024-11-04 18:59", "2024-11-04 19:00", "2024-11-04 19:59", "2024-11-04 20:00")
    assert list(pm["killzone_ny_pm"]) == [False, True, True, False]


def test_window_edges_half_open():
    df = flags("2024-06-03 13:59", "2024-06-03 14:00", "2024-06-03 14:59", "2024-06-03 15:00")  # EDT
    assert list(df["killzone_ny_am"]) == [False, True, True, False]
    assert list(flags("2024-06-03 19:59", "2024-06-03 20:00")["rth"]) == [True, False]  # 15:59 / 16:00 ET


def test_session_mask_handles_dst():
    idx = utc("2024-01-09 22:30", "2024-07-09 21:30", "2024-07-09 22:30")  # 17:30 ET break in both seasons
    assert list(in_session_mask(idx)) == [False, False, True]


def test_no_missing_minutes_across_dst_days():
    for start in ("2024-03-10 22:00", "2024-11-03 23:00"):  # Sunday open
        idx = pd.date_range(start, periods=60 * 22, freq="1min", tz="UTC")
        sd = session_date(idx)
        assert missing_minutes_by_year(idx, sd) == {}


# ---------------- synthetic triplets ----------------

def bars(start, periods, contract, base):
    idx = pd.date_range(start, periods=periods, freq="1min", tz="UTC", name="ts")
    close = base + np.arange(periods) * 0.25
    return pd.DataFrame({"open": close - 0.25, "high": close + 0.5, "low": close - 0.5,
                         "close": close, "volume": 100, "contract": contract}, index=idx)


def make_triplet(raw_dir, root="ES", adj=-30.0, shift_adj_ts=False, csv_adj=None, bump_close=False):
    """Mon 03-04 on ESH4, Tue 03-05 (roll day) onward on ESM4; plus a late 2025 day so a holdout exists."""
    parts = [bars("2024-03-04 14:00", 90, "ESH4", 5000), bars("2024-03-05 14:00", 90, "ESM4", 5030),
             bars("2025-06-10 14:00", 90, "ESU5", 6000)]
    un = pd.concat(parts)
    ba = un.copy()
    ba.loc[ba["contract"] == "ESH4", PRICE] += adj
    if bump_close:
        ba.iloc[5, ba.columns.get_loc("close")] -= 0.1
    if shift_adj_ts:
        ba.index = ba.index + pd.Timedelta(minutes=1)
    un.to_parquet(raw_dir / f"{root}_1m_unadjusted.parquet")
    ba.to_parquet(raw_dir / f"{root}_1m_backadjusted.parquet")
    pd.DataFrame({"date": ["2024-03-05"], "old_contract": ["ESH4"], "new_contract": ["ESM4"],
                  "adjustment": [csv_adj if csv_adj is not None else adj]}).to_csv(raw_dir / f"{root}_rolls.csv", index=False)


PRICE = ["open", "high", "low", "close"]


@pytest.fixture
def built(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir()
    make_triplet(raw)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]},
           "holdout": {"cutoff_date": "2025-01-01", "years": 2}}
    reports = build_mod.build(cfg, raw, out)
    return cfg, raw, out, reports


def kw(built):
    cfg, _, out, _ = built
    return dict(clean_dir=out, config=cfg)


def test_default_is_backadjusted_and_unadjusted_option(built):
    b = load_bars("ES", **kw(built))
    u = load_bars("ES", price_series="unadjusted", **kw(built))
    assert b.index.equals(u.index) and (b["volume"] == u["volume"]).all() and (b["contract"] == u["contract"]).all()
    h4 = b["contract"] == "ESH4"
    assert (u.loc[h4, "close"] - b.loc[h4, "close"] == 30.0).all()
    assert (u.loc[~h4, "close"] == b.loc[~h4, "close"]).all()
    with pytest.raises(ValueError):
        load_bars("ES", price_series="nope", **kw(built))


def test_is_roll_day_on_both_series_and_roll_log(built):
    for s in ("backadjusted", "unadjusted"):
        df = load_bars("ES", price_series=s, **kw(built))
        rd = df.groupby("session_date")["is_roll_day"].all()
        assert rd.to_dict() == {pd.Timestamp("2024-03-04"): False, pd.Timestamp("2024-03-05"): True}
    log = load_rolls("ES", clean_dir=built[2], config=built[0])
    assert list(log["new_contract"]) == ["ESM4"] and log["adjustment"].iloc[0] == -30.0


def test_holdout_excluded_by_default_and_requires_override(built):
    df = load_bars("ES", **kw(built))
    assert df["session_date"].max() < pd.Timestamp("2025-01-01")
    full = load_bars("ES", include_holdout=True, **kw(built))
    assert full["session_date"].max() == pd.Timestamp("2025-06-10") and len(full) == len(df) + 90
    # even an explicit date range cannot reach it
    assert load_bars("ES", start="2025-06-01", **kw(built)).empty
    assert load_rolls("ES", clean_dir=built[2], config=built[0], include_holdout=True).shape[0] == 1


def test_holdout_unknown_cutoff_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError):
        holdout_cutoff(tmp_path, {"holdout": {"cutoff_date": None}})


def test_auto_cutoff_recorded_in_meta(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"; raw.mkdir()
    make_triplet(raw)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]}, "holdout": {"cutoff_date": None, "years": 1}}
    build_mod.build(cfg, raw, out)
    assert json.loads((out / "meta.json").read_text())["holdout_cutoff"] == "2024-06-10"
    assert holdout_cutoff(out, cfg) == pd.Timestamp("2024-06-10")
    assert load_bars("ES", clean_dir=out, config=cfg)["session_date"].max() == pd.Timestamp("2024-03-05")


def test_cache_partitioned_by_symbol_series_year(built):
    base = built[2] / "bars"
    for s in ("backadjusted", "unadjusted"):
        years = sorted(p.name for p in (base / "symbol=ES" / f"series={s}").iterdir())
        assert years == ["year=2024", "year=2025"]


def test_build_reports_and_saves_quality(built):
    rep = built[3]["ES"]
    assert rep["holdout_rows_excluded"] == 90
    assert rep["date_range"]["end"].startswith("2024-03-05")        # holdout not in the range
    assert rep["roll_days_by_year"] == {"2024": 1} and rep["roll_findings"] == []
    assert rep["backadjusted_close_by_year"]["2024"]["cumulative_adjustment_max"] == 30.0
    assert rep["duplicate_timestamps"] == 0 and rep["zero_volume_bars"] == 0
    assert (built[2] / "quality" / "ES.json").exists() and (built[2] / "quality" / "ES.txt").exists()
    # 2024-03-04 and -05 are present, so no weekday gaps between them
    assert rep["missing_trading_days"] == []


def test_alignment_failure_is_loud(tmp_path):
    make_triplet(tmp_path, shift_adj_ts=True)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]}}
    with pytest.raises(AssertionError, match="timestamp index differs"):
        build_mod.build(cfg, tmp_path, tmp_path / "c")


def test_roll_mismatch_and_non_additive_reported(tmp_path):
    make_triplet(tmp_path, csv_adj=-12.5, bump_close=True)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]}, "holdout": {"cutoff_date": "2025-01-01", "years": 2}}
    f = build_mod.build(cfg, tmp_path, tmp_path / "c")["ES"]["roll_findings"]
    assert any("adjustment at" in x for x in f) and any("not constant" in x for x in f)


def test_quality_counts_bad_and_duplicate_rows(tmp_path):
    make_triplet(tmp_path)
    for name in ("unadjusted", "backadjusted"):
        p = tmp_path / f"ES_1m_{name}.parquet"
        df = pd.read_parquet(p)
        df.iloc[3, df.columns.get_loc("volume")] = 0
        df.iloc[7, df.columns.get_loc("high")] = df.iloc[7]["low"] - 1          # high < low
        df = pd.concat([df, df.iloc[[10]]])                                       # duplicate timestamp
        df.to_parquet(p)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]}, "holdout": {"cutoff_date": "2025-01-01", "years": 2}}
    r = build_mod.build(cfg, tmp_path, tmp_path / "c")["ES"]
    assert r["duplicate_timestamps"] == 1 and r["zero_volume_bars"] == 1
    assert r["bad_ohlc"]["unadjusted"]["high_lt_low"] == 1
    assert r["rows"]["raw"] - r["rows"]["clean"] == 2  # bad row + duplicate removed


def test_verification_window_et_and_holdout_guard(built):
    cfg, raw, out, _ = built
    w = window("ES", "2024-03-04 09:00", "2024-03-04 09:02", "unadjusted", raw_dir=raw, clean_dir=out, config=cfg)
    assert len(w) == 3 and str(w.index[0].tz) == "America/New_York" and w.index[0].hour == 9
    assert w["close"].iloc[0] == 5000.0                    # 09:00 EST == 14:00 UTC == first bar
    with pytest.raises(PermissionError):
        window("ES", "2025-06-10 10:00", "2025-06-10 10:05", raw_dir=raw, clean_dir=out, config=cfg)
    assert len(window("ES", "2025-06-10 10:00", "2025-06-10 10:05", include_holdout=True, raw_dir=raw, clean_dir=out, config=cfg)) == 6


def test_resample(built):
    df = load_bars("ES", **kw(built))
    r = resample(df, "5min")
    assert r["volume"].iloc[0] == 500 and r["high"].iloc[0] == df["high"].iloc[:5].max()


def test_empty_development_data_is_an_error(tmp_path):
    make_triplet(tmp_path)
    cfg = {**CFG, "instruments": {"ES": CFG["instruments"]["ES"]}, "holdout": {"cutoff_date": "2020-01-01", "years": 2}}
    with pytest.raises(ValueError, match="no development data"):
        build_mod.build(cfg, tmp_path, tmp_path / "c")
