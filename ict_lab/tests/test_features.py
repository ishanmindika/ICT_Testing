import numpy as np
import pandas as pd
import pytest

from ict_lab.data.sessions import add_session_columns
from ict_lab.features import FVGParams, LevelParams, SwingParams, detect_fvgs, detect_swings, param_grid, session_levels
from ict_lab.features.common import atr

T0 = pd.Timestamp("2024-03-05 15:00", tz="UTC")  # Tue 10:00 ET (EST)


def frame(highs, lows, start=T0, closes=None, freq="1min"):
    idx = pd.date_range(start, periods=len(highs), freq=freq, name="ts")
    highs, lows = np.asarray(highs, float), np.asarray(lows, float)
    closes = (highs + lows) / 2 if closes is None else np.asarray(closes, float)
    df = pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1}, index=idx)
    return add_session_columns(df)


# ---------------- FVG ----------------

def test_bullish_fvg_fields_and_mitigation():
    #        b1     b2     b3     after: touch  ce     fill
    hi = [100.0, 103.0, 104.0, 104.0, 103.0, 102.0]
    lo = [99.0, 100.5, 102.0, 101.5, 100.9, 99.5]
    f = detect_fvgs(frame(hi, lo))
    assert len(f) == 1
    r = f.iloc[0]
    assert (r.direction, r.bar_idx) == (1, 2)
    assert (r.bottom, r.top, r.mid, r.size_points) == (100.0, 102.0, 101.0, 2.0)
    assert r.ts_bar1 == T0 and r.ts_bar3 == T0 + pd.Timedelta(minutes=2)
    assert r.available_at == T0 + pd.Timedelta(minutes=3)          # bar 3 closes at 3
    assert (r.touch_idx, r.ce_idx, r.fill_idx) == (3, 4, 5)
    assert r.touch_ts == T0 + pd.Timedelta(minutes=3)


def test_bearish_fvg_and_never_filled():
    hi = [100.0, 99.0, 98.0, 98.5]
    lo = [99.0, 97.0, 96.0, 96.5]
    r = detect_fvgs(frame(hi, lo)).iloc[0]
    assert (r.direction, r.top, r.bottom, r.mid) == (-1, 99.0, 98.0, 98.5)
    assert (r.touch_idx, r.ce_idx) == (3, 3)
    assert pd.isna(r.fill_idx) and pd.isna(r.fill_ts)


def test_equal_levels_and_overlap_are_not_gaps():
    assert detect_fvgs(frame([100, 101, 102], [99, 100, 100])).empty   # low3 == high1 -> size 0
    assert detect_fvgs(frame([100, 101, 102], [99, 100, 99.5])).empty


def test_bar3_itself_does_not_count_as_mitigation():
    r = detect_fvgs(frame([100, 103, 104], [99, 101, 102])).iloc[0]
    assert pd.isna(r.touch_idx)


def test_touch_and_fill_same_bar():
    r = detect_fvgs(frame([100, 103, 104, 103], [99, 101, 102, 95])).iloc[0]
    assert (r.touch_idx, r.ce_idx, r.fill_idx) == (3, 3, 3)


def test_min_size_points_filter():
    f = frame([100, 103, 104], [99, 101, 102])                        # size 2
    assert len(detect_fvgs(f, FVGParams(min_size_points=2.0))) == 1
    assert detect_fvgs(f, FVGParams(min_size_points=2.5)).empty


def test_atr_multiple():
    # 20 flat bars (range 1, TR=1) then a gap of size 2 -> ATR(sma 3, bar3) = mean of last 3 TR
    hi = [100.5] * 20 + [100.5, 103.0, 104.0]
    lo = [99.5] * 20 + [99.5, 100.5, 102.0]
    f = frame(hi, lo, closes=[100.0] * 20 + [100.0, 102.0, 103.0])
    expected = atr(f, 3, "sma").iloc[-1]
    p = FVGParams(atr_period=3, atr_method="sma", min_size_atr_mult=0.5)
    r = detect_fvgs(f, p).iloc[0]
    assert r.atr == pytest.approx(expected) and r.size_atr == pytest.approx(r.size_points / expected)
    assert detect_fvgs(f, FVGParams(atr_period=3, atr_method="sma", min_size_atr_mult=50)).empty
    # during ATR warm-up the ATR threshold cannot be met
    assert detect_fvgs(frame([100, 103, 104], [99, 101, 102]), FVGParams(min_size_atr_mult=0.1)).empty


def test_timeframe_5m_uses_complete_bars_only():
    # 15 one-minute bars -> three 5m bars: [100,101], [102..], [105..] forming a bullish gap
    hi = [100.5] * 5 + [103.0] * 5 + [106.0] * 5
    lo = [100.0] * 5 + [102.0] * 5 + [105.0] * 5
    f = frame(hi, lo)
    r = detect_fvgs(f, FVGParams(timeframe="5m")).iloc[0]
    assert (r.bar_idx, r.bottom, r.top) == (2, 100.5, 105.0)
    assert r.available_at == T0 + pd.Timedelta(minutes=15)
    # drop one minute from the last 5m bar -> that bar is incomplete and dropped -> no FVG
    assert detect_fvgs(f.drop(f.index[12]), FVGParams(timeframe="5m")).empty


def test_contiguity_gaps_block_fvgs():
    f = frame([100, 103, 104], [99, 101, 102])
    gapped = f.copy()
    gapped.index = [T0, T0 + pd.Timedelta(minutes=1), T0 + pd.Timedelta(minutes=30)]
    assert detect_fvgs(gapped).shape[0] == 0
    assert len(detect_fvgs(gapped, FVGParams(require_contiguous=False))) == 1


def test_no_fvg_across_maintenance_break():
    # bars 16:58, 16:59 ET then 18:00 ET (Tue): not consecutive
    idx = pd.DatetimeIndex(["2024-03-05 21:58", "2024-03-05 21:59", "2024-03-05 23:00"], tz="UTC", name="ts")
    df = pd.DataFrame({"open": 1.0, "high": [100, 103, 104.0], "low": [99, 101, 102.0], "close": 1.0, "volume": 1}, index=idx)
    assert detect_fvgs(add_session_columns(df)).empty


def test_expire_bars_limits_mitigation_tracking():
    hi = [100, 103, 104, 104, 104, 104]
    lo = [99, 101, 102, 102.5, 102.5, 101]       # touch at idx 5
    f = frame(hi, lo)
    assert detect_fvgs(f).iloc[0].touch_idx == 5
    assert pd.isna(detect_fvgs(f, FVGParams(expire_bars=2)).iloc[0].touch_idx)


def test_param_grid_and_validation():
    grid = list(param_grid(FVGParams(), timeframe=["1m", "5m", "15m"], min_size_points=[0, 1]))
    assert len(grid) == 6 and {g.timeframe for g in grid} == {"1m", "5m", "15m"}
    with pytest.raises(ValueError):
        detect_fvgs(frame([1, 2, 3], [0, 1, 2]), FVGParams(timeframe="7m"))


def random_bars(n=600, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.6, n))
    return frame(close + rng.uniform(0.1, 0.8, n), close - rng.uniform(0.1, 0.8, n), closes=close)


@pytest.mark.parametrize("tf", ["1m", "5m"])
def test_fvg_formation_has_no_lookahead(tf):
    full = random_bars()
    p = FVGParams(timeframe=tf, atr_period=5)
    all_f = detect_fvgs(full, p)
    assert len(all_f) > 5
    form = ["direction", "ts_bar1", "ts_bar3", "available_at", "top", "bottom", "size_points", "atr"]
    for cut in (97, 250, 411):
        prefix = full.iloc[:cut]
        got = detect_fvgs(prefix, p)[form].reset_index(drop=True)
        end = prefix.index[-1] + pd.Timedelta(minutes=1)
        want = all_f[all_f["available_at"] <= end][form].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want)


# ---------------- swings ----------------

def test_swing_fractal_confirmation_and_sweep():
    #        0    1    2     3    4    5    6    7    8
    hi = [10.0, 11.0, 15.0, 12.0, 11.0, 13.0, 16.0, 12.0, 11.0]
    lo = [9.0, 10.0, 14.0, 11.0, 10.0, 12.0, 15.0, 11.0, 10.0]
    cl = [9.5, 10.5, 14.5, 11.5, 10.5, 12.5, 15.8, 11.5, 10.5]
    s = detect_swings(frame(hi, lo, closes=cl), SwingParams(n=2))
    sh = s[s.kind == "high"].reset_index(drop=True)
    assert list(sh.idx) == [2, 6] and sh.price[0] == 15.0
    assert sh.confirmed_idx[0] == 4 and sh.available_at[0] == T0 + pd.Timedelta(minutes=5)
    assert sh.swept_idx[0] == 6 and sh.close_through_idx[0] == 6      # bar 6 trades and closes above 15
    assert pd.isna(sh.swept_idx[1])                                    # 16 is never exceeded
    assert list(s[s.kind == "low"].idx) == [4]


def test_swing_low_wick_vs_close_and_unswept():
    hi = [20.0, 19.0, 18.0, 19.0, 20.0, 19.5, 19.0, 19.5]
    lo = [19.0, 18.0, 15.0, 18.0, 19.0, 14.0, 18.0, 18.5]      # swing low 15 at idx 2; bar 5 wicks to 14
    cl = [19.5, 18.5, 16.0, 18.5, 19.5, 17.0, 18.5, 19.0]      # ...but closes at 17
    s = detect_swings(frame(hi, lo, closes=cl), SwingParams(n=2))
    low = s[(s.kind == "low") & (s.idx == 2)].iloc[0]
    assert low.price == 15.0 and low.swept_idx == 5 and pd.isna(low.close_through_idx)


def test_swing_ties_strict_vs_not():
    hi = [10.0, 11.0, 15.0, 15.0, 11.0, 10.0]
    lo = [h - 1 for h in hi]
    f = frame(hi, lo)
    assert detect_swings(f, SwingParams(n=2)).query("kind == 'high'").empty           # equal highs: neither is strict
    assert len(detect_swings(f, SwingParams(n=2, strict=False)).query("kind == 'high'")) >= 1


def test_swing_no_lookahead_and_contiguity():
    full = random_bars(500, seed=3)
    p = SwingParams(n=3)
    a = detect_swings(full, p)
    assert len(a) > 20
    form = ["kind", "idx", "ts", "price", "available_at"]
    for cut in (120, 301):
        prefix = full.iloc[:cut]
        end = prefix.index[-1] + pd.Timedelta(minutes=1)
        got = detect_swings(prefix, p)[form].reset_index(drop=True)
        want = a[a["available_at"] <= end][form].reset_index(drop=True)
        pd.testing.assert_frame_equal(got, want)


def test_swing_requires_contiguous_window():
    hi = [10.0, 11.0, 15.0, 12.0, 11.0, 10.0]
    lo = [h - 1 for h in hi]
    f = frame(hi, lo)
    holey = f.drop(f.index[3])                                          # a missing minute inside the window
    assert 2 in list(detect_swings(f, SwingParams(n=2)).query("kind == 'high'").idx)
    assert detect_swings(holey, SwingParams(n=2)).query("kind == 'high'").empty
    assert not detect_swings(holey, SwingParams(n=2, require_contiguous=False)).query("kind == 'high'").empty


# ---------------- session / pre-window levels ----------------

def session_frame():
    """Mon 2024-03-04 and Tue 03-05 sessions, flat 99.5-100.5 with planted spikes (EST, UTC-5)."""
    idx = pd.date_range("2024-03-03 23:00", "2024-03-05 22:00", freq="1min", tz="UTC", name="ts")  # Sun 18:00 ET .. Tue 17:00 ET
    df = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx)
    # remove the daily maintenance break 17:00-18:00 ET Monday (22:00-23:00 UTC)
    df = df[~((df.index >= "2024-03-04 22:00") & (df.index < "2024-03-04 23:00"))]
    df.loc["2024-03-04 15:30", "high"] = 110.0      # 10:30 ET Mon, inside RTH
    df.loc["2024-03-04 08:30", "low"] = 90.0        # 03:30 ET Mon, outside RTH
    df.loc["2024-03-04 20:00", "low"] = 95.0        # 15:00 ET Mon, inside RTH -> RTH low
    df.loc["2024-03-05 02:00", "high"] = 105.0      # 21:00 ET Mon: belongs to the Tue session (after 18:00 ET)
    df.loc["2024-03-05 14:30", "high"] = 107.0      # 09:30 ET Tue (RTH, before NY AM)
    df.loc["2024-03-05 15:30", "high"] = 120.0      # 10:30 ET Tue, INSIDE the NY AM killzone
    return add_session_columns(df)


def test_prior_session_and_rth_levels():
    b = session_frame()
    lv = session_levels(b)
    tue = lv[b["session_date"] == pd.Timestamp("2024-03-05")]
    assert (tue["prior_session_high"] == 110.0).all() and (tue["prior_session_low"] == 90.0).all()
    assert (tue["prior_rth_high"] == 110.0).all() and (tue["prior_rth_low"] == 95.0).all()
    assert (tue["prior_session_date"] == pd.Timestamp("2024-03-04")).all()
    mon = lv[b["session_date"] == pd.Timestamp("2024-03-04")]
    assert mon["prior_session_high"].isna().all()   # nothing before the first session


def test_pre_window_levels_frozen_and_no_lookahead():
    b = session_frame()
    lv = session_levels(b)
    tue = b["session_date"] == pd.Timestamp("2024-03-05")
    pre = lv.loc[tue, "pre_ny_am_high"]
    start = pd.Timestamp("2024-03-05 15:00", tz="UTC")                       # 10:00 ET
    assert pre[pre.index < start].isna().all()
    # 18:00 ET Mon -> 10:00 ET Tue includes the 21:00 ET spike (105) and the 09:30 ET spike (107), not the 120 inside the zone
    assert (pre[pre.index >= start] == 107.0).all()
    assert (lv.loc[tue, "pre_ny_am_low"].dropna() == 99.5).all()
    # changing bars inside/after the window must not change the frozen value
    b2 = b.copy(); b2.loc["2024-03-05 16:00", "high"] = 500.0
    pre2 = session_levels(b2).loc[tue, "pre_ny_am_high"]
    pd.testing.assert_series_equal(pre, pre2)
    # london pre-window: 18:00 ET Mon -> 03:00 ET Tue = 08:00 UTC
    ln = lv.loc[tue, "pre_london_high"]
    assert ln[ln.index < pd.Timestamp("2024-03-05 08:00", tz="UTC")].isna().all()
    assert (ln[ln.index >= pd.Timestamp("2024-03-05 08:00", tz="UTC")] == 105.0).all()


def test_swept_flags():
    b = session_frame()
    lv = session_levels(b)
    t = lambda x: pd.Timestamp(x, tz="UTC")
    for col, first in (("swept_prior_session_high", "2024-03-05 15:30"),   # 120 beats 110
                       ("swept_pre_ny_am_high", "2024-03-05 15:30"),       # 120 beats the frozen 107
                       ("swept_prior_rth_high", "2024-03-05 15:30")):
        assert not lv.loc[t("2024-03-05 15:29"), col]
        assert lv.loc[t(first), col] and lv.loc[t("2024-03-05 21:00"), col]   # stays set for the session
    assert not lv["swept_prior_session_low"].any()                          # 90 never taken out
    assert not lv.loc[b["session_date"] == pd.Timestamp("2024-03-04"), "swept_prior_session_high"].any()
