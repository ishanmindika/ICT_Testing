import warnings

import numpy as np
import pandas as pd
import pytest

from ict_lab.features import (BiasParams, DisplacementParams, LevelParams, LookaheadWarning, MSSParams, SweepParams,
                              SwingParams, active_levels, detect_displacement, detect_mss, detect_sweeps,
                              displacement_mask, htf_bias, level_table, session_levels)
from ict_lab.tests.synth import make_sessions
from ict_lab.tests.test_features import T0, frame, session_frame

t = lambda x: pd.Timestamp(x, tz="UTC")
M = lambda k: T0 + pd.Timedelta(minutes=k)


# ---------------- level table: active until swept ----------------

def test_level_table_life_cycle():
    b = session_frame()
    tab = level_table(b)
    row = lambda typ, side, sd="2024-03-05": tab[(tab.type == typ) & (tab.side == side) & (tab.session_date == pd.Timestamp(sd))].iloc[0]
    ps = row("prior_session", "high")
    assert ps.price == 110.0 and ps.active_from == t("2024-03-04 23:00") and ps.expires == t("2024-03-05 22:01")   # one minute after the last bar
    assert ps.swept_ts == t("2024-03-05 15:30")                     # the 120 spike
    assert pd.isna(row("prior_rth", "low").swept_ts)               # 95 never taken
    am = row("pre_ny_am", "high")
    assert am.price == 107.0 and am.active_from == t("2024-03-05 15:00") and am.swept_ts == t("2024-03-05 15:30")
    live = lambda ts: set(zip(*(lambda d: (d.type, d.side))(active_levels(tab[tab.session_date == pd.Timestamp("2024-03-05")], t(ts)))))
    assert ("prior_session", "high") in live("2024-03-05 15:30")    # swept during this bar: still active at its open
    assert ("prior_session", "high") not in live("2024-03-05 15:31")
    assert ("pre_ny_am", "high") not in live("2024-03-05 14:59")    # not usable before the killzone starts


def test_swings_enter_level_table_and_live_until_swept():
    hi = [10.0, 11.0, 15.0, 12.0, 11.0, 13.0, 16.0, 12.0]
    f = frame(hi, [h - 1 for h in hi])
    tab = level_table(f, swing_p=SwingParams(n=2))
    sw = tab[(tab.type == "swing") & (tab.side == "high")].sort_values("active_from")
    first = sw.iloc[0]
    assert first.price == 15.0 and first.active_from == M(5) and first.swept_ts == M(6)
    assert pd.isna(first.expires)


# ---------------- sweeps ----------------

def lv(side, price, active_from=T0, typ="prior_session", expires=pd.NaT, lid=0):
    return pd.DataFrame({"level_id": [lid], "type": [typ], "side": [side], "price": [price],
                         "active_from": [active_from], "expires": [expires], "swept_ts": [pd.NaT],
                         "session_date": [pd.Timestamp("2024-03-05")]})


def test_sweep_wick_same_bar():
    hi = [99.0, 99.5, 99.4, 100.5, 99.6, 99.0]
    cl = [98.8, 99.2, 99.0, 99.8, 99.3, 98.9]
    f = frame(hi, [h - 1 for h in hi], closes=cl)
    s = detect_sweeps(f, lv("high", 100.0), SweepParams())
    r = s.iloc[0]
    assert len(s) == 1 and (r.sweep_direction, r.sweep_idx, r.confirm_idx) == (-1, 3, 3)
    assert r.penetration_points == 0.5 and r.penetration_ticks == 2.0 and r.extreme_price == 100.5
    assert r.available_at == M(4)
    assert detect_sweeps(f, lv("high", 100.0), SweepParams(min_penetration_ticks=3)).empty   # only 2 ticks through
    assert detect_sweeps(f, lv("high", 100.0), SweepParams(level_types=("swing",))).empty
    later = detect_sweeps(f, lv("high", 100.0), SweepParams(allow_same_bar=False)).iloc[0]
    assert (later.sweep_idx, later.confirm_idx) == (3, 4)                                     # must be a LATER close back


def test_sweep_close_back_within_k_bars_and_break():
    hi = [99.0, 99.0, 99.0, 101.0, 100.8, 100.2, 99.9, 99.5]
    cl = [98.5, 98.5, 98.5, 100.6, 100.4, 100.1, 99.8, 99.0]    # closes back below 100 at idx 6 (3 bars after breach)
    f = frame(hi, [h - 1 for h in hi], closes=cl)
    s = detect_sweeps(f, lv("high", 100.0), SweepParams(k=3))
    assert (s.iloc[0].sweep_idx, s.iloc[0].confirm_idx) == (3, 6) and s.iloc[0].penetration_points == 1.0
    assert detect_sweeps(f, lv("high", 100.0), SweepParams(k=2)).empty                         # too slow: a break, not a sweep


def test_sweep_low_side_and_level_not_active_yet():
    lo = [101.0, 100.0, 99.0, 100.5, 101.0, 100.0]
    hi = [x + 1 for x in lo]
    cl = [101.5, 100.5, 99.5, 100.6, 101.2, 100.6]
    f = frame(hi, lo, closes=cl)
    s = detect_sweeps(f, lv("low", 99.5, active_from=M(1)), SweepParams())
    assert len(s) == 1 and (s.iloc[0].sweep_direction, s.iloc[0].sweep_idx) == (1, 2) and s.iloc[0].confirm_idx == 3
    assert detect_sweeps(f, lv("low", 99.5, active_from=M(4)), SweepParams()).empty            # activated after the move


def test_level_yields_at_most_one_event():
    hi = [99.0, 100.5, 99.0, 100.6, 99.0]
    cl = [98.5, 99.5, 98.5, 99.5, 98.5]
    f = frame(hi, [h - 1 for h in hi], closes=cl)
    assert len(detect_sweeps(f, lv("high", 100.0), SweepParams())) == 1


# ---------------- MSS ----------------

HI = [10.0, 11.0, 12.0, 11.0, 10.5, 10.0, 9.5, 10.5, 12.2, 12.5, 12.0, 11.0]
LO = [9.0, 10.0, 11.0, 10.0, 9.5, 9.0, 8.0, 9.5, 10.0, 11.0, 11.0, 10.0]
CL = [9.5, 10.5, 11.5, 10.5, 10.0, 9.5, 9.2, 10.0, 11.5, 12.4, 11.5, 10.5]


def mss_inputs(hi=HI, lo=LO, cl=CL, level=("low", 8.5)):
    f = frame(hi, lo, closes=cl)
    sw = detect_sweeps(f, lv(level[0], level[1]), SweepParams())
    return f, sw


def test_bullish_mss_close_vs_wick():
    f, sw = mss_inputs()
    assert len(sw) == 1 and sw.iloc[0].sweep_direction == 1 and sw.iloc[0].confirm_idx == 6
    sp = SwingParams(n=1)
    close = detect_mss(f, sw, MSSParams(swing=sp, break_mode="close")).iloc[0]
    wick = detect_mss(f, sw, MSSParams(swing=sp, break_mode="wick")).iloc[0]
    assert (close.direction, close.mss_idx, close.broken_price) == (1, 9, 12.0)
    assert wick.mss_idx == 8 and close.available_at == M(10)
    assert close.swing_ts == M(2) and close.bars_after_confirm == 3


def test_bearish_mss_is_the_mirror():
    f, sw = mss_inputs([-x for x in LO], [-x for x in HI], [-x for x in CL], level=("high", -8.5))
    assert sw.iloc[0].sweep_direction == -1
    r = detect_mss(f, sw, MSSParams(swing=SwingParams(n=1))).iloc[0]
    assert (r.direction, r.mss_idx, r.broken_price) == (-1, 9, -12.0)


def test_mss_max_bars_and_min_break():
    f, sw = mss_inputs()
    sp = SwingParams(n=1)
    assert detect_mss(f, sw, MSSParams(swing=sp, max_bars=3)).empty                    # break arrives 3 bars after confirm: window 0..2
    assert len(detect_mss(f, sw, MSSParams(swing=sp, max_bars=4))) == 1
    assert detect_mss(f, sw, MSSParams(swing=sp, min_break_points=0.5)).empty         # close only clears 12.0 by 0.4


def test_mss_latest_reference_uses_newer_lower_high():
    hi = [10.0, 11.0, 12.0, 11.0, 10.5, 10.0, 9.5, 10.2, 10.8, 10.4, 10.3, 10.9]
    lo = [9.0, 10.0, 11.0, 10.0, 9.5, 9.0, 8.0, 9.5, 10.0, 9.8, 9.9, 10.0]
    cl = [9.5, 10.5, 11.5, 10.5, 10.0, 9.5, 9.2, 10.0, 10.5, 10.1, 10.0, 10.85]
    f, sw = mss_inputs(hi, lo, cl)
    sp = SwingParams(n=1)
    assert detect_mss(f, sw, MSSParams(swing=sp, reference="at_sweep")).empty          # 12.0 never taken
    r = detect_mss(f, sw, MSSParams(swing=sp, reference="latest")).iloc[0]            # lower high 10.8 (confirmed bar 9) broken at 11
    assert (r.mss_idx, r.broken_price) == (11, 10.8)


# ---------------- displacement ----------------

def flat(n=30):
    return [100.5] * n, [99.5] * n, [100.0] * n


def disp_frame(extra_hi, extra_lo, extra_cl, extra_op):
    hi, lo, cl = flat()
    f = frame(hi + extra_hi, lo + extra_lo, closes=cl + extra_cl)
    f["open"] = [100.0] * 30 + extra_op
    return f


def test_single_bar_atr_multiples():
    f = disp_frame([103.0], [100.0], [102.5], [100.0])             # range 3, ATR(prior) = 1
    for mult, expect in ((1.0, 1), (1.5, 1), (2.0, 1), (3.0, 1), (3.5, 0)):
        assert len(detect_displacement(f, DisplacementParams(atr_mult=mult))) == expect
    r = detect_displacement(f, DisplacementParams(atr_mult=2.0)).iloc[0]
    assert (r.direction, r.idx, r.range_points) == (1, 30, 3.0) and r.range_atr_mult == pytest.approx(3.0)
    assert r.available_at == M(31)
    assert displacement_mask(detect_displacement(f), f.index).sum() == 1


def test_run_of_bars_and_direction():
    up = disp_frame([101.2, 102.5], [99.9, 100.9], [101.0, 102.2], [100.0, 101.0])
    r = detect_displacement(up, DisplacementParams(run_length=2, atr_mult=2.0))
    assert len(r) == 1 and (r.iloc[0].start_idx, r.iloc[0].idx, r.iloc[0].range_points) == (30, 31, pytest.approx(2.6))
    mixed = disp_frame([101.2, 102.5], [99.9, 100.9], [101.0, 100.95], [100.0, 101.0])   # second bar closes down
    assert detect_displacement(mixed, DisplacementParams(run_length=2, atr_mult=2.0)).empty


def test_min_body_requirement():
    f = disp_frame([103.0], [100.0], [100.1], [100.0])             # big wicks, tiny body
    assert len(detect_displacement(f, DisplacementParams(atr_mult=2.0))) == 1
    assert detect_displacement(f, DisplacementParams(atr_mult=2.0, min_body_atr_mult=1.0)).empty


def test_percentile_method_flags_about_top_x_percent():
    rng = np.random.default_rng(5)
    n = 1500
    rng_pts = rng.uniform(0.5, 4.0, n)
    op = np.full(n, 100.0)
    f = frame(op + rng_pts, op, closes=op + 0.3)
    f["open"] = op
    r = detect_displacement(f, DisplacementParams(method="percentile", top_pct=10.0, lookback=100))
    assert 0.05 * (n - 100) < len(r) < 0.15 * (n - 100)
    assert r["idx"].min() >= 100                                   # needs a full lookback first
    with pytest.raises(ValueError):
        detect_displacement(f, DisplacementParams(method="nope"))


# ---------------- bias ----------------

def level_path(levels):
    return lambda i, k: np.full(len(k), levels[i], float)


def test_bias_none_and_prior_day_oc():
    assert (htf_bias(session_frame())["bias"] == 0).all()
    b = make_sessions(4, path=lambda i, k: np.concatenate([np.full(1000, 5000.0), np.full(len(k) - 1000, 5000.0 + [8, -8, 8, -8][i])]))
    out = htf_bias(b, BiasParams(method="prior_day_oc"))
    per = out["bias"].groupby(b["session_date"]).first().tolist()
    # session 1 up(open 5000 -> close 5008)... opens carry over so day 2 closes 4992: down, etc.
    assert per[0] == 0 and per[1] == 1 and per[2] == -1 and per[3] == 1
    assert out["bias"].groupby(b["session_date"]).nunique().max() == 1      # constant within a session


def test_prior_day_oc_rth_basis_differs():
    def path(i, k):
        et_min = (k + 18 * 60) % 1440                                      # ET minute of day (k=0 is 18:00)
        rth = (et_min >= 9 * 60 + 30) & (et_min < 16 * 60)
        px = np.full(len(k), 5010.0)                                       # rally into the open ...
        px[rth] = 5010.0 - 12.0 * (et_min[rth] - 570) / 390                # ... sell off through RTH
        px[et_min >= 16 * 60] = 5008.0                                     # ... settle above the session open
        return px + 10.0 * i                                               # each session ends 10 points above the last
    b = make_sessions(3, path=path)
    by_day = lambda m: htf_bias(b, BiasParams(method="prior_day_oc", daily_basis=m))["bias"].groupby(b["session_date"]).first().tolist()
    assert by_day("session")[1:] == [1, 1] and by_day("rth")[1:] == [-1, -1]


def test_daily_ma_slope():
    closes = [100, 101, 102, 103, 100, 95, 94]
    b = make_sessions(7, path=level_path(closes))
    out = htf_bias(b, BiasParams(method="daily_ma_slope", ma_period=3))["bias"].groupby(b["session_date"]).first().tolist()
    # MA3 of closes: d3=101 d4=102 d5=101.67 d6=99.33 d7=96.33; slope known after day d applies to day d+1
    assert out == [0, 0, 0, 0, 1, -1, -1]


def test_trend_swing_bias_uses_confirmed_swings_only():
    H = [10, 12, 11, 13, 12, 14, 13, 15]
    L = [8, 10, 9, 11, 10, 12, 11, 13]
    f = frame(np.repeat(H, 60), np.repeat(L, 60))
    out = htf_bias(f, BiasParams(method="trend_swing", trend_timeframe="1h", trend_swing_n=1))["bias"]
    assert (out.iloc[: 6 * 60] == 0).all()           # last needed swing (low at hour 4) is confirmed when hour 5 closes
    assert (out.iloc[6 * 60:] == 1).all()


def test_perfect_bias_is_flagged_and_uses_the_future():
    b = make_sessions(2, path=lambda i, k: 5000.0 + np.where(k > 1100, 20.0, 0.0))
    with pytest.warns(LookaheadWarning):
        out = htf_bias(b, BiasParams(method="perfect"))
    assert out["bias_lookahead"].all() and "LOOKAHEAD" in out["bias_method"].iloc[0] and out.attrs["lookahead"]
    assert (out["bias"] == 1).all()                  # whole session knows it will close higher, even at bar 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        early = htf_bias(b.iloc[:30], BiasParams(method="none"))
    assert not early["bias_lookahead"].any()
