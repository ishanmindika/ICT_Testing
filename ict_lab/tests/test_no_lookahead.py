"""CRITICAL: no detector may use bars after the current one (except bias method 'perfect', flagged as such).

Each detector runs twice: on the full history, and on history truncated at bar N. Everything the full run
says about the world up to bar N must equal what the truncated run says.
 - event tables: rows known by bar N's close (available_at <= N's close) must match exactly;
   outcome columns that happen after N are blanked in the full run before comparing
 - per-bar frames: rows up to N must match exactly
"""
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ict_lab.features import (BiasParams, DisplacementParams, FeatureConfig, FVGParams, LevelParams, LookaheadWarning,
                              MSSParams, SweepParams, SwingParams, compute_features, htf_bias)
from ict_lab.tests.synth import make_sessions

BARS = make_sessions(6, seed=11)
CUTS = [2500, 4100, 6003, 7777]          # mid-session cuts, different sessions

OUTCOMES = {  # table -> (index column, timestamp column) pairs describing things that happen AFTER formation
    "fvgs": [("touch_idx", "touch_ts"), ("ce_idx", "ce_ts"), ("fill_idx", "fill_ts")],
    "swings": [("swept_idx", "swept_ts"), ("close_through_idx", "close_through_ts")],
}

CONFIGS = {
    "default": FeatureConfig(),
    "htf_5m_wick_latest": FeatureConfig(
        fvg=FVGParams(timeframe="5m", min_size_atr_mult=0.5),
        swing=SwingParams(timeframe="5m", n=2),
        sweep=SweepParams(k=5, min_penetration_ticks=0, timeframe="1m"),
        displacement=DisplacementParams(method="percentile", top_pct=8, lookback=120, run_length=2, timeframe="1m"),
        mss=MSSParams(break_mode="wick", reference="latest", max_bars=20, swing=SwingParams(n=2)),
        bias=BiasParams(method="trend_swing", trend_timeframe="15m", trend_swing_n=2)),
    "daily_bias": FeatureConfig(
        displacement=DisplacementParams(atr_mult=2.0, run_length=3),
        bias=BiasParams(method="daily_ma_slope", ma_period=3)),
    "prior_day_rth": FeatureConfig(bias=BiasParams(method="prior_day_oc", daily_basis="rth")),
}


def censor(df: pd.DataFrame, pairs, last_open: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    for idx_c, ts_c in pairs:
        late = df[ts_c] > last_open
        df[idx_c] = df[idx_c].mask(late)
        df[ts_c] = df[ts_c].mask(late)
    return df


def known(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    return df[df["available_at"] <= end].reset_index(drop=True)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("cut", CUTS)
def test_every_detector_is_prefix_invariant(name, cut):
    cfg = CONFIGS[name]
    prefix = BARS.iloc[:cut]
    last_open = prefix.index[-1]
    end = last_open + pd.Timedelta(minutes=1)
    full, part = compute_features(BARS, cfg), compute_features(prefix, cfg)

    for table in ("fvgs", "swings", "sweeps", "displacement", "mss"):
        a, b = getattr(full, table), getattr(part, table)
        a = censor(known(a, end), OUTCOMES.get(table, []), last_open)
        b = known(b, end)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=f"{name}/{table}@{cut}")

    # per-bar frames
    pd.testing.assert_frame_equal(full.session_levels.loc[:last_open], part.session_levels, check_dtype=False,
                                  obj=f"{name}/session_levels@{cut}")
    pd.testing.assert_frame_equal(full.bias.loc[:last_open], part.bias, check_dtype=False, obj=f"{name}/bias@{cut}")

    # level registry: levels usable by bar N, with later sweeps blanked
    key = ["level_id", "type", "side", "price", "active_from", "session_date", "swept_ts"]
    fl = full.level_table[full.level_table["active_from"] <= last_open].copy()
    fl["swept_ts"] = fl["swept_ts"].mask(fl["swept_ts"] > last_open)
    pl = part.level_table[part.level_table["active_from"] <= last_open]
    pd.testing.assert_frame_equal(fl[key].reset_index(drop=True), pl[key].reset_index(drop=True), check_dtype=False,
                                  obj=f"{name}/level_table@{cut}")


def test_synthetic_data_actually_exercises_every_detector():
    """Guard against a vacuous pass: each detector must find something in the test data."""
    for name, cfg in CONFIGS.items():
        fs = compute_features(BARS, cfg)
        for table in ("fvgs", "swings", "sweeps", "displacement", "mss", "level_table"):
            assert len(getattr(fs, table)) > 3, (name, table)
    assert (compute_features(BARS, CONFIGS["daily_bias"]).bias["bias"] != 0).any()
    assert (compute_features(BARS, CONFIGS["htf_5m_wick_latest"]).bias["bias"] != 0).any()


def test_perfect_bias_is_the_one_detector_that_fails_the_test():
    cfg = BiasParams(method="perfect")
    cut = 5000
    prefix = BARS.iloc[:cut]
    with pytest.warns(LookaheadWarning):
        full = htf_bias(BARS, cfg)
    with pytest.warns(LookaheadWarning):
        part = htf_bias(prefix, cfg)
    assert full["bias_lookahead"].all() and part["bias_lookahead"].all()
    assert (full["bias"].loc[: prefix.index[-1]] != part["bias"]).any()      # the future leaks into the past
