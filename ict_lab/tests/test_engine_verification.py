"""End-to-end verification: cache-vs-scratch equality of TRADE logs, no-lookahead on the execution path
(multi-trade, multi-window, 15m liquidity), position non-overlap, named configs, diagnostics, handcheck."""
import numpy as np
import pandas as pd
import pytest

from ict_lab.analysis.diagnostics import diagnose, format_diagnostic
from ict_lab.analysis.handcheck import coverage, handcheck, select
from ict_lab.engine.config import (ExecConfig, Instrument, SignalConfig, StrategyConfig, config_from_dict,
                                   load_named_configs)
from ict_lab.engine.execute import Backtest, run_backtest
from ict_lab.engine.store import FeatureStore
from ict_lab.features import BiasParams, FVGParams, MSSParams, SwingParams
from ict_lab.tests.synth import make_sessions

NQ = Instrument("NQ", 0.25, 5.0)
BARS = make_sessions(9, first_day="2024-01-09", seed=21)          # "a month" of random but session-correct bars
SESSIONS = pd.DatetimeIndex(sorted(BARS["session_date"].unique()))


def run(cfg, bars=BARS, cache=None, sessions=None) -> Backtest:
    store = FeatureStore(bars, "NQ", cache)
    return run_backtest(store, cfg, NQ, sessions=sessions if sessions is not None else pd.DatetimeIndex(sorted(bars["session_date"].unique())))


def assert_no_overlap(trades: pd.DataFrame) -> None:
    t = trades.sort_values("entry_i")
    assert (t["entry_i"].to_numpy()[1:] > t["exit_i"].to_numpy()[:-1]).all(), "positions overlap in time"
    assert (t["exit_i"] >= t["entry_i"]).all()


# ---------------- named configs ----------------

def test_named_configs_match_the_spec():
    c = load_named_configs()
    assert set(c) == {"as_taught_5m", "as_taught_1m", "as_traded"}
    t5, t1, tr = c["as_taught_5m"], c["as_taught_1m"], c["as_traded"]
    for t, tf in ((t5, "5m"), (t1, "1m")):
        s, e = t.signal, t.exec
        assert (s.fvg.timeframe, s.fvg.min_size_points, s.fvg.min_size_atr_mult) == (tf, 0.0, 0.0)
        assert (s.bias.method, s.bias.trend_timeframe) == ("trend_swing", "15m")
        assert (s.sweep_required, s.sweep_universe, s.sweep_k, s.sweep_min_penetration_ticks) == (True, "bsl_ssl_15m", 3, 1.0)
        assert (s.displacement_atr_mult, s.displacement_atr_period, s.mss_required) == (1.5, 14, False)
        assert (e.entry, e.stop, e.target, e.fallback_r, e.max_trades_per_window) == ("mid", "swing", "liquidity", 2.0, 1)
        assert s.windows == ("london", "ny_am", "ny_pm")
    s, e = tr.signal, tr.exec
    assert (s.fvg.timeframe, s.bias.method, s.sweep_universe, s.sweep_k) == ("1m", "none", "session_refs_plus_swings", 3)
    assert (e.entry, e.stop, e.target, e.target_r, e.max_trades_per_window, e.trade_cap) == ("mid", "swing", "r", 2.0, "unlimited", 10)


def test_default_costs_and_instruments():
    e = ExecConfig()
    assert (e.commission_rt, e.stop_slippage_ticks, e.limit_through_ticks) == (4.0, 1, 1)
    nq, es = Instrument.from_config("NQ"), Instrument.from_config("ES")
    assert (nq.tick_size, nq.tick_value, es.tick_size, es.tick_value) == (0.25, 5.0, 0.25, 12.5)


def test_config_roundtrip_is_canonical():
    cfg = load_named_configs()["as_taught_5m"]
    again = config_from_dict(cfg.canonical(NQ))
    assert again.canonical_json(NQ) == cfg.canonical_json(NQ) and again.hash(NQ) == cfg.hash(NQ)


# ---------------- cache vs scratch ----------------

def random_config(rng: np.random.Generator) -> StrategyConfig:
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    wins = pick([("ny_am",), ("london", "ny_am"), ("london", "ny_am", "ny_pm")])
    sig = SignalConfig(
        windows=wins,
        fvg=FVGParams(timeframe=pick(["1m", "5m", "15m"]), min_size_points=pick([0.0, 0.5, 1.0])),
        bias=pick([BiasParams(), BiasParams(method="prior_day_oc"), BiasParams(method="daily_ma_slope", ma_period=3),
                   BiasParams(method="trend_swing", trend_timeframe="15m", trend_swing_n=1)]),
        sweep_required=True, sweep_universe=pick(["session_refs", "session_refs_plus_swings", "bsl_ssl_15m", "swings_only"]),
        sweep_k=pick([2, 3, 5]), sweep_min_penetration_ticks=pick([0.0, 1.0, 2.0]),
        mss_required=pick([False, False, True]), mss=MSSParams(swing=SwingParams(n=2)),
        displacement_atr_mult=pick([None, 1.0, 1.5]),
    )
    ex = ExecConfig(entry=pick(["proximal", "mid", "distal"]), stop=pick(["swing", "distal", "fixed"]),
                    stop_points=pick([2.0, 5.0]), target=pick(["r", "liquidity", "time"]), target_r=pick([1.0, 2.0, 3.0]),
                    target_time_minutes=pick([10, 30]), hard_exit=pick(["window_end", "rth_end"]),
                    max_trades_per_window=pick([1, "unlimited"]))
    return StrategyConfig("random", sig, ex)


def test_cache_vs_scratch_give_identical_trade_logs(tmp_path):
    rng = np.random.default_rng(7)
    total_trades = 0
    for i in range(3):
        cfg = random_config(rng)
        cache = tmp_path / f"c{i}"
        cold = run(cfg, cache=cache)                                      # computes and writes the cache
        warm_store = FeatureStore(BARS, "NQ", cache)                      # fresh process-like store: reads parquet
        warm = run_backtest(warm_store, cfg, NQ, sessions=SESSIONS)
        scratch = run(cfg, cache=None)                                    # no cache at all
        assert warm_store.stats["computed"] == 0 and warm_store.stats["disk"] > 0
        for other in (warm, scratch):
            pd.testing.assert_frame_equal(cold.trades, other.trades, check_dtype=False)
            pd.testing.assert_frame_equal(cold.signals, other.signals, check_dtype=False)
            pd.testing.assert_frame_equal(cold.no_trades, other.no_trades, check_dtype=False)
            pd.testing.assert_frame_equal(cold.windows, other.windows, check_dtype=False)
        assert_no_overlap(cold.trades)
        total_trades += len(cold.trades)
    assert total_trades > 0                                               # not vacuous


def test_new_parameter_variant_recomputes_only_its_own_streams(tmp_path):
    base = SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m"), sweep_universe="bsl_ssl_15m")
    s1 = FeatureStore(BARS, "NQ", tmp_path)
    run_backtest(s1, StrategyConfig("a", base), NQ, sessions=SESSIONS)
    s2 = FeatureStore(BARS, "NQ", tmp_path)
    from dataclasses import replace
    run_backtest(s2, StrategyConfig("b", replace(base, sweep_min_penetration_ticks=3.0)), NQ, sessions=SESSIONS)
    # only the sweep stream (and the MSS-free chain around it) is new; swings, level table, FVGs, displacement, bias come from disk
    assert 1 <= s2.stats["computed"] <= 2 and s2.stats["disk"] >= 4


# ---------------- no look-ahead on the execution path ----------------

PATHS = {
    "multi_trade_multi_window_1m": StrategyConfig("a", SignalConfig(
        windows=("london", "ny_am", "ny_pm"), fvg=FVGParams(timeframe="1m"), sweep_universe="session_refs_plus_swings",
        displacement_atr_mult=1.0), ExecConfig(max_trades_per_window="unlimited", target="r", target_r=1.0)),
    "taught_15m_liquidity_1m_fvg": StrategyConfig("b1", SignalConfig(
        windows=("london", "ny_am", "ny_pm"), fvg=FVGParams(timeframe="1m"), sweep_universe="bsl_ssl_15m",
        swing_15m=SwingParams(timeframe="15m", n=1), displacement_atr_mult=None, sweep_min_penetration_ticks=0.0),
        ExecConfig(target="liquidity", max_trades_per_window="unlimited")),
    "taught_15m_liquidity_5m_fvg": StrategyConfig("b", SignalConfig(
        windows=("london", "ny_am", "ny_pm"), fvg=FVGParams(timeframe="5m"), sweep_universe="bsl_ssl_15m",
        swing_15m=SwingParams(timeframe="15m", n=1), displacement_atr_mult=None, sweep_min_penetration_ticks=0.0,
        bias=BiasParams(method="none")), ExecConfig(target="liquidity", max_trades_per_window=3)),
    "taught_with_15m_bias_and_swing_target": StrategyConfig("c", SignalConfig(
        windows=("ny_am", "ny_pm"), fvg=FVGParams(timeframe="1m"), sweep_universe="swings_only",
        bias=BiasParams(method="trend_swing", trend_timeframe="15m", trend_swing_n=1), displacement_atr_mult=None),
        ExecConfig(target="liquidity", hard_exit="rth_end", max_trades_per_window="unlimited")),
}
CUTS = [3300, 5200, 8000]


@pytest.mark.parametrize("name", list(PATHS))
def test_truncated_run_matches_full_run_up_to_the_cut(name):
    cfg = PATHS[name]
    full = run(cfg)
    assert len(full.trades) >= 2, f"{name}: too few trades for a meaningful test"
    assert_no_overlap(full.trades)
    compared = 0
    for cut in CUTS:
        prefix = BARS.iloc[:cut]
        last_open = prefix.index[-1]
        part = run(cfg, bars=prefix)
        assert_no_overlap(part.trades)
        # signals knowable by the cut
        key = ["session_date", "window", "direction", "setup_ts", "fvg_top", "fvg_bottom", "sweep_level_type",
               "sweep_level_price", "sweep_ts", "swing_extreme", "disp_ts", "bias"]
        a = full.signals[full.signals["setup_ts"] <= last_open][key].reset_index(drop=True)
        b = part.signals[part.signals["setup_ts"] <= last_open][key].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, obj=f"{name}/signals@{cut}")
        # trades that finished before the cut (an exit forced by the end of data is not a finished trade)
        done = lambda t: t[(t["exit_i"] <= cut - 1) & (t["exit_reason"] != "data_end")].reset_index(drop=True)
        pd.testing.assert_frame_equal(done(full.trades), done(part.trades), check_dtype=False, obj=f"{name}/trades@{cut}")
        compared += len(done(part.trades))
    assert compared > 0, "no finished trade was compared: the test would pass vacuously"


# ---------------- diagnostics are counts only ----------------

def test_frequency_diagnostic_is_counts_and_reasons_only():
    cfg = PATHS["multi_trade_multi_window_1m"]
    bt = run(cfg)
    only_keys = Backtest(bt.trades[["session_date", "window"]], bt.windows, bt.no_trades, bt.signals)   # PnL columns removed
    d = diagnose(only_keys, SESSIONS)                                                                     # still works
    text = format_diagnostic("x", d).lower()
    for banned in ("pnl", "win rate", "winrate", "r_multiple", "expectancy", "profit", "sharpe", "drawdown", "gross", "net_"):
        assert banned not in text, banned
    assert set(d) >= {"windows_with_setup_pct_by_window", "windows_with_setup_pct_by_year_window", "days_with_trade_pct_overall",
                      "trades_per_year", "trades_per_day_distribution", "reason_mix_by_year", "reason_mix_by_window"}
    pct = d["days_with_trade_pct_overall"]
    assert pct["trading_days"] == len(SESSIONS) and pct["days_with_trade"] == bt.trades["session_date"].nunique()
    mix = d["reason_mix_by_window"].to_numpy().sum()
    assert mix == (bt.windows["status"] == "ok").sum()                                                   # every eligible window is classified


# ---------------- hand-check printout ----------------

def test_handcheck_prints_bars_and_covers_exit_types():
    bt = run(PATHS["multi_trade_multi_window_1m"])
    unadj = BARS.copy()
    for c in ("open", "high", "low", "close"):
        unadj[c] = BARS[c] + 12.0
    text, chosen = handcheck(bt, BARS, unadj, "demo", limit=12)
    assert "=== TRADE 1/" in text and "ENTRY" in text and "EXIT" in text and "adj->unadj offset +12.00" in text
    cov = coverage(chosen)
    assert cov["stop fill"] > 0 and cov["target fill (R)"] > 0 and len(chosen) >= 10
    first_of_each = select(bt.trades.reset_index(drop=True), 5)
    assert len(first_of_each) >= 4                                        # category-first selection
