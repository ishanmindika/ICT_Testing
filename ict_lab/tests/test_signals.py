import numpy as np
import pandas as pd
import pytest

from ict_lab.data.sessions import add_session_columns
from ict_lab.engine.config import ExecConfig, Instrument, SignalConfig, StrategyConfig
from ict_lab.engine.execute import run_backtest
from ict_lab.engine.signals import build_signals
from ict_lab.engine.store import FeatureStore
from ict_lab.features import BiasParams, FVGParams

NQ = Instrument("NQ", 0.25, 5.0)
TUE = pd.Timestamp("2024-03-05")
T0 = pd.Timestamp("2024-03-05 15:00", tz="UTC")           # 10:00 ET Tue (EST): NY AM window start
at = lambda k: T0 + pd.Timedelta(minutes=k)


def flat_two_sessions(mods=None):
    """Mon + Tue sessions, flat 100 (high 100.5 / low 99.5), with {bar_offset_from_T0: (o, h, l, c)} overrides."""
    idx = pd.date_range("2024-03-03 23:00", "2024-03-05 21:59", freq="1min", tz="UTC", name="ts")
    idx = idx[~((idx >= "2024-03-04 22:00") & (idx < "2024-03-04 23:00"))]            # Monday maintenance break
    df = pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx)
    for k, (o, h, l, c) in (mods or {}).items():
        df.loc[at(k), ["open", "high", "low", "close"]] = [o, h, l, c]
    return add_session_columns(df)


# sweep of the pre-window low (99.5) at +10, displacement bar at +11, bullish FVG [100.00, 100.75] closes at +13
GOOD = {
    10: (100.0, 100.0, 99.0, 99.8),        # breach by 2 ticks, closes back above 99.5
    11: (99.8, 101.6, 99.7, 101.5),        # displacement: range 1.9 vs ATR ~1
    12: (101.5, 102.2, 100.75, 102.0),     # low 100.75 > bar-1 high 100.0  -> bullish FVG [100.00, 100.75]
    13: (102.0, 102.0, 101.5, 101.8),
    14: (101.8, 101.8, 100.8, 101.0),
    15: (101.0, 101.0, 100.0, 100.5),      # trades one tick through the 100.25 limit
    16: (100.5, 101.0, 100.4, 100.9),
    17: (100.9, 101.4, 100.6, 101.3),      # from here the bars overlap, so no further gaps form
    18: (101.3, 102.2, 101.0, 101.6),
    19: (101.6, 102.4, 101.4, 102.3),
    20: (102.3, 103.5, 102.2, 103.0),      # target 103.25 + 1 tick through
}


def run(mods, sig=None, ex=None, sessions=(TUE,), cache_dir=None):
    bars = flat_two_sessions(mods)
    cfg = StrategyConfig("t", sig or SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m")), ex or ExecConfig())
    return run_backtest(FeatureStore(bars, "NQ", cache_dir), cfg, NQ, sessions=pd.DatetimeIndex(sessions))


def test_full_chain_to_trade():
    r = run(GOOD)
    s = r.signals.iloc[0]
    assert len(r.signals) == 1 and s.direction == 1 and s.setup_ts == at(13)
    assert (s.fvg_bottom, s.fvg_top) == (100.0, 100.75) and s.sweep_ts == at(10) and s.swing_extreme == 99.0
    assert s.disp_ts == at(11) and s.sweep_available_at == at(11)
    t = r.trades.iloc[0]
    assert (t.entry_ts, t.entry_price, t.stop_price, t.target_price) == (at(15), 100.25, 98.75, 103.25)
    assert (t.exit_ts, t.exit_price, t.exit_reason) == (at(20), 103.25, "target_r")
    assert (t.gross_pnl, t.net_pnl, t.r_multiple) == (60.0, 56.0, pytest.approx(2.0))
    assert r.no_trades.empty


@pytest.mark.parametrize("name,edit,reason", [
    ("no_sweep", lambda m: {k: v for k, v in m.items() if k != 10} | {10: (100.0, 100.5, 99.5, 100.0)}, "no_sweep"),
    ("no_displacement", lambda m: m | {11: (99.8, 100.95, 99.75, 100.9)}, "no_displacement"),
    ("no_fvg", lambda m: m | {12: (101.5, 102.2, 100.0, 102.0)}, "no_fvg"),
    ("limit_unfilled", lambda m: m | {15: (101.0, 101.0, 100.5, 100.5), 16: (100.5, 101.0, 100.5, 100.9)}
        | {k: (102.5, 103.0, 102.5, 102.5) for k in range(21, 60)}, "limit_unfilled"),
])
def test_first_failed_condition_is_recorded(name, edit, reason):
    r = run(edit(dict(GOOD)))
    assert r.trades.empty, name
    assert list(r.no_trades.reason) == [reason]


def test_bias_gate_and_mss_gate_reasons():
    r = run(GOOD, sig=SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m"), bias=BiasParams(method="prior_day_oc")))
    assert list(r.no_trades.reason) == ["bias_gate"]                 # flat Monday -> neutral bias -> gate closed
    r = run(GOOD, sig=SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m"), mss_required=True))
    assert list(r.no_trades.reason) == ["no_mss"]
    r = run(GOOD, sig=SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m"), displacement_atr_mult=None))
    assert len(r.trades) == 1                                        # displacement optional


def test_ineligible_window_is_not_a_no_trade():
    bars = flat_two_sessions(GOOD)
    cfg = SignalConfig(windows=("ny_am", "ny_pm"), fvg=FVGParams(timeframe="1m"))
    bars = bars[~((bars.index >= "2024-03-05 19:00") & (bars.index < "2024-03-05 20:00"))]   # no bars in the NY PM window
    sg = build_signals(FeatureStore(bars, "NQ"), cfg, NQ, pd.DatetimeIndex([TUE]))
    st = dict(zip(sg.windows.window, sg.windows.status))
    assert st == {"ny_am": "ok", "ny_pm": "ineligible"}


def test_every_valid_fvg_after_a_chain_is_emitted_once():
    m = dict(GOOD)
    m |= {21: (103.0, 103.5, 102.8, 103.2)}                         # bar-19 high 102.4 < bar-21 low 102.8: a 2nd bullish FVG
    sg = build_signals(FeatureStore(flat_two_sessions(m), "NQ"), SignalConfig(windows=("ny_am",), fvg=FVGParams(timeframe="1m")),
                       NQ, pd.DatetimeIndex([TUE]))
    assert list(sg.signals.setup_ts) == [at(13), at(22)] and list(sg.signals.direction) == [1, 1]
    assert sg.signals.fvg_row.is_unique
