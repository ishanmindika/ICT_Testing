import numpy as np
import pandas as pd
import pytest

from ict_lab.engine.config import ExecConfig, Instrument, SignalConfig, StrategyConfig
from ict_lab.engine.execute import run_backtest
from ict_lab.engine.signals import SIGNAL_COLS, Signals
from ict_lab.engine.store import FeatureStore
from ict_lab.tests.test_features import T0, frame

NQ, ES = Instrument("NQ", 0.25, 5.0), Instrument("ES", 0.25, 12.5)
SD = pd.Timestamp("2024-03-05")
M = lambda k: T0 + pd.Timedelta(minutes=k)


def bars(mods=None, n=75, base_hi=103.0, base_lo=102.0, mirror=False):
    """Flat far-away baseline; mods = {bar: (high, low[, open[, close]])}."""
    hi, lo = np.full(n, base_hi), np.full(n, base_lo)
    op = cl = (hi + lo) / 2
    op, cl = op.copy(), cl.copy()
    for i, m in (mods or {}).items():
        hi[i], lo[i] = m[0], m[1]
        op[i] = m[2] if len(m) > 2 else (m[0] + m[1]) / 2
        cl[i] = m[3] if len(m) > 3 else op[i]
    if mirror:
        hi, lo, op, cl = -lo, -hi, -op, -cl
    f = frame(hi, lo, closes=cl)
    f["open"] = op
    return f


def setup(bar=2, direction=1, top=102.0, bottom=100.0, extreme=99.0, window="ny_am"):
    if direction == -1:  # mirrored world
        top, bottom, extreme = -bottom, -top, -extreme
    r = dict.fromkeys(SIGNAL_COLS, np.nan)
    r.update(session_date=SD, symbol="NQ", window=window, direction=direction, setup_ts=M(bar), fvg_row=0, fvg_top=top,
             fvg_bottom=bottom, fvg_mid=(top + bottom) / 2, fvg_ts_bar1=M(bar - 3), fvg_ts_bar3=M(bar - 1), fvg_minutes=1,
             sweep_level_id=-1, sweep_level_type="prior_session", swing_extreme=extreme, bias=0,
             window_start=T0, window_end=M(60))
    return r


def run(b, setups, instrument=NQ, **ex):
    store = FeatureStore(b, instrument.symbol)
    sig = pd.DataFrame(setups, columns=SIGNAL_COLS)
    wl = pd.DataFrame([{"session_date": SD, "symbol": "NQ", "window": "ny_am", "window_start": T0, "window_end": M(60),
                        "status": "ok", "bias": 0, "n_sweeps": 1, "n_chains": 1, "n_setups": len(setups), "reason": ""}])
    cfg = StrategyConfig("t", SignalConfig(windows=("ny_am",)), ExecConfig(**ex))
    return run_backtest(store, cfg, instrument, signals=Signals(sig, wl))


FILL = {3: (101.5, 100.75)}               # low 100.75 = one tick THROUGH the 101.00 limit


def test_touch_does_not_fill_but_one_tick_through_does():
    r = run(bars({3: (102.0, 101.0)}), [setup()])
    assert r.trades.empty and list(r.no_trades.reason) == ["limit_unfilled"]
    r = run(bars(FILL), [setup()])
    t = r.trades.iloc[0]
    assert t.entry_price == 101.0 and t.entry_ts == M(3) and t.entry_i == 3


def test_order_becomes_active_the_bar_after_setup_and_cancels_at_window_end():
    r = run(bars({2: (101.5, 100.5)}), [setup(bar=3)])         # fill price trades on bar 2, before the order exists
    assert r.trades.empty
    r = run(bars({60: (101.5, 100.5)}), [setup()])             # bar 60 = 11:00 ET, outside the window
    assert r.trades.empty and list(r.no_trades.reason) == ["limit_unfilled"]


def test_target_fill_costs_and_r():
    b = bars({**FILL, 4: (105.5, 101.0), 5: (105.75, 102.0)})  # bar 4 only touches 105.5; bar 5 trades through
    t = run(b, [setup()]).trades.iloc[0]
    assert (t.stop_price, t.target_price, t.exit_price, t.exit_reason) == (98.75, 105.5, 105.5, "target_r")
    assert t.exit_ts == M(5) and t.bars_held == 2
    assert t.gross_pnl == 18 * 5.0 and t.commission == 4.0 and t.net_pnl == 86.0
    assert t.r_multiple == pytest.approx(2.0) and t.risk_points == 2.25 and not t.ambiguous_bar
    es = run(b, [setup()], instrument=ES).trades.iloc[0]
    assert es.gross_pnl == 18 * 12.5 and es.net_pnl == 18 * 12.5 - 4.0


def test_stop_slippage_and_gap_through_stop():
    t = run(bars({**FILL, 4: (101.5, 98.75)}), [setup()]).trades.iloc[0]
    assert (t.exit_reason, t.exit_price) == ("stop", 98.5) and t.gross_pnl == -10 * 5.0       # stop 98.75 - 1 tick
    gap = run(bars({**FILL, 4: (98.5, 97.5, 98.0)}), [setup()]).trades.iloc[0]
    assert gap.exit_price == 97.75                                                           # opens through the stop: worse fill
    none = run(bars({**FILL, 4: (101.5, 98.75)}), [setup()], stop_slippage_ticks=0).trades.iloc[0]
    assert none.exit_price == 98.75


def test_stop_on_entry_bar_and_target_never_on_entry_bar():
    t = run(bars({3: (101.5, 98.5)}), [setup()]).trades.iloc[0]
    assert (t.exit_reason, t.exit_i, t.bars_held, t.ambiguous_bar) == ("stop", 3, 0, False)
    # entry bar reaches target level but never the stop: the target cannot be assumed to come after the fill
    t = run(bars({3: (106.0, 100.75)}), [setup()]).trades.iloc[0]
    assert t.exit_reason == "hard_exit"


def test_ambiguous_bar_stop_wins_and_is_flagged():
    t = run(bars({**FILL, 4: (106.0, 98.5)}), [setup()]).trades.iloc[0]
    assert t.exit_reason == "stop" and t.ambiguous_bar
    assert t.exit_price == 98.5
    entry_bar = run(bars({3: (106.0, 98.5)}), [setup()]).trades.iloc[0]
    assert entry_bar.exit_reason == "stop" and entry_bar.ambiguous_bar


def test_target_needs_trade_through_unless_configured():
    b = bars({**FILL, 4: (105.5, 101.0)})                      # touches the 105.5 target exactly
    assert run(b, [setup()]).trades.iloc[0].exit_reason == "hard_exit"
    assert run(b, [setup()], target_through_ticks=0).trades.iloc[0].exit_reason == "target_r"


def test_hard_exit_at_window_end_open_with_slippage_and_rth_end():
    b = bars({**FILL, 60: (103.0, 102.0, 102.5)})              # 11:00 ET bar opens at 102.5
    t = run(b, [setup()]).trades.iloc[0]
    assert (t.exit_reason, t.exit_ts, t.exit_price) == ("hard_exit", M(60), 102.25)
    # rth_end = 16:00 ET = 21:00 UTC = bar 360 -> beyond this 75-bar frame -> data_end at the last close
    t = run(b, [setup()], hard_exit="rth_end").trades.iloc[0]
    assert t.exit_reason == "data_end" and t.exit_i == 74


def test_time_target():
    t = run(bars({**FILL}), [setup()], target="time", target_time_minutes=10).trades.iloc[0]
    assert (t.exit_reason, t.exit_ts, t.exit_i) == ("target_time", M(13), 13) and np.isnan(t.target_price)
    assert t.exit_price == 102.25                              # open of that bar (102.5) minus one tick


def test_entry_and_stop_options():
    f = {3: (101.5, 99.75)}                                    # reaches proximal 102 -> fill needs 101.75, mid 101, distal 100 -> 99.75
    assert run(bars(f), [setup()], entry="proximal").trades.iloc[0].entry_price == 102.0
    assert run(bars(f), [setup()], entry="distal").trades.iloc[0].entry_price == 100.0
    d = run(bars(f), [setup()], stop="distal", stop_buffer_ticks=2).trades.iloc[0]
    assert d.stop_price == 99.5
    fx = run(bars(f), [setup()], stop="fixed", stop_points=3.0).trades.iloc[0]
    assert fx.stop_price == 98.0
    r3 = run(bars(f), [setup()], target_r=3.0).trades.iloc[0]
    assert r3.target_price == 107.75


def test_mae_mfe_entry_bar_rules():
    b = bars({3: (104.0, 100.0), 4: (103.0, 101.0), 5: (105.75, 102.0)})
    t = run(b, [setup()]).trades.iloc[0]
    assert t.mae_points == 1.0                                 # entry bar low 100 vs entry 101
    assert t.mfe_points == 4.75                                # entry bar high (104) ignored; bar 5 high 105.75


def test_invalid_risk_reason():
    r = run(bars(FILL), [setup()], entry="distal", stop="distal", stop_buffer_ticks=0)
    assert r.trades.empty and list(r.no_trades.reason) == ["invalid_risk"]


def test_contracts_scale_costs():
    t = run(bars({**FILL, 4: (105.75, 102.0)}), [setup()], contracts=3).trades.iloc[0]
    assert t.gross_pnl == 18 * 5.0 * 3 and t.commission == 12.0


def test_fills_land_on_the_tick_grid_and_round_against_the_trader():
    # mid of 100.00..100.25 = 100.125 -> long entry rounds DOWN to 100.00; stop/target on grid
    t = run(bars({3: (101.0, 99.75)}), [setup(top=100.25, bottom=100.0, extreme=99.5)]).trades.iloc[0]
    assert t.entry_price == 100.0
    for col in ("entry_price", "stop_price", "target_price", "exit_price"):
        assert (t[col] / 0.25) == pytest.approx(round(t[col] / 0.25))
    # short mirror: mid of the same gap rounds UP
    s = run(bars({3: (101.0, 99.75)}, mirror=True), [setup(top=100.25, bottom=100.0, extreme=99.5, direction=-1)])
    assert s.trades.iloc[0].entry_price == -100.0 and s.trades.iloc[0].direction == -1


@pytest.mark.parametrize("mods,kw", [
    ({**FILL, 4: (105.75, 102.0)}, {}),                         # target
    ({**FILL, 4: (101.5, 98.75)}, {}),                          # stop
    ({**FILL, 4: (106.0, 98.5)}, {}),                           # ambiguous
    ({**FILL, 60: (103.0, 102.0, 102.5)}, {}),                  # hard exit
])
def test_short_is_the_exact_mirror_of_long(mods, kw):
    lg = run(bars(mods), [setup()], **kw).trades.iloc[0]
    sh = run(bars(mods, mirror=True), [setup(direction=-1)], **kw).trades.iloc[0]
    assert sh.direction == -1
    for col in ("entry_price", "stop_price", "exit_price"):
        assert sh[col] == -lg[col]
    for col in ("exit_reason", "bars_held", "gross_pnl", "net_pnl", "r_multiple", "ambiguous_bar", "mae_points", "mfe_points"):
        assert sh[col] == lg[col], col


def test_sequential_setups_skip_while_open_and_never_overlap():
    b = bars({3: (101.5, 100.75), 4: (101.5, 98.75),            # trade 1: fills bar 3, stopped bar 4
              7: (101.5, 100.75), 8: (105.75, 102.0)})          # trade 2: fills bar 7, target bar 8
    s = [setup(bar=2), setup(bar=4), setup(bar=6)]              # 2nd arrives while trade 1 is open (bar 4 = its exit bar)
    one = run(b, s)
    assert len(one.trades) == 1
    many = run(b, s, max_trades_per_window="unlimited")
    t = many.trades
    assert list(t.exit_reason) == ["stop", "target_r"] and list(t.trade_no) == [1, 2]
    assert (t.entry_i.iloc[1:].to_numpy() > t.exit_i.iloc[:-1].to_numpy()).all()
    assert many.windows.n_skipped_busy.iloc[0] == 1             # the setup that arrived on the exit bar


def test_unfilled_first_order_ends_the_window_and_cap_applies():
    r = run(bars({10: (101.5, 100.75)}), [setup(bar=2), setup(bar=8)], max_trades_per_window="unlimited")
    assert len(r.trades) == 1                                    # first order (bar 2) fills on bar 10: sequential, one order at a time
    with pytest.raises(ValueError):
        ExecConfig(max_trades_per_window=11)
    assert ExecConfig(max_trades_per_window="unlimited").trade_cap == 10


def test_trade_log_has_every_required_column_and_the_config():
    t = run(bars({**FILL, 4: (105.75, 102.0)}), [setup()]).trades
    need = ["session_date", "symbol", "window", "direction", "setup_ts", "entry_ts", "entry_price", "stop_price",
            "target_price", "exit_ts", "exit_price", "exit_reason", "bars_held", "gross_pnl", "net_pnl", "r_multiple",
            "mae_points", "mfe_points", "ambiguous_bar", "is_roll_day", "config"]
    assert not set(need) - set(t.columns)
    import json
    cfg = json.loads(t.config.iloc[0])
    assert cfg["exec"]["entry"] == "mid" and cfg["instrument"]["tick_value"] == 5.0 and cfg["signal"]["fvg"]["timeframe"] == "5m"


# ---------------- liquidity targets (levels from the cached streams) ----------------

def test_liquidity_target_uses_nearest_unswept_level_then_falls_back_to_r():
    from ict_lab.tests.test_features import session_frame
    sf = session_frame()                      # Tue: prior-session high 110, pre-NY-AM high 107; both swept by the 120 bar at 15:30
    store = FeatureStore(sf, "NQ")

    def go(setup_bar):
        sig = pd.DataFrame([setup(bar=setup_bar, top=100.25, bottom=99.25, extreme=99.0)], columns=SIGNAL_COLS)
        wl = pd.DataFrame([{"session_date": SD, "symbol": "NQ", "window": "ny_am", "window_start": T0, "window_end": M(60),
                            "status": "ok", "bias": 0, "n_sweeps": 1, "n_chains": 1, "n_setups": 1, "reason": ""}])
        cfg = StrategyConfig("t", SignalConfig(windows=("ny_am",)), ExecConfig(target="liquidity"))
        return run_backtest(store, cfg, NQ, signals=Signals(sig, wl)).trades.iloc[0]

    t = go(1)                                 # fills 15:02 (low 99.5 = one tick through 99.75)
    assert (t.target_source, t.target_price, t.exit_reason, t.exit_ts) == ("liquidity", 107.0, "target_liquidity", M(30))
    late = go(35)                             # after the 15:30 sweep nothing above is unswept
    assert (late.target_source, late.target_price) == ("fallback_r", 101.75)    # entry 99.75, risk 1.0 -> +2R
