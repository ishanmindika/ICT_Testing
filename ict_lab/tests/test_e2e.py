"""Raw triplet -> data build -> cache -> loaders -> named configs -> diagnostic / handcheck, on synthetic NQ."""
import pandas as pd
import pytest
import yaml
from pathlib import Path

from ict_lab.analysis.diagnostics import diagnose, format_diagnostic
from ict_lab.analysis.handcheck import handcheck
from ict_lab.data import build as data_build
from ict_lab.data.store import load_bars
from ict_lab.engine.config import load_named_configs
from ict_lab.engine.execute import run_backtest
from ict_lab.engine.runner import prepare, run_named
from ict_lab.tests.synth import make_sessions

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "data.yaml").read_text())
OFFSET = 50.0


@pytest.fixture(scope="module")
def nq_cache(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    raw, out = tmp / "raw", tmp / "clean"
    raw.mkdir()
    adj = make_sessions(14, first_day="2024-01-09", seed=5)
    days = sorted(adj["session_date"].unique())
    roll_day = pd.Timestamp(days[7])
    contract = pd.Series(["NQH4"] * len(adj), index=adj.index).where(adj["session_date"] < roll_day, "NQM4")
    un = adj[["open", "high", "low", "close", "volume"]].copy()
    old = (adj["session_date"] < roll_day).to_numpy()
    for c in ("open", "high", "low", "close"):            # back-adjusted: older contract shifted by -OFFSET relative to real prices
        un[c] = adj[c] + OFFSET
    ba = adj[["open", "high", "low", "close", "volume"]].copy()
    un["contract"] = ba["contract"] = contract
    un.to_parquet(raw / "NQ_1m_unadjusted.parquet")
    ba.to_parquet(raw / "NQ_1m_backadjusted.parquet")
    pd.DataFrame({"date": [f"{roll_day:%Y-%m-%d}"], "old_contract": ["NQH4"], "new_contract": ["NQM4"], "adjustment": [OFFSET]}
                 ).to_csv(raw / "NQ_rolls.csv", index=False)
    cfg = {**CFG, "instruments": {"NQ": CFG["instruments"]["NQ"]}, "holdout": {"cutoff_date": "2030-01-01", "years": 2}}
    data_build.build(cfg, raw, out)
    return cfg, out, roll_day


def test_named_configs_run_end_to_end_with_cache(nq_cache, tmp_path):
    cfg, out, roll_day = nq_cache
    results = {}
    for name in load_named_configs():
        bt = run_named(name, "NQ", "2024-01-15", "2024-01-31", cache_dir=tmp_path / "feat", clean_dir=out, config=cfg)
        results[name] = bt
        assert len(bt.windows) > 0 and set(bt.no_trades.columns) == {"session_date", "symbol", "window", "reason"}
    assert len(results["as_traded"].trades) > 0
    assert (tmp_path / "feat" / "NQ").exists()                              # features were cached to disk


def test_roll_day_flag_reaches_the_trade_log(nq_cache):
    cfg, out, roll_day = nq_cache
    bars = load_bars("NQ", clean_dir=out, config=cfg)
    assert bars.loc[bars.session_date == roll_day, "is_roll_day"].all() and not bars.loc[bars.session_date != roll_day, "is_roll_day"].any()
    bt = run_named("as_traded", "NQ", "2024-01-10", "2024-01-31", clean_dir=out, config=cfg)
    t = bt.trades
    assert len(t) > 0 and (t.is_roll_day == (t.session_date == roll_day)).all()


def test_diagnostic_and_handcheck_on_loaded_data(nq_cache):
    cfg, out, _ = nq_cache
    store, inst, days = prepare("NQ", "2024-01-15", "2024-01-31", clean_dir=out, config=cfg)
    bt = run_backtest(store, load_named_configs()["as_traded"], inst, sessions=days)
    text = format_diagnostic("as_traded", diagnose(bt, days))
    assert "reason_mix_by_year" in text and "days_with_trade_pct_overall" in text
    unadj = load_bars("NQ", "unadjusted", clean_dir=out, config=cfg).reindex(store.bars.index)
    out_text, chosen = handcheck(bt, store.bars, unadj, "as_traded", limit=3)
    assert f"offset +{OFFSET:.2f}".replace("+", "+") in out_text or "adj->unadj offset" in out_text


def test_holdout_is_respected_by_the_engine_path(nq_cache):
    cfg, out, _ = nq_cache
    tight = {**cfg, "holdout": {"cutoff_date": "2024-01-20", "years": 2}}
    store, inst, days = prepare("NQ", "2024-01-10", "2024-01-31", clean_dir=out, config=tight)
    assert store.bars["session_date"].max() < pd.Timestamp("2024-01-20") and days.max() < pd.Timestamp("2024-01-20")
    bt = run_backtest(store, load_named_configs()["as_traded"], inst, sessions=days)
    assert (bt.trades["session_date"] < pd.Timestamp("2024-01-20")).all() and (bt.windows["session_date"] < pd.Timestamp("2024-01-20")).all()
    store2, _, days2 = prepare("NQ", "2024-01-25", "2024-01-31", clean_dir=out, config=tight)
    assert len(days2) == 0                                                   # asking only for holdout dates yields nothing
