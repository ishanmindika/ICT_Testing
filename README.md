# ICT Silver Bullet backtesting lab

```
ict_lab/
  data/raw/      {ROOT}_1m_unadjusted.parquet, {ROOT}_1m_backadjusted.parquet, {ROOT}_rolls.csv (gitignored)
  data/clean/    bars/symbol=ES/series=backadjusted/year=2021/*.parquet, rolls/, quality/, meta.json (gitignored)
  data/          continuous.py quality.py sessions.py store.py build.py show.py loader.py
  configs/data.yaml
  features/ engine/ runs/ analysis/   (next steps)
```

## Data layer
1. Put the six files in `ict_lab/data/raw/`; check `configs/data.yaml` (`source.tz`, column names, instruments).
2. `pip install -e .[dev]`, then `python -m ict_lab.data.build` (prints and saves the quality report to `data/clean/quality/`).
3. Load: `load_bars("ES")` is **back-adjusted** (strategy logic); `load_bars("ES", price_series="unadjusted")` is for charts.
   `load_rolls("ES")` is the roll log; `is_roll_day` is on both series.
4. Verify against a chart (window is ET): `python -m ict_lab.data.show ES "2024-03-08 09:30" "2024-03-08 10:00" --series unadjusted`

Columns: open high low close volume contract session_date rth killzone_london killzone_ny_am killzone_ny_pm is_roll_day.
Timestamps are UTC bar-open times; windows are ET wall-clock, half-open `[start, end)`.

## Holdout
`holdout.cutoff_date` in the config (first held-out `session_date`). If unset, the build uses last session - 2 years and
tells you to pin it. Every loader (`load_bars`, `load_rolls`, `show`) excludes the holdout unless `include_holdout=True`,
and the quality report never inspects it.

## Rules
No price ratios or percentages anywhere (points, ticks, R only); `tests/test_no_ratios.py` enforces it.

## Feature layer (`ict_lab/features`)
Detectors run on the back-adjusted 1m frame; thresholds are points / ATR multiples. Defaults in `configs/features.yaml`.
- `detect_fvgs(bars, FVGParams(timeframe, min_size_points, min_size_atr_mult, ...))`: top/bottom/mid/size/ATR multiple,
  `bar_idx`, `available_at`, and first touch / 50% / full fill (index + timestamp).
- `session_levels(bars, LevelParams())`: per-bar prior session / prior RTH / pre-window (session open -> killzone start)
  highs and lows plus `swept_*` flags.
- `detect_swings(bars, SwingParams(n=3, timeframe, strict))`: N-bar fractals with confirmation time and first wick-sweep / close-through.
- Sweep a range with `param_grid(FVGParams(), timeframe=["1m","5m","15m"], min_size_points=[0,1,2])`.
No look-ahead: every record has `available_at` (close of the confirming bar); tests verify prefix-invariance.

### Layer 2 continued: sweeps, displacement, MSS, bias, charts
`compute_features(bars, FeatureConfig())` runs everything (`configs/features.yaml`, load with `load_feature_config`).
- `level_table`: every level with `active_from`, `expires` (session levels) and `swept_ts`; active until swept.
- `detect_sweeps`: trade-through by >= `min_penetration_ticks`, close back within `k` bars; level types selectable.
- `detect_displacement`: ATR multiple (1.0/1.5/2.0/3.0...) or top-X% of window ranges; optional run of N bars.
- `detect_mss`: break of the latest swing opposite the sweep; `close`|`wick`, `at_sweep`|`latest` reference.
- `htf_bias`: `none | prior_day_oc | trend_swing | daily_ma_slope | perfect`. **`perfect` is look-ahead**: it warns
  (`LookaheadWarning`) and is labelled `perfect(LOOKAHEAD)` with `bias_lookahead=True` in every output.
- `tests/test_no_lookahead.py` runs every detector on full vs truncated data and requires identical results.
- Charts (unadjusted candles, features detected on back-adjusted and shifted onto the same axis):
  `python -m ict_lab.analysis.charts ES --n 15 --killzone killzone_ny_am --seed 7` -> `ict_lab/analysis/charts/`.

## Layer 3: signals + execution (`ict_lab/engine`)
- `FeatureStore` caches every detector stream by its parameters (memory + parquet): a new variant computes only its own stream.
- `build_signals`: per session x killzone window, eligibility -> bias gate -> sweep -> [MSS] -> [displacement] -> every
  direction-matching FVG after a qualifying chain. First failing stage is logged.
- `run_backtest`: limit entry (fills only on trade-through), swing/distal/fixed stops, R / next-opposing-liquidity / time targets,
  hard exit, stop-first on ambiguous bars (`ambiguous_bar`), tick-grid fills (integer ticks, conservative rounding),
  costs gross + net, `max_trades_per_window` 1 or unlimited (cap 10), several windows per config, no overlapping positions.
- Named configs: `configs/grid.json` (`as_taught_5m`, `as_taught_1m`, `as_traded`).
- Run: `python -m ict_lab.engine.runner as_traded NQ 2024-03-01 2024-03-31`
- Hand-check: `python -m ict_lab.analysis.handcheck NQ 2024-03-01 2024-03-31 --configs as_traded as_taught_5m`
- Frequency diagnostic (counts and reasons only, never PnL): `python -m ict_lab.analysis.diagnostics NQ`
