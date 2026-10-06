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
