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
