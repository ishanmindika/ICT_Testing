# ICT Silver Bullet backtesting lab

```
ict_lab/
  data/raw/     your purchased 1-minute bars (gitignored)
  data/clean/   cleaned parquet + manifest/gaps/rolls (gitignored)
  data/         loader.py · cleaner.py · sessions.py · store.py · build.py
  configs/data.yaml
  features/ engine/ runs/ analysis/   (next steps)
```

## Data layer
1. Drop vendor files in `ict_lab/data/raw/` and edit `ict_lab/configs/data.yaml` (`source.tz`, column names).
2. `pip install -e .[dev]` then `python -m ict_lab.data.build`
3. In code: `from ict_lab.data.store import load_bars, resample; load_bars("ES", "2024-01-01")`

Guarantees: UTC index; invalid/duplicate bars dropped and counted; front-month chosen from the *prior*
session's volume (no look-ahead), prices unadjusted; gaps are reported, never filled; `session_date`
follows the CME 18:00 ET session. Each output has a `*.manifest.json` with the full quality report.

### Pre-built triplets (`{ROOT}_1m_unadjusted.parquet`, `{ROOT}_1m_backadjusted.parquet`, `{ROOT}_rolls.csv`)
`build` detects these automatically. Output per root in `data/clean/`: `{ROOT}_1m.parquet` with real prices in
`open/high/low/close`, back-adjusted in `adj_*`, plus `offset`, `contract`, `session_date`; `{ROOT}_1m.rolls.parquet`;
manifest. The two files must match exactly on timestamps, volume and contract (else the build fails); `rolls.csv`
is cross-checked against the price files and any disagreement is listed as `roll_findings` in the manifest.
