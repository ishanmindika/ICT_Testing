import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from ict_lab.analysis.charts import plot_window, random_day_charts, sample_dates, window_bounds
from ict_lab.tests.synth import make_sessions

ADJ = make_sessions(8, seed=3)
UN = ADJ.copy()
for c in ("open", "high", "low", "close"):
    UN[c] = ADJ[c] + 25.0


def test_window_bounds_are_dst_correct():
    assert window_bounds("2024-01-10", "killzone_ny_am")[0] == pd.Timestamp("2024-01-10 15:00", tz="UTC")   # EST
    assert window_bounds("2024-07-10", "killzone_ny_am")[0] == pd.Timestamp("2024-07-10 14:00", tz="UTC")   # EDT
    assert window_bounds("2024-01-10", "killzone_london")[1] == pd.Timestamp("2024-01-10 09:00", tz="UTC")


def test_chart_uses_unadjusted_prices_and_draws_overlays():
    d = sample_dates(ADJ, "killzone_ny_am", 1, 0)[0]
    fig = plot_window(ADJ, UN, d, "killzone_ny_am")
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    assert lo > 5000 + 20 - 200 and "UNADJUSTED" in ax.get_title(loc="left") and "+25.00" in ax.get_title(loc="left")
    view = UN.loc[window_bounds(d, "killzone_ny_am")[0]: window_bounds(d, "killzone_ny_am")[1]]
    assert lo < view["low"].min() and hi > view["high"].max()          # candles are the unadjusted series
    assert any(isinstance(p, Rectangle) for p in ax.patches)
    assert len(ax.collections) >= 2                                      # candle wicks + sweep markers/level lines
    plt.close(fig)


def test_detectors_never_see_unadjusted_prices(monkeypatch):
    import ict_lab.analysis.charts as charts
    seen = []
    real = charts.compute_features
    monkeypatch.setattr(charts, "compute_features", lambda bars, cfg: (seen.append(bars), real(bars, cfg))[1])
    d = sample_dates(ADJ, "killzone_ny_am", 1, 1)[0]
    shifted = UN.copy()
    for c in ("open", "high", "low", "close"):
        shifted[c] = ADJ[c] + 1000.0                                      # absurd offset
    fig = charts.plot_window(ADJ, shifted, d, "killzone_ny_am")
    assert "+1000.00" in fig.axes[0].get_title(loc="left")
    assert len(seen) == 1 and seen[0]["close"].equals(ADJ["close"].loc[seen[0].index])   # features ran on back-adjusted bars
    plt.close(fig)


def test_misaligned_series_refused():
    with pytest.raises(AssertionError):
        plot_window(ADJ, UN.iloc[1:], "2024-01-16")


def test_random_day_charts_written(tmp_path):
    paths = random_day_charts(ADJ, UN, tmp_path, "ES", "killzone_ny_pm", n=3, seed=2)
    assert len(paths) == 3 and all(p.exists() and p.stat().st_size > 10_000 for p in paths)
    assert paths == sorted(paths)
