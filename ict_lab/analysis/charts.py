"""Candlestick charts of a killzone window with every detected feature overlaid.

Candles are the UNADJUSTED series, so prices match what you would see on a live chart. Features are
detected on the BACK-ADJUSTED series (as the strategy sees them) and shifted onto the unadjusted axis by
the additive offset (unadjusted close - back-adjusted close) at the start of the chart.

    python -m ict_lab.analysis.charts ES --n 15 --killzone killzone_ny_am --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from ..data.sessions import ET, WINDOWS  # noqa: E402
from ..data.store import CLEAN_DIR, load_bars, load_config  # noqa: E402
from ..features import FeatureConfig, compute_features, displacement_mask, load_feature_config  # noqa: E402
from ..features.common import tf_minutes  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
UP, DOWN = "#26a69a", "#ef5350"
LEVEL_COLORS = {"prior_session": "#1f77b4", "prior_rth": "#9467bd", "pre_london": "#8c564b",
                "pre_ny_am": "#ff7f0e", "pre_ny_pm": "#e377c2", "swing": "#7f7f7f"}


def window_bounds(date: str | pd.Timestamp, killzone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC start/end of `killzone` on the ET calendar `date` (DST-correct)."""
    start, end = WINDOWS[killzone]
    day = pd.Timestamp(date).normalize()
    et = lambda hhmm: (day.tz_localize(ET) + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))).tz_convert("UTC")
    return et(start), et(end)


def plot_window(adj: pd.DataFrame, unadj: pd.DataFrame, date: str | pd.Timestamp, killzone: str = "killzone_ny_am",
                cfg: FeatureConfig = FeatureConfig(), pad_before: int = 30, pad_after: int = 30,
                context_days: int = 7, show_older_fvgs: bool = False) -> plt.Figure:
    """`adj` / `unadj` are aligned 1-minute frames (identical index). Detectors only see `adj`.
    FVGs formed before the chart starts are hidden unless `show_older_fvgs` (old unfilled gaps can dwarf the view)."""
    if not adj.index.equals(unadj.index):
        raise AssertionError("adjusted and unadjusted indexes differ")
    w0, w1 = window_bounds(date, killzone)
    c0, c1 = w0 - pd.Timedelta(minutes=pad_before), w1 + pd.Timedelta(minutes=pad_after)
    ctx = adj.loc[w0 - pd.Timedelta(days=context_days): c1]
    if ctx.loc[c0:c1].empty:
        raise ValueError(f"no bars for {date} {killzone} (holdout excluded, or a market holiday?)")
    fs = compute_features(ctx, cfg)

    view = unadj.loc[c0:c1]
    offset = (unadj["close"] - adj["close"]).loc[c0:c1]
    off0 = float(offset.iloc[0])
    pos = lambda ts: int(view.index.searchsorted(ts, side="left"))
    n = len(view)
    px = lambda p: p + off0

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axvspan(pos(w0) - 0.5, pos(w1) - 0.5, color="#bbbbbb", alpha=0.15, lw=0)

    # displacement: highlighted bars (behind candles)
    dis = fs.displacement[(fs.displacement["ts"] >= c0) & (fs.displacement["start_ts"] <= c1)]
    for r in dis.itertuples():
        ax.axvspan(pos(r.start_ts) - 0.5, pos(r.ts) + 0.5, color="#ffd54f", alpha=0.45, lw=0, zorder=1)

    # candles
    x = np.arange(n)
    o, h, l, c = (view[k].to_numpy() for k in ("open", "high", "low", "close"))
    col = np.where(c >= o, UP, DOWN)
    ax.vlines(x, l, h, colors=col, lw=1, zorder=3)
    ax.bar(x, np.maximum(np.abs(c - o), 1e-9), bottom=np.minimum(o, c), width=0.7, color=col, zorder=4)
    lo_y, hi_y = float(l.min()), float(h.max())
    pad = (hi_y - lo_y) * 0.12 + 0.5
    ylim = (lo_y - pad, hi_y + pad)

    # FVGs: shaded boxes from bar 1 until fully filled (or the chart edge)
    f = fs.fvgs
    f = f[(f["available_at"] <= c1) & (f["ts_bar1"] <= c1) & (f["fill_ts"].isna() | (f["fill_ts"] >= c0))]
    if not show_older_fvgs:
        f = f[f["ts_bar1"] >= c0]
    for r in f.itertuples():
        end = pos(r.fill_ts) if pd.notna(r.fill_ts) and r.fill_ts <= c1 else n - 1
        x0 = max(pos(r.ts_bar1), 0)
        ax.add_patch(Rectangle((x0 - 0.5, px(r.bottom)), end - x0 + 1, r.top - r.bottom,
                               facecolor=UP if r.direction == 1 else DOWN, alpha=0.22, edgecolor="none", zorder=2))

    # liquidity levels: horizontal lines while active
    lt = fs.level_table
    lt = lt[(lt["active_from"] <= c1) & (lt["expires"].isna() | (lt["expires"] > c0))]
    hidden = 0
    for r in lt.itertuples():
        y = px(r.price)
        if not ylim[0] <= y <= ylim[1]:
            hidden += 1
            continue
        end_ts = min([t for t in (r.swept_ts, r.expires, c1) if pd.notna(t)])
        x0, x1 = max(pos(r.active_from), 0), min(pos(end_ts), n - 1)
        if x1 <= x0 and not (x1 == x0 and r.type != "swing"):
            continue
        ax.hlines(y, x0 - 0.5, x1 + 0.5, colors=LEVEL_COLORS[r.type], lw=1.2,
                  linestyles="--" if r.type == "swing" else "-", zorder=2)
        ax.text(x1 + 0.6, y, f"{r.type} {r.side}", fontsize=7, va="center", color=LEVEL_COLORS[r.type], clip_on=True)

    # sweeps: markers at the extreme of the breach
    sw = fs.sweeps[(fs.sweeps["sweep_ts"] >= c0) & (fs.sweeps["sweep_ts"] <= c1)]
    for r in sw.itertuples():
        xs = pos(r.sweep_ts)
        up = r.sweep_direction == 1   # swept a low -> expect up
        ax.scatter([xs], [px(r.extreme_price) + (-0.35 if up else 0.35)], marker="^" if up else "v", s=90,
                   color="#2e7d32" if up else "#c62828", edgecolor="black", lw=0.6, zorder=6)

    # MSS: vertical line at the break bar
    ms = fs.mss[(fs.mss["mss_ts"] >= c0) & (fs.mss["mss_ts"] <= c1)]
    for r in ms.itertuples():
        xm = pos(r.mss_ts)
        colr = "#2e7d32" if r.direction == 1 else "#c62828"
        ax.axvline(xm, color=colr, ls="--", lw=1.4, zorder=5)
        ax.hlines(px(r.broken_price), pos(r.swing_ts) if r.swing_ts >= c0 else 0, xm, colors=colr, lw=1, zorder=5)
        ax.text(xm + 0.3, ylim[1] - pad * 0.3, f"MSS {'↑' if r.direction == 1 else '↓'}", color=colr, fontsize=8,
                rotation=90, va="top")

    ax.set_xlim(-1, n + 6)
    ax.set_ylim(*ylim)
    step = 5
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks, [view.index[i].tz_convert(ET).strftime("%H:%M") for i in ticks], rotation=0, fontsize=8)
    ax.set_xlabel("ET (bar open)")
    ax.set_ylabel("price (unadjusted)")
    ax.grid(alpha=0.2)

    bias = fs.bias["bias"].reindex(view.index).iloc[min(pos(w0), n - 1)]
    method = fs.bias["bias_method"].iloc[0]
    bias_txt = "" if method == "none" else f"   bias[{method}] = {int(bias):+d}"
    roll = "   ⚠ roll inside chart: overlay offset not constant" if offset.nunique() > 1 and offset.max() - offset.min() > 1e-6 else ""
    ax.set_title(f"{pd.Timestamp(date):%Y-%m-%d} {killzone}   ·   candles UNADJUSTED, overlays shifted {off0:+.2f} pts"
                 f"{bias_txt}{roll}", fontsize=10, loc="left")
    handles = [Patch(color=UP, alpha=0.3, label="bull FVG"), Patch(color=DOWN, alpha=0.3, label="bear FVG"),
               Patch(color="#ffd54f", alpha=0.6, label="displacement"),
               Line2D([], [], color="#7f7f7f", ls="--", label="swing level"),
               Line2D([], [], color="#1f77b4", label="session level"),
               Line2D([], [], marker="v", color="#c62828", ls="", label="sweep of high"),
               Line2D([], [], marker="^", color="#2e7d32", ls="", label="sweep of low"),
               Line2D([], [], color="#2e7d32", ls="--", label="MSS")]
    ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=4, framealpha=0.85)
    foot = f"{len(f)} FVG · {len(dis)} displacement · {len(sw)} sweeps · {len(ms)} MSS" + (f" · {hidden} levels off-screen" if hidden else "")
    fig.text(0.01, 0.005, foot, fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def sample_dates(adj: pd.DataFrame, killzone: str, n: int, seed: int, min_bars_frac: float = 0.9) -> list[pd.Timestamp]:
    """Random ET dates whose killzone window has (almost) all of its 1-minute bars."""
    minutes = tf_minutes("1m") * int((pd.Timestamp(f"2000-01-01 {WINDOWS[killzone][1]}") - pd.Timestamp(f"2000-01-01 {WINDOWS[killzone][0]}")).total_seconds() // 60)
    inside = adj[adj[killzone]]
    counts = inside.groupby(inside["session_date"]).size()
    ok = counts[(counts >= min_bars_frac * minutes) & (counts.index.dayofweek < 5)].index
    if len(ok) == 0:
        raise ValueError("no suitable days in the data")
    rng = np.random.default_rng(seed)
    return sorted(pd.Timestamp(d) for d in rng.choice(np.asarray(ok), size=min(n, len(ok)), replace=False))


def random_day_charts(adj: pd.DataFrame, unadj: pd.DataFrame, out_dir: Path, root: str = "X",
                      killzone: str = "killzone_ny_am", n: int = 15, seed: int = 0,
                      cfg: FeatureConfig = FeatureConfig(), **kw) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in sample_dates(adj, killzone, n, seed):
        fig = plot_window(adj, unadj, d, killzone, cfg, **kw)
        p = out_dir / f"{root}_{d:%Y-%m-%d}_{killzone}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--killzone", default="killzone_ny_am", choices=[k for k in WINDOWS if k != "rth"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", help="chart one specific ET date instead of random days (YYYY-MM-DD)")
    ap.add_argument("--features", default=str(LAB / "configs" / "features.yaml"))
    ap.add_argument("--out", default=str(LAB / "analysis" / "charts"))
    ap.add_argument("--include-holdout", action="store_true", help="allow holdout dates (don't, until the very end)")
    a = ap.parse_args()
    cfg = load_feature_config(a.features)
    kw = dict(include_holdout=a.include_holdout, clean_dir=CLEAN_DIR)
    adj, unadj = load_bars(a.root, "backadjusted", **kw), load_bars(a.root, "unadjusted", **kw)
    if a.date:
        fig = plot_window(adj, unadj, a.date, a.killzone, cfg)
        p = Path(a.out) / f"{a.root}_{a.date}_{a.killzone}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=110)
        print(p)
    else:
        for p in random_day_charts(adj, unadj, Path(a.out), a.root, a.killzone, a.n, a.seed, cfg):
            print(p)


if __name__ == "__main__":
    main()
