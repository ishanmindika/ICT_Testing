"""FeatureStore: computes each feature stream once and caches it, so configs share work.

Every stream is keyed by the parameters that define it (and a fingerprint of the bars), so asking for a
new variant (say a different sweep penetration) computes only that stream; everything it builds on is
reused. Streams are cached in memory and, when `cache_dir` is set, as parquet files. The detectors in
`ict_lab.features` are used as they are.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import pandas as pd

from ..features import (BiasParams, DisplacementParams, FVGParams, LevelParams, SweepParams, SwingParams,
                        detect_displacement, detect_fvgs, detect_mss, detect_swings, detect_sweeps, htf_bias,
                        level_table, session_levels)
from ..features.levels import LEVEL_TYPES
from ..features.mss import MSSParams


def _key(*objs) -> str:
    payload = json.dumps([asdict(o) if hasattr(o, "__dataclass_fields__") else o for o in objs],
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def fingerprint(bars: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(bars[["open", "high", "low", "close"]], index=True).to_numpy()
    return hashlib.sha1(h.tobytes() + str((len(bars), bars.index[0], bars.index[-1])).encode()).hexdigest()[:12]


class FeatureStore:
    """`bars` is the BACK-ADJUSTED 1-minute frame (see ict_lab.data.store.load_bars)."""

    def __init__(self, bars: pd.DataFrame, symbol: str, cache_dir: Path | str | None = None):
        self.bars, self.symbol = bars, symbol
        self.fp = fingerprint(bars)
        self.dir = None if cache_dir is None else Path(cache_dir) / symbol / self.fp
        self._mem: dict[str, pd.DataFrame] = {}
        self.stats = {"computed": 0, "disk": 0, "memory": 0}

    # ---- generic cache -------------------------------------------------------------------
    def _get(self, name: str, key: str, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        k = f"{name}-{key}"
        if k in self._mem:
            self.stats["memory"] += 1
            return self._mem[k]
        path = None if self.dir is None else self.dir / f"{k}.parquet"
        if path is not None and path.exists():
            df = pd.read_parquet(path)
            self.stats["disk"] += 1
        else:
            df = compute()
            self.stats["computed"] += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path)
        self._mem[k] = df
        return df

    # ---- streams ---------------------------------------------------------------------------
    def session_levels(self, p: LevelParams = LevelParams()) -> pd.DataFrame:
        return self._get("session_levels", _key(p), lambda: session_levels(self.bars, p))

    def swings(self, p: SwingParams) -> pd.DataFrame:
        return self._get("swings", _key(p), lambda: detect_swings(self.bars, p))

    def level_table(self, swing_p: SwingParams | None, p: LevelParams = LevelParams()) -> pd.DataFrame:
        """Session/pre-window levels plus the swings of `swing_p` (None = no swing levels)."""
        def make() -> pd.DataFrame:
            swings = self.swings(swing_p) if swing_p is not None else detect_swings(self.bars.iloc[:5], SwingParams()).iloc[:0]
            return level_table(self.bars, p, swing_p or SwingParams(), per_bar=self.session_levels(p), swings=swings)
        return self._get("level_table", _key(p, swing_p), make)

    def sweeps(self, swing_p: SwingParams | None, k: int, min_penetration_ticks: float, tick_size: float,
               p: LevelParams = LevelParams()) -> pd.DataFrame:
        """All level types at once; consumers filter by `level_type`."""
        sp = SweepParams(level_types=LEVEL_TYPES, k=k, min_penetration_ticks=min_penetration_ticks, tick_size=tick_size)
        return self._get("sweeps", _key(p, swing_p, sp),
                         lambda: detect_sweeps(self.bars, self.level_table(swing_p, p), sp))

    def mss(self, swing_p: SwingParams | None, k: int, min_penetration_ticks: float, tick_size: float, m: MSSParams) -> pd.DataFrame:
        sw = self.sweeps(swing_p, k, min_penetration_ticks, tick_size)
        return self._get("mss", _key(swing_p, k, min_penetration_ticks, tick_size, m),
                         lambda: detect_mss(self.bars, sw, m, swings=self.swings(m.swing)))

    def fvgs(self, p: FVGParams) -> pd.DataFrame:
        return self._get("fvgs", _key(p), lambda: detect_fvgs(self.bars, p))

    def displacement(self, p: DisplacementParams) -> pd.DataFrame:
        return self._get("displacement", _key(p), lambda: detect_displacement(self.bars, p))

    def bias(self, p: BiasParams) -> pd.DataFrame:
        return self._get("bias", _key(p), lambda: htf_bias(self.bars, p))
