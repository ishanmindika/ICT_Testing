"""Run every detector with one config and bundle the results."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .bias import BiasParams, htf_bias
from .displacement import DisplacementParams, detect_displacement
from .fvg import FVGParams, detect_fvgs
from .levels import level_table
from .liquidity import LevelParams, SwingParams, detect_swings, session_levels
from .mss import MSSParams, detect_mss
from .sweep import SweepParams, detect_sweeps


@dataclass(frozen=True)
class FeatureConfig:
    fvg: FVGParams = field(default_factory=FVGParams)
    swing: SwingParams = field(default_factory=SwingParams)
    levels: LevelParams = field(default_factory=LevelParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    displacement: DisplacementParams = field(default_factory=DisplacementParams)
    mss: MSSParams = field(default_factory=MSSParams)
    bias: BiasParams = field(default_factory=BiasParams)


@dataclass
class FeatureSet:
    session_levels: pd.DataFrame   # per bar
    level_table: pd.DataFrame
    swings: pd.DataFrame
    fvgs: pd.DataFrame
    sweeps: pd.DataFrame
    displacement: pd.DataFrame
    mss: pd.DataFrame
    bias: pd.DataFrame             # per bar


def compute_features(bars: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> FeatureSet:
    """`bars` must be the BACK-ADJUSTED 1-minute frame (see data.store.load_bars)."""
    per_bar = session_levels(bars, cfg.levels)
    swings = detect_swings(bars, cfg.swing)
    table = level_table(bars, cfg.levels, cfg.swing, per_bar=per_bar, swings=swings)
    sweeps = detect_sweeps(bars, table, cfg.sweep)
    mss_swings = swings if cfg.mss.swing == cfg.swing else None
    return FeatureSet(
        session_levels=per_bar, level_table=table, swings=swings,
        fvgs=detect_fvgs(bars, cfg.fvg), sweeps=sweeps,
        displacement=detect_displacement(bars, cfg.displacement),
        mss=detect_mss(bars, sweeps, cfg.mss, swings=mss_swings),
        bias=htf_bias(bars, cfg.bias),
    )


def config_from_dict(d: dict[str, Any]) -> FeatureConfig:
    """Build a FeatureConfig from the nested dict in configs/features.yaml (missing keys use defaults)."""
    kw: dict[str, Any] = {}
    for f in fields(FeatureConfig):
        sub = dict(d.get(f.name) or {})
        cls = f.default_factory  # the params dataclass
        if f.name == "mss" and "swing" in sub:
            sub["swing"] = SwingParams(**sub["swing"])
        if f.name == "sweep" and "level_types" in sub:
            sub["level_types"] = tuple(sub["level_types"])
        if f.name == "levels" and "killzones" in sub:
            sub["killzones"] = tuple(sub["killzones"])
        kw[f.name] = cls(**sub)
    return FeatureConfig(**kw)


def load_feature_config(path: Path | str) -> FeatureConfig:
    return config_from_dict(yaml.safe_load(Path(path).read_text()))
