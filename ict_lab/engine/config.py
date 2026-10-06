"""Strategy configuration: signal rules + execution rules, with a canonical JSON form for logging/hashing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from ..features import BiasParams, FVGParams, MSSParams, SwingParams

LAB = Path(__file__).resolve().parents[1]
HARD_CAP = 10                      # safety cap on trades per window when max_trades_per_window = "unlimited"
WINDOW_NAMES = ("london", "ny_am", "ny_pm")
UNIVERSES = ("session_refs", "session_refs_plus_swings", "bsl_ssl_15m", "swings_only")
ENTRIES = ("proximal", "mid", "distal")
STOPS = ("swing", "distal", "fixed")
TARGETS = ("r", "liquidity", "time")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    tick_value: float            # dollars per tick per contract

    @classmethod
    def from_config(cls, symbol: str, data_cfg: dict | None = None) -> "Instrument":
        data_cfg = data_cfg or yaml.safe_load((LAB / "configs" / "data.yaml").read_text())
        spec = data_cfg["instruments"][symbol]
        return cls(symbol, float(spec["tick_size"]), float(spec["point_value"]) * float(spec["tick_size"]))


@dataclass(frozen=True)
class SignalConfig:
    windows: tuple[str, ...] = ("ny_am",)
    fvg: FVGParams = field(default_factory=lambda: FVGParams(timeframe="5m"))
    bias: BiasParams = field(default_factory=BiasParams)                  # method 'none' = no gate
    sweep_required: bool = True
    sweep_universe: str = "session_refs"
    sweep_k: int = 3
    sweep_min_penetration_ticks: float = 1.0
    mss_required: bool = False
    mss: MSSParams = field(default_factory=MSSParams)
    displacement_atr_mult: float | None = 1.5                              # None = displacement not required
    displacement_atr_period: int = 14
    displacement_timeframe: str = "match_fvg"                              # or '1m' / '5m' / '15m'
    swing_1m: SwingParams = field(default_factory=lambda: SwingParams(timeframe="1m", n=3))
    swing_15m: SwingParams = field(default_factory=lambda: SwingParams(timeframe="15m", n=2))
    max_window_start_delay_min: int = 5                                    # eligibility: bars must exist at window start

    def __post_init__(self) -> None:
        if not set(self.windows) <= set(WINDOW_NAMES) or not self.windows:
            raise ValueError(f"windows must be a non-empty subset of {WINDOW_NAMES}")
        if self.sweep_universe not in UNIVERSES:
            raise ValueError(f"sweep_universe must be one of {UNIVERSES}")
        if self.mss_required and not self.sweep_required:
            raise ValueError("mss_required needs sweep_required")


@dataclass(frozen=True)
class ExecConfig:
    entry: str = "mid"                         # proximal | mid | distal edge of the FVG
    stop: str = "swing"                        # swing (beyond the swing that made the FVG) | distal | fixed
    stop_buffer_ticks: int = 1                 # distance beyond the swing / distal edge
    stop_points: float = 10.0                  # for stop == 'fixed'
    target: str = "r"                          # r | liquidity | time
    target_r: float = 2.0
    fallback_r: float = 2.0                    # liquidity target with nothing unswept
    target_time_minutes: int = 30              # for target == 'time'
    hard_exit: str = "window_end"              # window_end | rth_end | "HH:MM" (ET)
    max_trades_per_window: int | str = 1       # 1 or "unlimited" (hard cap 10)
    contracts: int = 1
    commission_rt: float = 4.0                 # dollars, round turn, per contract
    stop_slippage_ticks: int = 1               # market stops
    time_exit_slippage_ticks: int = 1          # market exits at the hard-exit / time-target
    limit_through_ticks: int = 1               # entry fills only when price trades this far THROUGH the level
    target_through_ticks: int = 1              # same rule for target limits (0 = a touch fills)

    def __post_init__(self) -> None:
        for name, allowed, val in (("entry", ENTRIES, self.entry), ("stop", STOPS, self.stop), ("target", TARGETS, self.target)):
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        m = self.max_trades_per_window
        if not (m == "unlimited" or (isinstance(m, int) and 1 <= m <= HARD_CAP)):
            raise ValueError(f"max_trades_per_window must be 1..{HARD_CAP} or 'unlimited'")

    @property
    def trade_cap(self) -> int:
        return HARD_CAP if self.max_trades_per_window == "unlimited" else int(self.max_trades_per_window)


@dataclass(frozen=True)
class StrategyConfig:
    name: str = "unnamed"
    signal: SignalConfig = field(default_factory=SignalConfig)
    exec: ExecConfig = field(default_factory=ExecConfig)

    def canonical(self, instrument: Instrument | None = None) -> dict[str, Any]:
        d = {"name": self.name, "signal": asdict(self.signal), "exec": asdict(self.exec)}
        if instrument is not None:
            d["instrument"] = asdict(instrument)
        return json.loads(json.dumps(d, sort_keys=True, default=list))

    def canonical_json(self, instrument: Instrument | None = None) -> str:
        return json.dumps(self.canonical(instrument), sort_keys=True, separators=(",", ":"))

    def hash(self, instrument: Instrument | None = None) -> str:
        return hashlib.sha1(self.canonical_json(instrument).encode()).hexdigest()[:12]


def _build(cls, d: dict[str, Any]):
    """dataclass from a (possibly partial) dict; nested params dataclasses are rebuilt recursively."""
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        if isinstance(v, dict):
            sub = {"fvg": FVGParams, "bias": BiasParams, "mss": MSSParams, "swing_1m": SwingParams,
                   "swing_15m": SwingParams, "swing": SwingParams, "signal": SignalConfig, "exec": ExecConfig}[f.name]
            v = _build(sub, v)
        elif f.name == "windows":
            v = tuple(v)
        kw[f.name] = v
    return cls(**kw)


def config_from_dict(d: dict[str, Any], name: str | None = None) -> StrategyConfig:
    d = dict(d)
    if name:
        d["name"] = name
    return _build(StrategyConfig, d)


def load_named_configs(path: Path | str = LAB / "configs" / "grid.json") -> dict[str, StrategyConfig]:
    raw = json.loads(Path(path).read_text())
    return {k: config_from_dict(v, k) for k, v in raw["configs"].items()}
