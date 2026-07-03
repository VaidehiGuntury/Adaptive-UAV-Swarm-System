"""YAML configuration loader for simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    """2D world dimensions and obstacle generation settings."""

    width: float
    height: float
    obstacle_count: int
    obstacle_min_radius: float
    obstacle_max_radius: float
    obstacle_seed: int | None


@dataclass(frozen=True)
class UAVConfig:
    """UAV motion and sensing limits (Paper 1 Assumption 2 / Sec. 2.3)."""

    max_speed: float
    max_angular_velocity: float
    sensing_range: float
    initial_spread_radius: float
    spawn_mode: str
    spawn_angular_noise: float


@dataclass(frozen=True)
class AggregationConfig:
    """
    Bio-inspired self-aggregation (BSA) parameters.

    Maps to Paper 1 Eqs. (6)–(10): utility U_a, costs J_C, J_V, J_L.
    """

    d_c: float
    d_0: float
    k_a: float
    turn_cost_weight: float
    trail_penalty: float
    cluster_penalty_weight: float
    turn_penalty_weight: float
    trail_penalty_weight: float
    candidates_per_frontier: int
    mission_region_radius: float
    replan_interval: float


@dataclass(frozen=True)
class IDEConfig:
    """Configuration for the IDE (Iterative Differential Evolution) allocator.

    Implements the parameters used in DEBS Paper 1 §4 (Algorithm 1 & 2).
    All values are supplied from the ``ide:`` block of ``simulation.yaml``;
    the block is entirely optional — when absent the allocator is disabled.

    Attributes
    ----------
    alpha:
        Controls the balance between exploration (high F) and exploitation
        (high CR). Used in Eq. 2 and Eq. 3 of DEBS §4. Must be in [0, 1].
    population_size:
        Number of candidate positions in the DE population (``N`` in the
        paper). Must be >= 4 so that ``x_best``, ``x_i``, ``x_r1``,
        ``x_r2`` can always be chosen as distinct individuals.
    fe_max:
        Maximum number of objective-function evaluations per pair
        interaction (``FE_max`` in Algorithm 1). Set to 0 to disable
        optimisation and return the LHS initialisation directly.
    t_att:
        Minimum elapsed-time (seconds) between two successive interactions
        of the *same* UAV pair (Algorithm 2, step 3). Prevents redundant
        renegotiation within a single BSA replan cycle.
    d_star:
        Target pairwise separation distance (metres). The objective Eq. 1
        reaches its minimum (0) when ``||p_i - p_j|| == d_star``.
        Revised from the paper's literal ``2 × sensing_range = 9 m`` to
        ``30 m`` based on fair-share area geometry for a 100 × 100 m world
        with 10 UAVs (see Stage 4 Q6 resolution).
    communication_range:
        Maximum distance (metres) within which two UAVs may interact.
        ``0.0`` means unlimited (all UAVs are always reachable), which
        matches the DEBS benchmark configuration.
    bounds_padding:
        Fractional margin applied to world bounds before LHS sampling,
        e.g. ``0.05`` keeps candidates 5 % inside each edge.
    seed:
        RNG seed for reproducibility. ``None`` means non-deterministic.
    """

    alpha: float = 0.5
    population_size: int = 20
    fe_max: int = 200
    t_att: float = 0.1
    d_star: float = 30.0
    communication_range: float = 0.0
    bounds_padding: float = 0.05
    seed: Optional[int] = 42

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError(
                "IDEConfig: population_size must be >= 4 "
                "(mutation requires x_best, x_i, x_r1, x_r2 "
                "as distinct individuals)"
            )
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("IDEConfig: alpha must be in [0, 1]")
        if self.fe_max < 0:
            raise ValueError("IDEConfig: fe_max must be >= 0")
        if self.d_star <= 0.0:
            raise ValueError("IDEConfig: d_star must be > 0")


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level simulation settings."""

    environment: EnvironmentConfig
    uav: UAVConfig
    aggregation: AggregationConfig
    num_uavs: int
    dt: float
    duration: float
    spawn_center_x: float
    spawn_center_y: float
    animation_interval_ms: int
    # Optional IDE allocator configuration (DEBS §4).
    # ``None`` disables the allocator; existing simulations without an
    # ``ide:`` YAML block remain fully valid.
    ide: Optional[IDEConfig] = None

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """Return world bounds in the format expected by ``IDEAllocator``.

        Returns ``(min_x, max_x, min_y, max_y)`` where the origin is
        ``(0, 0)`` and the arena spans ``environment.width`` ×
        ``environment.height`` metres.
        """
        return (0.0, self.environment.width, 0.0, self.environment.height)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate simulation configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    env = _require(raw, "environment")
    uav = _require(raw, "uav")
    agg = _require(raw, "aggregation")
    sim = _require(raw, "simulation")

    # -- Parse optional IDE block (DEBS §4) ---------------------------------
    ide_config: Optional[IDEConfig] = None
    if "ide" in raw:
        ide = raw["ide"]
        ide_config = IDEConfig(
            alpha=float(ide.get("alpha", 0.5)),
            population_size=int(ide.get("population_size", 20)),
            fe_max=int(ide.get("fe_max", 200)),
            t_att=float(ide.get("t_att", 0.1)),
            d_star=float(ide.get("d_star", 30.0)),
            communication_range=float(ide.get("communication_range", 0.0)),
            bounds_padding=float(ide.get("bounds_padding", 0.05)),
            seed=ide.get("seed", 42),  # None is valid (non-deterministic)
        )
    # -------------------------------------------------------------------------

    return SimulationConfig(
        environment=EnvironmentConfig(
            width=float(_require(env, "width")),
            height=float(_require(env, "height")),
            obstacle_count=int(_require(env, "obstacle_count")),
            obstacle_min_radius=float(_require(env, "obstacle_min_radius")),
            obstacle_max_radius=float(_require(env, "obstacle_max_radius")),
            obstacle_seed=env.get("obstacle_seed"),
        ),
        uav=UAVConfig(
            max_speed=float(_require(uav, "max_speed")),
            max_angular_velocity=float(_require(uav, "max_angular_velocity")),
            sensing_range=float(_require(uav, "sensing_range")),
            initial_spread_radius=float(_require(uav, "initial_spread_radius")),
            spawn_mode=str(uav.get("spawn_mode", "ring")),
            spawn_angular_noise=float(uav.get("spawn_angular_noise", 0.15)),
        ),
        aggregation=AggregationConfig(
            d_c=float(_require(agg, "d_c")),
            d_0=float(_require(agg, "d_0")),
            k_a=float(_require(agg, "k_a")),
            turn_cost_weight=float(_require(agg, "turn_cost_weight")),
            trail_penalty=float(_require(agg, "trail_penalty")),
            cluster_penalty_weight=float(_require(agg, "cluster_penalty_weight")),
            turn_penalty_weight=float(_require(agg, "turn_penalty_weight")),
            trail_penalty_weight=float(_require(agg, "trail_penalty_weight")),
            candidates_per_frontier=int(_require(agg, "candidates_per_frontier")),
            mission_region_radius=float(_require(agg, "mission_region_radius")),
            replan_interval=float(_require(agg, "replan_interval")),
        ),
        num_uavs=int(_require(sim, "num_uavs")),
        dt=float(_require(sim, "dt")),
        duration=float(_require(sim, "duration")),
        spawn_center_x=float(_require(sim, "spawn_center_x")),
        spawn_center_y=float(_require(sim, "spawn_center_y")),
        animation_interval_ms=int(_require(sim, "animation_interval_ms")),
        ide=ide_config,
    )
