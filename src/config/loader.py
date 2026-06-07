"""YAML configuration loader for simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    )
