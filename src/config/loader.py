"""YAML configuration loader for simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from src.config.search_config import (
    AssignmentConfig,
    DetectionConfig,
    MissionConfig,
    PriorityConfig,
    SearchBehaviourConfig,
    SearchConfig,
    TargetSpawnConfig,
    TrackingConfig,
)


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

    Maps to Paper 1 Eqs. (6)-(10): utility U_a, costs J_C, J_V, J_L.
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
class LinearMotionConfig:
    """Parameters for the constant-velocity obstacle motion model."""

    speed: float


@dataclass(frozen=True)
class WaypointMotionConfig:
    """Parameters for the waypoint-following obstacle motion model."""

    speed: float


@dataclass(frozen=True)
class RandomWalkMotionConfig:
    """Parameters for the random-walk obstacle motion model."""

    speed: float
    turn_noise: float


@dataclass(frozen=True)
class DynamicEnvironmentConfig:
    """
    Configuration for the Dynamic Environment Extension.

    When enabled is False the simulator behaves exactly like the
    original repository — no ObstacleManager is created, no dynamic
    obstacles are spawned, and no dynamic metrics are collected.
    """

    enabled: bool
    scenario: str
    obstacle_count: int
    collision_radius: float
    safety_margin: float
    random_seed: int | None
    linear: LinearMotionConfig
    waypoint: WaypointMotionConfig
    random_walk: RandomWalkMotionConfig


@dataclass(frozen=True)
class IDEConfig:
    """Configuration for the IDE (Iterative Differential Evolution) allocator.

    Implements the parameters used in DEBS Paper 1 §4 (Algorithm 1 & 2).
    All values are supplied from the ide: block of simulation.yaml;
    the block is entirely optional — when absent the allocator is disabled.

    Attributes
    ----------
    alpha:
        Controls the balance between exploration (high F) and exploitation
        (high CR). Used in Eq. 2 and Eq. 3 of DEBS §4. Must be in [0, 1].
    population_size:
        Number of candidate positions in the DE population (N in the
        paper). Must be >= 4 so that x_best, x_i, x_r1, x_r2 can always
        be chosen as distinct individuals.
    fe_max:
        Maximum number of objective-function evaluations per pair
        interaction (FE_max in Algorithm 1). Set to 0 to disable
        optimisation and return the LHS initialisation directly.
    t_att:
        Minimum elapsed-time (seconds) between two successive interactions
        of the same UAV pair (Algorithm 2, step 3).
    d_star:
        Target pairwise separation distance (metres). The objective Eq. 1
        reaches its minimum (0) when ||p_i - p_j|| == d_star.
    communication_range:
        Maximum distance (metres) within which two UAVs may interact.
        0.0 means unlimited. Paper value: 50.0m.
    bounds_padding:
        Local LHS sampling step size Qs (metres). Paper value: 0.5m.
        LHS samples within [centre - Qs, centre + Qs] per dimension.
    seed:
        RNG seed for reproducibility. None means non-deterministic.
    """

    alpha: float = 0.5
    population_size: int = 20
    fe_max: int = 200
    t_att: float = 2.0
    d_star: float = 30.0
    communication_range: float = 50.0
    bounds_padding: float = 0.5
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
    # None disables the allocator; existing simulations without an
    # ide: YAML block remain fully valid.
    ide: Optional[IDEConfig] = None

    # Optional search & tracking extension (Person 2).
    # None disables search extension.
    search: Optional[SearchConfig] = None

    # Optional dynamic environment extension (Person 1).
    # None disables dynamic obstacles.
    dynamic_environment: Optional[DynamicEnvironmentConfig] = None

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """Return world bounds in the format expected by IDEAllocator.

        Returns (min_x, max_x, min_y, max_y) where the origin is
        (0, 0) and the arena spans environment.width x environment.height metres.
        """
        return (0.0, self.environment.width, 0.0, self.environment.height)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration key: {key}")
    return mapping[key]


def _load_dynamic_environment_config(
    raw_dyn: dict[str, Any],
) -> DynamicEnvironmentConfig:
    """Parse the dynamic_environment YAML section into a typed config object."""
    lin_raw: dict[str, Any] = raw_dyn.get("linear", {})
    wp_raw: dict[str, Any] = raw_dyn.get("waypoint", {})
    rw_raw: dict[str, Any] = raw_dyn.get("random_walk", {})

    return DynamicEnvironmentConfig(
        enabled=bool(raw_dyn.get("enabled", False)),
        scenario=str(raw_dyn.get("scenario", "mixed")),
        obstacle_count=int(raw_dyn.get("obstacle_count", 12)),
        collision_radius=float(raw_dyn.get("collision_radius", 0.35)),
        safety_margin=float(raw_dyn.get("safety_margin", 0.75)),
        random_seed=raw_dyn.get("random_seed"),
        linear=LinearMotionConfig(
            speed=float(lin_raw.get("speed", 0.5)),
        ),
        waypoint=WaypointMotionConfig(
            speed=float(wp_raw.get("speed", 1.0)),
        ),
        random_walk=RandomWalkMotionConfig(
            speed=float(rw_raw.get("speed", 0.8)),
            turn_noise=float(rw_raw.get("turn_noise", 0.15)),
        ),
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load and validate simulation configuration from a YAML file."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    env = _require(raw, "environment")
    uav = _require(raw, "uav")
    agg = _require(raw, "aggregation")
    sim = _require(raw, "simulation")

    # Parse optional IDE block (DEBS §4) --------------------------------
    ide_config: Optional[IDEConfig] = None
    if "ide" in raw:
        ide = raw["ide"]
        ide_config = IDEConfig(
            alpha=float(ide.get("alpha", 0.5)),
            population_size=int(ide.get("population_size", 20)),
            fe_max=int(ide.get("fe_max", 200)),
            t_att=float(ide.get("t_att", 2.0)),
            d_star=float(ide.get("d_star", 30.0)),
            communication_range=float(ide.get("communication_range", 50.0)),
            bounds_padding=float(ide.get("bounds_padding", 0.5)),
            seed=ide.get("seed", 42),
        )

    # Parse optional dynamic_environment block (Person 1) ----------------
    raw_dyn: dict[str, Any] | None = raw.get("dynamic_environment")
    dyn_config: Optional[DynamicEnvironmentConfig] = (
        _load_dynamic_environment_config(raw_dyn) if raw_dyn is not None else None
    )

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
        search=_load_search_config(raw.get("search")),
        dynamic_environment=dyn_config,
    )


def _load_search_config(raw_search: dict[str, Any] | None) -> SearchConfig | None:
    """Load optional search extension configuration block."""
    if raw_search is None:
        return None
    if not raw_search.get("enabled", True):
        return None

    def _sub(key: str) -> dict[str, Any]:
        return raw_search.get(key, {})

    ts = _sub("targets")
    det = _sub("detection")
    pri = _sub("priority")
    beh = _sub("behaviour")
    asgn = _sub("assignment")
    trk = _sub("tracking")
    mis = _sub("mission")

    return SearchConfig(
        enabled=bool(raw_search.get("enabled", True)),
        targets=TargetSpawnConfig(
            count_static=int(ts.get("count_static", 3)),
            count_dynamic=int(ts.get("count_dynamic", 3)),
            count_time_varying=int(ts.get("count_time_varying", 2)),
            spawn_margin=float(ts.get("spawn_margin", 5.0)),
            min_separation=float(ts.get("min_separation", 8.0)),
            dynamic_speed_min=float(ts.get("dynamic_speed_min", 0.3)),
            dynamic_speed_max=float(ts.get("dynamic_speed_max", 0.8)),
            seed=int(ts.get("seed", 123)),
        ),
        detection=DetectionConfig(
            detection_radius=float(det.get("detection_radius", 0.0)),
            base_confidence=float(det.get("base_confidence", 0.85)),
            confidence_decay_rate=float(det.get("confidence_decay_rate", 0.05)),
            confidence_merge_alpha=float(det.get("confidence_merge_alpha", 0.6)),
            dedup_time_window_s=float(det.get("dedup_time_window_s", 2.0)),
            min_detection_confidence=float(det.get("min_detection_confidence", 0.3)),
        ),
        priority=PriorityConfig(
            w_type=float(pri.get("w_type", 0.30)),
            w_confidence=float(pri.get("w_confidence", 0.20)),
            w_movement=float(pri.get("w_movement", 0.15)),
            w_distance=float(pri.get("w_distance", 0.15)),
            w_urgency=float(pri.get("w_urgency", 0.10)),
            w_freshness=float(pri.get("w_freshness", 0.10)),
            score_moving_human=float(pri.get("score_moving_human", 1.0)),
            score_moving_vehicle=float(pri.get("score_moving_vehicle", 0.8)),
            score_static_human=float(pri.get("score_static_human", 0.7)),
            score_static_object=float(pri.get("score_static_object", 0.4)),
        ),
        behaviour=SearchBehaviourConfig(
            direct_nav_staleness_s=float(beh.get("direct_nav_staleness_s", 5.0)),
            spiral_initial_radius=float(beh.get("spiral_initial_radius", 3.0)),
            spiral_step=float(beh.get("spiral_step", 2.5)),
            spiral_max_radius=float(beh.get("spiral_max_radius", 20.0)),
            expanding_step_m=float(beh.get("expanding_step_m", 4.0)),
            expanding_max_radius=float(beh.get("expanding_max_radius", 25.0)),
            replan_interval_s=float(beh.get("replan_interval_s", 1.5)),
            arrival_threshold_m=float(beh.get("arrival_threshold_m", 1.0)),
            loss_timeout_s=float(beh.get("loss_timeout_s", 8.0)),
            max_recovery_attempts=int(beh.get("max_recovery_attempts", 5)),
        ),
        assignment=AssignmentConfig(
            max_targets_per_uav=int(asgn.get("max_targets_per_uav", 2)),
            reassignment_on_fail=bool(asgn.get("reassignment_on_fail", True)),
            fail_timeout_s=float(asgn.get("fail_timeout_s", 15.0)),
            cost_distance_weight=float(asgn.get("cost_distance_weight", 0.6)),
            cost_priority_weight=float(asgn.get("cost_priority_weight", 0.4)),
        ),
        tracking=TrackingConfig(
            required_tracking_duration_s=float(
                trk.get("required_tracking_duration_s", 10.0)
            ),
            history_max_length=int(trk.get("history_max_length", 200)),
            reacquisition_search_radius=float(
                trk.get("reacquisition_search_radius", 12.0)
            ),
            visibility_sample_interval_s=float(
                trk.get("visibility_sample_interval_s", 1.0)
            ),
        ),
        mission=MissionConfig(
            exploration_completion_threshold=float(
                mis.get("exploration_completion_threshold", 0.85)
            ),
            search_timeout_s=float(mis.get("search_timeout_s", 300.0)),
            min_coverage_before_search=float(
                mis.get("min_coverage_before_search", 0.70)
            ),
        ),
    )