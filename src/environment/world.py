"""
Configurable 2D simulation world.

Bundles spatial bounds, obstacles, and the exploration map used by BSA
viewpoint selection (Paper 1 Sec. 5). Optional subsystems support Papers 2–3.
"""

from __future__ import annotations

import numpy as np

from src.config.loader import DynamicEnvironmentConfig, EnvironmentConfig, UAVConfig
from src.environment.belief_map import BeliefMap
from src.environment.communication import CommunicationGraph
from src.environment.formation_spec import FormationSpec
from src.environment.map import ExplorationMap
from src.environment.obstacle_manager import ObstacleManager
from src.environment.obstacles import ObstacleField, generate_obstacles
from src.environment.target_region import TargetRegion

# ---------------------------------------------------------------------------
# Scenario speed-range tables (SDS §35)
# All values are multipliers of UAV max_speed (default 1.5 m/s).
# Each entry is (min_multiplier, max_multiplier) — a speed for each obstacle
# is drawn uniformly at random within this range using the seeded RNG.
#
# Scenario  | Description                     | Speed range relative to v_uav
# ----------|---------------------------------|------------------------------
# slow      | obstacles slower than UAV       | 0.27–0.47  (0.4–0.7 m/s @ 1.5)
# equal     | obstacles match UAV speed       | 0.93–1.07  (1.4–1.6 m/s @ 1.5)
# fast      | obstacles faster than UAV       | 1.20–1.53  (1.8–2.3 m/s @ 1.5)
#
# These multipliers are applied to the *actual* uav.max_speed so the
# behaviour scales correctly if the UAV speed is changed in the YAML.
# ---------------------------------------------------------------------------
_SLOW_RANGE = (0.267, 0.467)    # × v_uav → 0.4–0.7 m/s when v_uav = 1.5
_EQUAL_RANGE = (0.933, 1.067)   # × v_uav → 1.4–1.6 m/s when v_uav = 1.5
_FAST_RANGE = (1.200, 1.533)    # × v_uav → 1.8–2.3 m/s when v_uav = 1.5

# Supported scenario names and whether they produce dynamic obstacles
_DYNAMIC_SCENARIOS = frozenset({"slow", "equal_speed", "fast", "mixed", "custom"})
_STATIC_SCENARIOS = frozenset({"static"})


class World:
    """Top-level environment container."""

    def __init__(
        self,
        width: float,
        height: float,
        obstacles: ObstacleField,
        map_resolution: float = 0.5,
        communication_graph: CommunicationGraph | None = None,
        belief_map: BeliefMap | None = None,
        target_regions: list[TargetRegion] | None = None,
        formation_specs: list[FormationSpec] | None = None,
        obstacle_manager: ObstacleManager | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.communication_graph = communication_graph
        self.belief_map = belief_map
        self.target_regions: list[TargetRegion] = list(target_regions or [])
        self.formation_specs: list[FormationSpec] = list(formation_specs or [])
        # Dynamic obstacle manager (SDS §28).  None ⟹ static environment.
        self.obstacle_manager: ObstacleManager | None = obstacle_manager
        self.map = ExplorationMap(
            width=width,
            height=height,
            resolution=map_resolution,
            obstacles=obstacles,
        )

    @classmethod
    def from_config(
        cls,
        env_config: EnvironmentConfig,
        uav_config: UAVConfig,
        map_resolution: float | None = None,
        dynamic_config: DynamicEnvironmentConfig | None = None,
    ) -> World:
        """
        Build world from YAML environment and UAV settings.

        When *dynamic_config* is provided and ``enabled`` is True, an
        ``ObstacleManager`` is created and populated with dynamic obstacles
        according to the scenario configuration (SDS §26–28).

        Parameters
        ----------
        env_config:
            Static environment geometry (world size, static obstacles).
        uav_config:
            UAV parameters (used to derive map resolution).
        map_resolution:
            Override grid resolution [m/cell].  Defaults to
            ``sensing_range / 3``.
        dynamic_config:
            Dynamic environment parameters.  None ⟹ static world.
        """
        resolution = map_resolution if map_resolution is not None else uav_config.sensing_range / 3.0
        obstacles = generate_obstacles(
            count=env_config.obstacle_count,
            width=env_config.width,
            height=env_config.height,
            min_radius=env_config.obstacle_min_radius,
            max_radius=env_config.obstacle_max_radius,
            seed=env_config.obstacle_seed,
        )

        obstacle_manager: ObstacleManager | None = None
        if dynamic_config is not None and dynamic_config.enabled:
            # ``scenario: static`` is a named alias for "no dynamic obstacles"
            # even when ``enabled: true``.  This lets experimenters switch
            # scenarios without toggling the ``enabled`` flag.
            if dynamic_config.scenario not in _STATIC_SCENARIOS:
                obstacle_manager = _build_obstacle_manager(
                    dynamic_config=dynamic_config,
                    world_width=env_config.width,
                    world_height=env_config.height,
                    uav_max_speed=uav_config.max_speed,
                )

        return cls(
            width=env_config.width,
            height=env_config.height,
            obstacles=obstacles,
            map_resolution=resolution,
            obstacle_manager=obstacle_manager,
        )

    def clip_position(self, position: np.ndarray, margin: float = 0.5) -> np.ndarray:
        """Keep positions inside world bounds."""
        clipped = position.astype(np.float64).copy()
        clipped[0] = np.clip(clipped[0], margin, self.width - margin)
        clipped[1] = np.clip(clipped[1], margin, self.height - margin)
        return clipped

    def resolve_collisions(self, position: np.ndarray, margin: float = 0.3) -> np.ndarray:
        """Resolve obstacle collisions via projection."""
        return self.obstacles.nearest_free_point(
            position,
            world_width=self.width,
            world_height=self.height,
            margin=margin,
        )


# ---------------------------------------------------------------------------
# Private factory — builds and populates ObstacleManager from config
# ---------------------------------------------------------------------------


def _sample_speed(
    rng: np.random.Generator,
    speed_range: tuple[float, float],
    uav_max_speed: float,
) -> float:
    """
    Draw a single obstacle speed uniformly from a relative range.

    Parameters
    ----------
    rng:
        Seeded NumPy Generator — must be the same instance used throughout
        obstacle construction so all randomness is reproducible.
    speed_range:
        (min_multiplier, max_multiplier) applied to *uav_max_speed*.
    uav_max_speed:
        UAV maximum speed [m/s] used as the classification reference.

    Returns
    -------
    float
        Obstacle speed [m/s].
    """
    lo, hi = speed_range
    return float(rng.uniform(lo * uav_max_speed, hi * uav_max_speed))


def _build_obstacle_manager(
    dynamic_config: DynamicEnvironmentConfig,
    world_width: float,
    world_height: float,
    uav_max_speed: float,
) -> ObstacleManager:
    """
    Construct and populate an ``ObstacleManager`` from ``DynamicEnvironmentConfig``.

    Scenario behaviour
    ------------------
    The ``scenario`` field of *dynamic_config* controls how obstacle speeds
    are assigned.  All speed sampling uses the seeded ``rng`` so results are
    fully reproducible.

    ``static``
        No dynamic obstacles — caller (``World.from_config``) short-circuits
        before this function is reached.

    ``slow``
        Every obstacle receives a speed drawn uniformly from the slow range
        (0.267–0.467 × v_uav, equivalent to 0.4–0.7 m/s at v_uav = 1.5 m/s).
        All three motion model types are present.

    ``equal_speed``
        Every obstacle receives a speed drawn from the equal range
        (0.933–1.067 × v_uav, equivalent to 1.4–1.6 m/s at v_uav = 1.5 m/s).

    ``fast``
        Every obstacle receives a speed drawn from the fast range
        (1.200–1.533 × v_uav, equivalent to 1.8–2.3 m/s at v_uav = 1.5 m/s).

    ``mixed``
        The obstacle pool is divided into three equal thirds.  The first
        third receives slow speeds, the second equal speeds, the third fast
        speeds.  All three motion model types are distributed evenly across
        the full pool so every motion model appears in every speed category.

    ``custom`` (or any unrecognised value)
        Falls back to the explicit YAML speed values (``linear.speed``,
        ``waypoint.speed``, ``random_walk.speed``).  This is the legacy
        behaviour and preserves full backward compatibility for existing
        experiment configurations.

    Distribution of motion models
    ------------------------------
    For *N* total obstacles the split is always:
    - ``floor(N / 3)`` LinearObstacle instances
    - ``floor(N / 3)`` WaypointObstacle instances
    - ``N - 2 × floor(N / 3)`` RandomWalkObstacle instances

    For ``mixed`` the speed category is interleaved across the full pool
    so each model type appears in each speed category as evenly as possible.

    Parameters
    ----------
    dynamic_config:
        Parsed ``DynamicEnvironmentConfig`` with ``enabled == True`` and a
        scenario that is not ``static``.
    world_width, world_height:
        World dimensions [m].
    uav_max_speed:
        UAV maximum speed [m/s] used as the speed-classification reference.

    Returns
    -------
    ObstacleManager
        Populated manager ready to be owned by ``World``.
    """
    from src.environment.dynamic_obstacles import (
        LinearObstacle,
        RandomWalkObstacle,
        WaypointObstacle,
    )

    manager = ObstacleManager(
        collision_radius=dynamic_config.collision_radius,
        safety_margin=dynamic_config.safety_margin,
    )

    # Single seeded RNG for all random draws in this function — guarantees
    # full reproducibility from dynamic_environment.random_seed.
    rng = np.random.default_rng(dynamic_config.random_seed)

    scenario = dynamic_config.scenario
    n = dynamic_config.obstacle_count
    n_linear = n // 3
    n_waypoint = n // 3
    n_random = n - n_linear - n_waypoint

    bounds = (world_width, world_height)
    margin = 2.0          # minimum clearance from world edge [m]
    obstacle_radius = 0.5  # default dynamic obstacle radius [m]

    # ------------------------------------------------------------------
    # Speed assignment per scenario
    # ------------------------------------------------------------------

    def _speed_for_index(idx: int, total: int) -> float:
        """
        Return the speed [m/s] for obstacle at position *idx* out of *total*.

        ``mixed``: divide the pool into three equal thirds by index.
        Other scenarios: draw from the scenario range.
        ``custom``: falls through to per-model YAML value (handled below).
        """
        if scenario == "slow":
            return _sample_speed(rng, _SLOW_RANGE, uav_max_speed)
        if scenario == "equal_speed":
            return _sample_speed(rng, _EQUAL_RANGE, uav_max_speed)
        if scenario == "fast":
            return _sample_speed(rng, _FAST_RANGE, uav_max_speed)
        if scenario == "mixed":
            third = max(1, total // 3)
            if idx < third:
                return _sample_speed(rng, _SLOW_RANGE, uav_max_speed)
            if idx < 2 * third:
                return _sample_speed(rng, _EQUAL_RANGE, uav_max_speed)
            return _sample_speed(rng, _FAST_RANGE, uav_max_speed)
        # custom / unknown — caller provides speed from YAML config
        return float("nan")  # sentinel; replaced by per-model YAML value below

    def _rand_pos() -> np.ndarray:
        return np.array(
            [
                rng.uniform(margin, world_width - margin),
                rng.uniform(margin, world_height - margin),
            ],
            dtype=np.float64,
        )

    def _rand_angle() -> float:
        return float(rng.uniform(0.0, 2.0 * np.pi))

    # ------------------------------------------------------------------
    # Build obstacle pool — interleaved order for mixed speed distribution
    # ------------------------------------------------------------------
    # Obstacles are indexed globally (0 … n-1) so _speed_for_index can
    # distribute speed categories evenly across the full pool.

    global_idx = 0

    # --- Linear obstacles ---
    lin_yaml_speed = dynamic_config.linear.speed
    for i in range(n_linear):
        spd = _speed_for_index(global_idx, n)
        if np.isnan(spd):   # custom scenario
            spd = lin_yaml_speed
        global_idx += 1

        angle = _rand_angle()
        velocity = spd * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        manager.add_obstacle(
            LinearObstacle(
                obstacle_id=f"linear_{i}",
                position=_rand_pos(),
                velocity=velocity,
                radius=obstacle_radius,
                world_bounds=bounds,
            )
        )

    # --- Waypoint obstacles ---
    wp_yaml_speed = dynamic_config.waypoint.speed
    rw_noise = dynamic_config.random_walk.turn_noise
    for i in range(n_waypoint):
        spd = _speed_for_index(global_idx, n)
        if np.isnan(spd):   # custom scenario
            spd = wp_yaml_speed
        global_idx += 1

        centre = _rand_pos()
        patrol_radius = float(rng.uniform(5.0, 15.0))
        waypoints = [
            centre + patrol_radius * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
            for a in (0.0, 2.094, 4.189)  # 0°, 120°, 240°
        ]
        waypoints = [
            np.clip(wp, margin, [world_width - margin, world_height - margin])
            for wp in waypoints
        ]
        manager.add_obstacle(
            WaypointObstacle(
                obstacle_id=f"waypoint_{i}",
                position=_rand_pos(),
                radius=obstacle_radius,
                waypoints=waypoints,
                speed=spd,
            )
        )

    # --- Random walk obstacles ---
    rw_yaml_speed = dynamic_config.random_walk.speed
    for i in range(n_random):
        spd = _speed_for_index(global_idx, n)
        if np.isnan(spd):   # custom scenario
            spd = rw_yaml_speed
        global_idx += 1

        obs_seed = int(rng.integers(0, 2**31))
        manager.add_obstacle(
            RandomWalkObstacle(
                obstacle_id=f"random_walk_{i}",
                position=_rand_pos(),
                radius=obstacle_radius,
                speed=spd,
                heading=_rand_angle(),
                turn_noise=rw_noise,
                world_bounds=bounds,
                seed=obs_seed,
            )
        )

    return manager
