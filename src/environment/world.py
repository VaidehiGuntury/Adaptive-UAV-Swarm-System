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
            obstacle_manager = _build_obstacle_manager(
                dynamic_config=dynamic_config,
                world_width=env_config.width,
                world_height=env_config.height,
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


def _build_obstacle_manager(
    dynamic_config: DynamicEnvironmentConfig,
    world_width: float,
    world_height: float,
) -> ObstacleManager:
    """
    Construct and populate an ``ObstacleManager`` from ``DynamicEnvironmentConfig``.

    Obstacle types are distributed evenly across the three motion models.
    Positions are drawn uniformly at random inside the world bounds using
    the configured random seed for reproducibility.

    Distribution strategy
    ---------------------
    For *N* obstacles the split is:
    - floor(N / 3) linear obstacles
    - floor(N / 3) waypoint obstacles
    - remainder → random-walk obstacles

    This provides a balanced ``mixed`` scenario by default.  Named scenarios
    (``slow``, ``fast``, etc.) use the same distribution but different speeds
    taken from the per-model config.

    Parameters
    ----------
    dynamic_config:
        Parsed ``DynamicEnvironmentConfig`` with ``enabled == True``.
    world_width, world_height:
        World dimensions [m] used for obstacle placement and boundary
        reflection.

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

    rng = np.random.default_rng(dynamic_config.random_seed)
    n = dynamic_config.obstacle_count
    n_linear = n // 3
    n_waypoint = n // 3
    n_random = n - n_linear - n_waypoint

    bounds = (world_width, world_height)
    margin = 2.0  # minimum clearance from world edge [m]
    obstacle_radius = 0.5  # default dynamic obstacle radius [m]

    def _rand_pos() -> np.ndarray:
        """Sample a random position inside the world with margin."""
        return np.array(
            [
                rng.uniform(margin, world_width - margin),
                rng.uniform(margin, world_height - margin),
            ],
            dtype=np.float64,
        )

    def _rand_angle() -> float:
        return float(rng.uniform(0.0, 2.0 * np.pi))

    # --- Linear obstacles ---
    lin_speed = dynamic_config.linear.speed
    for i in range(n_linear):
        angle = _rand_angle()
        velocity = lin_speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
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
    wp_speed = dynamic_config.waypoint.speed
    for i in range(n_waypoint):
        # Generate a small patrol triangle as default waypoints
        centre = _rand_pos()
        patrol_radius = float(rng.uniform(5.0, 15.0))
        waypoints = [
            centre + patrol_radius * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
            for a in (0.0, 2.094, 4.189)  # 0°, 120°, 240°
        ]
        # Clip waypoints to world bounds
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
                speed=wp_speed,
            )
        )

    # --- Random walk obstacles ---
    rw_speed = dynamic_config.random_walk.speed
    rw_noise = dynamic_config.random_walk.turn_noise
    for i in range(n_random):
        # Each random-walk obstacle gets its own derived seed for independence
        obs_seed = int(rng.integers(0, 2**31))
        manager.add_obstacle(
            RandomWalkObstacle(
                obstacle_id=f"random_walk_{i}",
                position=_rand_pos(),
                radius=obstacle_radius,
                speed=rw_speed,
                heading=_rand_angle(),
                turn_noise=rw_noise,
                world_bounds=bounds,
                seed=obs_seed,
            )
        )

    return manager
