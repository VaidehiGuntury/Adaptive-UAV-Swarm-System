"""
Configurable 2D simulation world.

Bundles spatial bounds, obstacles, and the exploration map used by BSA
viewpoint selection (Paper 1 Sec. 5).
"""

from __future__ import annotations

import numpy as np

from src.config.loader import EnvironmentConfig, UAVConfig
from src.environment.map import ExplorationMap
from src.environment.obstacles import ObstacleField, generate_obstacles


class World:
    """Top-level environment container."""

    def __init__(
        self,
        width: float,
        height: float,
        obstacles: ObstacleField,
        map_resolution: float = 0.5,
    ) -> None:
        self.width = width
        self.height = height
        self.obstacles = obstacles
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
    ) -> World:
        """Build world from YAML environment and UAV settings."""
        resolution = map_resolution if map_resolution is not None else uav_config.sensing_range / 3.0
        obstacles = generate_obstacles(
            count=env_config.obstacle_count,
            width=env_config.width,
            height=env_config.height,
            min_radius=env_config.obstacle_min_radius,
            max_radius=env_config.obstacle_max_radius,
            seed=env_config.obstacle_seed,
        )
        return cls(
            width=env_config.width,
            height=env_config.height,
            obstacles=obstacles,
            map_resolution=resolution,
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
