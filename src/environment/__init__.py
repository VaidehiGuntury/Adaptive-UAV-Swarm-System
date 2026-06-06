"""2D environment: world bounds, occupancy map, and obstacles."""

from src.environment.map import ExplorationMap
from src.environment.obstacles import CircularObstacle, ObstacleField, generate_obstacles
from src.environment.world import World

__all__ = [
    "CircularObstacle",
    "ExplorationMap",
    "ObstacleField",
    "World",
    "generate_obstacles",
]
