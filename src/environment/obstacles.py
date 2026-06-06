"""
Static obstacle representation and procedural generation.

Paper 1 uses cluttered forest scenes; this module provides a reusable 2D
obstacle model for the simulator foundation (full voxel mapping is future work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CircularObstacle:
    """Disk obstacle in the horizontal plane."""

    center: NDArray[np.float64]
    radius: float

    def contains(self, point: NDArray[np.float64]) -> bool:
        """Return True if point lies inside the obstacle."""
        return float(np.linalg.norm(point - self.center)) <= self.radius

    def distance_to(self, point: NDArray[np.float64]) -> float:
        """Signed distance: negative inside, positive outside."""
        return float(np.linalg.norm(point - self.center) - self.radius)


class ObstacleField:
    """Collection of obstacles with vectorized collision queries."""

    def __init__(self, obstacles: Sequence[CircularObstacle] | None = None) -> None:
        self._obstacles: list[CircularObstacle] = list(obstacles or [])

    @property
    def obstacles(self) -> tuple[CircularObstacle, ...]:
        return tuple(self._obstacles)

    def is_collision(self, point: NDArray[np.float64], margin: float = 0.0) -> bool:
        """Check whether a point collides with any obstacle."""
        for obstacle in self._obstacles:
            if obstacle.distance_to(point) <= margin:
                return True
        return False

    def nearest_free_point(
        self,
        point: NDArray[np.float64],
        world_width: float,
        world_height: float,
        margin: float = 0.2,
    ) -> NDArray[np.float64]:
        """Project point to nearest collision-free location (simple repulsion)."""
        adjusted = point.astype(np.float64).copy()
        for _ in range(8):
            if not self.is_collision(adjusted, margin=margin):
                break
            for obstacle in self._obstacles:
                delta = adjusted - obstacle.center
                dist = float(np.linalg.norm(delta))
                if dist < obstacle.radius + margin:
                    direction = delta / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    adjusted = obstacle.center + direction * (obstacle.radius + margin)
            adjusted[0] = np.clip(adjusted[0], margin, world_width - margin)
            adjusted[1] = np.clip(adjusted[1], margin, world_height - margin)
        return adjusted


def generate_obstacles(
    count: int,
    width: float,
    height: float,
    min_radius: float,
    max_radius: float,
    seed: int | None = None,
    margin: float = 2.0,
    max_attempts: int = 500,
) -> ObstacleField:
    """
    Place non-overlapping circular obstacles inside the world bounds.

    Parameters
    ----------
    count:
        Number of obstacles to place.
    width, height:
        World dimensions [m].
    min_radius, max_radius:
        Obstacle radius range [m].
    seed:
        RNG seed for reproducibility.
  """
    rng = np.random.default_rng(seed)
    placed: list[CircularObstacle] = []

    for _ in range(count):
        for _attempt in range(max_attempts):
            radius = float(rng.uniform(min_radius, max_radius))
            center = np.array(
                [
                    rng.uniform(margin + radius, width - margin - radius),
                    rng.uniform(margin + radius, height - margin - radius),
                ],
                dtype=np.float64,
            )
            candidate = CircularObstacle(center=center, radius=radius)
            overlap = any(
                np.linalg.norm(candidate.center - other.center)
                < candidate.radius + other.radius + 0.5
                for other in placed
            )
            if not overlap:
                placed.append(candidate)
                break

    return ObstacleField(placed)
