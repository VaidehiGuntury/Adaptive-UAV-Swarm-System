"""
2D exploration map for frontier detection.

Paper 1 uses a voxel grid with frontier clustering (Sec. 4). This module
provides a lightweight grid abstraction so Paper 2/3 layers can extend it
without refactoring the simulation engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.environment.obstacles import ObstacleField

# Geographic key (grid col, grid row) for stable trail identity (Paper 1 Eq. 10).
RegionKey = tuple[int, int]


@dataclass(frozen=True)
class FrontierCluster:
    """Frontier cluster centroid used for viewpoint sampling (Paper 1 Sec. 5)."""

    centroid: NDArray[np.float64]
    is_trail: bool
    cluster_id: int
    region_key: RegionKey


class ExplorationMap:
    """
    Occupancy / exploration grid over the 2D world.

    Cells are marked explored when a UAV sensing disk overlaps them.
  """

    def __init__(
        self,
        width: float,
        height: float,
        resolution: float,
        obstacles: ObstacleField,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.obstacles = obstacles

        self._nx = max(1, int(np.ceil(width / resolution)))
        self._ny = max(1, int(np.ceil(height / resolution)))
        self._explored = np.zeros((self._ny, self._nx), dtype=bool)
        self._obstacle_mask = self._build_obstacle_mask()
        self._trail_region_keys: set[RegionKey] = set()
        self._next_cluster_id = 0

    def grid_shape(self) -> tuple[int, int]:
        """Grid dimensions as (rows, cols) matching ``explored_mask()`` indexing."""
        return self._ny, self._nx

    def explored_mask(self) -> NDArray[np.bool_]:
        """Read-only copy of the explored-cell boolean grid."""
        return self._explored.copy()

    def obstacle_mask(self) -> NDArray[np.bool_]:
        """Read-only copy of the obstacle-cell boolean grid."""
        return self._obstacle_mask.copy()

    def frontier_mask(self) -> NDArray[np.bool_]:
        """Read-only copy of frontier cells (explored cells bordering unexplored free space)."""
        return self._compute_frontier_mask().copy()

    def region_key_for_point(self, point: NDArray[np.float64]) -> RegionKey:
        """Stable geographic key for a world point (centroid grid cell)."""
        col, row = self.world_to_grid(point)
        return col, row

    def mark_cluster_as_trail(self, region_key: RegionKey) -> None:
        """Mark a frontier region as trail for J_L penalty (Paper 1 Eq. 10)."""
        self._trail_region_keys.add(region_key)

    def is_trail_region(self, region_key: RegionKey) -> bool:
        """Return True if the geographic region was marked as trail."""
        return region_key in self._trail_region_keys

    def _build_obstacle_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros((self._ny, self._nx), dtype=bool)
        xs = (np.arange(self._nx) + 0.5) * self.resolution
        ys = (np.arange(self._ny) + 0.5) * self.resolution
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                point = np.array([x, y], dtype=np.float64)
                if self.obstacles.is_collision(point):
                    mask[j, i] = True
        return mask

    def _compute_frontier_mask(self) -> NDArray[np.bool_]:
        """Frontier cells: explored free cells adjacent to unexplored free cells."""
        unexplored_free = (~self._explored) & (~self._obstacle_mask)
        frontier_mask = np.zeros_like(self._explored, dtype=bool)
        for j in range(self._ny):
            for i in range(self._nx):
                if not self._explored[j, i] or self._obstacle_mask[j, i]:
                    continue
                neighbors = [
                    (i + 1, j),
                    (i - 1, j),
                    (i, j + 1),
                    (i, j - 1),
                ]
                for ni, nj in neighbors:
                    if 0 <= ni < self._nx and 0 <= nj < self._ny:
                        if unexplored_free[nj, ni]:
                            frontier_mask[j, i] = True
                            break
        return frontier_mask

    def world_to_grid(self, point: NDArray[np.float64]) -> tuple[int, int]:
        """Convert world coordinates to grid indices (col, row), clamped."""
        col = int(np.clip(point[0] / self.resolution, 0, self._nx - 1))
        row = int(np.clip(point[1] / self.resolution, 0, self._ny - 1))
        return col, row

    def grid_to_world(self, i: int, j: int) -> NDArray[np.float64]:
        """Convert grid indices to world coordinates (cell center)."""
        return np.array(
            [(i + 0.5) * self.resolution, (j + 0.5) * self.resolution],
            dtype=np.float64,
        )

    def mark_explored(self, position: NDArray[np.float64], radius: float) -> None:
        """Mark cells within sensing radius as explored (Paper 1 depth camera model)."""
        i_center, j_center = self.world_to_grid(position)
        cells_radius = int(np.ceil(radius / self.resolution))
        for j in range(
            max(0, j_center - cells_radius),
            min(self._ny, j_center + cells_radius + 1),
        ):
            for i in range(
                max(0, i_center - cells_radius),
                min(self._nx, i_center + cells_radius + 1),
            ):
                cell_center = self.grid_to_world(i, j)
                if np.linalg.norm(cell_center - position) <= radius:
                    if not self._obstacle_mask[j, i]:
                        self._explored[j, i] = True

    def explored_fraction(self) -> float:
        """Fraction of free cells that have been explored."""
        free = ~self._obstacle_mask
        if not np.any(free):
            return 1.0
        return float(np.mean(self._explored[free]))

    def extract_frontier_clusters(self, min_cluster_size: int = 3) -> list[FrontierCluster]:
        """
        Extract frontier cells: explored cells adjacent to unexplored free cells.

        Clusters are connected components (4-neighborhood) for viewpoint sampling.
        """
        frontier_mask = self._compute_frontier_mask()
        visited = np.zeros_like(frontier_mask, dtype=bool)
        clusters: list[FrontierCluster] = []

        for j in range(self._ny):
            for i in range(self._nx):
                if not frontier_mask[j, i] or visited[j, i]:
                    continue
                stack = [(i, j)]
                component: list[tuple[int, int]] = []
                while stack:
                    ci, cj = stack.pop()
                    if visited[cj, ci] or not frontier_mask[cj, ci]:
                        continue
                    visited[cj, ci] = True
                    component.append((ci, cj))
                    for ni, nj in ((ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)):
                        if 0 <= ni < self._nx and 0 <= nj < self._ny:
                            if frontier_mask[nj, ni] and not visited[nj, ni]:
                                stack.append((ni, nj))

                if len(component) < min_cluster_size:
                    continue

                points = np.array(
                    [self.grid_to_world(ci, cj) for ci, cj in component],
                    dtype=np.float64,
                )
                centroid = np.mean(points, axis=0)
                region_key = self.region_key_for_point(centroid)
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
                clusters.append(
                    FrontierCluster(
                        centroid=centroid,
                        is_trail=region_key in self._trail_region_keys,
                        cluster_id=cluster_id,
                        region_key=region_key,
                    )
                )

        return clusters
