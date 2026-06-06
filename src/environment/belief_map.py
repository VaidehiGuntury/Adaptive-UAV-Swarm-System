"""
Belief map grid container for Paper 3 target search.

Stores cell probabilities for visualization and future Bayesian updates
(Eqs. 3–6). No update logic in Week 1.5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BeliefMap:
    """
    2D belief grid over the search area Φ (Paper 3).

    Cell values represent target presence probability; ``normalized_array``
    ensures Σ b(h) = 1 over free cells.
    """

    def __init__(self, width: float, height: float, resolution: float) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self._nx = max(1, int(np.ceil(width / resolution)))
        self._ny = max(1, int(np.ceil(height / resolution)))
        self._beliefs = np.zeros((self._ny, self._nx), dtype=np.float32)

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Grid dimensions (rows, cols)."""
        return self._ny, self._nx

    def world_to_grid(self, point: NDArray[np.float64]) -> tuple[int, int]:
        """Convert world coordinates to grid indices (clamped)."""
        i = int(np.clip(point[0] / self.resolution, 0, self._nx - 1))
        j = int(np.clip(point[1] / self.resolution, 0, self._ny - 1))
        return i, j

    def grid_to_world(self, i: int, j: int) -> NDArray[np.float64]:
        """Convert grid indices to world coordinates (cell center)."""
        return np.array(
            [(i + 0.5) * self.resolution, (j + 0.5) * self.resolution],
            dtype=np.float64,
        )

    def set_belief(self, i: int, j: int, value: float) -> None:
        """Set belief probability at grid cell (i, j)."""
        if 0 <= i < self._nx and 0 <= j < self._ny:
            self._beliefs[j, i] = max(0.0, float(value))

    def get_belief(self, i: int, j: int) -> float:
        """Return belief probability at grid cell (i, j)."""
        if 0 <= i < self._nx and 0 <= j < self._ny:
            return float(self._beliefs[j, i])
        return 0.0

    def normalized_array(self) -> NDArray[np.float32]:
        """
        Return belief grid normalized to sum 1 (Paper 3 Eq. 1).

        Returns zeros if the grid is empty.
        """
        total = float(np.sum(self._beliefs))
        if total <= 0.0:
            return self._beliefs.copy()
        return (self._beliefs / total).astype(np.float32)

    def set_uniform(self) -> None:
        """Initialize uniform belief over all cells."""
        self._beliefs.fill(1.0)
