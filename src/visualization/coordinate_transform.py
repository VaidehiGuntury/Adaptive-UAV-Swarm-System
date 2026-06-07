"""
World-to-screen coordinate mapping for Pygame rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CoordinateTransform:
    """
    Affine map from world coordinates [m] to screen pixels.

    Preserves aspect ratio and applies a uniform margin around the world bounds.
    """

    world_width: float
    world_height: float
    screen_width: int
    screen_height: int
    margin_px: int = 40

    @property
    def scale(self) -> float:
        """Pixels per world meter."""
        usable_w = self.screen_width - 2 * self.margin_px
        usable_h = self.screen_height - 2 * self.margin_px
        if self.world_width <= 0.0 or self.world_height <= 0.0:
            return 1.0
        return min(usable_w / self.world_width, usable_h / self.world_height)

    def world_to_screen(self, point: NDArray[np.float64]) -> tuple[int, int]:
        """Convert a 2D world point to integer screen coordinates."""
        sx = self.margin_px + int(point[0] * self.scale)
        sy = self.screen_height - self.margin_px - int(point[1] * self.scale)
        return sx, sy

    def world_radius_to_pixels(self, radius: float) -> int:
        """Convert a world-space radius to pixel radius."""
        return max(1, int(radius * self.scale))

    def cell_screen_rect(self, col: int, row: int, resolution: float) -> tuple[int, int, int, int]:
        """
        Return pygame-style screen rect (x, y, width, height) for a grid cell.

        ``col`` / ``row`` are grid indices (i, j); y increases upward in world space.
        """
        x0 = col * resolution
        y0 = row * resolution
        x1 = x0 + resolution
        y1 = y0 + resolution
        sx0, sy0 = self.world_to_screen(np.array([x0, y1], dtype=np.float64))
        sx1, sy1 = self.world_to_screen(np.array([x1, y0], dtype=np.float64))
        left = min(sx0, sx1)
        top = min(sy0, sy1)
        width = max(1, abs(sx1 - sx0))
        height = max(1, abs(sy1 - sy0))
        return left, top, width, height
