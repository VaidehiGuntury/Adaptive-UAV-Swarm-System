"""
Target and search zone geometry for Paper 3.

Static region definitions for mission setup and visualization.
Probability fields (α, β, γ) are deferred to the search algorithm week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CircularTargetRegion:
    """Circular search or prior-knowledge zone on the 2D plane."""

    region_id: str
    center: NDArray[np.float64]
    radius: float
    label: str = ""

    def contains(self, point: NDArray[np.float64]) -> bool:
        """Return True if ``point`` lies inside the region."""
        return float(np.linalg.norm(point - self.center)) <= self.radius


@dataclass(frozen=True)
class RectangularTargetRegion:
    """Axis-aligned rectangular search zone."""

    region_id: str
    origin: NDArray[np.float64]
    width: float
    height: float
    label: str = ""

    def contains(self, point: NDArray[np.float64]) -> bool:
        """Return True if ``point`` lies inside the region."""
        return bool(
            self.origin[0] <= point[0] <= self.origin[0] + self.width
            and self.origin[1] <= point[1] <= self.origin[1] + self.height
        )


TargetRegion = CircularTargetRegion | RectangularTargetRegion
