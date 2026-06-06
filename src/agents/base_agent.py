"""
Abstract agent interface for the swarm simulator.

Paper 2 formation agents and Paper 3 search agents can extend this base
without changing the simulation engine contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseAgent(ABC):
    """Minimal agent contract used by the simulation engine."""

    agent_id: int
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]

    @abstractmethod
    def update(self, dt: float) -> None:
        """Advance agent state by one timestep."""

    @abstractmethod
    def move(self, dt: float) -> None:
        """Apply motion toward the current target."""

    @abstractmethod
    def compute_distance(self, other: BaseAgent) -> float:
        """Euclidean distance to another agent."""

    @abstractmethod
    def set_target(self, target: NDArray[np.float64]) -> None:
        """Assign a new navigation target."""
