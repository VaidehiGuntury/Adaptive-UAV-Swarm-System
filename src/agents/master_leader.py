"""
Master leader agent stub for Paper 2 global reference trajectory.

The master broadcasts state to UGV leaders; no controller logic in Week 1.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.base_agent import AgentRole, BaseAgent
from src.agents.uav import _wrap_angle


@dataclass
class MasterLeader(BaseAgent):
    """
    Global reference agent T_0 (Paper 2, index 0).

    Other agents track this leader through UGV intermediaries once
    formation control is implemented.
    """

    agent_id: int
    position: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    heading: float = 0.0
    altitude: float = 2.0
    assigned_target: NDArray[np.float64] | None = None
    max_speed: float = 0.8
    max_angular_velocity: float = 0.5

    @property
    def role(self) -> AgentRole:
        return AgentRole.MASTER

    def set_target(self, target: NDArray[np.float64]) -> None:
        self.assigned_target = target.astype(np.float64).copy()

    def compute_distance(self, other: BaseAgent) -> float:
        return float(np.linalg.norm(self.position - other.position))

    def update(self, dt: float) -> None:
        self.move(dt)

    def move(self, dt: float) -> None:
        """Move toward assigned target with bounded speed (stub kinematics)."""
        if self.assigned_target is None:
            return

        direction = self.assigned_target - self.position
        distance = float(np.linalg.norm(direction))
        if distance < 1e-6:
            self.velocity = np.zeros(2, dtype=np.float64)
            return

        desired_heading = float(np.arctan2(direction[1], direction[0]))
        heading_delta = _wrap_angle(desired_heading - self.heading)
        max_delta = self.max_angular_velocity * dt
        heading_delta = float(np.clip(heading_delta, -max_delta, max_delta))
        self.heading = _wrap_angle(self.heading + heading_delta)

        speed = min(self.max_speed, distance / dt)
        self.velocity = speed * np.array(
            [np.cos(self.heading), np.sin(self.heading)],
            dtype=np.float64,
        )
        step = self.velocity * dt
        if float(np.linalg.norm(step)) > distance:
            self.position = self.assigned_target.copy()
            self.velocity = np.zeros(2, dtype=np.float64)
        else:
            self.position = self.position + step
