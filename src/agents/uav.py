"""
Quadrotor UAV agent for Paper 1 decentralized exploration.

State fields support BSA viewpoint selection (Sec. 5) and future task
allocation (Sec. 4) without implementing IDE in this iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.base_agent import BaseAgent


@dataclass(frozen=True)
class UAVRegion:
    """
    Circular mission region (Paper 1 task allocation model).

    Each UAV owns a disk centered at its allocated target with radius equal
    to maximum detection range.
    """

    center: NDArray[np.float64]
    radius: float


@dataclass
class UAV(BaseAgent):
    """
    Homogeneous UAV with planar kinematics.

    Paper 1 notation:
    - position p_i
    - velocity v_i
    - allocated target p̃_i* (assigned_target)
    - mission region A_i (assigned_region)
    """

    agent_id: int
    position: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    heading: float = 0.0
    assigned_region: UAVRegion | None = None
    assigned_target: NDArray[np.float64] | None = None
    max_speed: float = 1.5
    max_angular_velocity: float = 0.9

    def set_target(self, target: NDArray[np.float64]) -> None:
        """Assign navigation target p̃_i* for BSA / execution."""
        self.assigned_target = target.astype(np.float64).copy()

    def set_region(self, center: NDArray[np.float64], radius: float) -> None:
        """Assign circular mission region centered at allocated target."""
        self.assigned_region = UAVRegion(
            center=center.astype(np.float64).copy(),
            radius=radius,
        )
        self.assigned_target = center.astype(np.float64).copy()

    def compute_distance(self, other: BaseAgent) -> float:
        """Euclidean distance ||p_i - p_j||_2."""
        return float(np.linalg.norm(self.position - other.position))

    def update(self, dt: float) -> None:
        """Update kinematics for one simulation step."""
        self.move(dt)

    def move(self, dt: float) -> None:
        """
        Move toward assigned target with bounded speed and heading rate.

        Implements simplified execution of Paper 1 trajectories (Sec. 6 is
        deferred; this uses direct velocity steering toward the BSA target).
        """
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


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def spawn_uavs(
    count: int,
    center: NDArray[np.float64],
    spread_radius: float,
    mission_radius: float,
    max_speed: float,
    max_angular_velocity: float,
    seed: int | None = None,
) -> list[UAV]:
    """
    Initialize UAV fleet near world center (Paper 1 experimental setup).

    Assigned targets are distributed on a ring for initial decentralization.
    Full IDE pairwise allocation (Paper 1 Sec. 4) is a future integration point.
    """
    rng = np.random.default_rng(seed)
    agents: list[UAV] = []
    for agent_id in range(count):
        angle = 2.0 * np.pi * agent_id / count
        offset = spread_radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        position = center + offset + rng.normal(scale=0.3, size=2)
        initial_target = center + 2.0 * spread_radius * np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float64,
        )
        uav = UAV(
            agent_id=agent_id,
            position=position.astype(np.float64),
            max_speed=max_speed,
            max_angular_velocity=max_angular_velocity,
        )
        uav.set_region(initial_target, mission_radius)
        agents.append(uav)
    return agents
