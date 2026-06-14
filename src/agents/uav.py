"""
Quadrotor UAV agent for Paper 1 decentralized exploration.

State fields support BSA viewpoint selection (Sec. 5) and future task
allocation (Sec. 4) without implementing IDE in this iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.agents.base_agent import AgentRole, BaseAgent

SpawnMode = Literal["ring", "legacy"]


@dataclass(frozen=True)
class UAVRegion:
    """
    Circular mission region (Paper 1 task allocation model).

    Each UAV owns a disk centred at its allocated target p̃_i* with radius
    equal to maximum detection range.
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
    - allocated target p̃_i* (assigned_region.center)
    - selected viewpoint vp_c (assigned_target — navigation execution)
    """

    agent_id: int
    position: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    heading: float = 0.0
    altitude: float = 2.0
    assigned_region: UAVRegion | None = None
    assigned_target: NDArray[np.float64] | None = None
    max_speed: float = 1.5
    max_angular_velocity: float = 0.9

    @property
    def role(self) -> AgentRole:
        return AgentRole.UAV

    def set_target(self, target: NDArray[np.float64]) -> None:
        """Assign BSA navigation viewpoint vp_c (Paper 1 Sec. 5 execution)."""
        self.assigned_target = target.astype(np.float64).copy()

    def set_region(self, center: NDArray[np.float64], radius: float) -> None:
        """
        Assign circular mission region p̃_i* (Paper 1 task allocation model).

        Does not update the navigation viewpoint; IDE / spawn sets this once
        until explicit reassignment is implemented.
        """
        self.assigned_region = UAVRegion(
            center=center.astype(np.float64).copy(),
            radius=radius,
        )

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

    def apply_velocity_command(self, velocity_command: NDArray[np.float64], dt: float) -> None:
        """Apply a direct velocity command with bounded speed (formation control)."""
        velocity = np.asarray(velocity_command, dtype=np.float64)
        norm = float(np.linalg.norm(velocity))
        if norm < 1e-9:
            self.velocity = np.zeros(2, dtype=np.float64)
            return
        if norm > self.max_speed:
            velocity = velocity / norm * self.max_speed
        self.velocity = velocity
        self.heading = float(np.arctan2(velocity[1], velocity[0]))
        self.position = self.position + velocity * dt


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
    spawn_mode: SpawnMode = "ring",
    spawn_angular_noise: float = 0.15,
) -> list[UAV]:
    """
    Initialize UAV fleet for Paper 1 experiments.

    ``ring`` mode (default): evenly spaced on a circle with small angular noise;
    p̃_i* is initialised at the spawn position.

    ``legacy`` mode: pre-correction spawn (tight cluster + outward allocation).

    Full IDE pairwise allocation (Paper 1 Sec. 4) is a future integration point.
    """
    rng = np.random.default_rng(seed)
    agents: list[UAV] = []

    for agent_id in range(count):
        base_angle = 2.0 * np.pi * agent_id / count

        if spawn_mode == "legacy":
            angle = base_angle
            offset = spread_radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            position = center + offset + rng.normal(scale=0.3, size=2)
            allocation = center + 2.0 * spread_radius * np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float64,
            )
        else:
            angle = base_angle + float(rng.uniform(-spawn_angular_noise, spawn_angular_noise))
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            position = center + spread_radius * direction
            allocation = position.copy()

        uav = UAV(
            agent_id=agent_id,
            position=position.astype(np.float64),
            max_speed=max_speed,
            max_angular_velocity=max_angular_velocity,
        )
        uav.set_region(allocation, mission_radius)
        uav.set_target(position.astype(np.float64))
        agents.append(uav)

    return agents
