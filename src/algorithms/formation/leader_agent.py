"""
Leader reference frame for Paper 2 leader-follower formations.

Composition-based abstraction: any agent pose can supply a ``LeaderAgent``
without inheriting from UAV/UGV types. No formation-control logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_types import (
    Vector2,
    rotation_matrix,
    transform_to_local,
    transform_to_world,
)


@dataclass(frozen=True)
class LeaderAgent:
    """
    Formation reference leader (Paper 2 local frame origin).

    ``formation_target`` is an optional global formation motion goal.
    It is intentionally decoupled from Paper 1 ``assigned_target``.
    """

    agent_id: int
    position: Vector2
    heading: float
    formation_target: Vector2 | None = None

    @property
    def forward_direction(self) -> Vector2:
        """Unit vector along leader +x (forward) axis."""
        return np.array(
            [np.cos(self.heading), np.sin(self.heading)],
            dtype=np.float64,
        )

    @property
    def left_direction(self) -> Vector2:
        """Unit vector along leader +y (left) axis."""
        return np.array(
            [-np.sin(self.heading), np.cos(self.heading)],
            dtype=np.float64,
        )

    def rotation_matrix(self) -> NDArray[np.float64]:
        """Return R(theta) mapping leader-local offsets to world axes."""
        return rotation_matrix(self.heading)

    def to_world(self, local: Vector2) -> Vector2:
        """Map leader-local offset to world coordinates."""
        return transform_to_world(local, self.position, self.heading)

    def to_local(self, world: Vector2) -> Vector2:
        """Map world coordinates into the leader-local frame."""
        return transform_to_local(world, self.position, self.heading)


def leader_agent_from_pose(
    agent_id: int,
    position: Vector2,
    heading: float,
    formation_target: Vector2 | None = None,
) -> LeaderAgent:
    """Build a leader frame from pose data (composition, not inheritance)."""
    return LeaderAgent(
        agent_id=agent_id,
        position=np.asarray(position, dtype=np.float64).copy(),
        heading=float(heading),
        formation_target=(
            None
            if formation_target is None
            else np.asarray(formation_target, dtype=np.float64).copy()
        ),
    )
