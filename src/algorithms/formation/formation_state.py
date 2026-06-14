"""
Runtime formation assignment containers for Paper 2 (no control logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_types import (
    DEFAULT_SLOT_TOLERANCE,
    FormationType,
    Vector2,
    is_slot_occupied_by_distance,
    transform_to_world,
)

Vector2Map = dict[int, NDArray[np.float64]]


def compute_formation_centroid(
    active_members: tuple[int, ...],
    positions: Vector2Map,
) -> Vector2:
    """Mean world position of active formation members (computed, never cached)."""
    if not active_members:
        return np.zeros(2, dtype=np.float64)
    pts = [positions[aid] for aid in active_members if aid in positions]
    if not pts:
        return np.zeros(2, dtype=np.float64)
    return np.mean(np.stack(pts, axis=0), axis=0)


@dataclass
class FormationState:
    """
    Mutable formation assignment snapshot.

    Slot-space maps support future controllers; ``desired_offsets`` mirrors
    agent-local offsets for convenience. Centroid is derived on demand via
    :func:`compute_formation_centroid`.
    """

    leader_id: int
    formation_type: FormationType
    leader_heading: float = 0.0
    slot_assignments: dict[int, int] = field(default_factory=dict)
    slot_offsets: dict[int, Vector2] = field(default_factory=dict)
    desired_offsets: dict[int, Vector2] = field(default_factory=dict)
    active_members: tuple[int, ...] = ()
    slot_tolerance: float = DEFAULT_SLOT_TOLERANCE

    def desired_world_position(
        self,
        agent_id: int,
        leader_position: Vector2,
    ) -> Vector2 | None:
        """World-frame desired position for ``agent_id`` using leader heading."""
        local = self.desired_offsets.get(agent_id)
        if local is None:
            slot_idx = self.slot_assignments.get(agent_id)
            if slot_idx is None:
                return None
            local = self.slot_offsets.get(slot_idx)
        if local is None:
            return None
        return transform_to_world(local, leader_position, self.leader_heading)

    def slot_world_position(
        self,
        slot_index: int,
        leader_position: Vector2,
    ) -> Vector2 | None:
        """World-frame position of a formation slot."""
        local = self.slot_offsets.get(slot_index)
        if local is None:
            return None
        return transform_to_world(local, leader_position, self.leader_heading)

    def is_slot_occupied(
        self,
        agent_id: int,
        agent_position: Vector2,
        leader_position: Vector2,
    ) -> bool:
        """True when ``agent_position`` is within ``slot_tolerance`` of its slot."""
        desired = self.desired_world_position(agent_id, leader_position)
        if desired is None:
            return False
        distance = float(np.linalg.norm(agent_position - desired))
        return is_slot_occupied_by_distance(distance, self.slot_tolerance)

    def centroid(self, positions: Vector2Map) -> Vector2:
        """Dynamic centroid from current member world positions."""
        return compute_formation_centroid(self.active_members, positions)
