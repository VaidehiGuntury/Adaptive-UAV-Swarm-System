"""
Immutable formation specifications for Paper 2 (Paper 2 Eqs. 27–29).

``FormationSpec`` is the serializable desired-shape contract for renderers,
planners, and future network synchronization. Control logic lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_state import FormationState
from src.algorithms.formation.formation_types import (
    DEFAULT_FORMATION_SPACING,
    FormationType,
    get_formation_definition,
    transform_to_world,
)

Vector2 = NDArray[np.float64]


@dataclass(frozen=True)
class FormationSpec:
    """
    Desired group formation for visualization and future controllers.

    All offsets are leader-local. World positions use
    ``leader_pos + R(leader_heading) @ local_offset``.
    """

    group_id: int
    leader_id: int
    formation_type: FormationType
    member_ids: tuple[int, ...]
    leader_heading: float = 0.0
    spacing: float = DEFAULT_FORMATION_SPACING
    slot_assignments: dict[int, int] = field(default_factory=dict)
    slot_offsets: dict[int, Vector2] = field(default_factory=dict)
    desired_offsets: dict[int, Vector2] = field(default_factory=dict)
    edges: tuple[tuple[int, int], ...] = ()

    @classmethod
    def from_state(cls, group_id: int, state: FormationState, spacing: float = DEFAULT_FORMATION_SPACING) -> FormationSpec:
        """Build an immutable spec from a runtime :class:`FormationState`."""
        n = len(state.active_members)
        definition = get_formation_definition(state.formation_type)
        return cls(
            group_id=group_id,
            leader_id=state.leader_id,
            formation_type=state.formation_type,
            member_ids=state.active_members,
            leader_heading=state.leader_heading,
            spacing=spacing,
            slot_assignments=dict(state.slot_assignments),
            slot_offsets={k: v.copy() for k, v in state.slot_offsets.items()},
            desired_offsets={k: v.copy() for k, v in state.desired_offsets.items()},
            edges=definition.edges(n),
        )

    def desired_local_position(self, member_id: int) -> Vector2 | None:
        """Leader-local desired offset for ``member_id``."""
        if member_id in self.desired_offsets:
            return self.desired_offsets[member_id]
        slot_idx = self.slot_assignments.get(member_id)
        if slot_idx is None:
            return None
        return self.slot_offsets.get(slot_idx)

    def desired_world_position(
        self,
        leader_position: Vector2,
        member_id: int,
    ) -> Vector2 | None:
        """Compute desired world position for a group member."""
        local = self.desired_local_position(member_id)
        if local is None:
            return None
        return transform_to_world(local, leader_position, self.leader_heading)

    def slot_world_position(
        self,
        slot_index: int,
        leader_position: Vector2,
    ) -> Vector2 | None:
        """World position of a formation slot."""
        local = self.slot_offsets.get(slot_index)
        if local is None:
            return None
        return transform_to_world(local, leader_position, self.leader_heading)

    def edge_world_segments(
        self,
        leader_position: Vector2,
    ) -> list[tuple[Vector2, Vector2]]:
        """World-frame line segments for formation edges."""
        segments: list[tuple[Vector2, Vector2]] = []
        for a, b in self.edges:
            pa = self.slot_world_position(a, leader_position)
            pb = self.slot_world_position(b, leader_position)
            if pa is not None and pb is not None:
                segments.append((pa, pb))
        return segments
