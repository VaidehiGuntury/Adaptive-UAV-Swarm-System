"""
Follower slot representation for Paper 2 leader-follower formations.

All world positions and error vectors are computed on demand; no control logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_state import FormationState
from src.algorithms.formation.leader_agent import LeaderAgent

Vector2 = NDArray[np.float64]


from src.algorithms.formation.formation_types import is_slot_occupied_by_distance


@dataclass(frozen=True)
class FollowerSlotState:
    """
    Per-follower slot snapshot (representational only).

    ``formation_error`` is world-frame ``p_agent - p_desired``; it is not
    consumed by any controller in this phase.
    """

    agent_id: int
    slot_index: int
    local_offset: Vector2
    world_slot_position: Vector2
    formation_error: Vector2
    error_magnitude: float
    is_occupied: bool


def compute_follower_slot_state(
    leader: LeaderAgent,
    formation_state: FormationState,
    agent_id: int,
    agent_position: Vector2,
) -> FollowerSlotState | None:
    """Compute follower slot snapshot for one agent."""
    slot_index = formation_state.slot_assignments.get(agent_id)
    if slot_index is None:
        return None

    local_offset = formation_state.desired_offsets.get(agent_id)
    if local_offset is None:
        local_offset = formation_state.slot_offsets.get(slot_index)
    if local_offset is None:
        return None

    local = np.asarray(local_offset, dtype=np.float64)
    world_slot = leader.to_world(local)
    agent_pos = np.asarray(agent_position, dtype=np.float64)
    error = agent_pos - world_slot
    magnitude = float(np.linalg.norm(error))
    occupied = is_slot_occupied_by_distance(magnitude, formation_state.slot_tolerance)

    return FollowerSlotState(
        agent_id=agent_id,
        slot_index=slot_index,
        local_offset=local.copy(),
        world_slot_position=world_slot,
        formation_error=error,
        error_magnitude=magnitude,
        is_occupied=occupied,
    )


def compute_follower_slot_states(
    leader: LeaderAgent,
    formation_state: FormationState,
    positions: dict[int, Vector2],
) -> tuple[FollowerSlotState, ...]:
    """Compute slot snapshots for all non-leader active members."""
    followers: list[FollowerSlotState] = []
    for agent_id in formation_state.active_members:
        if agent_id == formation_state.leader_id:
            continue
        position = positions.get(agent_id)
        if position is None:
            continue
        state = compute_follower_slot_state(
            leader,
            formation_state,
            agent_id,
            position,
        )
        if state is not None:
            followers.append(state)
    return tuple(followers)
