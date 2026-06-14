"""
Geometric slot assignment utilities for Paper 2 formations.

Assigns agents to formation slots without modifying agent motion.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_state import FormationState
from src.algorithms.formation.formation_types import (
    DEFAULT_FORMATION_SPACING,
    DEFAULT_FORMATION_TYPE,
    FormationType,
    get_formation_definition,
)

Vector2 = NDArray[np.float64]


def assign_formation_slots(
    agent_ids: list[int] | tuple[int, ...],
    leader_id: int,
    formation_type: FormationType = DEFAULT_FORMATION_TYPE,
    spacing: float = DEFAULT_FORMATION_SPACING,
    leader_heading: float = 0.0,
) -> FormationState:
    """
    Assign ``agent_ids`` to formation slots for ``formation_type``.

    Leader always occupies slot 0. Remaining agents are sorted by ``agent_id``
    and mapped to slots 1 … N-1. No agent positions are modified.
    """
    unique_ids = sorted(set(agent_ids))
    if leader_id not in unique_ids:
        raise ValueError(f"leader_id {leader_id} must appear in agent_ids")

    followers = [aid for aid in unique_ids if aid != leader_id]
    ordered = [leader_id, *followers]
    n = len(ordered)

    definition = get_formation_definition(formation_type)
    offsets = definition.slot_offsets(n, spacing)

    slot_offsets = {idx: offsets[idx].copy() for idx in range(n)}
    slot_assignments = {agent_id: idx for idx, agent_id in enumerate(ordered)}
    desired_offsets = {
        agent_id: slot_offsets[slot_assignments[agent_id]].copy()
        for agent_id in ordered
    }

    return FormationState(
        leader_id=leader_id,
        formation_type=formation_type,
        leader_heading=leader_heading,
        slot_assignments=slot_assignments,
        slot_offsets=slot_offsets,
        desired_offsets=desired_offsets,
        active_members=tuple(ordered),
    )
