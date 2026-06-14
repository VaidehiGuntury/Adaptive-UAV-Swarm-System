"""
Leader-follower group snapshot assembly for Paper 2.

Builds synchronized, read-only views for visualization, replay, and future
controller debugging. No simulation-engine integration in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.follower_state import (
    FollowerSlotState,
    compute_follower_slot_states,
)
from src.algorithms.formation.formation_metrics import (
    FormationMetricsSnapshot,
    compute_formation_metrics,
)
from src.algorithms.formation.formation_state import FormationState
from src.algorithms.formation.formation_types import DEFAULT_FORMATION_SPACING
from src.algorithms.formation.leader_agent import LeaderAgent, leader_agent_from_pose

Vector2 = NDArray[np.float64]
HeadingMap = dict[int, float]


@dataclass(frozen=True)
class FormationGroupSnapshot:
    """
    Synchronized leader-follower snapshot at one simulation instant.

    ``timestep`` and ``time_s`` support replay, metrics sync, and future
    stability analysis.
    """

    timestep: int
    time_s: float
    leader: LeaderAgent
    formation_state: FormationState
    followers: tuple[FollowerSlotState, ...]
    metrics: FormationMetricsSnapshot
    axis_length: float = DEFAULT_FORMATION_SPACING


def build_formation_group_snapshot(
    formation_state: FormationState,
    positions: dict[int, Vector2],
    *,
    timestep: int = 0,
    time_s: float = 0.0,
    headings: HeadingMap | None = None,
    axis_length: float = DEFAULT_FORMATION_SPACING,
) -> FormationGroupSnapshot | None:
    """
    Assemble a leader-follower snapshot from formation state and agent poses.

    ``formation_target`` on the leader is left ``None``; Paper 1 BSA targets
    are not copied automatically.
    """
    leader_pos = positions.get(formation_state.leader_id)
    if leader_pos is None:
        return None

    heading = formation_state.leader_heading
    if headings is not None and formation_state.leader_id in headings:
        heading = headings[formation_state.leader_id]

    leader = leader_agent_from_pose(
        agent_id=formation_state.leader_id,
        position=leader_pos,
        heading=heading,
        formation_target=None,
    )
    followers = compute_follower_slot_states(leader, formation_state, positions)
    metrics = compute_formation_metrics(followers, len(formation_state.active_members))

    return FormationGroupSnapshot(
        timestep=timestep,
        time_s=time_s,
        leader=leader,
        formation_state=formation_state,
        followers=followers,
        metrics=metrics,
        axis_length=axis_length,
    )


def build_formation_group_snapshots(
    formation_states: list[FormationState],
    positions: dict[int, Vector2],
    *,
    timestep: int = 0,
    time_s: float = 0.0,
    headings: HeadingMap | None = None,
    axis_length: float = DEFAULT_FORMATION_SPACING,
) -> tuple[FormationGroupSnapshot, ...]:
    """Build snapshots for every formation group in ``formation_states``."""
    snapshots: list[FormationGroupSnapshot] = []
    for state in formation_states:
        snapshot = build_formation_group_snapshot(
            state,
            positions,
            timestep=timestep,
            time_s=time_s,
            headings=headings,
            axis_length=axis_length,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
    return tuple(snapshots)
