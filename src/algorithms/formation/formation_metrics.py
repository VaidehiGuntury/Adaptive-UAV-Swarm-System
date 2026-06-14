"""
Formation metrics infrastructure for Paper 2 (observational only).

No controller stability metrics in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.algorithms.formation.follower_state import FollowerSlotState


@dataclass(frozen=True)
class FormationMetricsSnapshot:
    """Observational formation quality metrics at a single instant."""

    mean_formation_error: float
    slot_occupancy_completeness: float
    occupied_slot_count: int
    vacant_slot_count: int


def compute_formation_metrics(
    followers: tuple[FollowerSlotState, ...],
    total_slot_count: int,
    *,
    leader_slot_occupied: bool = True,
) -> FormationMetricsSnapshot:
    """
    Compute formation metrics from follower slot snapshots.

    Mean error is averaged over followers only (leader slot 0 excluded).
    Occupancy counts include the leader slot when ``leader_slot_occupied`` is True.
    """
    if total_slot_count <= 0:
        return FormationMetricsSnapshot(
            mean_formation_error=0.0,
            slot_occupancy_completeness=0.0,
            occupied_slot_count=0,
            vacant_slot_count=0,
        )

    follower_occupied = sum(1 for follower in followers if follower.is_occupied)
    occupied = int(leader_slot_occupied) + follower_occupied
    occupied = min(occupied, total_slot_count)
    vacant = total_slot_count - occupied
    completeness = occupied / total_slot_count

    if followers:
        mean_error = sum(f.error_magnitude for f in followers) / len(followers)
    else:
        mean_error = 0.0

    return FormationMetricsSnapshot(
        mean_formation_error=mean_error,
        slot_occupancy_completeness=completeness,
        occupied_slot_count=occupied,
        vacant_slot_count=vacant,
    )
