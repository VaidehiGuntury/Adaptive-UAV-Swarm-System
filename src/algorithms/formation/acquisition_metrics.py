"""
Formation acquisition metrics for Paper 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.algorithms.formation.follower_state import FollowerSlotState


@dataclass(frozen=True)
class FormationAcquisitionMetrics:
    """Runtime formation quality metrics during acquisition."""

    rms_formation_error: float
    mean_follower_slot_error: float
    occupied_slot_percentage: float
    formation_convergence_time_s: float | None


def compute_acquisition_metrics(
    followers: tuple[FollowerSlotState, ...],
    total_slot_count: int,
    leader_slot_occupied: bool = True,
    convergence_time_s: float | None = None,
) -> FormationAcquisitionMetrics:
    """Compute RMS error, mean follower error, and occupancy percentage."""
    if not followers:
        return FormationAcquisitionMetrics(
            rms_formation_error=0.0,
            mean_follower_slot_error=0.0,
            occupied_slot_percentage=0.0,
            formation_convergence_time_s=convergence_time_s,
        )

    errors = [f.error_magnitude for f in followers]
    mean_error = float(np.mean(errors))
    rms_error = float(np.sqrt(np.mean(np.square(errors))))

    follower_occupied = sum(1 for f in followers if f.is_occupied)
    occupied = int(leader_slot_occupied) + follower_occupied
    occupied = min(occupied, max(total_slot_count, 1))
    percentage = occupied / max(total_slot_count, 1)

    return FormationAcquisitionMetrics(
        rms_formation_error=rms_error,
        mean_follower_slot_error=mean_error,
        occupied_slot_percentage=percentage,
        formation_convergence_time_s=convergence_time_s,
    )
