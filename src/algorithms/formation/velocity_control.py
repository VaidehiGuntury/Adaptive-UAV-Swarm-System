"""
Velocity helpers for Paper 2 formation control.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector2 = NDArray[np.float64]


def clamp_velocity(command: Vector2, max_speed: float) -> Vector2:
    """Normalize then clamp velocity to ``max_speed``."""
    vec = np.asarray(command, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.zeros(2, dtype=np.float64)
    if norm <= max_speed:
        return vec
    return (vec / norm) * max_speed


def proportional_slot_command(
    slot_error: Vector2,
    proportional_gain: float,
    dead_zone: float,
) -> Vector2:
    """
    Proportional slot-tracking command from ``slot_error = p* - p``.

    Returns zero when ``‖slot_error‖ <= dead_zone``.
    """
    error = np.asarray(slot_error, dtype=np.float64)
    magnitude = float(np.linalg.norm(error))
    if magnitude <= dead_zone:
        return np.zeros(2, dtype=np.float64)
    return proportional_gain * error
