"""
Distributed consensus correction for Paper 2 formation tracking.

Lightweight neighbor-position consensus layered on proportional slot tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.environment.communication import CommunicationGraph

Vector2 = NDArray[np.float64]


@dataclass
class ConsensusConfig:
    """Parameters for local neighbor consensus correction."""

    consensus_gain: float = 0.15
    enabled: bool = True


def follower_neighbor_ids(
    graph: CommunicationGraph,
    agent_id: int,
    follower_ids: frozenset[int],
) -> tuple[int, ...]:
    """Return communicating neighbors that are followers (excludes leader/self)."""
    if agent_id not in follower_ids:
        return ()
    return tuple(
        neighbor_id
        for neighbor_id in graph.neighbors(agent_id)
        if neighbor_id in follower_ids and neighbor_id != agent_id
    )


def compute_consensus_correction(
    agent_id: int,
    positions: dict[int, Vector2],
    neighbor_ids: tuple[int, ...],
) -> Vector2:
    """
    Raw consensus term: Σ_{j∈N(i)} (p_j - p_i).

    Returns zero when there are no follower neighbors.
    """
    position = positions.get(agent_id)
    if position is None or not neighbor_ids:
        return np.zeros(2, dtype=np.float64)

    p_i = np.asarray(position, dtype=np.float64)
    total = np.zeros(2, dtype=np.float64)
    for neighbor_id in neighbor_ids:
        p_j = positions.get(neighbor_id)
        if p_j is None:
            continue
        delta = np.asarray(p_j, dtype=np.float64) - p_i
        if not np.all(np.isfinite(delta)):
            continue
        total += delta

    if not np.all(np.isfinite(total)):
        return np.zeros(2, dtype=np.float64)
    return total


def scaled_consensus_command(raw_correction: Vector2, consensus_gain: float) -> Vector2:
    """Apply consensus gain to the raw neighbor sum."""
    return consensus_gain * np.asarray(raw_correction, dtype=np.float64)
