"""
Consensus observability metrics for Paper 2 distributed formation tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.environment.communication import CommunicationGraph

Vector2 = NDArray[np.float64]


@dataclass(frozen=True)
class ConsensusMetricsSnapshot:
    """Observational consensus metrics at one instant."""

    average_neighbor_count: float
    graph_connectivity_status: bool
    consensus_residual_magnitude: float
    local_spacing_variance: float
    isolated_follower_count: int


def compute_consensus_metrics(
    follower_ids: frozenset[int],
    graph: CommunicationGraph,
    positions: dict[int, Vector2],
    scaled_corrections: dict[int, Vector2],
    neighbor_map: dict[int, tuple[int, ...]],
) -> ConsensusMetricsSnapshot:
    """Compute neighbor, connectivity, and consensus residual metrics."""
    if not follower_ids:
        return ConsensusMetricsSnapshot(
            average_neighbor_count=0.0,
            graph_connectivity_status=True,
            consensus_residual_magnitude=0.0,
            local_spacing_variance=0.0,
            isolated_follower_count=0,
        )

    neighbor_counts = [len(neighbor_map.get(fid, ())) for fid in follower_ids]
    avg_neighbors = float(np.mean(neighbor_counts)) if neighbor_counts else 0.0
    isolated = sum(1 for count in neighbor_counts if count == 0)

    residuals = [
        float(np.linalg.norm(scaled_corrections[fid]))
        for fid in follower_ids
        if fid in scaled_corrections
    ]
    residual_mag = float(np.mean(residuals)) if residuals else 0.0

    local_variances: list[float] = []
    for follower_id in follower_ids:
        neighbors = neighbor_map.get(follower_id, ())
        if len(neighbors) < 2:
            continue
        p_i = positions.get(follower_id)
        if p_i is None:
            continue
        distances = [
            float(np.linalg.norm(np.asarray(positions[n], dtype=np.float64) - np.asarray(p_i, dtype=np.float64)))
            for n in neighbors
            if n in positions
        ]
        if len(distances) >= 2:
            local_variances.append(float(np.var(distances)))

    spacing_var = float(np.mean(local_variances)) if local_variances else 0.0
    connected = graph.is_follower_subgraph_connected(follower_ids)

    return ConsensusMetricsSnapshot(
        average_neighbor_count=avg_neighbors,
        graph_connectivity_status=connected,
        consensus_residual_magnitude=residual_mag,
        local_spacing_variance=spacing_var,
        isolated_follower_count=isolated,
    )
