"""
Communication and consensus overlays for Pygame (Paper 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.environment.communication import CommunicationGraph
from src.visualization.coordinate_transform import CoordinateTransform
from src.visualization.render_palette import (
    COLOR_COMM_ACTIVE_EDGE,
    COLOR_COMM_EDGE,
    COLOR_COMM_ISOLATED,
    COLOR_CONSENSUS_VECTOR,
    CONSENSUS_VECTOR_SCALE,
)

Vector2 = NDArray[np.float64]


def draw_communication_overlay(
    surface: Any,
    pg: Any,
    graph: CommunicationGraph | None,
    agents: list[Any],
    transform: CoordinateTransform,
    *,
    neighbor_map: dict[int, tuple[int, ...]] | None = None,
    scaled_consensus: dict[int, Vector2] | None = None,
    isolated_followers: frozenset[int] | None = None,
    consensus_active: bool = False,
    agent_radius_px: int = 6,
) -> None:
    """Draw comm links, consensus vectors, and isolation warnings."""
    if graph is None:
        return

    positions = {agent.agent_id: agent.position for agent in agents}
    active_pairs: set[tuple[int, int]] = set()
    if neighbor_map is not None:
        for source, neighbors in neighbor_map.items():
            for target in neighbors:
                active_pairs.add((min(source, target), max(source, target)))

    for source, target, _weight in graph.edges():
        p0 = transform.world_to_screen(positions[source])
        p1 = transform.world_to_screen(positions[target])
        pair = (min(source, target), max(source, target))
        color = COLOR_COMM_ACTIVE_EDGE if pair in active_pairs else COLOR_COMM_EDGE
        pg.draw.line(surface, color, p0, p1, width=1)

    if consensus_active and scaled_consensus:
        for agent_id, correction in scaled_consensus.items():
            if agent_id not in positions:
                continue
            norm = float(np.linalg.norm(correction))
            if norm < 1e-6:
                continue
            start = positions[agent_id]
            end = start + correction * CONSENSUS_VECTOR_SCALE
            p0 = transform.world_to_screen(start)
            p1 = transform.world_to_screen(end)
            pg.draw.line(surface, COLOR_CONSENSUS_VECTOR, p0, p1, width=2)

    if isolated_followers:
        for agent_id in isolated_followers:
            if agent_id not in positions:
                continue
            cx, cy = transform.world_to_screen(positions[agent_id])
            pg.draw.circle(surface, COLOR_COMM_ISOLATED, (cx, cy), agent_radius_px + 8, width=2)
