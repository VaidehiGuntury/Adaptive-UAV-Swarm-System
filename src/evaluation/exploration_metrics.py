"""
Observational exploration metrics for Paper 1 behavioral analysis.

Pure functions only — metrics do not influence control decisions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.environment.map import ExplorationMap, RegionKey


def allocated_center(agent: UAV) -> NDArray[np.float64] | None:
    """Return mission allocation centre p̃* (Paper 1), or None if unset."""
    if agent.assigned_region is not None:
        return agent.assigned_region.center
    return None


def mean_target_separation(agents: list[UAV]) -> float:
    """
    Mean pairwise distance between allocated mission centres p̃_i*.

    Returns 0.0 when fewer than two agents have assigned regions.
    """
    centers = [allocated_center(agent) for agent in agents]
    centers = [c for c in centers if c is not None]
    if len(centers) < 2:
        return 0.0

    distances: list[float] = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            distances.append(float(np.linalg.norm(centers[i] - centers[j])))
    return float(np.mean(distances))


def frontier_reuse_frequency(region_keys: list[RegionKey]) -> float:
    """
    Cumulative fraction of frontier replans that re-selected a prior region.

    Iterates replan region keys in mission order; a replan counts as reuse when
    its ``region_key`` was already chosen by any earlier replan. Returns 0.0
    when there are no frontier replans.
    """
    if not region_keys:
        return 0.0
    seen: set[RegionKey] = set()
    reuses = 0
    for key in region_keys:
        if key in seen:
            reuses += 1
        seen.add(key)
    return float(reuses / len(region_keys))


def revisit_ratio(
    agent_histories: dict[int, list[NDArray[np.float64]]],
    exploration_map: ExplorationMap,
) -> float:
    """
    Fleet-wide ratio of revisits to total grid-cell visits.

    revisit_ratio = (total_visits - unique_visits) / total_visits
    """
    total_visits = 0
    unique_visits: set[tuple[int, int, int]] = set()

    for agent_id, trail in agent_histories.items():
        for position in trail:
            col, row = exploration_map.world_to_grid(position)
            total_visits += 1
            unique_visits.add((agent_id, col, row))

    if total_visits == 0:
        return 0.0
    return float((total_visits - len(unique_visits)) / total_visits)
