"""
Greedy nearest-frontier baseline allocator (Phase H comparison).

Naive alternative to SelfAggregationController (+ IDE): each UAV
independently targets its own nearest frontier cell by Euclidean
distance, on the same replan cadence, with no coordination, no
candidate scoring, and no region/fair-share allocation. Duplicate
targets across UAVs are expected and allowed — that is the point of
the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.config.loader import AggregationConfig, UAVConfig
from src.environment.map import RegionKey
from src.environment.world import World


@dataclass
class GreedyFrontierAllocator:
    """
    Naive nearest-frontier-cell baseline allocator.

    Drop-in replacement for SelfAggregationController in SimulationEngine:
    implements the same begin_step() / update() / replan_region_history /
    step_reassignment_count interface (see SimulationEngine.step() and
    SimulationEngine._collect_metrics()), so no engine changes are
    required to run it.
    """

    config: AggregationConfig
    uav_config: UAVConfig
    _time_since_replan: dict[int, float] = field(default_factory=dict)

    def begin_step(self) -> None:
        """No per-step instrumentation state to reset (naive baseline)."""

    @property
    def replan_region_history(self) -> list[RegionKey]:
        """Stub — greedy has no frontier-cluster/region concept to report."""
        return []

    @property
    def step_reassignment_count(self) -> int:
        """Stub — greedy does not track BSA-style replan counts."""
        return 0

    def update(
        self,
        agent: UAV,
        all_agents: list[UAV],
        world: World,
        dt: float,
    ) -> None:
        """
        Assign the nearest frontier cell on the replan cadence.

        Reassigns whenever the replan interval elapses, or the UAV has
        already arrived at its current target (using the same ``d_c``
        threshold BSA uses) — the arrival check exists because
        ``spawn_uavs`` sets each UAV's initial ``assigned_target`` to its
        own spawn position, so without it a fleet would idle for one full
        ``replan_interval`` before its first real assignment while DEBS
        (whose own arrival check has the same effect) moves immediately;
        this keeps the two allocators comparable from t=0. No hysteresis,
        no measurable-improvement gate, no coordination with other UAVs —
        ``all_agents`` is accepted only to match the interface.
        """
        elapsed = self._time_since_replan.get(agent.agent_id, 0.0) + dt
        self._time_since_replan[agent.agent_id] = elapsed

        has_target = agent.assigned_target is not None
        arrived = False
        if has_target:
            dist_to_target = float(
                np.linalg.norm(agent.position - agent.assigned_target)
            )
            arrived = dist_to_target <= self.config.d_c

        due_for_replan = elapsed >= self.config.replan_interval
        if has_target and not arrived and not due_for_replan:
            return

        self._time_since_replan[agent.agent_id] = 0.0

        target = self._nearest_frontier_target(agent.position, world)
        if target is None:
            # No frontier cells left (e.g. fully explored) — hold position
            # rather than crash or chase a stale target.
            agent.set_target(agent.position.copy())
            return

        agent.set_target(target)

    def _nearest_frontier_target(
        self,
        position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64] | None:
        """
        Nearest frontier-cell world position by Euclidean distance.

        Uses ``ExplorationMap.frontier_mask()`` directly (raw per-cell
        frontiers), not ``extract_frontier_clusters()`` — deliberately
        skips BSA's clustering step to keep this baseline maximally naive.

        Returns None when the map has no frontier cells (fully explored).
        """
        mask = world.map.frontier_mask()
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            return None

        best_point: NDArray[np.float64] | None = None
        best_dist = float("inf")
        for row, col in zip(rows.tolist(), cols.tolist()):
            point = world.map.grid_to_world(col, row)
            dist = float(np.linalg.norm(point - position))
            if dist < best_dist:
                best_dist = dist
                best_point = point
        return best_point
