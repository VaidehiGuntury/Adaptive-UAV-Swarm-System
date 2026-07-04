"""
SearchAssigner — target-to-UAV assignment after exploration completes.

Design principles
-----------------
- Consumes IDE outputs (agent.assigned_region) as read-only
- Minimizes travel distance using a greedy cost matrix
- Respects max_targets_per_uav constraint
- Avoids duplicate assignments
- Supports reassignment when a UAV fails (no position update timeout)
- Does NOT modify IDE allocation or BSA in any way

Assignment cost
---------------
  cost(uav_i, target_j) = w_dist * normalized_dist(uav_i, target_j)
                         - w_priority * normalized_priority(target_j)

The solver uses a greedy assignment on the cost matrix (sufficient for
the fleet sizes considered: 10–30 UAVs × 8 targets).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.config.search_config import AssignmentConfig
from src.search.target import Target, TargetStatus
from src.search.target_manager import TargetManager


@dataclass
class AssignmentRecord:
    """Tracks current assignments for audit and rebalancing."""

    # uav_id → set of assigned target_ids
    uav_to_targets: dict[int, set[int]] = field(default_factory=dict)
    # target_id → uav_id
    target_to_uav: dict[int, int] = field(default_factory=dict)
    # uav_id → last known position (for failure detection)
    uav_last_position: dict[int, NDArray[np.float64]] = field(default_factory=dict)
    # uav_id → last update time
    uav_last_update_time: dict[int, float] = field(default_factory=dict)


class SearchAssigner:
    """
    Assigns detected targets to UAVs after exploration completes.

    Parameters
    ----------
    config : AssignmentConfig
    """

    def __init__(self, config: AssignmentConfig) -> None:
        self._config = config
        self._record = AssignmentRecord()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(
        self,
        agents: list[UAV],
        target_manager: TargetManager,
        current_time: float,
    ) -> dict[int, list[int]]:
        """
        Perform initial assignment of detected targets to UAVs.

        Returns a mapping {uav_id: [target_id, ...]} for the new assignments.
        Also updates TargetManager with ASSIGNED status.
        """
        targets = target_manager.detected_targets()
        if not targets:
            return {}

        # Initialize records for all agents
        for agent in agents:
            if agent.agent_id not in self._record.uav_to_targets:
                self._record.uav_to_targets[agent.agent_id] = set()
            self._record.uav_last_position[agent.agent_id] = agent.position.copy()
            self._record.uav_last_update_time[agent.agent_id] = current_time

        assignments: dict[int, list[int]] = {a.agent_id: [] for a in agents}
        unassigned = list(targets)

        # Build cost matrix: rows = agents, cols = targets
        # Only consider agents with available slots
        max_tpu = self._config.max_targets_per_uav

        while unassigned:
            # Find agent with most capacity
            eligible = [
                a for a in agents
                if len(self._record.uav_to_targets[a.agent_id]) < max_tpu
            ]
            if not eligible:
                break

            # For each unassigned target, find cheapest eligible agent
            best_cost = float("inf")
            best_agent_id: int | None = None
            best_target: Target | None = None

            max_priority = max((t.priority for t in unassigned), default=1.0)
            if max_priority <= 0.0:
                max_priority = 1.0

            for target in unassigned:
                dists = [
                    float(np.linalg.norm(a.position - target.position))
                    for a in eligible
                ]
                max_dist = max(dists) if dists else 1.0
                if max_dist <= 0.0:
                    max_dist = 1.0

                for agent, dist in zip(eligible, dists):
                    norm_dist = dist / max_dist
                    norm_priority = target.priority / max_priority
                    cost = (
                        self._config.cost_distance_weight * norm_dist
                        - self._config.cost_priority_weight * norm_priority
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_agent_id = agent.agent_id
                        best_target = target

            if best_agent_id is None or best_target is None:
                break

            # Commit assignment
            self._record.uav_to_targets[best_agent_id].add(best_target.target_id)
            self._record.target_to_uav[best_target.target_id] = best_agent_id
            assignments[best_agent_id].append(best_target.target_id)
            target_manager.assign_target(best_target.target_id, best_agent_id)
            unassigned.remove(best_target)

        return assignments

    def rebalance(
        self,
        agents: list[UAV],
        target_manager: TargetManager,
        current_time: float,
    ) -> dict[int, list[int]]:
        """
        Detect failed UAVs and reassign their targets.

        A UAV is considered failed if it has not been updated within
        config.fail_timeout_s seconds.

        Returns new assignments created by rebalancing.
        """
        if not self._config.reassignment_on_fail:
            return {}

        failed_targets: list[Target] = []
        active_ids = {a.agent_id for a in agents}

        for uav_id, last_time in list(self._record.uav_last_update_time.items()):
            if uav_id not in active_ids:
                continue
            if current_time - last_time > self._config.fail_timeout_s:
                # UAV failed — release its targets
                for tid in self._record.uav_to_targets.get(uav_id, set()):
                    target = target_manager.get_target(tid)
                    if target is not None and target.status not in (
                        TargetStatus.COMPLETED,
                        TargetStatus.LOST,
                    ):
                        target_manager.unassign_target(tid)
                        failed_targets.append(target)
                self._record.uav_to_targets[uav_id] = set()

        if not failed_targets:
            return {}

        # Re-score and reassign
        return self.assign(agents, target_manager, current_time)

    def update_agent_position(
        self,
        uav_id: int,
        position: NDArray[np.float64],
        current_time: float,
    ) -> None:
        """Record latest agent position for failure detection."""
        self._record.uav_last_position[uav_id] = position.copy()
        self._record.uav_last_update_time[uav_id] = current_time

    def release_target(self, target_id: int, target_manager: TargetManager) -> None:
        """Release a completed or permanently-lost target from assignment records."""
        uav_id = self._record.target_to_uav.pop(target_id, None)
        if uav_id is not None:
            self._record.uav_to_targets.get(uav_id, set()).discard(target_id)

    def get_assignments(self) -> dict[int, set[int]]:
        """Return a snapshot of current uav_id → {target_ids} mapping."""
        return {k: set(v) for k, v in self._record.uav_to_targets.items()}
