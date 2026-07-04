"""
Simulation loop for Paper 1 decentralized exploration.

Responsibilities:
  - advance time
  - update BSA aggregation decisions (EXPLORING phase)
  - update UAV kinematics
  - collect metrics and agent trajectories
  - drive MissionOrchestrator for search phase (SEARCHING phase)

Visualization is intentionally excluded (see src/visualization/renderer.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import UAV
from src.config.loader import SimulationConfig
from src.environment.world import World
from src.evaluation.exploration_metrics import (
    frontier_reuse_frequency,
    mean_target_separation,
    revisit_ratio,
)

if TYPE_CHECKING:
    from src.search.mission_phase import MissionOrchestrator, MissionPhase


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregated metrics aligned with Paper 1 evaluation (mission progress)."""

    timestep: int
    time_s: float
    explored_fraction: float
    mean_speed: float
    mean_pairwise_distance: float
    mean_target_separation: float
    frontier_reuse_frequency: float
    target_reassignment_count: int
    revisit_ratio: float
    active_frontier_count: int
    # Search extension metrics (None during exploration phase)
    mission_phase: str = "exploring"
    detected_targets: int = 0
    assigned_targets: int = 0
    tracking_targets: int = 0
    completed_targets: int = 0
    lost_targets: int = 0


@dataclass
class SimulationState:
    """Snapshot of simulation state for rendering or logging."""

    timestep: int
    time_s: float
    agents: list[UAV]
    metrics: SimulationMetrics


class SimulationEngine:
    """Discrete-time simulator with history recording."""

    def __init__(
        self,
        world: World,
        agents: list[UAV],
        aggregation: SelfAggregationController,
        config: SimulationConfig,
    ) -> None:
        self.world = world
        self.agents = agents
        self.aggregation = aggregation
        self.config = config
        self.timestep = 0
        self.time_s = 0.0
        self.metrics_history: list[SimulationMetrics] = []
        self.agent_histories: dict[int, list[NDArray[np.float64]]] = {
            agent.agent_id: [agent.position.copy()] for agent in agents
        }
        # Search extension: populated in build_simulation() when search config present
        self.mission_orchestrator: MissionOrchestrator | None = None

    @property
    def total_steps(self) -> int:
        return int(self.config.duration / self.config.dt)

    def step(self) -> SimulationMetrics:
        """Execute one simulation timestep."""
        dt = self.config.dt

        # Determine current mission phase
        is_searching = (
            self.mission_orchestrator is not None
            and self.mission_orchestrator.phase.name in ("SEARCHING", "COMPLETED")
        )
        is_transitioning = (
            self.mission_orchestrator is not None
            and self.mission_orchestrator.phase.name == "SEARCH_TRANSITION"
        )

        if not is_searching and not is_transitioning:
            # ── EXPLORATION PHASE ── BSA is active
            self.aggregation.begin_step()
            for agent in self.agents:
                self.aggregation.update(agent, self.agents, self.world, dt)

        # Kinematics always advance
        for agent in self.agents:
            agent.update(dt)
            agent.position = self.world.resolve_collisions(agent.position)
            agent.position = self.world.clip_position(agent.position)
            if not is_searching and not is_transitioning:
                # Only mark explored during exploration phase
                self.world.map.mark_explored(agent.position, self.config.uav.sensing_range)
            self.agent_histories[agent.agent_id].append(agent.position.copy())

        # Also mark explored during transition (UAVs still cover ground)
        if is_transitioning:
            for agent in self.agents:
                self.world.map.mark_explored(agent.position, self.config.uav.sensing_range)

        self.timestep += 1
        self.time_s += dt

        # Advance mission orchestrator (handles phase transitions and search logic)
        if self.mission_orchestrator is not None:
            self.mission_orchestrator.step(self.time_s, dt)

        metrics = self._collect_metrics()
        self.metrics_history.append(metrics)
        return metrics

    def run(self) -> list[SimulationMetrics]:
        """Run the full simulation until duration is reached."""
        results: list[SimulationMetrics] = []
        while self.time_s < self.config.duration:
            results.append(self.step())
        return results

    def get_state(self) -> SimulationState:
        """Return current state snapshot for visualization."""
        latest = self.metrics_history[-1] if self.metrics_history else self._collect_metrics()
        return SimulationState(
            timestep=self.timestep,
            time_s=self.time_s,
            agents=list(self.agents),
            metrics=latest,
        )

    def _collect_metrics(self) -> SimulationMetrics:
        speeds = [float(np.linalg.norm(agent.velocity)) for agent in self.agents]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        pairwise: list[float] = []
        for i, agent_i in enumerate(self.agents):
            for agent_j in self.agents[i + 1 :]:
                pairwise.append(agent_i.compute_distance(agent_j))
        mean_pairwise = float(np.mean(pairwise)) if pairwise else 0.0

        frontier_clusters = self.world.map.extract_frontier_clusters()

        # Search phase metrics
        phase_name = "exploring"
        detected = assigned = tracking = completed = lost = 0
        if self.mission_orchestrator is not None:
            phase_name = self.mission_orchestrator.phase.name.lower()
            tm = self.mission_orchestrator.target_manager
            from src.search.target import TargetStatus
            for t in tm.all_targets():
                if t.status == TargetStatus.DETECTED:
                    detected += 1
                elif t.status in (TargetStatus.ASSIGNED, TargetStatus.SEARCHING):
                    assigned += 1
                elif t.status == TargetStatus.TRACKING:
                    tracking += 1
                elif t.status == TargetStatus.COMPLETED:
                    completed += 1
                elif t.status == TargetStatus.LOST:
                    lost += 1

        return SimulationMetrics(
            timestep=self.timestep,
            time_s=self.time_s,
            explored_fraction=self.world.map.explored_fraction(),
            mean_speed=mean_speed,
            mean_pairwise_distance=mean_pairwise,
            mean_target_separation=mean_target_separation(self.agents),
            frontier_reuse_frequency=frontier_reuse_frequency(
                self.aggregation.replan_region_history
            ),
            target_reassignment_count=self.aggregation.step_reassignment_count,
            revisit_ratio=revisit_ratio(self.agent_histories, self.world.map),
            active_frontier_count=len(frontier_clusters),
            mission_phase=phase_name,
            detected_targets=detected,
            assigned_targets=assigned,
            tracking_targets=tracking,
            completed_targets=completed,
            lost_targets=lost,
        )
