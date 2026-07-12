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
from src.algorithms.allocation import IDEAllocator
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

        # Mission radius read from config once, reused in set_region() calls.
        self._mission_radius: float = config.aggregation.mission_region_radius

        # IDE allocator (DEBS §4). None when the ide: block is absent from YAML.
        # When None the simulation is identical to the pre-IDE baseline.
        if config.ide is not None:
            self._ide_allocator: IDEAllocator | None = IDEAllocator(
                config=config.ide,
                world_bounds=config.world_bounds,
            )
        else:
            self._ide_allocator = None

        # Gate: tracks last simulation time at which IDE allocation ran.
        # IDE runs once per BSA replan cycle, not every step.
        self._last_ide_time: float = -float("inf")

        # Search extension: populated in build_simulation() when search
        # config is present. None means search extension is disabled.
        self.mission_orchestrator: MissionOrchestrator | None = None

    @property
    def total_steps(self) -> int:
        return int(self.config.duration / self.config.dt)

    def step(self) -> SimulationMetrics:
        """
        Execute one simulation timestep.

        Step order:
        1. Obstacle Update   - advance all dynamic obstacles (if enabled).
        2. IDE Allocation    - update p~* before BSA reads J_C (DEBS §4).
        3. BSA Aggregation   - viewpoint selection per UAV (exploration only).
        4. UAV Kinematics    - motion toward assigned target.
        5. Collision Resolve - push UAVs out of static obstacles.
        6. Boundary Clamp    - keep UAVs inside world bounds.
        7. Map Update        - mark explored cells.
        8. Mission Phase     - advance search orchestrator if active.
        9. Metrics           - collect and record.
        """
        dt = self.config.dt

        # 1. Obstacle Update — advance dynamic obstacles before aggregation
        #    so UAVs react to the latest obstacle state.
        if self.world.obstacle_manager is not None:
            self.world.obstacle_manager.update(dt)

        # Determine current mission phase
        is_searching = (
            self.mission_orchestrator is not None
            and self.mission_orchestrator.phase.name in ("SEARCHING", "COMPLETED")
        )
        is_transitioning = (
            self.mission_orchestrator is not None
            and self.mission_orchestrator.phase.name == "SEARCH_TRANSITION"
        )

        # 2 & 3. IDE + BSA — only during exploration phase.
        #    IDE updates p~* BEFORE BSA reads J_C (DEBS §4 Algorithm 2).
        if not is_searching and not is_transitioning:
            self.aggregation.begin_step()

            # IDE allocation: runs once per replan cycle, gated by _last_ide_time.
            if self._ide_allocator is not None:
                elapsed_since_ide = self.time_s - self._last_ide_time
                if elapsed_since_ide >= self.config.aggregation.replan_interval:
                    new_allocations = self._ide_allocator.allocate(
                        self.agents,
                        self.time_s,
                    )
                    for agent in self.agents:
                        if agent.agent_id in new_allocations:
                            agent.set_region(
                                new_allocations[agent.agent_id],
                                self._mission_radius,
                            )
                    self._last_ide_time = self.time_s

            # BSA aggregation: reads fresh p~* set by IDE above.
            for agent in self.agents:
                self.aggregation.update(agent, self.agents, self.world, dt)

        # 4-6. UAV kinematics, collision resolution, boundary clamp.
        for agent in self.agents:
            agent.update(dt)
            agent.position = self.world.resolve_collisions(agent.position)
            agent.position = self.world.clip_position(agent.position)
            # 7. Map update — only mark explored during exploration phase.
            if not is_searching and not is_transitioning:
                self.world.map.mark_explored(
                    agent.position, self.config.uav.sensing_range
                )
            self.agent_histories[agent.agent_id].append(agent.position.copy())

        # Also mark explored during transition (UAVs still cover ground).
        if is_transitioning:
            for agent in self.agents:
                self.world.map.mark_explored(
                    agent.position, self.config.uav.sensing_range
                )

        self.timestep += 1
        self.time_s += dt

        # 8. Advance mission orchestrator (handles phase transitions and
        #    search logic). Must run after kinematics.
        if self.mission_orchestrator is not None:
            self.mission_orchestrator.step(self.time_s, dt)

        # 9. Metrics.
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
        latest = (
            self.metrics_history[-1]
            if self.metrics_history
            else self._collect_metrics()
        )
        return SimulationState(
            timestep=self.timestep,
            time_s=self.time_s,
            agents=list(self.agents),
            metrics=latest,
        )

    def _collect_metrics(self) -> SimulationMetrics:
        speeds = [
            float(np.linalg.norm(agent.velocity)) for agent in self.agents
        ]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        pairwise: list[float] = []
        for i, agent_i in enumerate(self.agents):
            for agent_j in self.agents[i + 1:]:
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