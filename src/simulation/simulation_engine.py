"""
Simulation loop for Paper 1 decentralized exploration.

Responsibilities:
  - advance time
  - update BSA aggregation decisions
  - update UAV kinematics
  - collect metrics and agent trajectories

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
    from src.algorithms.formation.formation_controller import FormationAcquisitionController


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
    formation_active: bool = False
    consensus_active: bool = False
    rms_formation_error: float = 0.0
    mean_follower_slot_error: float = 0.0
    occupied_slot_percentage: float = 0.0
    formation_convergence_time_s: float | None = None
    average_neighbor_count: float = 0.0
    graph_connectivity_status: bool = False
    consensus_residual_magnitude: float = 0.0
    local_spacing_variance: float = 0.0
    isolated_follower_count: int = 0


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
        formation_controller: FormationAcquisitionController | None = None,
    ) -> None:
        self.world = world
        self.agents = agents
        self.aggregation = aggregation
        self.config = config
        self.formation_controller = formation_controller
        self.timestep = 0
        self.time_s = 0.0
        self.metrics_history: list[SimulationMetrics] = []
        self.agent_histories: dict[int, list[NDArray[np.float64]]] = {
            agent.agent_id: [agent.position.copy()] for agent in agents
        }

    @property
    def total_steps(self) -> int:
        return int(self.config.duration / self.config.dt)

    def step(self) -> SimulationMetrics:
        """Execute one simulation timestep."""
        dt = self.config.dt
        self.aggregation.begin_step()

        mean_pairwise = self._mean_pairwise_distance()
        if self.formation_controller is not None:
            self.formation_controller.update_activation(
                self.agents,
                self.world,
                mean_pairwise,
                self.time_s,
            )

        formation_active = (
            self.formation_controller is not None and self.formation_controller.is_active
        )
        leader_id = self.formation_controller.leader_id if formation_active else None

        for agent in self.agents:
            if not formation_active or agent.agent_id == leader_id:
                self.aggregation.update(agent, self.agents, self.world, dt)

        if formation_active and self.formation_controller is not None:
            self.formation_controller.apply_follower_control(
                self.agents,
                self.world,
                dt,
                self.config.uav.sensing_range,
                self.time_s,
            )

        for agent in self.agents:
            if formation_active and self.formation_controller is not None:
                if self.formation_controller.is_follower(agent.agent_id):
                    agent.position = self.world.resolve_collisions(agent.position)
                    agent.position = self.world.clip_position(agent.position)
                    self.world.map.mark_explored(agent.position, self.config.uav.sensing_range)
                    self.agent_histories[agent.agent_id].append(agent.position.copy())
                    continue
            agent.update(dt)
            agent.position = self.world.resolve_collisions(agent.position)
            agent.position = self.world.clip_position(agent.position)
            self.world.map.mark_explored(agent.position, self.config.uav.sensing_range)
            self.agent_histories[agent.agent_id].append(agent.position.copy())

        self.timestep += 1
        self.time_s += dt
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

    def _mean_pairwise_distance(self) -> float:
        pairwise: list[float] = []
        for i, agent_i in enumerate(self.agents):
            for agent_j in self.agents[i + 1 :]:
                pairwise.append(agent_i.compute_distance(agent_j))
        return float(np.mean(pairwise)) if pairwise else 0.0

    def _collect_metrics(self) -> SimulationMetrics:
        speeds = [float(np.linalg.norm(agent.velocity)) for agent in self.agents]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        mean_pairwise = self._mean_pairwise_distance()
        frontier_clusters = self.world.map.extract_frontier_clusters()

        formation_active = False
        consensus_active = False
        rms_formation_error = 0.0
        mean_follower_slot_error = 0.0
        occupied_slot_percentage = 0.0
        formation_convergence_time_s = None
        average_neighbor_count = 0.0
        graph_connectivity_status = False
        consensus_residual_magnitude = 0.0
        local_spacing_variance = 0.0
        isolated_follower_count = 0

        controller = self.formation_controller
        if controller is not None and controller.is_active:
            formation_active = True
            consensus_active = controller.consensus_active
            if controller.last_acquisition_metrics is not None:
                acq = controller.last_acquisition_metrics
                rms_formation_error = acq.rms_formation_error
                mean_follower_slot_error = acq.mean_follower_slot_error
                occupied_slot_percentage = acq.occupied_slot_percentage
                formation_convergence_time_s = acq.formation_convergence_time_s
            if controller.last_consensus_metrics is not None:
                con = controller.last_consensus_metrics
                average_neighbor_count = con.average_neighbor_count
                graph_connectivity_status = con.graph_connectivity_status
                consensus_residual_magnitude = con.consensus_residual_magnitude
                local_spacing_variance = con.local_spacing_variance
                isolated_follower_count = con.isolated_follower_count

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
            formation_active=formation_active,
            consensus_active=consensus_active,
            rms_formation_error=rms_formation_error,
            mean_follower_slot_error=mean_follower_slot_error,
            occupied_slot_percentage=occupied_slot_percentage,
            formation_convergence_time_s=formation_convergence_time_s,
            average_neighbor_count=average_neighbor_count,
            graph_connectivity_status=graph_connectivity_status,
            consensus_residual_magnitude=consensus_residual_magnitude,
            local_spacing_variance=local_spacing_variance,
            isolated_follower_count=isolated_follower_count,
        )
