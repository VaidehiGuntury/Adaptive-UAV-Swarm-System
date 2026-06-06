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

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import UAV
from src.config.loader import SimulationConfig
from src.environment.world import World


@dataclass(frozen=True)
class SimulationMetrics:
    """Aggregated metrics aligned with Paper 1 evaluation (mission progress)."""

    timestep: int
    time_s: float
    explored_fraction: float
    mean_speed: float
    mean_pairwise_distance: float


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

    @property
    def total_steps(self) -> int:
        return int(self.config.duration / self.config.dt)

    def step(self) -> SimulationMetrics:
        """Execute one simulation timestep."""
        dt = self.config.dt

        for agent in self.agents:
            self.aggregation.update(agent, self.agents, self.world, dt)

        for agent in self.agents:
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

    def _collect_metrics(self) -> SimulationMetrics:
        speeds = [float(np.linalg.norm(agent.velocity)) for agent in self.agents]
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        pairwise: list[float] = []
        for i, agent_i in enumerate(self.agents):
            for agent_j in self.agents[i + 1 :]:
                pairwise.append(agent_i.compute_distance(agent_j))
        mean_pairwise = float(np.mean(pairwise)) if pairwise else 0.0

        return SimulationMetrics(
            timestep=self.timestep,
            time_s=self.time_s,
            explored_fraction=self.world.map.explored_fraction(),
            mean_speed=mean_speed,
            mean_pairwise_distance=mean_pairwise,
        )
