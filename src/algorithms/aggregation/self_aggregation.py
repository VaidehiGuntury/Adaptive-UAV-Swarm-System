"""
Bio-inspired self-aggregation controller (Paper 1 Sec. 5).

Separates:
  - aggregation decision logic (candidate generation + selection)
  - fitness evaluation (fitness_functions.py)
  - movement execution (UAV.move via simulation engine)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.algorithms.aggregation.fitness_functions import ViewpointCandidate, evaluate_viewpoint_cost
from src.config.loader import AggregationConfig, UAVConfig
from src.environment.map import FrontierCluster
from src.environment.world import World


@dataclass
class SelfAggregationController:
    """
    Decentralized BSA viewpoint selector for each UAV.

    Paper 1 workflow (Sec. 5):
      1. Extract frontier clusters from map M.
      2. Sample candidate viewpoints vp_c around cluster centroids.
      3. Score each ξ_c with J_C, J_V, J_L (Eqs. 6–10).
      4. Select minimal-cost viewpoint as next target pose.
    """

    config: AggregationConfig
    uav_config: UAVConfig
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    _time_since_replan: dict[int, float] = field(default_factory=dict)

    def update(
        self,
        agent: UAV,
        all_agents: list[UAV],
        world: World,
        dt: float,
    ) -> None:
        """
        Run BSA decision logic and assign a new target when replan interval elapses.
        """
        elapsed = self._time_since_replan.get(agent.agent_id, 0.0) + dt
        self._time_since_replan[agent.agent_id] = elapsed

        if elapsed < self.config.replan_interval and agent.assigned_target is not None:
            return

        self._time_since_replan[agent.agent_id] = 0.0
        clusters = world.map.extract_frontier_clusters()
        candidates = self._generate_candidates(agent, clusters, world)

        if not candidates:
            fallback = self._fallback_target(agent, world)
            agent.set_target(fallback)
            agent.set_region(fallback, self.config.mission_region_radius)
            return

        best = max(
            candidates,
            key=lambda c: evaluate_viewpoint_cost(c, agent, all_agents, self.config),
        )
        agent.set_target(best.viewpoint)
        agent.set_region(best.viewpoint, self.config.mission_region_radius)

        if best.is_trail:
            world.map.mark_cluster_as_trail(best.cluster_id)

    def _generate_candidates(
        self,
        agent: UAV,
        clusters: list[FrontierCluster],
        world: World,
    ) -> list[ViewpointCandidate]:
        """Sample candidate viewpoints around frontier cluster centroids."""
        candidates: list[ViewpointCandidate] = []
        sensing = self.uav_config.sensing_range

        for cluster in clusters:
            for _ in range(self.config.candidates_per_frontier):
                angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
                radius = float(self.rng.uniform(0.5, sensing))
                offset = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
                viewpoint = cluster.centroid + offset
                viewpoint = world.clip_position(viewpoint)
                viewpoint = world.resolve_collisions(viewpoint)

                yaw = float(np.arctan2(
                    cluster.centroid[1] - viewpoint[1],
                    cluster.centroid[0] - viewpoint[0],
                ))
                candidates.append(
                    ViewpointCandidate(
                        viewpoint=viewpoint,
                        yaw=yaw,
                        cluster_id=cluster.cluster_id,
                        is_trail=cluster.is_trail,
                    )
                )

        return candidates

    def _fallback_target(self, agent: UAV, world: World) -> NDArray[np.float64]:
        """
        Explore outward when no frontiers exist yet.

        Uses a random direction within sensing range — placeholder until full
        mapping / frontier pipeline is active at simulation start.
        """
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        radius = self.uav_config.sensing_range
        direction = radius * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        target = agent.position + direction
        target = world.clip_position(target)
        return world.resolve_collisions(target)
