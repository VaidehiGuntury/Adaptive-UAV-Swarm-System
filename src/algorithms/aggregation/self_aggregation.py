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
from src.environment.map import FrontierCluster, RegionKey
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
    _step_region_keys: list[RegionKey] = field(default_factory=list)
    _replan_region_history: list[RegionKey] = field(default_factory=list)
    _step_reassignment_count: int = 0
    # Execution-commitment state (per agent):
    #   _target_timeout    - safety-net deadline (elapsed seconds) after
    #                        which an unreachable target is abandoned.
    #   _target_region_key - region_key of the frontier the current
    #                        target came from, used to re-score
    #                        "continuing toward the current target" for
    #                        the measurable-improvement comparison.
    _target_timeout: dict[int, float] = field(default_factory=dict)
    _target_region_key: dict[int, RegionKey] = field(default_factory=dict)

    def begin_step(self) -> None:
        """Reset per-step observational counters (instrumentation only)."""
        self._step_region_keys.clear()
        self._step_reassignment_count = 0

    @property
    def step_reassignment_count(self) -> int:
        """Number of BSA replans executed in the current step."""
        return self._step_reassignment_count

    @property
    def step_region_keys(self) -> list[RegionKey]:
        """Frontier region keys selected during replans this step."""
        return list(self._step_region_keys)

    @property
    def replan_region_history(self) -> list[RegionKey]:
        """Cumulative frontier region keys from all BSA replans (instrumentation)."""
        return list(self._replan_region_history)

    def update(
        self,
        agent: UAV,
        all_agents: list[UAV],
        world: World,
        dt: float,
        communication_range: float | None = None,
    ) -> None:
        """
        Run BSA decision logic with hysteresis-based viewpoint commitment.

        ``communication_range``: forwarded to J_C (Paper 1 Eq. 6) so the
        dispersal repulsion term only considers UAVs within range, matching
        IDEAllocator's partner-eligibility distance basis. None (default)
        preserves unlimited-range behaviour.

        Candidates are re-evaluated on the existing ``replan_interval``
        cadence (Paper 1 §5), but the current target is only *replaced*
        when one of the following holds:
          - no target is currently assigned (bootstrap),
          - the UAV has arrived within ``d_c`` of the target (Paper 1
            Eq. 7's own "very close" threshold — reused here rather than
            adding a new tolerance constant),
          - a safety-net deadline elapses: the time needed to cross the
            entire world at max_speed (guards only against a genuinely
            stuck target, e.g. obstacle-blocked), or
          - the newly re-evaluated best candidate scores strictly higher
            than continuing toward the current target under the same
            cost function (Eqs. 6-10) — i.e. a measurable improvement,
            not merely a different candidate.

        Two simpler alternatives were tried and both regressed coverage
        below the original fixed-timer baseline (58%):
          1. Switching every ``replan_interval`` regardless of progress
             (the original behaviour): max_speed(1.5m/s) x
             replan_interval(2.0s) caps travel at 3.0m/cycle, so most
             selected viewpoints — often tens of metres away — were
             abandoned before being reached (measured: only ~14% of
             cycles got a UAV within sensing_range of its target).
          2. Committing unconditionally until arrival/timeout: this
             over-corrected — a UAV committed to one distant target for
             its entire (now much longer) journey ignores any closer,
             more useful frontier that appears while it travels, since
             BSA no longer re-evaluates until arrival. Measured coverage
             dropped further (to ~41%) because long, stale commitments
             replaced short, frequent (if truncated) redirections.
        Re-evaluating on the same cadence as before but only switching on
        a measurable improvement keeps decisions current without
        discarding an in-progress trajectory for a merely-different one.

        Updates vp_c via ``set_target`` only; allocated region p̃* remains
        stable (still updated on its own cadence by IDE, Paper 1 §4).
        """
        elapsed = self._time_since_replan.get(agent.agent_id, 0.0) + dt
        self._time_since_replan[agent.agent_id] = elapsed

        has_target = agent.assigned_target is not None
        arrived = False
        timed_out = False
        if has_target:
            dist_to_target = float(
                np.linalg.norm(agent.position - agent.assigned_target)
            )
            arrived = dist_to_target <= self.config.d_c
            deadline = self._target_timeout.get(agent.agent_id, float("inf"))
            timed_out = elapsed >= deadline

        due_for_check = elapsed >= self.config.replan_interval
        if has_target and not arrived and not timed_out and not due_for_check:
            return

        self._time_since_replan[agent.agent_id] = 0.0
        self._step_reassignment_count += 1

        clusters = world.map.extract_frontier_clusters()
        candidates = self._generate_candidates(agent, clusters, world)

        world_diagonal = float(np.hypot(world.width, world.height))
        safety_net_deadline = world_diagonal / self.uav_config.max_speed

        if not candidates:
            if not has_target or arrived or timed_out:
                fallback = self._fallback_target(agent, world)
                agent.set_target(fallback)
                self._target_region_key.pop(agent.agent_id, None)
                self._target_timeout[agent.agent_id] = safety_net_deadline
            # else: nothing better available — keep the current target.
            return

        best = max(
            candidates,
            key=lambda c: evaluate_viewpoint_cost(
                c, agent, all_agents, self.config, communication_range=communication_range
            ),
        )

        if has_target and not arrived and not timed_out:
            best_score = evaluate_viewpoint_cost(
                best, agent, all_agents, self.config, communication_range=communication_range
            )
            current = ViewpointCandidate(
                viewpoint=agent.assigned_target,
                yaw=0.0,
                cluster_id=-1,
                is_trail=world.map.is_trail_region(
                    self._target_region_key.get(agent.agent_id, (0, 0))
                ),
                region_key=self._target_region_key.get(agent.agent_id, (0, 0)),
            )
            current_score = evaluate_viewpoint_cost(
                current, agent, all_agents, self.config, communication_range=communication_range
            )
            if best_score <= current_score:
                return  # not a measurable improvement — keep the current target

        agent.set_target(best.viewpoint)
        self._step_region_keys.append(best.region_key)
        self._replan_region_history.append(best.region_key)
        self._target_region_key[agent.agent_id] = best.region_key
        self._target_timeout[agent.agent_id] = safety_net_deadline

        if best.is_trail:
            world.map.mark_cluster_as_trail(best.region_key)

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
                        region_key=cluster.region_key,
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
