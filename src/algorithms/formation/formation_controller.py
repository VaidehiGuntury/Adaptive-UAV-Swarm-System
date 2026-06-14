"""
Centralized formation acquisition + distributed consensus composition (Paper 2).

Mission flow: BSA aggregation → compactness threshold → WEDGE slot tracking
with optional neighbor consensus correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.algorithms.formation.acquisition_metrics import (
    FormationAcquisitionMetrics,
    compute_acquisition_metrics,
)
from src.algorithms.formation.comm_refresh import refresh_formation_comm_graph
from src.algorithms.formation.consensus_controller import (
    ConsensusConfig,
    compute_consensus_correction,
    follower_neighbor_ids,
    scaled_consensus_command,
)
from src.algorithms.formation.consensus_metrics import (
    ConsensusMetricsSnapshot,
    compute_consensus_metrics,
)
from src.algorithms.formation.follower_state import compute_follower_slot_state
from src.algorithms.formation.formation_state import FormationState
from src.algorithms.formation.formation_types import (
    DEFAULT_FORMATION_SPACING,
    DEFAULT_FORMATION_TYPE,
    DEFAULT_SLOT_TOLERANCE,
    FormationType,
)
from src.algorithms.formation.leader_agent import leader_agent_from_pose
from src.algorithms.formation.slot_assignment import assign_formation_slots
from src.algorithms.formation.velocity_control import clamp_velocity, proportional_slot_command
from src.environment.formation_spec import FormationSpec
from src.environment.world import World

Vector2 = NDArray[np.float64]


@dataclass
class FormationAcquisitionConfig:
    """Parameters for formation acquisition and slot tracking."""

    proportional_gain: float = 0.8
    dead_zone: float = 0.15
    activation_mean_pairwise: float = 10.0
    formation_type: FormationType = DEFAULT_FORMATION_TYPE
    formation_spacing: float = DEFAULT_FORMATION_SPACING
    slot_tolerance: float = DEFAULT_SLOT_TOLERANCE


@dataclass
class FormationAcquisitionController:
    """
    Formation acquisition controller with optional consensus extension.

    Slot tracking is centralized via leader frame; consensus uses local neighbors.
    """

    config: FormationAcquisitionConfig
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)
    is_active: bool = False
    activated_at_time_s: float | None = None
    formation_convergence_time_s: float | None = None
    formation_state: FormationState | None = None
    last_acquisition_metrics: FormationAcquisitionMetrics | None = None
    last_consensus_metrics: ConsensusMetricsSnapshot | None = None
    scaled_consensus_commands: dict[int, Vector2] = field(default_factory=dict)
    neighbor_map: dict[int, tuple[int, ...]] = field(default_factory=dict)
    isolated_followers: frozenset[int] = frozenset()

    @property
    def leader_id(self) -> int | None:
        if self.formation_state is None:
            return None
        return self.formation_state.leader_id

    @property
    def consensus_active(self) -> bool:
        return self.is_active and self.consensus_config.enabled

    def is_follower(self, agent_id: int) -> bool:
        if not self.is_active or self.formation_state is None:
            return False
        return agent_id in self.formation_state.active_members and agent_id != self.formation_state.leader_id

    def update_activation(
        self,
        agents: list[UAV],
        world: World,
        mean_pairwise_distance: float,
        time_s: float,
    ) -> None:
        """Latch formation acquisition when swarm compactness threshold is met."""
        if self.is_active:
            return
        if mean_pairwise_distance >= self.config.activation_mean_pairwise:
            return
        self._activate(agents, world, time_s)

    def _activate(self, agents: list[UAV], world: World, time_s: float) -> None:
        leader_id = min(agent.agent_id for agent in agents)
        leader = next(agent for agent in agents if agent.agent_id == leader_id)
        self.formation_state = assign_formation_slots(
            [agent.agent_id for agent in agents],
            leader_id=leader_id,
            formation_type=self.config.formation_type,
            spacing=self.config.formation_spacing,
            leader_heading=leader.heading,
        )
        self.formation_state.slot_tolerance = self.config.slot_tolerance
        world.formation_states = [self.formation_state]
        world.formation_specs = [
            FormationSpec.from_state(0, self.formation_state, spacing=self.config.formation_spacing)
        ]
        self.is_active = True
        self.activated_at_time_s = time_s

    def apply_follower_control(
        self,
        agents: list[UAV],
        world: World,
        dt: float,
        communication_range: float,
        time_s: float,
    ) -> None:
        """Apply u_total = u_slot + k_consensus * u_consensus to followers."""
        if not self.is_active or self.formation_state is None:
            return

        state = self.formation_state
        leader_uav = next((a for a in agents if a.agent_id == state.leader_id), None)
        if leader_uav is None:
            return

        state.leader_heading = float(leader_uav.heading)
        leader = leader_agent_from_pose(
            agent_id=leader_uav.agent_id,
            position=leader_uav.position,
            heading=leader_uav.heading,
        )

        formation_agents = [a for a in agents if a.agent_id in state.active_members]
        graph = refresh_formation_comm_graph(world, formation_agents, communication_range)

        positions = {agent.agent_id: agent.position for agent in agents}
        follower_ids = frozenset(
            aid for aid in state.active_members if aid != state.leader_id
        )

        self.neighbor_map = {}
        self.scaled_consensus_commands = {}
        follower_states = []
        isolated: set[int] = set()

        for agent in agents:
            if not self.is_follower(agent.agent_id):
                continue

            follower_state = compute_follower_slot_state(
                leader, state, agent.agent_id, agent.position
            )
            if follower_state is None:
                continue
            follower_states.append(follower_state)

            slot_error = follower_state.world_slot_position - agent.position
            u_slot = proportional_slot_command(
                slot_error,
                self.config.proportional_gain,
                self.config.dead_zone,
            )

            neighbors = follower_neighbor_ids(graph, agent.agent_id, follower_ids)
            self.neighbor_map[agent.agent_id] = neighbors
            if not neighbors:
                isolated.add(agent.agent_id)

            u_consensus_raw = compute_consensus_correction(
                agent.agent_id, positions, neighbors
            )
            u_consensus = (
                scaled_consensus_command(u_consensus_raw, self.consensus_config.consensus_gain)
                if self.consensus_config.enabled
                else np.zeros(2, dtype=np.float64)
            )
            self.scaled_consensus_commands[agent.agent_id] = u_consensus

            u_total = u_slot + u_consensus
            velocity = clamp_velocity(u_total, agent.max_speed)
            agent.apply_velocity_command(velocity, dt)

        self.isolated_followers = frozenset(isolated)

        if (
            self.formation_convergence_time_s is None
            and follower_states
            and all(f.is_occupied for f in follower_states)
        ):
            self.formation_convergence_time_s = time_s

        self.last_acquisition_metrics = compute_acquisition_metrics(
            tuple(follower_states),
            len(state.active_members),
            leader_slot_occupied=True,
            convergence_time_s=self.formation_convergence_time_s,
        )

        self.last_consensus_metrics = compute_consensus_metrics(
            follower_ids,
            graph,
            positions,
            self.scaled_consensus_commands,
            self.neighbor_map,
        )
