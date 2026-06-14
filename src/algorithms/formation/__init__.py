"""
Paper 2 formation infrastructure and controllers.

Centralized slot acquisition with optional distributed consensus correction.
APF and full Paper 2 Eqs. (6–8) dynamics are deferred.
"""

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
from src.algorithms.formation.follower_state import (
    FollowerSlotState,
    compute_follower_slot_state,
    compute_follower_slot_states,
)
from src.algorithms.formation.formation_controller import (
    FormationAcquisitionConfig,
    FormationAcquisitionController,
)
from src.algorithms.formation.formation_metrics import (
    FormationMetricsSnapshot,
    compute_formation_metrics,
)
from src.algorithms.formation.formation_state import (
    FormationState,
    compute_formation_centroid,
)
from src.algorithms.formation.formation_types import (
    DEFAULT_FORMATION_SPACING,
    DEFAULT_FORMATION_TYPE,
    DEFAULT_SLOT_TOLERANCE,
    FormationDefinition,
    FormationType,
    get_formation_definition,
    is_slot_occupied_by_distance,
    register_formation_definition,
    rotate_offset,
    transform_to_local,
    transform_to_world,
)
from src.algorithms.formation.group_snapshot import (
    FormationGroupSnapshot,
    build_formation_group_snapshot,
    build_formation_group_snapshots,
)
from src.algorithms.formation.leader_agent import LeaderAgent, leader_agent_from_pose
from src.algorithms.formation.slot_assignment import assign_formation_slots
from src.algorithms.formation.velocity_control import clamp_velocity, proportional_slot_command

__all__ = [
    "DEFAULT_FORMATION_SPACING",
    "DEFAULT_FORMATION_TYPE",
    "DEFAULT_SLOT_TOLERANCE",
    "ConsensusConfig",
    "ConsensusMetricsSnapshot",
    "FormationAcquisitionConfig",
    "FormationAcquisitionController",
    "FormationAcquisitionMetrics",
    "FormationDefinition",
    "FormationGroupSnapshot",
    "FormationMetricsSnapshot",
    "FormationState",
    "FormationType",
    "FollowerSlotState",
    "LeaderAgent",
    "assign_formation_slots",
    "build_formation_group_snapshot",
    "build_formation_group_snapshots",
    "clamp_velocity",
    "compute_acquisition_metrics",
    "compute_consensus_correction",
    "compute_consensus_metrics",
    "compute_follower_slot_state",
    "compute_follower_slot_states",
    "compute_formation_centroid",
    "compute_formation_metrics",
    "follower_neighbor_ids",
    "get_formation_definition",
    "is_slot_occupied_by_distance",
    "leader_agent_from_pose",
    "proportional_slot_command",
    "refresh_formation_comm_graph",
    "register_formation_definition",
    "rotate_offset",
    "scaled_consensus_command",
    "transform_to_local",
    "transform_to_world",
]
