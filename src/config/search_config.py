"""
Configuration dataclasses for the Target Search & Tracking extension.

All parameters are YAML-driven. No hardcoded constants.
Maps to configs/simulation.yaml `search:` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TargetSpawnConfig:
    """
    Parameters controlling target population at mission start.

    count_static           -- number of static targets
    count_dynamic          -- number of continuously moving targets (constant velocity)
    count_time_varying     -- number of targets that change state during mission
    count_random_walk      -- number of random-walk targets (direction perturbed each step)
    count_waypoint_patrol  -- number of waypoint-patrol targets (cycle through fixed points)
    spawn_margin           -- minimum distance from world edge for spawning [m]
    min_separation         -- minimum inter-target distance at spawn [m]
    dynamic_speed_min      -- minimum speed of moving targets [m/s]
    dynamic_speed_max      -- maximum speed of moving targets [m/s]
    random_walk_turn_rad   -- max heading perturbation per second for random-walk [rad/s]
    waypoint_count         -- number of waypoints per patrol target
    seed                   -- RNG seed for reproducible target placement
    """

    count_static: int = 3
    count_dynamic: int = 3
    count_time_varying: int = 2
    count_random_walk: int = 0
    count_waypoint_patrol: int = 0
    spawn_margin: float = 5.0
    min_separation: float = 8.0
    dynamic_speed_min: float = 0.3
    dynamic_speed_max: float = 0.8
    random_walk_turn_rad: float = 0.5
    waypoint_count: int = 4
    seed: int = 123


@dataclass(frozen=True)
class DetectionConfig:
    """
    Parameters for the per-UAV detection model.

    detection_radius       -- sensing radius for target detection [m]
                             defaults to UAV sensing_range if 0.0
    base_confidence        -- initial confidence on first detection [0,1]
    confidence_decay_rate  -- per-second confidence decay when not seen [1/s]
    confidence_merge_alpha -- weighted merge factor for multi-UAV observations
    dedup_time_window_s    -- seconds within which duplicate detections are merged
    min_detection_confidence -- confidence threshold to register a detection
    """

    detection_radius: float = 0.0        # 0.0 ? use UAV sensing_range
    base_confidence: float = 0.85
    confidence_decay_rate: float = 0.05
    confidence_merge_alpha: float = 0.6
    dedup_time_window_s: float = 2.0
    min_detection_confidence: float = 0.3


@dataclass(frozen=True)
class PriorityConfig:
    """
    Weights for the multi-factor priority scorer.

    Priority = w_type * type_score
             + w_confidence * confidence
             + w_movement * movement_bonus
             + w_distance * distance_score (inverse)
             + w_urgency * urgency
             + w_freshness * freshness_score

    All weights should sum to 1.0 for a normalized score but are not required to.
    """

    w_type: float = 0.30
    w_confidence: float = 0.20
    w_movement: float = 0.15
    w_distance: float = 0.15
    w_urgency: float = 0.10
    w_freshness: float = 0.10

    # Type-specific base scores (higher = higher priority)
    score_moving_human: float = 1.0
    score_moving_vehicle: float = 0.8
    score_static_human: float = 0.7
    score_static_object: float = 0.4


@dataclass(frozen=True)
class SearchBehaviourConfig:
    """
    Parameters governing individual search behaviour strategies.

    direct_nav_staleness_s  -- max age of last_seen before switching away
                              from DirectNav [s]
    spiral_initial_radius   -- starting radius of spiral search [m]
    spiral_step             -- radius increment per spiral loop [m]
    spiral_max_radius       -- maximum spiral expansion radius [m]
    expanding_step_m        -- expanding search grid step size [m]
    expanding_max_radius    -- maximum expanding search radius [m]
    replan_interval_s       -- search waypoint replan interval [s]
    arrival_threshold_m     -- distance to waypoint considered "arrived" [m]
    loss_timeout_s          -- seconds without detection before LOST transition
    max_recovery_attempts   -- max spiral/expand loops before permanent LOST
    """

    direct_nav_staleness_s: float = 5.0
    spiral_initial_radius: float = 3.0
    spiral_step: float = 2.5
    spiral_max_radius: float = 20.0
    expanding_step_m: float = 4.0
    expanding_max_radius: float = 25.0
    replan_interval_s: float = 1.5
    arrival_threshold_m: float = 1.0
    loss_timeout_s: float = 8.0
    max_recovery_attempts: int = 5


@dataclass(frozen=True)
class AssignmentConfig:
    """
    Parameters for the search target assignment algorithm.

    max_targets_per_uav  -- upper bound on targets assigned to one UAV
    reassignment_on_fail -- whether to reassign on UAV failure detection
    fail_timeout_s       -- seconds without position update = UAV failure
    cost_distance_weight -- weight of travel distance in assignment cost
    cost_priority_weight -- weight of target priority in assignment cost
    """

    max_targets_per_uav: int = 2
    reassignment_on_fail: bool = True
    fail_timeout_s: float = 15.0
    cost_distance_weight: float = 0.6
    cost_priority_weight: float = 0.4


@dataclass(frozen=True)
class TrackingConfig:
    """
    Parameters for the target tracking subsystem.

    required_tracking_duration_s -- seconds of continuous tracking to
                                    mark a target COMPLETED
    history_max_length           -- max position history entries per target
    reacquisition_search_radius  -- search radius after LOST transition [m]
    visibility_sample_interval_s -- interval for logging visibility samples [s]
    """

    required_tracking_duration_s: float = 10.0
    history_max_length: int = 200
    reacquisition_search_radius: float = 12.0
    visibility_sample_interval_s: float = 1.0


@dataclass(frozen=True)
class HandoverConfig:
    """
    Parameters for multi-UAV handover logic.

    handover_enabled          -- whether to perform proximity-based handovers
    handover_distance_ratio   -- handover triggers when another UAV is within
                                (handover_distance_ratio ? current_dist) of target
    handover_hysteresis_s     -- minimum seconds between handovers for same target
    """

    handover_enabled: bool = True
    handover_distance_ratio: float = 0.6
    handover_hysteresis_s: float = 10.0


@dataclass(frozen=True)
class MissionConfig:
    """
    Parameters for mission phase transitions.

    exploration_completion_threshold -- explored_fraction to trigger
                                       search phase [0,1]
    search_timeout_s                 -- hard timeout for search phase [s]
    min_coverage_before_search       -- minimum coverage fraction required
                                       before search can begin
    """

    exploration_completion_threshold: float = 0.85
    search_timeout_s: float = 300.0
    min_coverage_before_search: float = 0.70


@dataclass(frozen=True)
class SearchConfig:
    """
    Root search configuration bundling all sub-configs.

    Optional in SimulationConfig -- absence disables search extension entirely.
    """

    targets: TargetSpawnConfig = field(default_factory=TargetSpawnConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    priority: PriorityConfig = field(default_factory=PriorityConfig)
    behaviour: SearchBehaviourConfig = field(default_factory=SearchBehaviourConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    enabled: bool = True
