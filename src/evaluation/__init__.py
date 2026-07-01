"""Experiment logging and observational exploration metrics."""

from src.evaluation.exploration_metrics import (
    frontier_reuse_frequency,
    mean_target_separation,
    revisit_ratio,
)
from src.evaluation.dynamic_environment_metrics import (
    InteractionRecord,
    average_obstacle_speed,
    blocked_path_events,
    collision_count,
    coverage_degradation,
    mission_completion_time,
    near_miss_count,
    obstacle_encounters,
)

__all__ = [
    # Paper 1 exploration metrics
    "frontier_reuse_frequency",
    "mean_target_separation",
    "revisit_ratio",
    # Dynamic environment metrics
    "InteractionRecord",
    "average_obstacle_speed",
    "blocked_path_events",
    "collision_count",
    "coverage_degradation",
    "mission_completion_time",
    "near_miss_count",
    "obstacle_encounters",
]
