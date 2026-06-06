"""Paper 1 bio-inspired self-aggregation (BSA) module."""

from src.aggregation.fitness_functions import ViewpointCandidate, evaluate_viewpoint_cost
from src.aggregation.self_aggregation import SelfAggregationController

__all__ = [
    "SelfAggregationController",
    "ViewpointCandidate",
    "evaluate_viewpoint_cost",
]
