"""
Deprecated package — use src.algorithms.aggregation.

Retained for backward-compatible imports during the Week 1.5 transition.
"""

from src.algorithms.aggregation import (
    SelfAggregationController,
    ViewpointCandidate,
    evaluate_viewpoint_cost,
)

__all__ = [
    "SelfAggregationController",
    "ViewpointCandidate",
    "evaluate_viewpoint_cost",
]
