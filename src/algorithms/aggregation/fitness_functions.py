"""
BSA fitness functions from Paper 1 Eqs. (6)–(10).

Each function maps directly to the DEBS viewpoint selection cost terms:
  J_C  — bio-inspired self-aggregation cost (Eq. 6)
  U_a  — pairwise utility (Eq. 7–8)
  J_V  — turning / navigation cost (Eq. 9)
  J_L  — frontier / trail penalty (Eq. 10)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.config.loader import AggregationConfig
from src.environment.map import RegionKey


def allocation_center(agent: UAV) -> NDArray[np.float64] | None:
    """Allocated mission centre p̃_i* (Paper 1); falls back to navigation target if unset."""
    if agent.assigned_region is not None:
        return agent.assigned_region.center
    return agent.assigned_target


@dataclass(frozen=True)
class ViewpointCandidate:
    """
    Candidate viewpoint ξ_c = {vp_c, vr_c} (Paper 1 Sec. 5).

    vr_c (yaw) is retained for future 3D integration; 2D simulation uses vp_c.
    """

    viewpoint: NDArray[np.float64]
    yaw: float
    cluster_id: int
    is_trail: bool
    region_key: RegionKey = (0, 0)


def compute_lambda_beta(config: AggregationConfig) -> float:
    """Continuity scaling factor λ_β (Paper 1 Eq. 8)."""
    d_c = config.d_c
    d_0 = config.d_0
    if d_0 <= d_c:
        raise ValueError("AggregationConfig requires d_0 > d_c for Eq. (8).")
    numerator = config.k_a * (d_c - d_0) ** 2 * np.sqrt(d_c * d_0)
    denominator = np.sqrt(d_0) - np.sqrt(d_c)
    return float(numerator / denominator)


def utility_u_a(
    point_a: NDArray[np.float64],
    point_b: NDArray[np.float64],
    config: AggregationConfig,
    lambda_beta: float | None = None,
) -> float:
    """
    Utility U_a(p_A, p_B) (Paper 1 Eq. 7).

    Piecewise attraction/repulsion shaping distance between two positions.
    """
    d_ab = float(np.linalg.norm(point_a - point_b))
    d_c = config.d_c
    d_0 = config.d_0
    k_a = config.k_a

    if d_ab >= d_0:
        return 0.0
    if d_ab < d_c:
        beta = lambda_beta if lambda_beta is not None else compute_lambda_beta(config)
        if d_ab <= 1e-12:
            return beta * (1.0 / np.sqrt(d_c) - 1.0 / np.sqrt(d_0))
        return float(beta * (1.0 / np.sqrt(d_ab) - 1.0 / np.sqrt(d_0)))
    return float(k_a * (d_ab - d_0) ** 2)


def aggregation_utility_j_c(
    candidate: ViewpointCandidate,
    agent: UAV,
    all_agents: list[UAV],
    config: AggregationConfig,
    lambda_beta: float | None = None,
) -> float:
    """
    Bio-inspired self-aggregation utility J_C (Paper 1 Eq. 6).

    Higher values attract toward the own allocated target and repel from
    others' targets and positions. Viewpoint selection maximizes this term.
    """
    vp_c = candidate.viewpoint
    own_allocated = allocation_center(agent)
    if own_allocated is None:
        return float("-inf")

    utility = utility_u_a(vp_c, own_allocated, config, lambda_beta)

    for other in all_agents:
        if other.agent_id == agent.agent_id:
            continue
        other_allocated = allocation_center(other)
        if other_allocated is not None:
            utility -= utility_u_a(vp_c, other_allocated, config, lambda_beta)
        utility += utility_u_a(vp_c, other.position, config, lambda_beta)

    return float(utility)


# Backward-compatible alias used in tests and documentation.
aggregation_cost_j_c = aggregation_utility_j_c


def turning_cost_j_v(
    candidate: ViewpointCandidate,
    agent: UAV,
    config: AggregationConfig,
) -> float:
    """
    Turning / navigation cost (Paper 1 Eq. 9).

    Eq. (9) uses a·cos(θ). For minimization we apply a·(1 - cos(θ)) so aligned
    headings incur lower cost, matching the paper's intent to penalize turns.
    """
    direction = candidate.viewpoint - agent.position
    dist = float(np.linalg.norm(direction))
    speed = float(np.linalg.norm(agent.velocity))
    if dist < 1e-9 or speed < 1e-9:
        return 0.0

    dir_hat = direction / dist
    vel_hat = agent.velocity / speed
    cos_theta = float(np.clip(np.dot(vel_hat, dir_hat), -1.0, 1.0))
    return float(config.turn_cost_weight * (1.0 - cos_theta))


def trail_penalty_j_l(candidate: ViewpointCandidate, config: AggregationConfig) -> float:
    """Frontier / trail penalty J_L(c) (Paper 1 Eq. 10)."""
    if candidate.is_trail:
        return float(config.trail_penalty)
    return 0.0


def evaluate_viewpoint_cost(
    candidate: ViewpointCandidate,
    agent: UAV,
    all_agents: list[UAV],
    config: AggregationConfig,
    lambda_beta: float | None = None,
) -> float:
    """
    Composite viewpoint score for argmax selection (Paper 1 Sec. 5).

    Eq. (6) is a utility (maximize); Eqs. (9)–(10) are penalties (minimize):

        score = w_C·J_C − w_V·J_V − w_L·J_L
    """
    beta = lambda_beta if lambda_beta is not None else compute_lambda_beta(config)
    j_c = aggregation_utility_j_c(candidate, agent, all_agents, config, beta)
    j_v = turning_cost_j_v(candidate, agent, config)
    j_l = trail_penalty_j_l(candidate, config)
    return float(
        config.cluster_penalty_weight * j_c
        - config.turn_penalty_weight * j_v
        - config.trail_penalty_weight * j_l
    )
