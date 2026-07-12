"""
Dynamic environment evaluation metrics (SDS §34).

All functions are pure — they accept data structures and return scalar values
or simple aggregates.  They never modify simulation state, never import
renderer modules, and never import aggregation or BSA logic.

These metrics complement the existing Paper 1 exploration metrics in
``exploration_metrics.py`` without replacing or modifying them.

Metrics implemented (SDS §34)
------------------------------
collision_count           — SDS §34.1
near_miss_count           — SDS §34.2
obstacle_encounters       — SDS §34.3
coverage_degradation      — SDS §34.4
mission_completion_time   — SDS §34.5
blocked_path_events       — SDS §34.6
average_obstacle_speed    — SDS §34.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.environment.obstacle_manager import CollisionResult, ObstacleManager


# ---------------------------------------------------------------------------
# Supporting data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteractionRecord:
    """
    Record of a single UAV–obstacle interaction at one simulation timestep.

    Attributes
    ----------
    timestep : int
        Simulation step at which the interaction was observed.
    time_s : float
        Simulation time [s] at which the interaction was observed.
    agent_id : int
        Identifier of the UAV involved.
    result : CollisionResult
        Full collision query result containing zone classification and distance.
    """

    timestep: int
    time_s: float
    agent_id: int
    result: CollisionResult


# ---------------------------------------------------------------------------
# SDS §34.1 — Collision Count
# ---------------------------------------------------------------------------


def collision_count(interactions: Sequence[InteractionRecord]) -> int:
    """
    Count the total number of physical collisions in the interaction log.

    A collision is defined as ``result.colliding == True``, which corresponds
    to the condition d ≤ r_collision (SDS §34.1 Eq.).

    Parameters
    ----------
    interactions:
        Sequence of ``InteractionRecord`` objects collected during a mission.

    Returns
    -------
    int
        Total collision count across all UAVs and timesteps.
    """
    return sum(1 for record in interactions if record.result.colliding)


# ---------------------------------------------------------------------------
# SDS §34.2 — Near-Miss Count
# ---------------------------------------------------------------------------


def near_miss_count(interactions: Sequence[InteractionRecord]) -> int:
    """
    Count interactions that entered the safety margin without collision.

    A near miss satisfies ``result.near_miss == True``, which corresponds to
    r_collision < d ≤ r_safety (SDS §34.2).

    Parameters
    ----------
    interactions:
        Sequence of ``InteractionRecord`` objects collected during a mission.

    Returns
    -------
    int
        Total near-miss count across all UAVs and timesteps.
    """
    return sum(1 for record in interactions if record.result.near_miss)


# ---------------------------------------------------------------------------
# SDS §34.3 — Obstacle Encounter Count
# ---------------------------------------------------------------------------


def obstacle_encounters(
    interactions: Sequence[InteractionRecord],
    cooldown_steps: int = 10,
) -> int:
    """
    Count unique obstacle encounters across the UAV fleet.

    Repeated interaction with the same (agent_id, obstacle_id) pair within
    *cooldown_steps* timesteps counts as a single encounter.  This prevents
    a UAV that moves slowly past an obstacle from generating many spurious
    events (SDS §34.3).

    Parameters
    ----------
    interactions:
        Sequence of ``InteractionRecord`` objects.  May be unordered; the
        function sorts by timestep internally.
    cooldown_steps:
        Minimum gap between two records involving the same agent and obstacle
        to count as separate encounters.  Defaults to 10 steps.

    Returns
    -------
    int
        Number of unique encounters satisfying the cooldown criterion.
    """
    # Key: (agent_id, obstacle_id) → last timestep this pair was counted
    last_seen: dict[tuple[int, str], int] = {}
    count = 0

    # Filter to records where an obstacle was involved
    relevant = [r for r in interactions if r.result.obstacle_id is not None]
    # Sort by timestep for deterministic cooldown behaviour
    relevant.sort(key=lambda r: r.timestep)

    for record in relevant:
        if record.result.obstacle_id is None:
            continue
        key = (record.agent_id, record.result.obstacle_id)
        last_step = last_seen.get(key, -(cooldown_steps + 1))
        if record.timestep - last_step >= cooldown_steps:
            count += 1
            last_seen[key] = record.timestep

    return count


# ---------------------------------------------------------------------------
# SDS §34.4 — Coverage Degradation
# ---------------------------------------------------------------------------


def coverage_degradation(
    static_coverage: float,
    dynamic_coverage: float,
) -> float:
    """
    Compute the reduction in exploration coverage caused by dynamic obstacles.

    CoverageLoss = Coverage_static − Coverage_dynamic  (SDS §34.4)

    A positive value indicates the dynamic environment caused lower coverage.
    A negative value indicates the dynamic environment unexpectedly increased
    coverage (this is possible if obstacles pushed UAVs into unexplored areas).

    Parameters
    ----------
    static_coverage:
        Final explored fraction achieved in the equivalent static environment.
        In [0, 1].
    dynamic_coverage:
        Final explored fraction achieved with dynamic obstacles active.
        In [0, 1].

    Returns
    -------
    float
        Signed coverage loss.  Positive ⟹ degradation; negative ⟹ gain.
    """
    return float(static_coverage - dynamic_coverage)


# ---------------------------------------------------------------------------
# SDS §34.5 — Mission Completion Time
# ---------------------------------------------------------------------------


def mission_completion_time(
    time_series: Sequence[tuple[float, float]],
    coverage_threshold: float = 0.95,
) -> float | None:
    """
    Find the elapsed time at which exploration first reaches *coverage_threshold*.

    Parameters
    ----------
    time_series:
        Ordered sequence of ``(time_s, explored_fraction)`` pairs from the
        simulation metrics history.
    coverage_threshold:
        Coverage level at which the mission is considered complete.
        Defaults to 0.95 (95 %).

    Returns
    -------
    float | None
        Elapsed mission time [s] at first reaching the threshold, or None if
        the threshold was never reached during the recorded mission.
    """
    for time_s, fraction in time_series:
        if fraction >= coverage_threshold:
            return float(time_s)
    return None


# ---------------------------------------------------------------------------
# SDS §34.6 — Blocked Path Events
# ---------------------------------------------------------------------------


def blocked_path_events(
    target_histories: Sequence[Sequence[NDArray[np.float64] | None]],
) -> int:
    """
    Count the number of times a UAV's assigned target changed during the mission.

    A blocked path event is defined as a target reassignment — a change in
    ``assigned_target`` between consecutive timesteps for any UAV (SDS §34.6).
    This is a conservative proxy for path blocking: if an obstacle forces a
    replan, the target changes.

    Parameters
    ----------
    target_histories:
        One inner sequence per UAV, ordered by timestep.  Each element is
        either a 2-D ``NDArray[np.float64]`` (the assigned target position)
        or ``None`` (no target assigned).

    Returns
    -------
    int
        Fleet-wide count of target reassignments across the full mission.
    """
    events = 0
    for history in target_histories:
        prev: NDArray[np.float64] | None = None
        for target in history:
            if prev is not None and target is not None:
                if not np.allclose(prev, target, atol=1e-6):
                    events += 1
            elif prev is None and target is not None:
                pass  # first assignment is not a reassignment
            prev = target
    return events


# ---------------------------------------------------------------------------
# SDS §34.7 — Average Obstacle Speed
# ---------------------------------------------------------------------------


def average_obstacle_speed(manager: ObstacleManager) -> float:
    """
    Compute the mean travel speed of all active dynamic obstacles.

    v̄ = (1/N) Σ ||v_i||  (SDS §34.7)

    Parameters
    ----------
    manager:
        The ``ObstacleManager`` owning the active obstacle collection.

    Returns
    -------
    float
        Mean speed [m/s].  Returns 0.0 when no active obstacles exist.
    """
    speeds: list[float] = []
    for obstacle in manager:
        if obstacle.active:
            speeds.append(float(np.linalg.norm(obstacle.velocity)))
    if not speeds:
        return 0.0
    return float(np.mean(speeds))
