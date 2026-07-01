"""
Obstacle lifecycle manager for the Dynamic Environment Extension (SDS §25).

ObstacleManager is the single point of truth for all dynamic obstacles.
It is designed to be owned by ``World`` (Phase 2 integration), but in
Phase 1 it operates as a fully self-contained, independently testable unit.

Responsibilities (SDS §25)
--------------------------
- Maintain the collection of active dynamic obstacles.
- Add and remove obstacles by identifier.
- Update all active obstacles every simulation timestep.
- Answer point-collision queries against all active obstacles.
- Return the nearest active obstacle to a given point.
- Predict future positions of all active obstacles.

Dependency rules (SDS §31)
---------------------------
This module MUST NOT import UAV, Aggregation, SimulationEngine, or Renderer.
It imports only ``dynamic_obstacles`` from the environment package and NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from src.environment.dynamic_obstacles import DynamicObstacle


# ---------------------------------------------------------------------------
# Collision / near-miss result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollisionResult:
    """
    Result of a collision or near-miss query at a single point.

    Attributes
    ----------
    colliding : bool
        True when the query point lies inside at least one active obstacle.
    near_miss : bool
        True when the query point is within the safety margin but outside
        all obstacle boundaries (i.e., *not* colliding but dangerously close).
    obstacle_id : str | None
        Identifier of the nearest obstacle that triggered the result, or
        None when no relevant obstacle was found.
    distance : float
        Signed surface distance to the nearest active obstacle.
        Negative ⟹ inside the obstacle; positive ⟹ outside.
        ``float('inf')`` when no active obstacles exist.
    """

    colliding: bool
    near_miss: bool
    obstacle_id: str | None
    distance: float


@dataclass(frozen=True)
class NearestObstacleResult:
    """
    Result of a nearest-obstacle query.

    Attributes
    ----------
    obstacle : DynamicObstacle | None
        The closest active obstacle, or None when no obstacles are present.
    distance : float
        Signed surface distance to the nearest obstacle boundary [m].
        ``float('inf')`` when no obstacles are present.
    """

    obstacle: DynamicObstacle | None
    distance: float


@dataclass(frozen=True)
class FuturePositionEntry:
    """
    Predicted future position entry for a single obstacle.

    Attributes
    ----------
    obstacle_id : str
        Obstacle identifier.
    predicted_position : NDArray[np.float64]
        Estimated 2-D centre position at the queried future time [m].
    """

    obstacle_id: str
    predicted_position: NDArray[np.float64]


# ---------------------------------------------------------------------------
# ObstacleManager
# ---------------------------------------------------------------------------


@dataclass
class ObstacleManager:
    """
    Central coordinator for the dynamic obstacle collection (SDS §25).

    Internal storage uses an ordered ``dict`` keyed on ``obstacle_id`` to
    guarantee O(1) add, remove, and lookup while preserving insertion order
    for deterministic iteration (important for reproducible experiments).

    Parameters
    ----------
    collision_radius : float
        Surface-distance threshold below which a point is considered to be
        in collision.  A point at surface distance ≤ 0 always collides;
        setting this to a positive value adds a safety margin.
        Defaults to 0.0 (exact boundary contact).
    safety_margin : float
        Surface-distance threshold defining the near-miss zone:
        ``collision_radius < d < safety_margin``.
        Must be greater than or equal to *collision_radius*.
        Defaults to 0.75 [m] (SDS §27 default).

    Raises
    ------
    ValueError
        When ``safety_margin`` is less than ``collision_radius``.
    """

    collision_radius: float = 0.0
    safety_margin: float = 0.75
    _obstacles: dict[str, DynamicObstacle] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.safety_margin < self.collision_radius:
            raise ValueError(
                f"safety_margin ({self.safety_margin}) must be >= "
                f"collision_radius ({self.collision_radius})"
            )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def add_obstacle(self, obstacle: DynamicObstacle) -> None:
        """
        Register a dynamic obstacle with the manager.

        If an obstacle with the same ``obstacle_id`` already exists it is
        silently replaced.  This behaviour supports re-spawning scenarios
        where an obstacle is reset to a new initial state.

        Parameters
        ----------
        obstacle:
            Fully initialised ``DynamicObstacle`` instance.
        """
        self._obstacles[obstacle.obstacle_id] = obstacle

    def remove_obstacle(self, obstacle_id: str) -> DynamicObstacle | None:
        """
        Deregister and return the obstacle identified by *obstacle_id*.

        Parameters
        ----------
        obstacle_id:
            Identifier of the obstacle to remove.

        Returns
        -------
        DynamicObstacle | None
            The removed obstacle, or None if no obstacle with that id exists.
        """
        return self._obstacles.pop(obstacle_id, None)

    def get_obstacle(self, obstacle_id: str) -> DynamicObstacle | None:
        """
        Return the obstacle for the given *obstacle_id* without removing it.

        Parameters
        ----------
        obstacle_id:
            Identifier to look up.

        Returns
        -------
        DynamicObstacle | None
            The obstacle, or None if not present.
        """
        return self._obstacles.get(obstacle_id)

    def obstacle_count(self) -> int:
        """Return the total number of registered obstacles (active and inactive)."""
        return len(self._obstacles)

    def active_obstacle_count(self) -> int:
        """Return the number of obstacles whose ``active`` flag is True."""
        return sum(1 for obs in self._obstacles.values() if obs.active)

    def __iter__(self) -> Iterator[DynamicObstacle]:
        """Iterate over all registered obstacles in insertion order."""
        return iter(self._obstacles.values())

    def __contains__(self, obstacle_id: str) -> bool:
        """Support ``obstacle_id in manager`` membership test."""
        return obstacle_id in self._obstacles

    # ------------------------------------------------------------------
    # Simulation update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Advance every *active* obstacle by one simulation timestep.

        Inactive obstacles (``active == False``) are skipped.  This is the
        only method that mutates obstacle state and should be called once
        per simulation step, before UAV kinematics are updated (SDS §29).

        Parameters
        ----------
        dt:
            Simulation timestep [s].  Must be positive.
        """
        if dt <= 0.0:
            return
        for obstacle in self._obstacles.values():
            if obstacle.active:
                obstacle.update(dt)

    # ------------------------------------------------------------------
    # Collision queries
    # ------------------------------------------------------------------

    def check_collision(
        self,
        point: NDArray[np.float64],
    ) -> CollisionResult:
        """
        Check whether *point* collides with or is near any active obstacle.

        The result encodes three mutually exclusive zones:

        - **Collision zone**: surface distance ≤ ``collision_radius``.
        - **Near-miss zone**: ``collision_radius < d ≤ safety_margin``.
        - **Safe zone**: surface distance > ``safety_margin``.

        Parameters
        ----------
        point:
            2-D query position [m].

        Returns
        -------
        CollisionResult
            Populated result referencing the nearest relevant obstacle.
            When no active obstacles exist, returns a safe result with
            distance ``float('inf')``.
        """
        nearest_id: str | None = None
        nearest_dist = float("inf")

        for obstacle in self._obstacles.values():
            if not obstacle.active:
                continue
            d = obstacle.distance_to(point)
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = obstacle.obstacle_id

        if nearest_id is None:
            return CollisionResult(
                colliding=False,
                near_miss=False,
                obstacle_id=None,
                distance=float("inf"),
            )

        colliding = nearest_dist <= self.collision_radius
        near_miss = (not colliding) and (nearest_dist <= self.safety_margin)

        return CollisionResult(
            colliding=colliding,
            near_miss=near_miss,
            obstacle_id=nearest_id,
            distance=nearest_dist,
        )

    def is_collision(self, point: NDArray[np.float64]) -> bool:
        """
        Return True when *point* lies within the collision zone of any active obstacle.

        Convenience wrapper around ``check_collision`` for callers that only
        need a boolean result.

        Parameters
        ----------
        point:
            2-D query position [m].

        Returns
        -------
        bool
        """
        for obstacle in self._obstacles.values():
            if obstacle.active and obstacle.distance_to(point) <= self.collision_radius:
                return True
        return False

    # ------------------------------------------------------------------
    # Nearest obstacle
    # ------------------------------------------------------------------

    def nearest_obstacle(
        self,
        point: NDArray[np.float64],
    ) -> NearestObstacleResult:
        """
        Return the nearest active obstacle and its signed surface distance.

        Parameters
        ----------
        point:
            2-D query position [m].

        Returns
        -------
        NearestObstacleResult
            Contains the nearest ``DynamicObstacle`` and its surface distance
            [m].  When no active obstacles exist, ``obstacle`` is None and
            ``distance`` is ``float('inf')``.
        """
        nearest_obs: DynamicObstacle | None = None
        nearest_dist = float("inf")

        for obstacle in self._obstacles.values():
            if not obstacle.active:
                continue
            d = obstacle.distance_to(point)
            if d < nearest_dist:
                nearest_dist = d
                nearest_obs = obstacle

        return NearestObstacleResult(obstacle=nearest_obs, distance=nearest_dist)

    # ------------------------------------------------------------------
    # Future position prediction
    # ------------------------------------------------------------------

    def predict_all_positions(
        self,
        time: float,
    ) -> list[FuturePositionEntry]:
        """
        Predict the positions of all active obstacles at a future time *time* [s].

        Uses each obstacle's ``predict_position(time)`` method.  For
        ``LinearObstacle`` this is an exact constant-velocity prediction.
        For ``WaypointObstacle`` and ``RandomWalkObstacle`` it is a
        first-order linear extrapolation from the current velocity — an
        approximation sufficient for short prediction horizons.

        Parameters
        ----------
        time:
            Prediction horizon [s] measured from the *current* state.
            Must be non-negative.

        Returns
        -------
        list[FuturePositionEntry]
            One entry per active obstacle, in insertion order.
        """
        if time < 0.0:
            raise ValueError(f"prediction time must be non-negative, got {time}")

        results: list[FuturePositionEntry] = []
        for obstacle in self._obstacles.values():
            if not obstacle.active:
                continue
            predicted = obstacle.predict_position(time)
            results.append(
                FuturePositionEntry(
                    obstacle_id=obstacle.obstacle_id,
                    predicted_position=predicted,
                )
            )
        return results

    def predict_obstacle_position(
        self,
        obstacle_id: str,
        time: float,
    ) -> NDArray[np.float64] | None:
        """
        Predict the position of a specific obstacle at future time *time* [s].

        Parameters
        ----------
        obstacle_id:
            Identifier of the obstacle to query.
        time:
            Prediction horizon [s].  Must be non-negative.

        Returns
        -------
        NDArray[np.float64] | None
            Predicted 2-D position [m], or None if the obstacle does not exist
            or is inactive.
        """
        obstacle = self._obstacles.get(obstacle_id)
        if obstacle is None or not obstacle.active:
            return None
        return obstacle.predict_position(time)

    # ------------------------------------------------------------------
    # Serialization / diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> list[dict[str, object]]:
        """
        Return serialized state of all registered obstacles.

        Intended for CSV logging and experiment reproducibility.

        Returns
        -------
        list[dict[str, object]]
            One dictionary per obstacle (including inactive ones), in
            insertion order.  Each dictionary is produced by
            ``DynamicObstacle.serialize()``.
        """
        return [obs.serialize() for obs in self._obstacles.values()]

    def __repr__(self) -> str:
        return (
            f"ObstacleManager("
            f"total={self.obstacle_count()}, "
            f"active={self.active_obstacle_count()}, "
            f"collision_radius={self.collision_radius}, "
            f"safety_margin={self.safety_margin})"
        )
