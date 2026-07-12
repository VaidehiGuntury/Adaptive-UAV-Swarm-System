"""
Dynamic obstacle representation and motion models for Phase 1 of the
Dynamic Environment Extension (SDS §22–24).

This module introduces moving obstacles with multiple motion models while
remaining entirely independent of the existing static obstacle pipeline.
It intentionally imports only NumPy and the standard library so that it
can be integrated into World / SimulationEngine in a later phase without
creating circular dependencies.

Motion models implemented
--------------------------
LinearObstacle     — constant velocity (deterministic).
WaypointObstacle   — patrol along a sequence of waypoints.
RandomWalkObstacle — stochastic heading perturbation (Brownian-like).

Dependency rules (SDS §31)
---------------------------
This module MUST NOT import UAV, Aggregation, SimulationEngine, or Renderer.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Motion type discriminator
# ---------------------------------------------------------------------------


class MotionType(Enum):
    """Discriminator tag for the obstacle's motion model."""

    LINEAR = auto()
    WAYPOINT = auto()
    RANDOM_WALK = auto()


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class DynamicObstacle(ABC):
    """
    Abstract base for all moving obstacles.

    State fields (SDS §23)
    ----------------------
    obstacle_id  : str   — unique identifier within the ObstacleManager.
    position     : ndarray[float64, shape(2,)] — current 2-D centre [m].
    velocity     : ndarray[float64, shape(2,)] — current velocity [m/s].
    radius       : float — collision radius [m].
    active       : bool  — when False the obstacle is ignored by queries.
    motion_type  : MotionType — discriminator for the concrete model.

    Subclasses must implement ``update(dt)`` which advances the obstacle
    state by one simulation timestep.  All other public methods are
    implemented here using the ``position`` and ``velocity`` fields.
    """

    def __init__(
        self,
        obstacle_id: str,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        radius: float,
        active: bool = True,
        motion_type: MotionType = MotionType.LINEAR,
    ) -> None:
        if radius <= 0.0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.obstacle_id: str = obstacle_id
        self.position: NDArray[np.float64] = position.astype(np.float64).copy()
        self.velocity: NDArray[np.float64] = velocity.astype(np.float64).copy()
        self.radius: float = float(radius)
        self.active: bool = active
        self.motion_type: MotionType = motion_type

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def update(self, dt: float) -> None:
        """
        Advance obstacle state by one simulation timestep of length *dt* [s].

        Implementations must update ``self.position`` (and ``self.velocity``
        if the model changes speed/direction) in-place.
        """

    # ------------------------------------------------------------------
    # Concrete query methods — derived from position/velocity/radius
    # ------------------------------------------------------------------

    def predict_position(self, time: float) -> NDArray[np.float64]:
        """
        Predict obstacle centre position at absolute future time *time* [s].

        Uses the current velocity as a linear first-order approximation.
        Subclasses may override this for higher-fidelity prediction.

        Parameters
        ----------
        time:
            Duration ahead of the *current* state to predict [s].
            Must be non-negative.

        Returns
        -------
        NDArray[np.float64]
            Predicted 2-D position [m].
        """
        if time < 0.0:
            raise ValueError(f"prediction time must be non-negative, got {time}")
        return self.position + self.velocity * time

    def distance_to(self, point: NDArray[np.float64]) -> float:
        """
        Signed surface distance from *point* to this obstacle's boundary.

        Negative values indicate the point is inside the obstacle.
        This convention is consistent with ``CircularObstacle.distance_to``
        in ``obstacles.py``.

        Parameters
        ----------
        point:
            2-D world position [m].

        Returns
        -------
        float
            Signed distance [m].  Negative ⟹ inside obstacle.
        """
        return float(np.linalg.norm(point - self.position)) - self.radius

    def collides(self, point: NDArray[np.float64]) -> bool:
        """
        Return True when *point* lies at or inside the obstacle boundary.

        Parameters
        ----------
        point:
            2-D world position [m].

        Returns
        -------
        bool
            True ⟹ collision (point inside or on boundary).
        """
        return self.distance_to(point) <= 0.0

    def bounding_circle(self) -> tuple[NDArray[np.float64], float]:
        """
        Return the current bounding circle as (centre, radius).

        Convenience accessor for the renderer and collision prediction layer.

        Returns
        -------
        tuple[NDArray[np.float64], float]
            (centre [m], radius [m]).
        """
        return self.position.copy(), self.radius

    def serialize(self) -> dict[str, object]:
        """
        Serialize obstacle state to a plain dictionary.

        Intended for CSV logging and experiment reproducibility.

        Returns
        -------
        dict[str, object]
            Keys: obstacle_id, motion_type, position_x, position_y,
            velocity_x, velocity_y, radius, active.
        """
        return {
            "obstacle_id": self.obstacle_id,
            "motion_type": self.motion_type.name,
            "position_x": float(self.position[0]),
            "position_y": float(self.position[1]),
            "velocity_x": float(self.velocity[0]),
            "velocity_y": float(self.velocity[1]),
            "radius": self.radius,
            "active": self.active,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.obstacle_id!r}, "
            f"pos=({self.position[0]:.2f}, {self.position[1]:.2f}), "
            f"radius={self.radius:.2f}, active={self.active})"
        )


# ---------------------------------------------------------------------------
# LinearObstacle — constant velocity motion (SDS §24)
# ---------------------------------------------------------------------------


@dataclass
class LinearObstacle(DynamicObstacle):
    """
    Moving obstacle with constant velocity (SDS §24 — Linear Motion).

    Mathematical model
    ------------------
    p(t + Δt) = p(t) + v · Δt

    The velocity remains constant unless the obstacle reaches a world
    boundary, in which case the component normal to the boundary is
    reflected (elastic bounce).  This prevents obstacles from drifting
    off-screen while preserving the deterministic character of the model.

    Parameters
    ----------
    obstacle_id:
        Unique string identifier.
    position:
        Initial 2-D centre [m].
    velocity:
        Constant velocity vector [m/s].
    radius:
        Collision radius [m].
    world_bounds:
        (width, height) of the simulation world [m].  Used for boundary
        reflection.  Defaults to (100.0, 100.0).
    active:
        Whether the obstacle participates in collision queries.
    """

    # ``@dataclass`` generates ``__init__`` only for fields declared here.
    # We delegate to the parent ``__init__`` manually because
    # ``DynamicObstacle`` is not a dataclass (it is abstract).

    def __init__(
        self,
        obstacle_id: str,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        radius: float,
        world_bounds: tuple[float, float] = (100.0, 100.0),
        active: bool = True,
    ) -> None:
        super().__init__(
            obstacle_id=obstacle_id,
            position=position,
            velocity=velocity,
            radius=radius,
            active=active,
            motion_type=MotionType.LINEAR,
        )
        self.world_bounds: tuple[float, float] = world_bounds

    def update(self, dt: float) -> None:
        """
        Advance position by one timestep using constant velocity.

        Reflects velocity components at world boundaries so the obstacle
        remains within [0, width] × [0, height].

        Parameters
        ----------
        dt:
            Simulation timestep [s].  Must be positive.
        """
        if dt <= 0.0:
            return

        self.position = self.position + self.velocity * dt

        width, height = self.world_bounds
        margin = self.radius

        # Reflect at left/right boundaries
        if self.position[0] <= margin:
            self.position[0] = margin
            self.velocity[0] = abs(self.velocity[0])
        elif self.position[0] >= width - margin:
            self.position[0] = width - margin
            self.velocity[0] = -abs(self.velocity[0])

        # Reflect at bottom/top boundaries
        if self.position[1] <= margin:
            self.position[1] = margin
            self.velocity[1] = abs(self.velocity[1])
        elif self.position[1] >= height - margin:
            self.position[1] = height - margin
            self.velocity[1] = -abs(self.velocity[1])


# ---------------------------------------------------------------------------
# WaypointObstacle — patrol along a fixed sequence of waypoints (SDS §24)
# ---------------------------------------------------------------------------


class WaypointObstacle(DynamicObstacle):
    """
    Moving obstacle that follows a closed sequence of waypoints (SDS §24).

    Algorithm
    ---------
    At each timestep the obstacle steers toward the current target waypoint.
    When it arrives within *arrival_tolerance* [m] it advances to the next
    waypoint in the list (wrapping around to index 0 after the last).

    The velocity vector is kept aligned with the direction to the current
    waypoint at the configured *speed*.

    Parameters
    ----------
    obstacle_id:
        Unique string identifier.
    position:
        Initial 2-D centre [m].  Need not lie on a waypoint.
    radius:
        Collision radius [m].
    waypoints:
        Ordered sequence of 2-D world positions [m] forming the patrol route.
        Must contain at least one waypoint.
    speed:
        Constant travel speed [m/s].
    arrival_tolerance:
        Distance threshold for waypoint switching [m].  Defaults to the
        obstacle radius.
    active:
        Whether the obstacle participates in collision queries.

    Raises
    ------
    ValueError
        When ``waypoints`` is empty or ``speed`` is non-positive.
    """

    def __init__(
        self,
        obstacle_id: str,
        position: NDArray[np.float64],
        radius: float,
        waypoints: Sequence[NDArray[np.float64]],
        speed: float,
        arrival_tolerance: float | None = None,
        active: bool = True,
    ) -> None:
        if not waypoints:
            raise ValueError("WaypointObstacle requires at least one waypoint")
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")

        # Compute initial velocity toward first waypoint
        first_wp = np.asarray(waypoints[0], dtype=np.float64)
        direction = first_wp - position.astype(np.float64)
        dist = float(np.linalg.norm(direction))
        initial_velocity = (direction / dist * speed) if dist > 1e-9 else np.zeros(2, dtype=np.float64)

        super().__init__(
            obstacle_id=obstacle_id,
            position=position,
            velocity=initial_velocity,
            radius=radius,
            active=active,
            motion_type=MotionType.WAYPOINT,
        )
        self._waypoints: list[NDArray[np.float64]] = [
            np.asarray(wp, dtype=np.float64).copy() for wp in waypoints
        ]
        self._speed: float = float(speed)
        self._current_index: int = 0
        self._arrival_tolerance: float = (
            float(arrival_tolerance) if arrival_tolerance is not None else self.radius
        )

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def waypoints(self) -> tuple[NDArray[np.float64], ...]:
        """Ordered patrol waypoints (read-only tuple copy)."""
        return tuple(wp.copy() for wp in self._waypoints)

    @property
    def current_waypoint_index(self) -> int:
        """Index of the waypoint the obstacle is currently steering toward."""
        return self._current_index

    @property
    def current_waypoint(self) -> NDArray[np.float64]:
        """Current target waypoint position [m]."""
        return self._waypoints[self._current_index].copy()

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Advance the patrol obstacle by one timestep.

        Steps
        -----
        1. Compute direction vector to the current waypoint.
        2. Set velocity toward that waypoint at the configured speed.
        3. Integrate position.
        4. If the obstacle has reached the waypoint, advance the index.

        Parameters
        ----------
        dt:
            Simulation timestep [s].  Must be positive.
        """
        if dt <= 0.0:
            return

        target = self._waypoints[self._current_index]
        direction = target - self.position
        dist = float(np.linalg.norm(direction))

        if dist <= self._arrival_tolerance:
            # Snap to the waypoint and advance
            self.position = target.copy()
            self._current_index = (self._current_index + 1) % len(self._waypoints)
            target = self._waypoints[self._current_index]
            direction = target - self.position
            dist = float(np.linalg.norm(direction))

        if dist > 1e-9:
            self.velocity = direction / dist * self._speed
        else:
            self.velocity = np.zeros(2, dtype=np.float64)

        self.position = self.position + self.velocity * dt


# ---------------------------------------------------------------------------
# RandomWalkObstacle — stochastic heading perturbation (SDS §24)
# ---------------------------------------------------------------------------


class RandomWalkObstacle(DynamicObstacle):
    """
    Moving obstacle with stochastic heading updates (SDS §24 — Random Walk).

    Algorithm
    ---------
    At each timestep, the current heading is perturbed by Gaussian noise
    scaled by *turn_noise* [rad/s]:

        heading(t + Δt) = heading(t) + N(0, turn_noise) · Δt

    The velocity is then updated from the new heading at the fixed *speed*.
    Boundary reflection is applied using the same elastic model as
    ``LinearObstacle`` to prevent drift outside the world.

    This models pedestrian-like motion where direction is mostly maintained
    but gradually drifts over time.

    Parameters
    ----------
    obstacle_id:
        Unique string identifier.
    position:
        Initial 2-D centre [m].
    radius:
        Collision radius [m].
    speed:
        Travel speed [m/s].  Remains constant; only heading changes.
    heading:
        Initial heading angle [rad].  Defaults to 0.0 (east).
    turn_noise:
        Standard deviation of per-second heading perturbation [rad/s].
    world_bounds:
        (width, height) of the simulation world [m] for boundary reflection.
    seed:
        RNG seed for reproducibility.  None ⟹ non-deterministic.
    active:
        Whether the obstacle participates in collision queries.

    Raises
    ------
    ValueError
        When ``speed`` or ``turn_noise`` is non-positive.
    """

    def __init__(
        self,
        obstacle_id: str,
        position: NDArray[np.float64],
        radius: float,
        speed: float,
        heading: float = 0.0,
        turn_noise: float = 0.3,
        world_bounds: tuple[float, float] = (100.0, 100.0),
        seed: int | None = None,
        active: bool = True,
    ) -> None:
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")
        if turn_noise <= 0.0:
            raise ValueError(f"turn_noise must be positive, got {turn_noise}")

        initial_velocity = speed * np.array(
            [math.cos(heading), math.sin(heading)], dtype=np.float64
        )
        super().__init__(
            obstacle_id=obstacle_id,
            position=position,
            velocity=initial_velocity,
            radius=radius,
            active=active,
            motion_type=MotionType.RANDOM_WALK,
        )
        self._speed: float = float(speed)
        self._heading: float = float(heading)
        self._turn_noise: float = float(turn_noise)
        self.world_bounds: tuple[float, float] = world_bounds
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def heading(self) -> float:
        """Current heading angle [rad], wrapped to (−π, π]."""
        return self._heading

    @property
    def speed(self) -> float:
        """Constant travel speed [m/s]."""
        return self._speed

    @property
    def turn_noise(self) -> float:
        """Heading perturbation standard deviation [rad/s]."""
        return self._turn_noise

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """
        Advance the random-walk obstacle by one timestep.

        Steps
        -----
        1. Draw heading perturbation Δθ ~ N(0, turn_noise · √dt).
        2. Update heading: θ ← wrap(θ + Δθ).
        3. Recompute velocity from heading and fixed speed.
        4. Integrate position.
        5. Reflect at world boundaries.

        The perturbation is scaled by √dt so that the diffusion coefficient
        remains consistent regardless of the simulation timestep (Itô
        convention for Wiener-process-driven heading).

        Parameters
        ----------
        dt:
            Simulation timestep [s].  Must be positive.
        """
        if dt <= 0.0:
            return

        # Heading perturbation (Itô scaling)
        delta_theta = float(self._rng.normal(0.0, self._turn_noise * math.sqrt(dt)))
        self._heading = _wrap_angle(self._heading + delta_theta)

        # Rebuild velocity from heading
        self.velocity = self._speed * np.array(
            [math.cos(self._heading), math.sin(self._heading)], dtype=np.float64
        )

        # Integrate position
        self.position = self.position + self.velocity * dt

        # Boundary reflection — same logic as LinearObstacle
        width, height = self.world_bounds
        margin = self.radius

        if self.position[0] <= margin:
            self.position[0] = margin
            self._heading = _wrap_angle(math.pi - self._heading)
        elif self.position[0] >= width - margin:
            self.position[0] = width - margin
            self._heading = _wrap_angle(math.pi - self._heading)

        if self.position[1] <= margin:
            self.position[1] = margin
            self._heading = _wrap_angle(-self._heading)
        elif self.position[1] >= height - margin:
            self.position[1] = height - margin
            self._heading = _wrap_angle(-self._heading)

        # Re-sync velocity after possible reflection
        self.velocity = self._speed * np.array(
            [math.cos(self._heading), math.sin(self._heading)], dtype=np.float64
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _wrap_angle(angle: float) -> float:
    """Wrap *angle* to the interval (−π, π]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
