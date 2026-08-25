"""
Target data model for the Search & Tracking extension.

Defines Target objects, type/status enumerations, detection events,
and tracking records. Pure data -- no algorithm logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
from numpy.typing import NDArray


class TargetType(Enum):
    """
    Target classification used by the priority scorer.

    STATIC          -- never moves
    DYNAMIC         -- constant-velocity motion with boundary bounce
    TIME_VARYING    -- changes velocity on a schedule
    RANDOM_WALK     -- random-walk motion (direction perturbed each tick)
    WAYPOINT_PATROL -- cycles through a fixed list of waypoints
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    TIME_VARYING = "time_varying"
    RANDOM_WALK = "random_walk"
    WAYPOINT_PATROL = "waypoint_patrol"


class TargetStatus(Enum):
    """
    Target lifecycle state machine.

    UNDISCOVERED ? DETECTED ? ASSIGNED ? SEARCHING ? TRACKING ? COMPLETED
                                                    ? LOST ?
    """

    UNDISCOVERED = "undiscovered"
    DETECTED = "detected"
    ASSIGNED = "assigned"
    SEARCHING = "searching"
    TRACKING = "tracking"
    LOST = "lost"
    COMPLETED = "completed"


@dataclass
class Target:
    """
    Represents a single search target in the environment.

    Fields
    ------
    target_id        : unique integer identifier
    target_type      : TargetType enum
    position         : current 2D world position [m]
    velocity         : current 2D velocity [m/s] (zero for static)
    confidence       : detection confidence [0, 1]
    priority         : computed numeric priority score (higher = more urgent)
    status           : TargetStatus lifecycle state
    creation_time    : simulation time when target was spawned [s]
    detection_time   : simulation time of first detection (None if undiscovered)
    last_seen        : simulation time of most recent detection (None if undiscovered)
    last_seen_position : world position at last observation
    assigned_uav     : agent_id of assigned UAV (None if unassigned)
    tracking_history : ordered list of (time_s, position) observation tuples
    is_moving        : True if target currently has non-zero velocity
    state_schedule   : for TIME_VARYING: list of (time_s, new_velocity) events
    """

    target_id: int
    target_type: TargetType
    position: NDArray[np.float64]
    velocity: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    confidence: float = 0.0
    priority: float = 0.0
    status: TargetStatus = TargetStatus.UNDISCOVERED
    creation_time: float = 0.0
    detection_time: Optional[float] = None
    last_seen: Optional[float] = None
    last_seen_position: Optional[NDArray[np.float64]] = None
    assigned_uav: Optional[int] = None
    tracking_history: list[tuple[float, NDArray[np.float64]]] = field(
        default_factory=list
    )
    is_moving: bool = False
    # TIME_VARYING schedule: list of (trigger_time_s, new_velocity_array)
    state_schedule: list[tuple[float, NDArray[np.float64]]] = field(
        default_factory=list
    )
    # RANDOM_WALK: max heading perturbation per second [rad/s]
    random_walk_turn_rad: float = 0.5
    # WAYPOINT_PATROL: ordered list of waypoints + current index
    patrol_waypoints: list[NDArray[np.float64]] = field(default_factory=list)
    patrol_index: int = 0
    patrol_arrival_radius: float = 2.0
    # Speed magnitude (used by RANDOM_WALK and WAYPOINT_PATROL)
    speed: float = 0.0
    # Prediction: last two observations for linear extrapolation
    predicted_position: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.is_moving = float(np.linalg.norm(self.velocity)) > 1e-6

    def update_position(
        self,
        dt: float,
        world_width: float,
        world_height: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Advance target position by dt seconds using the appropriate motion model.

        DYNAMIC         -- constant velocity with boundary bounce
        RANDOM_WALK     -- velocity direction perturbed by Gaussian noise each tick
        WAYPOINT_PATROL -- steer toward next waypoint at fixed speed, cycle on arrival
        TIME_VARYING    -- same as DYNAMIC (velocity set by schedule)
        """
        if not self.is_moving:
            return

        if self.target_type == TargetType.RANDOM_WALK:
            self._update_random_walk(dt, world_width, world_height, rng)
        elif self.target_type == TargetType.WAYPOINT_PATROL:
            self._update_waypoint_patrol(dt, world_width, world_height)
        else:
            # DYNAMIC and TIME_VARYING: constant-velocity with bounce
            self._update_constant_velocity(dt, world_width, world_height)

        # Update linear-extrapolation prediction
        self._update_prediction(dt)

    def _update_constant_velocity(
        self, dt: float, world_width: float, world_height: float
    ) -> None:
        """Constant velocity motion with boundary bounce."""
        new_pos = self.position + self.velocity * dt
        if new_pos[0] <= 0.0 or new_pos[0] >= world_width:
            self.velocity = self.velocity * np.array([-1.0, 1.0])
            new_pos[0] = float(np.clip(new_pos[0], 0.5, world_width - 0.5))
        if new_pos[1] <= 0.0 or new_pos[1] >= world_height:
            self.velocity = self.velocity * np.array([1.0, -1.0])
            new_pos[1] = float(np.clip(new_pos[1], 0.5, world_height - 0.5))
        self.position = new_pos

    def _update_random_walk(
        self,
        dt: float,
        world_width: float,
        world_height: float,
        rng: Optional[np.random.Generator],
    ) -> None:
        """
        Random walk: perturb heading by Gaussian noise scaled by random_walk_turn_rad.

        Speed magnitude is preserved (set at spawn via self.speed).
        """
        _rng = rng if rng is not None else np.random.default_rng()
        current_heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
        turn = float(_rng.normal(0.0, self.random_walk_turn_rad * dt))
        new_heading = current_heading + turn
        spd = self.speed if self.speed > 1e-9 else float(np.linalg.norm(self.velocity))
        self.velocity = spd * np.array([np.cos(new_heading), np.sin(new_heading)], dtype=np.float64)
        new_pos = self.position + self.velocity * dt
        if new_pos[0] <= 0.5 or new_pos[0] >= world_width - 0.5:
            self.velocity[0] *= -1.0
            new_pos[0] = float(np.clip(new_pos[0], 0.5, world_width - 0.5))
        if new_pos[1] <= 0.5 or new_pos[1] >= world_height - 0.5:
            self.velocity[1] *= -1.0
            new_pos[1] = float(np.clip(new_pos[1], 0.5, world_height - 0.5))
        self.position = new_pos

    def _update_waypoint_patrol(
        self, dt: float, world_width: float, world_height: float
    ) -> None:
        """
        Waypoint patrol: steer toward next waypoint at fixed speed.

        On arrival (within patrol_arrival_radius) advance to next waypoint (cyclic).
        """
        if not self.patrol_waypoints:
            return
        target_wp = self.patrol_waypoints[self.patrol_index % len(self.patrol_waypoints)]
        direction = target_wp - self.position
        dist = float(np.linalg.norm(direction))
        if dist < self.patrol_arrival_radius:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_waypoints)
            target_wp = self.patrol_waypoints[self.patrol_index]
            direction = target_wp - self.position
            dist = float(np.linalg.norm(direction))
        if dist > 1e-6:
            spd = self.speed if self.speed > 1e-9 else float(np.linalg.norm(self.velocity))
            self.velocity = spd * direction / dist
        step = self.velocity * dt
        new_pos = self.position + step
        new_pos[0] = float(np.clip(new_pos[0], 0.5, world_width - 0.5))
        new_pos[1] = float(np.clip(new_pos[1], 0.5, world_height - 0.5))
        self.position = new_pos

    def _update_prediction(self, dt: float) -> None:
        """
        Update linear-extrapolation predicted position.

        predicted_position = current_position + velocity * dt
        Used by DirectNavBehaviour to lead a moving target.
        """
        if self.is_moving:
            self.predicted_position = self.position + self.velocity * dt
        else:
            self.predicted_position = self.position.copy()

    def apply_state_schedule(self, current_time: float) -> None:
        """
        Apply TIME_VARYING state transitions whose trigger time has elapsed.

        Processed entries are removed from the schedule.
        """
        if self.target_type != TargetType.TIME_VARYING:
            return
        pending = []
        for trigger_time, new_velocity in self.state_schedule:
            if current_time >= trigger_time:
                self.velocity = new_velocity.copy()
                self.is_moving = float(np.linalg.norm(new_velocity)) > 1e-6
            else:
                pending.append((trigger_time, new_velocity))
        self.state_schedule = pending

    def distance_to(self, point: NDArray[np.float64]) -> float:
        """Euclidean distance from target position to a given point."""
        return float(np.linalg.norm(self.position - point))

    def record_tracking_observation(
        self,
        time_s: float,
        position: NDArray[np.float64],
        max_history: int = 200,
    ) -> None:
        """Append an observation to the tracking history (bounded length)."""
        self.tracking_history.append((time_s, position.copy()))
        if len(self.tracking_history) > max_history:
            self.tracking_history = self.tracking_history[-max_history:]

    def __repr__(self) -> str:
        pos = f"({self.position[0]:.1f}, {self.position[1]:.1f})"
        return (
            f"Target(id={self.target_id}, type={self.target_type.value}, "
            f"status={self.status.value}, pos={pos}, "
            f"conf={self.confidence:.2f}, pri={self.priority:.2f})"
        )


@dataclass(frozen=True)
class DetectionEvent:
    """
    Immutable record of a single target detection by one UAV.

    Generated by DetectionSystem.scan() and stored in DetectionDatabase.
    """

    target_id: int
    uav_id: int
    position: NDArray[np.float64]
    timestamp: float
    confidence: float
    target_type: TargetType

    def __hash__(self) -> int:
        return hash((self.target_id, self.uav_id, round(self.timestamp, 2)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DetectionEvent):
            return NotImplemented
        return (
            self.target_id == other.target_id
            and self.uav_id == other.uav_id
            and abs(self.timestamp - other.timestamp) < 1e-9
        )


@dataclass
class TrackingRecord:
    """
    Per-target tracking metadata maintained by TargetTracker.

    tracking_duration_s   -- cumulative seconds of active tracking
    loss_events           -- number of times target was lost
    reacquisition_events  -- number of times target was reacquired
    visibility_log        -- list of (time_s, is_visible) samples
    last_tracked_time     -- simulation time of most recent tracking tick
    recovery_attempts     -- current recovery attempt count
    handover_events       -- number of successful UAV handovers for this target
    failed_handovers      -- number of handover attempts that failed
    localization_errors   -- list of ||observed_pos - true_pos|| samples [m]
    last_handover_time    -- simulation time of most recent handover
    """

    target_id: int
    tracking_duration_s: float = 0.0
    loss_events: int = 0
    reacquisition_events: int = 0
    visibility_log: list[tuple[float, bool]] = field(default_factory=list)
    last_tracked_time: Optional[float] = None
    recovery_attempts: int = 0
    handover_events: int = 0
    failed_handovers: int = 0
    localization_errors: list[float] = field(default_factory=list)
    last_handover_time: Optional[float] = None


@dataclass
class HandoverEvent:
    """Record of a single multi-UAV handover."""

    target_id: int
    from_uav: int
    to_uav: int
    timestamp: float
    success: bool
