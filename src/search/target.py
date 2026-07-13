"""
Target data model for the Search & Tracking extension.

Defines Target objects, type/status enumerations, detection events,
and tracking records. Pure data — no algorithm logic lives here.
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

    STATIC       — never moves (injured human, rescue kit, building)
    DYNAMIC      — moves continuously (walking person, vehicle, animal)
    TIME_VARYING — changes state during mission (moving→stopped, appearing/disappearing)
    """

    STATIC = "static"
    DYNAMIC = "dynamic"
    TIME_VARYING = "time_varying"


class TargetStatus(Enum):
    """
    Target lifecycle state machine.

    UNDISCOVERED → DETECTED → ASSIGNED → SEARCHING → TRACKING → COMPLETED
                                                    ↘ LOST ↗
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
    is_human         : for STATIC targets, distinguishes an injured-human
                       target (score_static_human) from an inanimate object
                       (score_static_object) in priority scoring. Unused
                       for DYNAMIC/TIME_VARYING targets.
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
    is_human: bool = False
    # TIME_VARYING schedule: list of (trigger_time_s, new_velocity_array)
    state_schedule: list[tuple[float, NDArray[np.float64]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.is_moving = float(np.linalg.norm(self.velocity)) > 1e-6

    def update_position(self, dt: float, world_width: float, world_height: float) -> None:
        """
        Advance target position by dt seconds.

        Bounces off world boundary edges so dynamic targets stay in-world.
        """
        if not self.is_moving:
            return
        new_pos = self.position + self.velocity * dt

        # Boundary bounce
        if new_pos[0] <= 0.0 or new_pos[0] >= world_width:
            self.velocity = self.velocity * np.array([-1.0, 1.0])
            new_pos[0] = float(np.clip(new_pos[0], 0.5, world_width - 0.5))
        if new_pos[1] <= 0.0 or new_pos[1] >= world_height:
            self.velocity = self.velocity * np.array([1.0, -1.0])
            new_pos[1] = float(np.clip(new_pos[1], 0.5, world_height - 0.5))

        self.position = new_pos

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

    tracking_duration_s — cumulative seconds of active tracking
    loss_events         — number of times target was lost
    reacquisition_events — number of times target was reacquired
    visibility_log      — list of (time_s, is_visible) samples
    last_tracked_time   — simulation time of most recent tracking tick
    recovery_attempts   — current recovery attempt count
    """

    target_id: int
    tracking_duration_s: float = 0.0
    loss_events: int = 0
    reacquisition_events: int = 0
    visibility_log: list[tuple[float, bool]] = field(default_factory=list)
    last_tracked_time: Optional[float] = None
    recovery_attempts: int = 0
