"""
TargetManager -- central lifecycle manager for all search targets.

Supports five motion models:
  STATIC, DYNAMIC (constant velocity), TIME_VARYING,
  RANDOM_WALK, WAYPOINT_PATROL
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from src.config.search_config import SearchConfig, TargetSpawnConfig
from src.search.target import Target, TargetStatus, TargetType


class TargetManager:
    """
    Manages the full population of search targets.

    Parameters
    ----------
    config : SearchConfig
    world_width : float
    world_height : float
    """

    def __init__(
        self,
        config: SearchConfig,
        world_width: float,
        world_height: float,
    ) -> None:
        self._config = config
        self._world_width = world_width
        self._world_height = world_height
        self._targets: dict[int, Target] = {}
        self._next_id = 0
        self._completed_ids: set[int] = set()
        # Shared RNG for random-walk motion (deterministic per-run)
        self._motion_rng = np.random.default_rng(config.targets.seed + 9999)

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------

    def spawn_targets(self, current_time: float = 0.0) -> list[Target]:
        """Create and register all targets defined in config."""
        spawn_cfg = self._config.targets
        rng = np.random.default_rng(spawn_cfg.seed)
        spawned: list[Target] = []
        occupied: list[NDArray[np.float64]] = []

        # Static
        for _ in range(spawn_cfg.count_static):
            pos = self._sample_position(rng, occupied, spawn_cfg)
            t = self._make_target(TargetType.STATIC, pos,
                                  np.zeros(2, dtype=np.float64), current_time)
            self._register(t, occupied, pos, spawned)

        # Dynamic (constant velocity)
        for _ in range(spawn_cfg.count_dynamic):
            pos = self._sample_position(rng, occupied, spawn_cfg)
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min,
                                      spawn_cfg.dynamic_speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            t = self._make_target(TargetType.DYNAMIC, pos, vel, current_time,
                                  speed=speed)
            self._register(t, occupied, pos, spawned)

        # Time-varying
        for _ in range(spawn_cfg.count_time_varying):
            pos = self._sample_position(rng, occupied, spawn_cfg)
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min,
                                      spawn_cfg.dynamic_speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            schedule = [
                (current_time + 20.0, np.zeros(2, dtype=np.float64)),
                (current_time + 50.0, vel * 0.6),
                (current_time + 90.0, np.zeros(2, dtype=np.float64)),
            ]
            t = self._make_target(TargetType.TIME_VARYING, pos, vel, current_time,
                                  speed=speed, state_schedule=schedule)
            self._register(t, occupied, pos, spawned)

        # Random walk
        for _ in range(spawn_cfg.count_random_walk):
            pos = self._sample_position(rng, occupied, spawn_cfg)
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min,
                                      spawn_cfg.dynamic_speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            t = self._make_target(
                TargetType.RANDOM_WALK, pos, vel, current_time,
                speed=speed,
                random_walk_turn_rad=spawn_cfg.random_walk_turn_rad,
            )
            self._register(t, occupied, pos, spawned)

        # Waypoint patrol
        for _ in range(spawn_cfg.count_waypoint_patrol):
            pos = self._sample_position(rng, occupied, spawn_cfg)
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min,
                                      spawn_cfg.dynamic_speed_max))
            waypoints = [
                self._sample_position(rng, [], spawn_cfg)
                for _ in range(spawn_cfg.waypoint_count)
            ]
            t = self._make_target(
                TargetType.WAYPOINT_PATROL, pos,
                np.zeros(2, dtype=np.float64), current_time,
                speed=speed, patrol_waypoints=waypoints,
            )
            t.is_moving = True
            self._register(t, occupied, pos, spawned)

        return spawned

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update(self, dt: float, current_time: float) -> None:
        """Advance all active targets by one timestep."""
        decay = self._config.detection.confidence_decay_rate
        for target in self._targets.values():
            if target.status == TargetStatus.COMPLETED:
                continue

            target.apply_state_schedule(current_time)

            if target.is_moving or target.target_type == TargetType.WAYPOINT_PATROL:
                target.update_position(
                    dt,
                    self._world_width,
                    self._world_height,
                    rng=self._motion_rng,
                )

            if target.status not in (TargetStatus.UNDISCOVERED, TargetStatus.COMPLETED):
                if target.last_seen is not None:
                    age = current_time - target.last_seen
                    if age > 0:
                        target.confidence = max(
                            0.0,
                            target.confidence - decay * dt,
                        )

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def register_detection(
        self,
        target_id: int,
        confidence: float,
        position: NDArray[np.float64],
        current_time: float,
    ) -> None:
        """Transition target to DETECTED and record observation."""
        target = self._targets.get(target_id)
        if target is None:
            return
        if target.status == TargetStatus.UNDISCOVERED:
            target.detection_time = current_time
            target.status = TargetStatus.DETECTED
        target.confidence = confidence
        target.last_seen = current_time
        target.last_seen_position = position.copy()

    def assign_target(self, target_id: int, uav_id: int) -> None:
        """Mark target as ASSIGNED to a specific UAV."""
        target = self._targets.get(target_id)
        if target is None:
            return
        target.assigned_uav = uav_id
        if target.status == TargetStatus.DETECTED:
            target.status = TargetStatus.ASSIGNED

    def begin_searching(self, target_id: int) -> None:
        """Transition target to SEARCHING state."""
        target = self._targets.get(target_id)
        if target is not None and target.status == TargetStatus.ASSIGNED:
            target.status = TargetStatus.SEARCHING

    def begin_tracking(
        self,
        target_id: int,
        position: NDArray[np.float64],
        current_time: float,
    ) -> None:
        """Transition target to TRACKING and record observation."""
        target = self._targets.get(target_id)
        if target is None:
            return
        target.status = TargetStatus.TRACKING
        target.last_seen = current_time
        target.last_seen_position = position.copy()
        target.record_tracking_observation(
            current_time, position,
            self._config.tracking.history_max_length,
        )

    def mark_lost(self, target_id: int) -> None:
        """Transition target to LOST state."""
        target = self._targets.get(target_id)
        if target is not None and target.status == TargetStatus.TRACKING:
            target.status = TargetStatus.LOST

    def mark_completed(self, target_id: int) -> None:
        """Transition target to COMPLETED."""
        target = self._targets.get(target_id)
        if target is not None:
            target.status = TargetStatus.COMPLETED
            self._completed_ids.add(target_id)

    def unassign_target(self, target_id: int) -> None:
        """Remove UAV assignment and revert to DETECTED."""
        target = self._targets.get(target_id)
        if target is None:
            return
        target.assigned_uav = None
        if target.status in (TargetStatus.ASSIGNED, TargetStatus.SEARCHING):
            target.status = TargetStatus.DETECTED

    def handover_target(self, target_id: int, new_uav_id: int) -> bool:
        """Reassign target tracking from current UAV to new_uav_id."""
        target = self._targets.get(target_id)
        if target is None:
            return False
        if target.assigned_uav == new_uav_id:
            return False
        target.assigned_uav = new_uav_id
        return True

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_target(self, target_id: int) -> Target | None:
        return self._targets.get(target_id)

    def all_targets(self) -> list[Target]:
        return list(self._targets.values())

    def active_targets(self) -> list[Target]:
        return [t for t in self._targets.values()
                if t.status != TargetStatus.COMPLETED]

    def detected_targets(self) -> list[Target]:
        return [t for t in self._targets.values()
                if t.status == TargetStatus.DETECTED]

    def assigned_targets(self) -> list[Target]:
        active = {TargetStatus.ASSIGNED, TargetStatus.SEARCHING, TargetStatus.TRACKING}
        return [t for t in self._targets.values() if t.status in active]

    def lost_targets(self) -> list[Target]:
        return [t for t in self._targets.values()
                if t.status == TargetStatus.LOST]

    def completed_targets(self) -> list[Target]:
        return [t for t in self._targets.values()
                if t.status == TargetStatus.COMPLETED]

    def undiscovered_targets(self) -> list[Target]:
        return [t for t in self._targets.values()
                if t.status == TargetStatus.UNDISCOVERED]

    def targets_for_uav(self, uav_id: int) -> list[Target]:
        return [
            t for t in self._targets.values()
            if t.assigned_uav == uav_id
            and t.status not in (TargetStatus.COMPLETED, TargetStatus.UNDISCOVERED)
        ]

    def all_missions_complete(self) -> bool:
        for target in self._targets.values():
            if target.status not in (TargetStatus.COMPLETED, TargetStatus.LOST):
                return False
        return len(self._targets) > 0

    def iter_targets(self) -> Iterator[Target]:
        yield from self._targets.values()

    @property
    def total_count(self) -> int:
        return len(self._targets)

    @property
    def completed_count(self) -> int:
        return len(self.completed_targets())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(
        self,
        target: Target,
        occupied: list[NDArray[np.float64]],
        pos: NDArray[np.float64],
        spawned: list[Target],
    ) -> None:
        self._targets[target.target_id] = target
        occupied.append(pos)
        spawned.append(target)

    def _make_target(
        self,
        target_type: TargetType,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        creation_time: float,
        speed: float = 0.0,
        state_schedule: list[tuple[float, NDArray[np.float64]]] | None = None,
        random_walk_turn_rad: float = 0.5,
        patrol_waypoints: list[NDArray[np.float64]] | None = None,
    ) -> Target:
        tid = self._next_id
        self._next_id += 1
        return Target(
            target_id=tid,
            target_type=target_type,
            position=position.copy(),
            velocity=velocity.copy(),
            creation_time=creation_time,
            speed=speed,
            state_schedule=list(state_schedule or []),
            random_walk_turn_rad=random_walk_turn_rad,
            patrol_waypoints=list(patrol_waypoints or []),
        )

    def _sample_position(
        self,
        rng: np.random.Generator,
        occupied: list[NDArray[np.float64]],
        spawn_cfg: TargetSpawnConfig,
        max_attempts: int = 200,
    ) -> NDArray[np.float64]:
        """Sample spawn position respecting margin and minimum separation."""
        margin = spawn_cfg.spawn_margin
        min_sep = spawn_cfg.min_separation
        for _ in range(max_attempts):
            pos = np.array(
                [
                    rng.uniform(margin, self._world_width - margin),
                    rng.uniform(margin, self._world_height - margin),
                ],
                dtype=np.float64,
            )
            if not any(float(np.linalg.norm(pos - o)) < min_sep for o in occupied):
                return pos
        return np.array(
            [
                rng.uniform(margin, self._world_width - margin),
                rng.uniform(margin, self._world_height - margin),
            ],
            dtype=np.float64,
        )
