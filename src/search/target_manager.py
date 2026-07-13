"""
TargetManager — central lifecycle manager for all search targets.

Responsibilities
----------------
- Spawn targets at mission start (static, dynamic, time-varying)
- Update moving target positions each simulation tick
- Apply TIME_VARYING state schedule transitions
- Provide read-only query API for detection and assignment modules
- Manage status transitions throughout the target lifecycle
- Remove completed targets from the active pool

Integration
-----------
Stored as world.target_manager.
SimulationEngine calls target_manager.update(dt, time_s) each tick
during the SEARCHING phase.
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
        Root search configuration.
    world_width : float
        World X dimension [m].
    world_height : float
        World Y dimension [m].
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

    # ------------------------------------------------------------------
    # Public spawn API
    # ------------------------------------------------------------------

    def spawn_targets(self, current_time: float = 0.0) -> list[Target]:
        """
        Create and register all targets defined in config.

        Should be called once at search phase start (after exploration).
        Returns the list of spawned Target objects.
        """
        spawn_cfg = self._config.targets
        rng = np.random.default_rng(spawn_cfg.seed)
        spawned: list[Target] = []

        occupied_positions: list[NDArray[np.float64]] = []

        for _ in range(spawn_cfg.count_static):
            pos = self._sample_position(rng, occupied_positions, spawn_cfg)
            target = self._make_target(
                target_type=TargetType.STATIC,
                position=pos,
                velocity=np.zeros(2, dtype=np.float64),
                creation_time=current_time,
            )
            self._targets[target.target_id] = target
            occupied_positions.append(pos)
            spawned.append(target)

        for _ in range(spawn_cfg.count_static_human):
            pos = self._sample_position(rng, occupied_positions, spawn_cfg)
            target = self._make_target(
                target_type=TargetType.STATIC,
                position=pos,
                velocity=np.zeros(2, dtype=np.float64),
                creation_time=current_time,
                is_human=True,
            )
            self._targets[target.target_id] = target
            occupied_positions.append(pos)
            spawned.append(target)

        for _ in range(spawn_cfg.count_dynamic):
            pos = self._sample_position(rng, occupied_positions, spawn_cfg)
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min, spawn_cfg.dynamic_speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            target = self._make_target(
                target_type=TargetType.DYNAMIC,
                position=pos,
                velocity=vel,
                creation_time=current_time,
            )
            self._targets[target.target_id] = target
            occupied_positions.append(pos)
            spawned.append(target)

        for _ in range(spawn_cfg.count_time_varying):
            pos = self._sample_position(rng, occupied_positions, spawn_cfg)
            # Start moving, then stop at a random time, then move again
            speed = float(rng.uniform(spawn_cfg.dynamic_speed_min, spawn_cfg.dynamic_speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            vel = speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

            # State schedule: stop at t+20s, resume at t+50s, stop at t+90s
            schedule = [
                (current_time + 20.0, np.zeros(2, dtype=np.float64)),
                (current_time + 50.0, vel * 0.6),
                (current_time + 90.0, np.zeros(2, dtype=np.float64)),
            ]
            target = self._make_target(
                target_type=TargetType.TIME_VARYING,
                position=pos,
                velocity=vel,
                creation_time=current_time,
                state_schedule=schedule,
            )
            self._targets[target.target_id] = target
            occupied_positions.append(pos)
            spawned.append(target)

        return spawned

    # ------------------------------------------------------------------
    # Public update API (called by SimulationEngine each search tick)
    # ------------------------------------------------------------------

    def update(self, dt: float, current_time: float) -> None:
        """
        Advance all active target states by one timestep.

        - Moves dynamic and time-varying targets
        - Applies scheduled state changes for TIME_VARYING targets
        - Decays confidence of unseen targets
        """
        decay = self._config.detection.confidence_decay_rate
        for target in self._targets.values():
            if target.status == TargetStatus.COMPLETED:
                continue

            # Apply state schedule transitions
            target.apply_state_schedule(current_time)

            # Move position for non-static targets
            if target.is_moving:
                target.update_position(dt, self._world_width, self._world_height)

            # Decay confidence for detected-but-unseen targets
            if target.status not in (TargetStatus.UNDISCOVERED, TargetStatus.COMPLETED):
                if target.last_seen is not None:
                    age = current_time - target.last_seen
                    if age > 0:
                        target.confidence = max(
                            0.0,
                            target.confidence - decay * dt,
                        )

    # ------------------------------------------------------------------
    # Status transition API
    # ------------------------------------------------------------------

    def register_detection(
        self,
        target_id: int,
        confidence: float,
        position: NDArray[np.float64],
        current_time: float,
    ) -> None:
        """
        Transition target to DETECTED and record observation.

        Safe to call multiple times — only sets detection_time on first call.
        """
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
        """Transition target to COMPLETED and remove from active pool."""
        target = self._targets.get(target_id)
        if target is not None:
            target.status = TargetStatus.COMPLETED
            self._completed_ids.add(target_id)

    def unassign_target(self, target_id: int) -> None:
        """Remove UAV assignment and revert to DETECTED for reassignment."""
        target = self._targets.get(target_id)
        if target is None:
            return
        target.assigned_uav = None
        if target.status in (TargetStatus.ASSIGNED, TargetStatus.SEARCHING):
            target.status = TargetStatus.DETECTED

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_target(self, target_id: int) -> Target | None:
        """Return target by ID, or None."""
        return self._targets.get(target_id)

    def all_targets(self) -> list[Target]:
        """Return all targets (all statuses)."""
        return list(self._targets.values())

    def active_targets(self) -> list[Target]:
        """Return targets that are not yet COMPLETED."""
        return [t for t in self._targets.values() if t.status != TargetStatus.COMPLETED]

    def detected_targets(self) -> list[Target]:
        """Return targets that have been detected but not yet assigned."""
        return [t for t in self._targets.values() if t.status == TargetStatus.DETECTED]

    def assigned_targets(self) -> list[Target]:
        """Return targets that are ASSIGNED, SEARCHING, or TRACKING."""
        active_statuses = {
            TargetStatus.ASSIGNED,
            TargetStatus.SEARCHING,
            TargetStatus.TRACKING,
        }
        return [t for t in self._targets.values() if t.status in active_statuses]

    def lost_targets(self) -> list[Target]:
        """Return targets currently in LOST state."""
        return [t for t in self._targets.values() if t.status == TargetStatus.LOST]

    def completed_targets(self) -> list[Target]:
        """Return all COMPLETED targets."""
        return [t for t in self._targets.values() if t.status == TargetStatus.COMPLETED]

    def undiscovered_targets(self) -> list[Target]:
        """Return targets not yet detected."""
        return [t for t in self._targets.values() if t.status == TargetStatus.UNDISCOVERED]

    def targets_for_uav(self, uav_id: int) -> list[Target]:
        """Return all targets currently assigned to a given UAV."""
        return [
            t for t in self._targets.values()
            if t.assigned_uav == uav_id
            and t.status not in (TargetStatus.COMPLETED, TargetStatus.UNDISCOVERED)
        ]

    def all_missions_complete(self) -> bool:
        """
        Return True when every target is either COMPLETED or permanently LOST.

        Permanently LOST targets have status LOST with max recovery attempts exceeded
        (set externally by SearchController). For mission completion purposes both
        COMPLETED and LOST (exhausted) count as done.
        """
        for target in self._targets.values():
            if target.status not in (TargetStatus.COMPLETED, TargetStatus.LOST):
                return False
        return len(self._targets) > 0

    def iter_targets(self) -> Iterator[Target]:
        """Iterate over all registered targets."""
        yield from self._targets.values()

    @property
    def total_count(self) -> int:
        """Total number of registered targets."""
        return len(self._targets)

    @property
    def completed_count(self) -> int:
        """Number of COMPLETED targets."""
        return len(self.completed_targets())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_target(
        self,
        target_type: TargetType,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        creation_time: float,
        state_schedule: list[tuple[float, NDArray[np.float64]]] | None = None,
        is_human: bool = False,
    ) -> Target:
        tid = self._next_id
        self._next_id += 1
        return Target(
            target_id=tid,
            target_type=target_type,
            position=position.copy(),
            velocity=velocity.copy(),
            creation_time=creation_time,
            state_schedule=list(state_schedule or []),
            is_human=is_human,
        )

    def _sample_position(
        self,
        rng: np.random.Generator,
        occupied: list[NDArray[np.float64]],
        spawn_cfg: TargetSpawnConfig,
        max_attempts: int = 200,
    ) -> NDArray[np.float64]:
        """Sample a valid spawn position respecting margins and separation."""
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
            too_close = any(
                float(np.linalg.norm(pos - other)) < min_sep
                for other in occupied
            )
            if not too_close:
                return pos

        # Fallback: return random position even if separation not met
        return np.array(
            [
                rng.uniform(margin, self._world_width - margin),
                rng.uniform(margin, self._world_height - margin),
            ],
            dtype=np.float64,
        )
