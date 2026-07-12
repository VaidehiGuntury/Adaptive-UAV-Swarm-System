"""
TargetTracker — per-target tracking state machine.

Responsibilities
----------------
- Record continuous observations while target is visible
- Accumulate tracking_duration_s
- Detect target loss (no observation for > loss_timeout_s)
- Attempt reacquisition when target becomes visible again
- Log visibility history for metrics

Integration
-----------
SearchController calls tracker.update() each tick with the visibility map
produced by DetectionSystem.update_tracking_observations().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.config.search_config import SearchBehaviourConfig, TrackingConfig
from src.search.target import Target, TargetStatus, TrackingRecord
from src.search.target_manager import TargetManager


class TargetTracker:
    """
    Manages tracking state for all active targets.

    Parameters
    ----------
    tracking_config : TrackingConfig
    behaviour_config : SearchBehaviourConfig
        Used for loss_timeout_s and max_recovery_attempts.
    """

    def __init__(
        self,
        tracking_config: TrackingConfig,
        behaviour_config: SearchBehaviourConfig,
    ) -> None:
        self._t_cfg = tracking_config
        self._b_cfg = behaviour_config
        # target_id → TrackingRecord
        self._records: dict[int, TrackingRecord] = {}
        # target_id → last visibility sample time
        self._last_visibility_sample: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        target_manager: TargetManager,
        visibility_map: dict[int, list[int]],
        current_time: float,
    ) -> dict[int, str]:
        """
        Process one tracking tick for all active targets.

        visibility_map: uav_id → list of visible target_ids (from DetectionSystem)

        Returns a dict of target_id → event_type for events this tick:
            "lost"         — target transitioned to LOST
            "reacquired"   — target reacquired after LOST
            "completed"    — tracking duration met
        """
        # Invert visibility map: target_id → list of observing uav_ids
        target_visibility: dict[int, list[int]] = {}
        for uav_id, target_ids in visibility_map.items():
            for tid in target_ids:
                target_visibility.setdefault(tid, []).append(uav_id)

        events: dict[int, str] = {}

        for target in target_manager.assigned_targets():
            record = self._get_or_create_record(target.target_id)
            is_visible = target.target_id in target_visibility

            if is_visible:
                # Get best observing UAV position (closest, implicit via record)
                self._record_observation(target, record, current_time)

                if target.status == TargetStatus.LOST:
                    # Reacquisition
                    record.reacquisition_events += 1
                    record.recovery_attempts = 0
                    target_manager.begin_tracking(
                        target.target_id, target.position, current_time
                    )
                    events[target.target_id] = "reacquired"
                elif target.status in (TargetStatus.SEARCHING, TargetStatus.ASSIGNED):
                    target_manager.begin_tracking(
                        target.target_id, target.position, current_time
                    )

                # Accumulate tracking duration
                if record.last_tracked_time is not None:
                    elapsed = current_time - record.last_tracked_time
                    record.tracking_duration_s += elapsed
                record.last_tracked_time = current_time

                # Log visibility
                self._log_visibility(target.target_id, current_time, True, record)

                # Check completion
                if (
                    record.tracking_duration_s
                    >= self._t_cfg.required_tracking_duration_s
                    and target.status == TargetStatus.TRACKING
                ):
                    target_manager.mark_completed(target.target_id)
                    events[target.target_id] = "completed"

            else:
                # Not visible this tick
                self._log_visibility(target.target_id, current_time, False, record)

                if target.status == TargetStatus.TRACKING:
                    # Check loss timeout
                    last_seen = target.last_seen
                    if last_seen is not None:
                        unseen_duration = current_time - last_seen
                        if unseen_duration > self._b_cfg.loss_timeout_s:
                            record.loss_events += 1
                            record.recovery_attempts += 1
                            target_manager.mark_lost(target.target_id)
                            events[target.target_id] = "lost"

        # Handle permanently lost targets
        for target in target_manager.lost_targets():
            record = self._get_or_create_record(target.target_id)
            if record.recovery_attempts >= self._b_cfg.max_recovery_attempts:
                # Exhausted recovery — mark completed (permanently lost)
                target_manager.mark_completed(target.target_id)
                events[target.target_id] = "completed"

        return events

    def get_record(self, target_id: int) -> TrackingRecord | None:
        """Return the tracking record for a given target ID."""
        return self._records.get(target_id)

    def get_all_records(self) -> dict[int, TrackingRecord]:
        """Return all tracking records (snapshot)."""
        return dict(self._records)

    def tracking_duration(self, target_id: int) -> float:
        """Return cumulative tracking duration for a target [s]."""
        record = self._records.get(target_id)
        return record.tracking_duration_s if record else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_record(self, target_id: int) -> TrackingRecord:
        if target_id not in self._records:
            self._records[target_id] = TrackingRecord(target_id=target_id)
        return self._records[target_id]

    def _record_observation(
        self,
        target: Target,
        record: TrackingRecord,
        current_time: float,
    ) -> None:
        """Append tracking history entry."""
        target.record_tracking_observation(
            current_time,
            target.position,
            self._t_cfg.history_max_length,
        )

    def _log_visibility(
        self,
        target_id: int,
        current_time: float,
        is_visible: bool,
        record: TrackingRecord,
    ) -> None:
        """Append a visibility sample if the sampling interval has elapsed."""
        last = self._last_visibility_sample.get(target_id, -999.0)
        if current_time - last >= self._t_cfg.visibility_sample_interval_s:
            record.visibility_log.append((current_time, is_visible))
            self._last_visibility_sample[target_id] = current_time
