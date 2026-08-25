"""
TargetTracker -- per-target tracking with handover and localization error.

New capabilities vs original:
  - Processes LOST targets so reacquisition can be triggered
  - Multi-UAV handover when a closer UAV is available
  - Localization error per tick (||last_seen - true_pos||)
  - Full HandoverEvent log
"""

from __future__ import annotations

import numpy as np

from src.agents.uav import UAV
from src.config.search_config import HandoverConfig, SearchBehaviourConfig, TrackingConfig
from src.search.target import HandoverEvent, Target, TargetStatus, TrackingRecord
from src.search.target_manager import TargetManager


class TargetTracker:
    """
    Manages tracking state for all active targets.

    Parameters
    ----------
    tracking_config  : TrackingConfig
    behaviour_config : SearchBehaviourConfig
    handover_config  : HandoverConfig (optional)
    """

    def __init__(
        self,
        tracking_config: TrackingConfig,
        behaviour_config: SearchBehaviourConfig,
        handover_config: HandoverConfig | None = None,
    ) -> None:
        self._t_cfg = tracking_config
        self._b_cfg = behaviour_config
        self._h_cfg = handover_config or HandoverConfig()
        self._records: dict[int, TrackingRecord] = {}
        self._last_visibility_sample: dict[int, float] = {}
        self.handover_log: list[HandoverEvent] = []

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def update(
        self,
        target_manager: TargetManager,
        visibility_map: dict[int, list[int]],
        current_time: float,
        agents: list[UAV] | None = None,
    ) -> dict[int, str]:
        """
        Process one tracking tick.

        Returns dict[target_id -> event_str] where event_str is one of:
          "lost", "reacquired", "completed", "handover"
        """
        # Invert visibility map: target_id -> [uav_ids seeing it]
        target_visibility: dict[int, list[int]] = {}
        for uav_id, tids in visibility_map.items():
            for tid in tids:
                target_visibility.setdefault(tid, []).append(uav_id)

        events: dict[int, str] = {}

        # Process assigned AND lost targets (lost targets can reacquire)
        all_active = target_manager.assigned_targets() + target_manager.lost_targets()

        for target in all_active:
            record = self._get_or_create_record(target.target_id)
            is_visible = target.target_id in target_visibility

            if is_visible:
                self._record_observation(target, record, current_time)
                self._record_localization_error(target, record)

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
                    record.tracking_duration_s += current_time - record.last_tracked_time
                record.last_tracked_time = current_time

                self._log_visibility(target.target_id, current_time, True, record)

                # Completion check
                if (
                    record.tracking_duration_s >= self._t_cfg.required_tracking_duration_s
                    and target.status == TargetStatus.TRACKING
                ):
                    target_manager.mark_completed(target.target_id)
                    events[target.target_id] = "completed"
                    continue

                # Handover check
                if (
                    self._h_cfg.handover_enabled
                    and agents is not None
                    and target.status == TargetStatus.TRACKING
                ):
                    ho = self._check_handover(target, record, agents, current_time)
                    if ho is not None:
                        target_manager.handover_target(target.target_id, ho.to_uav)
                        self.handover_log.append(ho)
                        record.handover_events += 1
                        events[target.target_id] = "handover"

            else:
                # Target not visible this tick
                self._log_visibility(target.target_id, current_time, False, record)

                if target.status == TargetStatus.TRACKING:
                    last_seen = target.last_seen
                    if last_seen is not None:
                        if current_time - last_seen > self._b_cfg.loss_timeout_s:
                            record.loss_events += 1
                            record.recovery_attempts += 1
                            target_manager.mark_lost(target.target_id)
                            events[target.target_id] = "lost"

        # Permanently lost: exhausted recovery
        for target in target_manager.lost_targets():
            record = self._get_or_create_record(target.target_id)
            if record.recovery_attempts >= self._b_cfg.max_recovery_attempts:
                target_manager.mark_completed(target.target_id)
                events[target.target_id] = "completed"

        return events

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_record(self, target_id: int) -> TrackingRecord | None:
        return self._records.get(target_id)

    def get_all_records(self) -> dict[int, TrackingRecord]:
        return dict(self._records)

    def tracking_duration(self, target_id: int) -> float:
        record = self._records.get(target_id)
        return record.tracking_duration_s if record else 0.0

    def total_handovers(self) -> int:
        return sum(1 for e in self.handover_log if e.success)

    def total_failed_handovers(self) -> int:
        return sum(r.failed_handovers for r in self._records.values())

    def avg_localization_error(self) -> float:
        all_errors: list[float] = []
        for rec in self._records.values():
            all_errors.extend(rec.localization_errors)
        return float(np.mean(all_errors)) if all_errors else 0.0

    # ------------------------------------------------------------------
    # Private helpers
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
        target.record_tracking_observation(current_time, target.position, 200)

    def _record_localization_error(
        self,
        target: Target,
        record: TrackingRecord,
    ) -> None:
        if target.last_seen_position is None:
            return
        err = float(np.linalg.norm(target.last_seen_position - target.position))
        record.localization_errors.append(err)

    def _log_visibility(
        self,
        target_id: int,
        current_time: float,
        is_visible: bool,
        record: TrackingRecord,
    ) -> None:
        last = self._last_visibility_sample.get(target_id, -999.0)
        if current_time - last >= self._t_cfg.visibility_sample_interval_s:
            record.visibility_log.append((current_time, is_visible))
            self._last_visibility_sample[target_id] = current_time

    def _check_handover(
        self,
        target: Target,
        record: TrackingRecord,
        agents: list[UAV],
        current_time: float,
    ) -> HandoverEvent | None:
        cfg = self._h_cfg
        # Hysteresis gate
        if (
            record.last_handover_time is not None
            and current_time - record.last_handover_time < cfg.handover_hysteresis_s
        ):
            return None

        owner_id = target.assigned_uav
        if owner_id is None:
            return None

        owner_dist = float("inf")
        for a in agents:
            if a.agent_id == owner_id:
                owner_dist = target.distance_to(a.position)
                break

        if owner_dist == float("inf"):
            return None

        threshold = owner_dist * cfg.handover_distance_ratio
        best_dist = owner_dist
        best_uav_id: int | None = None
        for a in agents:
            if a.agent_id == owner_id:
                continue
            d = target.distance_to(a.position)
            if d < threshold and d < best_dist:
                best_dist = d
                best_uav_id = a.agent_id

        if best_uav_id is None:
            return None

        record.last_handover_time = current_time
        return HandoverEvent(
            target_id=target.target_id,
            from_uav=owner_id,
            to_uav=best_uav_id,
            timestamp=current_time,
            success=True,
        )
