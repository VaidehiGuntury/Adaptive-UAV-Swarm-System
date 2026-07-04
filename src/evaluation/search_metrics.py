"""
Search, tracking, and mission metrics for the Target Search extension.

All collectors are pure functions or lightweight dataclass accumulators.
No algorithm logic lives here — metrics are observational only.

CSV export follows the same pattern as run_experiment.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.search.target import Target, TargetStatus, TargetType
from src.search.target_manager import TargetManager
from src.search.tracker import TargetTracker


# ------------------------------------------------------------------
# Per-target detection metrics
# ------------------------------------------------------------------

@dataclass
class TargetDetectionMetrics:
    """Detection timing metrics for a single target."""

    target_id: int
    target_type: str
    first_detection_time: Optional[float] = None    # simulation time of first detection
    rediscovery_time: Optional[float] = None        # time from LOST to reacquisition
    assignment_delay: Optional[float] = None        # time from detection to assignment


# ------------------------------------------------------------------
# Per-target search metrics
# ------------------------------------------------------------------

@dataclass
class TargetSearchMetrics:
    """Search outcome metrics for a single target."""

    target_id: int
    target_type: str
    search_start_time: Optional[float] = None       # time when SEARCHING began
    search_end_time: Optional[float] = None         # time when TRACKING began
    search_duration_s: float = 0.0                  # total active search time
    search_success: bool = False                    # True if target was tracked
    assigned_uav: Optional[int] = None


# ------------------------------------------------------------------
# Per-target tracking metrics
# ------------------------------------------------------------------

@dataclass
class TargetTrackingMetrics:
    """Tracking quality metrics for a single target."""

    target_id: int
    target_type: str
    tracking_duration_s: float = 0.0
    loss_events: int = 0
    reacquisition_events: int = 0
    tracking_accuracy: float = 0.0                  # fraction of time target was visible
    assigned_uav: Optional[int] = None


# ------------------------------------------------------------------
# Mission-level aggregated metrics
# ------------------------------------------------------------------

@dataclass
class MissionSearchMetrics:
    """
    Aggregated mission-level search and tracking metrics.

    Exported to CSV for experiment analysis.
    """

    # Mission config
    num_uavs: int = 0
    environment_type: str = "static"                # static / dynamic / mixed

    # Coverage
    coverage_at_search_start: float = 0.0           # explored_fraction when search began
    coverage_at_mission_end: float = 0.0

    # Target population
    total_targets: int = 0
    static_targets: int = 0
    dynamic_targets: int = 0
    time_varying_targets: int = 0

    # Detection
    detected_count: int = 0
    detection_rate: float = 0.0                     # detected / total
    avg_first_detection_time: float = 0.0
    avg_assignment_delay: float = 0.0

    # Search
    successful_searches: int = 0
    failed_searches: int = 0
    search_success_rate: float = 0.0
    avg_search_duration_s: float = 0.0

    # Tracking
    total_tracking_duration_s: float = 0.0
    avg_tracking_duration_s: float = 0.0
    avg_tracking_accuracy: float = 0.0
    total_loss_events: int = 0
    total_reacquisition_events: int = 0

    # Mission completion
    completed_targets: int = 0
    lost_targets: int = 0
    mission_completion_time: Optional[float] = None
    search_phase_duration_s: float = 0.0


# ------------------------------------------------------------------
# Metrics collector
# ------------------------------------------------------------------

class SearchMetricsCollector:
    """
    Collects and computes all search/tracking metrics from simulation state.

    Parameters
    ----------
    num_uavs : int
    environment_type : str
    """

    def __init__(self, num_uavs: int, environment_type: str = "static") -> None:
        self._num_uavs = num_uavs
        self._env_type = environment_type
        self._detection_metrics: dict[int, TargetDetectionMetrics] = {}
        self._search_metrics: dict[int, TargetSearchMetrics] = {}
        self._tracking_metrics: dict[int, TargetTrackingMetrics] = {}
        self._search_start_time: float | None = None
        self._mission_end_time: float | None = None
        self._coverage_at_start: float = 0.0
        self._coverage_at_end: float = 0.0
        self._prev_statuses: dict[int, TargetStatus] = {}

    def record_search_start(self, current_time: float, coverage: float) -> None:
        """Record the time and coverage when search phase began."""
        self._search_start_time = current_time
        self._coverage_at_start = coverage

    def record_mission_end(self, current_time: float, coverage: float) -> None:
        """Record mission completion time."""
        self._mission_end_time = current_time
        self._coverage_at_end = coverage

    def update(
        self,
        target_manager: TargetManager,
        tracker: TargetTracker,
        current_time: float,
    ) -> None:
        """
        Update metrics from current target and tracker state.

        Should be called each tick during search phase.
        """
        for target in target_manager.all_targets():
            tid = target.target_id
            prev_status = self._prev_statuses.get(tid, TargetStatus.UNDISCOVERED)
            curr_status = target.status

            # ── Detection metrics ──
            if tid not in self._detection_metrics:
                self._detection_metrics[tid] = TargetDetectionMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                )
            dm = self._detection_metrics[tid]
            if (
                dm.first_detection_time is None
                and target.detection_time is not None
            ):
                dm.first_detection_time = target.detection_time

            # Assignment delay: detection → assigned
            if (
                dm.assignment_delay is None
                and curr_status in (
                    TargetStatus.ASSIGNED,
                    TargetStatus.SEARCHING,
                    TargetStatus.TRACKING,
                )
                and dm.first_detection_time is not None
            ):
                dm.assignment_delay = current_time - dm.first_detection_time

            # ── Search metrics ──
            if tid not in self._search_metrics:
                self._search_metrics[tid] = TargetSearchMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                    assigned_uav=target.assigned_uav,
                )
            sm = self._search_metrics[tid]
            if prev_status != TargetStatus.SEARCHING and curr_status == TargetStatus.SEARCHING:
                sm.search_start_time = current_time
            if prev_status == TargetStatus.SEARCHING and curr_status == TargetStatus.TRACKING:
                sm.search_end_time = current_time
                sm.search_success = True
                if sm.search_start_time is not None:
                    sm.search_duration_s = current_time - sm.search_start_time

            # ── Tracking metrics ──
            if tid not in self._tracking_metrics:
                self._tracking_metrics[tid] = TargetTrackingMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                    assigned_uav=target.assigned_uav,
                )
            tkm = self._tracking_metrics[tid]
            record = tracker.get_record(tid)
            if record is not None:
                tkm.tracking_duration_s = record.tracking_duration_s
                tkm.loss_events = record.loss_events
                tkm.reacquisition_events = record.reacquisition_events
                # Tracking accuracy: fraction of visibility_log entries where visible
                if record.visibility_log:
                    visible_count = sum(1 for _, vis in record.visibility_log if vis)
                    tkm.tracking_accuracy = visible_count / len(record.visibility_log)

            self._prev_statuses[tid] = curr_status

    def compute_mission_metrics(
        self,
        target_manager: TargetManager,
    ) -> MissionSearchMetrics:
        """Compute final aggregated mission metrics."""
        all_targets = target_manager.all_targets()
        total = len(all_targets)

        static_n = sum(1 for t in all_targets if t.target_type == TargetType.STATIC)
        dynamic_n = sum(1 for t in all_targets if t.target_type == TargetType.DYNAMIC)
        tv_n = sum(
            1 for t in all_targets if t.target_type == TargetType.TIME_VARYING
        )

        detected_n = sum(
            1 for t in all_targets if t.status != TargetStatus.UNDISCOVERED
        )
        completed_n = len(target_manager.completed_targets())
        lost_n = len(target_manager.lost_targets())

        # Detection times
        det_times = [
            dm.first_detection_time
            for dm in self._detection_metrics.values()
            if dm.first_detection_time is not None
        ]
        avg_det_time = float(sum(det_times) / len(det_times)) if det_times else 0.0

        # Assignment delays
        delays = [
            dm.assignment_delay
            for dm in self._detection_metrics.values()
            if dm.assignment_delay is not None
        ]
        avg_delay = float(sum(delays) / len(delays)) if delays else 0.0

        # Search metrics
        search_durations = [
            sm.search_duration_s
            for sm in self._search_metrics.values()
            if sm.search_success and sm.search_duration_s > 0.0
        ]
        successful = sum(1 for sm in self._search_metrics.values() if sm.search_success)
        failed = sum(1 for sm in self._search_metrics.values() if not sm.search_success)
        avg_search_dur = (
            float(sum(search_durations) / len(search_durations))
            if search_durations
            else 0.0
        )

        # Tracking metrics
        track_durs = [
            tkm.tracking_duration_s for tkm in self._tracking_metrics.values()
        ]
        total_track_dur = float(sum(track_durs))
        avg_track_dur = float(sum(track_durs) / len(track_durs)) if track_durs else 0.0
        accuracies = [
            tkm.tracking_accuracy
            for tkm in self._tracking_metrics.values()
            if tkm.tracking_accuracy > 0.0
        ]
        avg_accuracy = float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
        total_loss = sum(
            tkm.loss_events for tkm in self._tracking_metrics.values()
        )
        total_reacq = sum(
            tkm.reacquisition_events for tkm in self._tracking_metrics.values()
        )

        # Mission timing
        search_dur = 0.0
        if self._search_start_time is not None and self._mission_end_time is not None:
            search_dur = self._mission_end_time - self._search_start_time

        return MissionSearchMetrics(
            num_uavs=self._num_uavs,
            environment_type=self._env_type,
            coverage_at_search_start=self._coverage_at_start,
            coverage_at_mission_end=self._coverage_at_end,
            total_targets=total,
            static_targets=static_n,
            dynamic_targets=dynamic_n,
            time_varying_targets=tv_n,
            detected_count=detected_n,
            detection_rate=detected_n / total if total > 0 else 0.0,
            avg_first_detection_time=avg_det_time,
            avg_assignment_delay=avg_delay,
            successful_searches=successful,
            failed_searches=failed,
            search_success_rate=successful / total if total > 0 else 0.0,
            avg_search_duration_s=avg_search_dur,
            total_tracking_duration_s=total_track_dur,
            avg_tracking_duration_s=avg_track_dur,
            avg_tracking_accuracy=avg_accuracy,
            total_loss_events=total_loss,
            total_reacquisition_events=total_reacq,
            completed_targets=completed_n,
            lost_targets=lost_n,
            mission_completion_time=self._mission_end_time,
            search_phase_duration_s=search_dur,
        )


# ------------------------------------------------------------------
# CSV export utilities
# ------------------------------------------------------------------

MISSION_CSV_FIELDS = [
    "num_uavs",
    "environment_type",
    "coverage_at_search_start",
    "coverage_at_mission_end",
    "total_targets",
    "static_targets",
    "dynamic_targets",
    "time_varying_targets",
    "detected_count",
    "detection_rate",
    "avg_first_detection_time",
    "avg_assignment_delay",
    "successful_searches",
    "failed_searches",
    "search_success_rate",
    "avg_search_duration_s",
    "total_tracking_duration_s",
    "avg_tracking_duration_s",
    "avg_tracking_accuracy",
    "total_loss_events",
    "total_reacquisition_events",
    "completed_targets",
    "lost_targets",
    "mission_completion_time",
    "search_phase_duration_s",
]

PER_TARGET_CSV_FIELDS = [
    "target_id",
    "target_type",
    "first_detection_time",
    "assignment_delay",
    "search_success",
    "search_duration_s",
    "tracking_duration_s",
    "loss_events",
    "reacquisition_events",
    "tracking_accuracy",
    "assigned_uav",
    "final_status",
]


def write_mission_metrics_csv(
    metrics: MissionSearchMetrics,
    output_path: Path,
) -> None:
    """Append one row to the mission metrics CSV (creates file + header if new)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MISSION_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                field: getattr(metrics, field, "")
                for field in MISSION_CSV_FIELDS
            }
        )


def write_per_target_csv(
    collector: SearchMetricsCollector,
    target_manager: TargetManager,
    output_path: Path,
) -> None:
    """Write per-target detail rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_TARGET_CSV_FIELDS)
        writer.writeheader()
        for target in target_manager.all_targets():
            tid = target.target_id
            dm = collector._detection_metrics.get(tid)
            sm = collector._search_metrics.get(tid)
            tkm = collector._tracking_metrics.get(tid)
            writer.writerow(
                {
                    "target_id": tid,
                    "target_type": target.target_type.value,
                    "first_detection_time": dm.first_detection_time if dm else "",
                    "assignment_delay": dm.assignment_delay if dm else "",
                    "search_success": sm.search_success if sm else False,
                    "search_duration_s": sm.search_duration_s if sm else 0.0,
                    "tracking_duration_s": tkm.tracking_duration_s if tkm else 0.0,
                    "loss_events": tkm.loss_events if tkm else 0,
                    "reacquisition_events": tkm.reacquisition_events if tkm else 0,
                    "tracking_accuracy": tkm.tracking_accuracy if tkm else 0.0,
                    "assigned_uav": target.assigned_uav,
                    "final_status": target.status.value,
                }
            )
