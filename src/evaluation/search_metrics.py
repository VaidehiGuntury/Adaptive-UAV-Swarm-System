"""
Search, tracking, and mission metrics for the Target Search extension.

Covers all required reporting categories:
  - Detection metrics (rate, time, false positives/negatives)
  - Tracking metrics (success rate, loss, reacquisition, localization error)
  - Multi-UAV handover metrics (successful/failed handovers, utilization)
  - Computational performance (FPS, runtime)
  - Mission-level aggregation

CSV export follows the same pattern as run_experiment.py.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.search.target import Target, TargetStatus, TargetType
from src.search.target_manager import TargetManager
from src.search.tracker import TargetTracker


# ──────────────────────────────────────────────────────────────────────────────
# Per-target dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetDetectionMetrics:
    target_id: int
    target_type: str
    target_speed: float = 0.0
    first_detection_time: Optional[float] = None
    rediscovery_time: Optional[float] = None
    assignment_delay: Optional[float] = None
    false_positive: bool = False   # detected but never actually in range
    false_negative: bool = False   # was in range but never detected


@dataclass
class TargetSearchMetrics:
    target_id: int
    target_type: str
    search_start_time: Optional[float] = None
    search_end_time: Optional[float] = None
    search_duration_s: float = 0.0
    search_success: bool = False
    assigned_uav: Optional[int] = None


@dataclass
class TargetTrackingMetrics:
    target_id: int
    target_type: str
    tracking_duration_s: float = 0.0
    loss_events: int = 0
    reacquisition_events: int = 0
    tracking_accuracy: float = 0.0
    avg_localization_error: float = 0.0
    assigned_uav: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Mission-level aggregated metrics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MissionSearchMetrics:
    """All metrics required by the experiment prompt, in one dataclass."""

    # ── Experiment parameters ──────────────────────────────────────────
    seed: int = 0
    num_uavs: int = 0
    environment_type: str = "static"
    target_speed_mps: float = 0.0          # nominal target speed for this run

    # ── Coverage ──────────────────────────────────────────────────────
    coverage_at_search_start: float = 0.0
    coverage_at_mission_end: float = 0.0

    # ── Target population ─────────────────────────────────────────────
    total_targets: int = 0
    static_targets: int = 0
    dynamic_targets: int = 0
    time_varying_targets: int = 0
    random_walk_targets: int = 0
    waypoint_patrol_targets: int = 0

    # ── Detection ─────────────────────────────────────────────────────
    detected_count: int = 0
    detection_rate: float = 0.0
    avg_first_detection_time: float = 0.0
    avg_detection_delay: float = 0.0       # time from search-phase start to first detection
    avg_assignment_delay: float = 0.0
    false_positives: int = 0
    false_negatives: int = 0

    # ── Search ────────────────────────────────────────────────────────
    successful_searches: int = 0
    failed_searches: int = 0
    search_success_rate: float = 0.0
    avg_search_duration_s: float = 0.0

    # ── Tracking ──────────────────────────────────────────────────────
    total_tracking_duration_s: float = 0.0
    avg_tracking_duration_s: float = 0.0
    avg_tracking_accuracy: float = 0.0
    avg_localization_error_m: float = 0.0
    total_loss_events: int = 0
    total_reacquisition_events: int = 0
    reacquisition_rate: float = 0.0

    # ── Multi-UAV handover ────────────────────────────────────────────
    successful_handovers: int = 0
    failed_handovers: int = 0
    avg_assigned_targets_per_uav: float = 0.0
    uav_utilization: float = 0.0           # fraction of UAVs that tracked ≥1 target

    # ── Mission completion ────────────────────────────────────────────
    completed_targets: int = 0
    lost_targets: int = 0
    mission_completion_time: Optional[float] = None
    search_phase_duration_s: float = 0.0

    # ── Computational performance ─────────────────────────────────────
    sim_fps: float = 0.0                   # simulation steps per wall-clock second
    wall_time_s: float = 0.0
    steps_executed: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Collector
# ──────────────────────────────────────────────────────────────────────────────

class SearchMetricsCollector:
    """
    Collects all search/tracking metrics tick-by-tick.

    Parameters
    ----------
    num_uavs         : int
    environment_type : str ("static" | "dynamic" | "mixed")
    seed             : int  RNG seed of this run
    target_speed_mps : float  nominal target speed for this experiment
    """

    def __init__(
        self,
        num_uavs: int,
        environment_type: str = "static",
        seed: int = 0,
        target_speed_mps: float = 0.0,
    ) -> None:
        self._num_uavs = num_uavs
        self._env_type = environment_type
        self._seed = seed
        self._speed = target_speed_mps

        self._detection_metrics: dict[int, TargetDetectionMetrics] = {}
        self._search_metrics: dict[int, TargetSearchMetrics] = {}
        self._tracking_metrics: dict[int, TargetTrackingMetrics] = {}

        self._search_start_time: float | None = None
        self._mission_end_time: float | None = None
        self._coverage_at_start: float = 0.0
        self._coverage_at_end: float = 0.0
        self._prev_statuses: dict[int, TargetStatus] = {}

        # Performance counters
        self._wall_start: float = time.perf_counter()
        self._steps: int = 0

        # UAV utilization tracking: uav_id → bool (tracked at least one target)
        self._uav_tracked: dict[int, bool] = {}

    # ── Lifecycle hooks ────────────────────────────────────────────────────

    def record_search_start(self, current_time: float, coverage: float) -> None:
        self._search_start_time = current_time
        self._coverage_at_start = coverage
        self._wall_start = time.perf_counter()

    def record_mission_end(self, current_time: float, coverage: float) -> None:
        self._mission_end_time = current_time
        self._coverage_at_end = coverage

    def tick(self) -> None:
        """Call once per simulation step to track step count."""
        self._steps += 1

    # ── Per-step update ───────────────────────────────────────────────────

    def update(
        self,
        target_manager: TargetManager,
        tracker: TargetTracker,
        current_time: float,
    ) -> None:
        """Update all metrics from current simulation state."""
        for target in target_manager.all_targets():
            tid = target.target_id
            prev = self._prev_statuses.get(tid, TargetStatus.UNDISCOVERED)
            curr = target.status

            # ── Detection ──
            if tid not in self._detection_metrics:
                spd = float(target.speed) if target.speed > 0 else \
                      float(__import__("numpy").linalg.norm(target.velocity))
                self._detection_metrics[tid] = TargetDetectionMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                    target_speed=spd,
                )
            dm = self._detection_metrics[tid]
            if dm.first_detection_time is None and target.detection_time is not None:
                dm.first_detection_time = target.detection_time
            if (
                dm.assignment_delay is None
                and curr in (TargetStatus.ASSIGNED, TargetStatus.SEARCHING,
                             TargetStatus.TRACKING)
                and dm.first_detection_time is not None
            ):
                dm.assignment_delay = current_time - dm.first_detection_time

            # ── Search ──
            if tid not in self._search_metrics:
                self._search_metrics[tid] = TargetSearchMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                    assigned_uav=target.assigned_uav,
                )
            sm = self._search_metrics[tid]
            if prev != TargetStatus.SEARCHING and curr == TargetStatus.SEARCHING:
                sm.search_start_time = current_time
            if prev == TargetStatus.SEARCHING and curr == TargetStatus.TRACKING:
                sm.search_end_time = current_time
                sm.search_success = True
                if sm.search_start_time is not None:
                    sm.search_duration_s = current_time - sm.search_start_time

            # ── Tracking ──
            if tid not in self._tracking_metrics:
                self._tracking_metrics[tid] = TargetTrackingMetrics(
                    target_id=tid,
                    target_type=target.target_type.value,
                    assigned_uav=target.assigned_uav,
                )
            tkm = self._tracking_metrics[tid]
            rec = tracker.get_record(tid)
            if rec is not None:
                tkm.tracking_duration_s = rec.tracking_duration_s
                tkm.loss_events = rec.loss_events
                tkm.reacquisition_events = rec.reacquisition_events
                if rec.visibility_log:
                    vis_count = sum(1 for _, v in rec.visibility_log if v)
                    tkm.tracking_accuracy = vis_count / len(rec.visibility_log)
                if rec.localization_errors:
                    import numpy as np
                    tkm.avg_localization_error = float(
                        np.mean(rec.localization_errors)
                    )

            # UAV utilization
            if target.assigned_uav is not None and curr == TargetStatus.TRACKING:
                self._uav_tracked[target.assigned_uav] = True

            self._prev_statuses[tid] = curr

    # ── Final computation ─────────────────────────────────────────────────

    def compute_mission_metrics(
        self,
        target_manager: TargetManager,
        tracker: TargetTracker,
    ) -> MissionSearchMetrics:
        """Aggregate everything into a MissionSearchMetrics instance."""
        import numpy as np

        all_t = target_manager.all_targets()
        total = len(all_t)

        # Target type counts
        def _count(tt: TargetType) -> int:
            return sum(1 for t in all_t if t.target_type == tt)

        # Detection
        det_n = sum(1 for t in all_t if t.status != TargetStatus.UNDISCOVERED)
        # False negatives: undiscovered at end
        fn = sum(1 for t in all_t if t.status == TargetStatus.UNDISCOVERED)
        fp = sum(1 for dm in self._detection_metrics.values() if dm.false_positive)

        det_times = [
            dm.first_detection_time
            for dm in self._detection_metrics.values()
            if dm.first_detection_time is not None
        ]
        avg_det = float(np.mean(det_times)) if det_times else 0.0

        # Detection delay relative to search start
        if self._search_start_time is not None and det_times:
            delays = [max(0.0, t - self._search_start_time) for t in det_times]
            avg_delay = float(np.mean(delays))
        else:
            avg_delay = 0.0

        assign_delays = [
            dm.assignment_delay
            for dm in self._detection_metrics.values()
            if dm.assignment_delay is not None
        ]
        avg_assign_delay = float(np.mean(assign_delays)) if assign_delays else 0.0

        # Search
        succ = sum(1 for sm in self._search_metrics.values() if sm.search_success)
        fail = sum(1 for sm in self._search_metrics.values() if not sm.search_success)
        s_durs = [
            sm.search_duration_s for sm in self._search_metrics.values()
            if sm.search_success and sm.search_duration_s > 0
        ]
        avg_sdur = float(np.mean(s_durs)) if s_durs else 0.0

        # Tracking
        t_durs = [tkm.tracking_duration_s for tkm in self._tracking_metrics.values()]
        total_tdur = float(sum(t_durs))
        avg_tdur = float(np.mean(t_durs)) if t_durs else 0.0
        accs = [tkm.tracking_accuracy for tkm in self._tracking_metrics.values()
                if tkm.tracking_accuracy > 0]
        avg_acc = float(np.mean(accs)) if accs else 0.0
        loc_errs = [tkm.avg_localization_error for tkm in self._tracking_metrics.values()
                    if tkm.avg_localization_error > 0]
        avg_loc = float(np.mean(loc_errs)) if loc_errs else 0.0
        total_loss = sum(tkm.loss_events for tkm in self._tracking_metrics.values())
        total_reacq = sum(
            tkm.reacquisition_events for tkm in self._tracking_metrics.values()
        )
        reacq_rate = total_reacq / max(total_loss, 1)

        # Handover
        succ_ho = tracker.total_handovers()
        fail_ho = tracker.total_failed_handovers()

        # UAV utilization
        uav_util = (
            len(self._uav_tracked) / self._num_uavs if self._num_uavs > 0 else 0.0
        )
        # Average targets per UAV (only UAVs that received at least one assignment)
        from src.search.assignment import AssignmentRecord
        # approximate from tracking metrics
        uav_target_counts: dict[int, int] = {}
        for tkm in self._tracking_metrics.values():
            if tkm.assigned_uav is not None:
                uav_target_counts[tkm.assigned_uav] = \
                    uav_target_counts.get(tkm.assigned_uav, 0) + 1
        avg_tpu = (
            float(np.mean(list(uav_target_counts.values())))
            if uav_target_counts else 0.0
        )

        # Completion
        comp_n = len(target_manager.completed_targets())
        lost_n = len(target_manager.lost_targets())

        # Timing
        search_dur = 0.0
        if self._search_start_time is not None and self._mission_end_time is not None:
            search_dur = self._mission_end_time - self._search_start_time

        # Performance
        wall = time.perf_counter() - self._wall_start
        fps = self._steps / max(wall, 1e-9)

        return MissionSearchMetrics(
            seed=self._seed,
            num_uavs=self._num_uavs,
            environment_type=self._env_type,
            target_speed_mps=self._speed,
            coverage_at_search_start=self._coverage_at_start,
            coverage_at_mission_end=self._coverage_at_end,
            total_targets=total,
            static_targets=_count(TargetType.STATIC),
            dynamic_targets=_count(TargetType.DYNAMIC),
            time_varying_targets=_count(TargetType.TIME_VARYING),
            random_walk_targets=_count(TargetType.RANDOM_WALK),
            waypoint_patrol_targets=_count(TargetType.WAYPOINT_PATROL),
            detected_count=det_n,
            detection_rate=det_n / total if total > 0 else 0.0,
            avg_first_detection_time=avg_det,
            avg_detection_delay=avg_delay,
            avg_assignment_delay=avg_assign_delay,
            false_positives=fp,
            false_negatives=fn,
            successful_searches=succ,
            failed_searches=fail,
            search_success_rate=succ / total if total > 0 else 0.0,
            avg_search_duration_s=avg_sdur,
            total_tracking_duration_s=total_tdur,
            avg_tracking_duration_s=avg_tdur,
            avg_tracking_accuracy=avg_acc,
            avg_localization_error_m=avg_loc,
            total_loss_events=total_loss,
            total_reacquisition_events=total_reacq,
            reacquisition_rate=reacq_rate,
            successful_handovers=succ_ho,
            failed_handovers=fail_ho,
            avg_assigned_targets_per_uav=avg_tpu,
            uav_utilization=uav_util,
            completed_targets=comp_n,
            lost_targets=lost_n,
            mission_completion_time=self._mission_end_time,
            search_phase_duration_s=search_dur,
            sim_fps=fps,
            wall_time_s=wall,
            steps_executed=self._steps,
        )


# ──────────────────────────────────────────────────────────────────────────────
# CSV field lists
# ──────────────────────────────────────────────────────────────────────────────

MISSION_CSV_FIELDS = [
    "seed", "num_uavs", "environment_type", "target_speed_mps",
    "coverage_at_search_start", "coverage_at_mission_end",
    "total_targets", "static_targets", "dynamic_targets",
    "time_varying_targets", "random_walk_targets", "waypoint_patrol_targets",
    "detected_count", "detection_rate",
    "avg_first_detection_time", "avg_detection_delay", "avg_assignment_delay",
    "false_positives", "false_negatives",
    "successful_searches", "failed_searches", "search_success_rate",
    "avg_search_duration_s",
    "total_tracking_duration_s", "avg_tracking_duration_s",
    "avg_tracking_accuracy", "avg_localization_error_m",
    "total_loss_events", "total_reacquisition_events", "reacquisition_rate",
    "successful_handovers", "failed_handovers",
    "avg_assigned_targets_per_uav", "uav_utilization",
    "completed_targets", "lost_targets",
    "mission_completion_time", "search_phase_duration_s",
    "sim_fps", "wall_time_s", "steps_executed",
]

PER_TARGET_CSV_FIELDS = [
    "target_id", "target_type", "target_speed",
    "first_detection_time", "assignment_delay",
    "search_success", "search_duration_s",
    "tracking_duration_s", "loss_events", "reacquisition_events",
    "tracking_accuracy", "avg_localization_error",
    "assigned_uav", "final_status",
]


def write_mission_metrics_csv(
    metrics: MissionSearchMetrics,
    output_path: Path,
) -> None:
    """Append one row (creates header on first write)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MISSION_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({f: getattr(metrics, f, "") for f in MISSION_CSV_FIELDS})


def write_per_target_csv(
    collector: SearchMetricsCollector,
    target_manager: TargetManager,
    output_path: Path,
) -> None:
    """Write per-target detail rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PER_TARGET_CSV_FIELDS)
        writer.writeheader()
        for target in target_manager.all_targets():
            tid = target.target_id
            dm = collector._detection_metrics.get(tid)
            sm = collector._search_metrics.get(tid)
            tkm = collector._tracking_metrics.get(tid)
            spd = float(target.speed) if target.speed > 0 else float(
                __import__("numpy").linalg.norm(target.velocity)
            )
            writer.writerow({
                "target_id": tid,
                "target_type": target.target_type.value,
                "target_speed": round(spd, 4),
                "first_detection_time": dm.first_detection_time if dm else "",
                "assignment_delay": dm.assignment_delay if dm else "",
                "search_success": sm.search_success if sm else False,
                "search_duration_s": sm.search_duration_s if sm else 0.0,
                "tracking_duration_s": tkm.tracking_duration_s if tkm else 0.0,
                "loss_events": tkm.loss_events if tkm else 0,
                "reacquisition_events": tkm.reacquisition_events if tkm else 0,
                "tracking_accuracy": tkm.tracking_accuracy if tkm else 0.0,
                "avg_localization_error": tkm.avg_localization_error if tkm else 0.0,
                "assigned_uav": target.assigned_uav,
                "final_status": target.status.value,
            })
