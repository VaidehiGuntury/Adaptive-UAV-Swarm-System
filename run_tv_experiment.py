"""
Time-Varying Target Experiment Runner
Generates real simulation results for the report.
Runs 5 seeds x 3 scenario types on a 60x60 world.
"""
from __future__ import annotations
import csv, time, numpy as np
from dataclasses import replace, dataclass, field
from pathlib import Path
from typing import Optional

# ── patch sys.path ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config.loader import load_config, EnvironmentConfig
from src.config.search_config import (
    SearchConfig, TargetSpawnConfig, DetectionConfig, SearchBehaviourConfig,
    TrackingConfig, MissionConfig, AssignmentConfig, PriorityConfig,
    HandoverConfig
)
from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs, UAV
from src.environment.world import World
from src.search.mission_phase import MissionOrchestrator, MissionPhase
from src.search.target import TargetType, TargetStatus
from src.simulation.simulation_engine import SimulationEngine

# ── Constants ──────────────────────────────────────────────────────────────────
SEEDS        = [42, 73, 137, 256, 999]
WORLD_W      = 60.0
WORLD_H      = 60.0
N_UAVS       = 10
DT           = 0.1
SEARCH_STEPS = 800           # 80 s of search phase at dt=0.1
OUT_DIR      = Path("experiments/results/time_varying")

# ── Search config factory ──────────────────────────────────────────────────────

def make_search_config(
    n_static: int,
    n_dynamic: int,
    n_tv: int,
    seed: int,
    speed_min: float = 0.4,
    speed_max: float = 0.9,
) -> SearchConfig:
    return SearchConfig(
        targets=TargetSpawnConfig(
            count_static=n_static,
            count_dynamic=n_dynamic,
            count_time_varying=n_tv,
            count_random_walk=0,
            count_waypoint_patrol=0,
            spawn_margin=4.0,
            min_separation=6.0,
            dynamic_speed_min=speed_min,
            dynamic_speed_max=speed_max,
            seed=seed,
        ),
        detection=DetectionConfig(
            detection_radius=4.5,
            base_confidence=0.85,
            confidence_decay_rate=0.04,
            confidence_merge_alpha=0.6,
            dedup_time_window_s=2.0,
            min_detection_confidence=0.25,
        ),
        priority=PriorityConfig(),
        behaviour=SearchBehaviourConfig(
            direct_nav_staleness_s=4.0,
            spiral_initial_radius=3.0,
            spiral_step=2.5,
            spiral_max_radius=18.0,
            expanding_step_m=4.0,
            expanding_max_radius=22.0,
            replan_interval_s=1.5,
            arrival_threshold_m=1.0,
            loss_timeout_s=6.0,
            max_recovery_attempts=4,
        ),
        tracking=TrackingConfig(
            required_tracking_duration_s=8.0,
            history_max_length=200,
            reacquisition_search_radius=10.0,
            visibility_sample_interval_s=0.5,
        ),
        assignment=AssignmentConfig(
            max_targets_per_uav=3,
            reassignment_on_fail=True,
            fail_timeout_s=30.0,
            cost_distance_weight=0.6,
            cost_priority_weight=0.4,
        ),
        mission=MissionConfig(
            exploration_completion_threshold=0.01,   # start search immediately
            min_coverage_before_search=0.01,
            search_timeout_s=SEARCH_STEPS * DT + 10.0,
        ),
        handover=HandoverConfig(
            handover_enabled=True,
            handover_distance_ratio=0.55,
            handover_hysteresis_s=8.0,
        ),
    )


def build_engine(seed: int, search_cfg: SearchConfig) -> SimulationEngine:
    """Build a minimal simulation engine with the given search config."""
    from src.config.loader import (SimulationConfig, EnvironmentConfig,
                                   UAVConfig, AggregationConfig)

    env = EnvironmentConfig(
        width=WORLD_W, height=WORLD_H,
        obstacle_count=8,
        obstacle_min_radius=1.2, obstacle_max_radius=2.5,
        obstacle_seed=seed,
    )
    uav_cfg = UAVConfig(
        max_speed=1.5, max_angular_velocity=0.9,
        sensing_range=4.5, initial_spread_radius=12.0,
        spawn_mode="ring", spawn_angular_noise=0.15,
    )
    agg_cfg = AggregationConfig(
        d_c=0.5, d_0=10.0, k_a=1.0,
        turn_cost_weight=1.0, trail_penalty=8.0,
        cluster_penalty_weight=1.0, turn_penalty_weight=0.5,
        trail_penalty_weight=1.0, candidates_per_frontier=6,
        mission_region_radius=4.5, replan_interval=2.0,
    )
    sim_cfg = SimulationConfig(
        environment=env, uav=uav_cfg, aggregation=agg_cfg,
        num_uavs=N_UAVS, dt=DT,
        duration=SEARCH_STEPS * DT + 20.0,
        spawn_center_x=30.0, spawn_center_y=30.0,
        animation_interval_ms=50,
        search=search_cfg,
    )

    world = World.from_config(env, uav_cfg)
    agents = spawn_uavs(
        count=N_UAVS, center=np.array([30.0, 30.0]),
        spread_radius=12.0, mission_radius=4.5,
        max_speed=1.5, max_angular_velocity=0.9,
        seed=seed, spawn_mode="ring",
    )
    agg = SelfAggregationController(
        config=agg_cfg, uav_config=uav_cfg,
        rng=np.random.default_rng(seed),
    )
    engine = SimulationEngine(world, agents, agg, sim_cfg)
    orch = MissionOrchestrator(
        config=search_cfg, world=world, agents=agents,
        rng=np.random.default_rng(seed + 7),
    )
    engine.mission_orchestrator = orch
    return engine


# ── Per-target event log ───────────────────────────────────────────────────────

@dataclass
class TargetLog:
    target_id: int
    target_type: str
    spawn_time: float
    detection_time: Optional[float] = None
    detection_delay: Optional[float] = None   # time from search start to detection
    assigned_uav: Optional[int] = None
    assignment_time: Optional[float] = None
    tracking_start: Optional[float] = None
    tracking_end: Optional[float] = None
    tracking_duration: float = 0.0
    loss_events: int = 0
    reacq_events: int = 0
    handover_events: int = 0
    localization_errors: list = field(default_factory=list)
    avg_loc_error: float = 0.0
    tracking_accuracy: float = 0.0
    final_status: str = "undiscovered"
    # TIME_VARYING specific
    state_changes: list = field(default_factory=list)   # (time, was_moving)
    # confidence timeline: list of (time, confidence)
    confidence_timeline: list = field(default_factory=list)


def run_scenario(
    seed: int,
    scenario_name: str,
    n_static: int, n_dynamic: int, n_tv: int,
) -> tuple[list[TargetLog], dict]:
    """Run one seed of one scenario. Returns (target_logs, summary_metrics)."""

    search_cfg = make_search_config(n_static, n_dynamic, n_tv, seed)
    engine = build_engine(seed, search_cfg)
    orch = engine.mission_orchestrator

    target_logs: dict[int, TargetLog] = {}
    prev_status: dict[int, TargetStatus] = {}
    prev_moving: dict[int, bool] = {}
    search_start_time = None
    steps_in_search = 0

    t_wall_start = time.perf_counter()

    step = 0
    while step < SEARCH_STEPS + 200:   # extra 200 exploration steps
        m = engine.step()
        step += 1
        phase = orch.phase

        if phase == MissionPhase.SEARCHING:
            if search_start_time is None:
                search_start_time = engine.time_s
            steps_in_search += 1

        if phase in (MissionPhase.SEARCHING, MissionPhase.COMPLETED):
            tm = orch.target_manager

            # Initialise logs for new targets
            for target in tm.all_targets():
                tid = target.target_id
                if tid not in target_logs:
                    target_logs[tid] = TargetLog(
                        target_id=tid,
                        target_type=target.target_type.value,
                        spawn_time=target.creation_time,
                    )

                log = target_logs[tid]
                curr_status = target.status
                old_status = prev_status.get(tid, TargetStatus.UNDISCOVERED)

                # Detection event
                if (
                    log.detection_time is None
                    and target.detection_time is not None
                ):
                    log.detection_time = target.detection_time
                    if search_start_time is not None:
                        log.detection_delay = target.detection_time - search_start_time

                # Assignment event
                if (
                    log.assignment_time is None
                    and curr_status in (TargetStatus.ASSIGNED, TargetStatus.SEARCHING,
                                        TargetStatus.TRACKING)
                    and target.assigned_uav is not None
                ):
                    log.assignment_time = engine.time_s
                    log.assigned_uav = target.assigned_uav

                # Tracking start
                if old_status != TargetStatus.TRACKING and curr_status == TargetStatus.TRACKING:
                    if log.tracking_start is None:
                        log.tracking_start = engine.time_s

                # Tracking end
                if curr_status == TargetStatus.COMPLETED and log.tracking_end is None:
                    log.tracking_end = engine.time_s

                # Update from tracker record
                rec = orch.tracker.get_record(tid)
                if rec is not None:
                    log.tracking_duration = rec.tracking_duration_s
                    log.loss_events = rec.loss_events
                    log.reacq_events = rec.reacquisition_events
                    log.handover_events = rec.handover_events
                    if rec.localization_errors:
                        log.localization_errors = rec.localization_errors.copy()
                        log.avg_loc_error = float(np.mean(rec.localization_errors))
                    if rec.visibility_log:
                        vis_count = sum(1 for _, v in rec.visibility_log if v)
                        log.tracking_accuracy = vis_count / len(rec.visibility_log)

                # TIME_VARYING: detect state changes
                if target.target_type == TargetType.TIME_VARYING:
                    was_moving = prev_moving.get(tid, target.is_moving)
                    if was_moving != target.is_moving:
                        log.state_changes.append((engine.time_s, target.is_moving))

                # Confidence timeline (sample every 5 steps)
                if step % 5 == 0 and target.confidence > 0:
                    log.confidence_timeline.append(
                        (engine.time_s, target.confidence)
                    )

                # Final status
                log.final_status = curr_status.value

                prev_status[tid] = curr_status
                prev_moving[tid] = target.is_moving

        if phase == MissionPhase.COMPLETED:
            break

    wall_time = time.perf_counter() - t_wall_start

    # Build summary metrics
    logs = list(target_logs.values())
    detected = [l for l in logs if l.detection_time is not None]
    tracked = [l for l in logs if l.tracking_duration > 0]
    completed = [l for l in logs if l.final_status == "completed"]
    lost_perm = [l for l in logs if l.final_status == "lost"]
    tv_logs = [l for l in logs if l.target_type == "time_varying"]

    det_delays = [l.detection_delay for l in detected if l.detection_delay is not None]
    det_times  = [l.detection_time  for l in detected if l.detection_time  is not None]
    track_durs = [l.tracking_duration for l in tracked if l.tracking_duration > 0]
    loc_errs   = [l.avg_loc_error    for l in tracked if l.avg_loc_error > 0]
    accs       = [l.tracking_accuracy for l in tracked if l.tracking_accuracy > 0]

    all_loss = sum(l.loss_events for l in logs)
    all_reacq = sum(l.reacq_events for l in logs)
    all_ho    = sum(l.handover_events for l in logs)

    summary = {
        "seed":                seed,
        "scenario":            scenario_name,
        "total_targets":       len(logs),
        "n_static":            sum(1 for l in logs if l.target_type == "static"),
        "n_dynamic":           sum(1 for l in logs if l.target_type == "dynamic"),
        "n_time_varying":      len(tv_logs),
        "detected":            len(detected),
        "detection_rate":      len(detected) / max(len(logs), 1),
        "avg_det_delay_s":     float(np.mean(det_delays)) if det_delays else 0.0,
        "avg_det_time_s":      float(np.mean(det_times))  if det_times  else 0.0,
        "search_success_rate": len(completed) / max(len(logs), 1),
        "completed":           len(completed),
        "permanently_lost":    len(lost_perm),
        "avg_tracking_dur_s":  float(np.mean(track_durs)) if track_durs else 0.0,
        "avg_loc_error_m":     float(np.mean(loc_errs))   if loc_errs   else 0.0,
        "avg_tracking_acc":    float(np.mean(accs))        if accs       else 0.0,
        "total_loss_events":   all_loss,
        "total_reacq_events":  all_reacq,
        "reacq_rate":          all_reacq / max(all_loss, 1),
        "total_handovers":     all_ho,
        # TIME_VARYING specific
        "tv_detected":         sum(1 for l in tv_logs if l.detection_time is not None),
        "tv_detection_rate":   sum(1 for l in tv_logs if l.detection_time is not None) / max(len(tv_logs), 1),
        "tv_completed":        sum(1 for l in tv_logs if l.final_status == "completed"),
        "tv_state_changes":    sum(len(l.state_changes) for l in tv_logs),
        "tv_avg_loc_error":    float(np.mean([l.avg_loc_error for l in tv_logs if l.avg_loc_error > 0])) if any(l.avg_loc_error > 0 for l in tv_logs) else 0.0,
        "tv_avg_tracking_dur": float(np.mean([l.tracking_duration for l in tv_logs if l.tracking_duration > 0])) if any(l.tracking_duration > 0 for l in tv_logs) else 0.0,
        "search_start_time_s": search_start_time or 0.0,
        "search_steps":        steps_in_search,
        "wall_time_s":         round(wall_time, 2),
        "sim_fps":             round(step / wall_time, 1),
    }
    return logs, summary


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("Static Only",          8, 0, 0),
    ("Dynamic Only",         0, 8, 0),
    ("Time-Varying Only",    0, 0, 8),
    ("Mixed (2S+2D+4TV)",    2, 2, 4),
    ("Mixed (0S+4D+4TV)",    0, 4, 4),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    all_target_logs = []

    for scenario_name, ns, nd, ntv in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}  (static={ns} dynamic={nd} tv={ntv})")
        print('='*60)
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            try:
                logs, summary = run_scenario(seed, scenario_name, ns, nd, ntv)
                all_summaries.append(summary)
                for log in logs:
                    log_dict = {
                        "scenario": scenario_name,
                        "seed": seed,
                        **log.__dict__,
                    }
                    # Flatten non-serialisable fields
                    log_dict["localization_errors"] = f"{log.avg_loc_error:.4f}"
                    log_dict["state_changes"] = len(log.state_changes)
                    log_dict["confidence_timeline"] = len(log.confidence_timeline)
                    all_target_logs.append(log_dict)
                print(
                    f"det={summary['detection_rate']:.0%}  "
                    f"track={summary['search_success_rate']:.0%}  "
                    f"tv_det={summary['tv_detected']}/{summary['n_time_varying']}  "
                    f"tv_comp={summary['tv_completed']}  "
                    f"loc={summary['avg_loc_error_m']:.2f}m  "
                    f"ho={summary['total_handovers']}  "
                    f"wall={summary['wall_time_s']}s"
                )
            except Exception as exc:
                import traceback
                print(f"  FAIL: {exc}")
                traceback.print_exc()

    # Write summary CSV
    if all_summaries:
        summary_path = OUT_DIR / "scenario_summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)
        print(f"\nSummary CSV: {summary_path}")

    # Write per-target CSV
    if all_target_logs:
        safe_cols = [k for k in all_target_logs[0].keys()
                     if k not in ("confidence_timeline",)]
        pt_path = OUT_DIR / "per_target_detail.csv"
        with open(pt_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=safe_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_target_logs)
        print(f"Per-target CSV: {pt_path}")

    # Print final aggregated table
    _print_report(all_summaries)
    return all_summaries


def _print_report(summaries: list[dict]):
    import statistics

    print("\n" + "=" * 90)
    print("FINAL RESULTS: TIME-VARYING TARGET EXPERIMENT")
    print("=" * 90)

    by_scenario = {}
    for s in summaries:
        by_scenario.setdefault(s["scenario"], []).append(s)

    cols = [
        ("Detection Rate",      "detection_rate",      ".1%"),
        ("TV Detection Rate",   "tv_detection_rate",   ".1%"),
        ("Search Success Rate", "search_success_rate", ".1%"),
        ("TV Completed",        "tv_completed",        ".1f"),
        ("Avg Det Delay (s)",   "avg_det_delay_s",     ".1f"),
        ("Avg Track Dur (s)",   "avg_tracking_dur_s",  ".1f"),
        ("TV Avg Track Dur(s)", "tv_avg_tracking_dur", ".1f"),
        ("Avg Loc Error (m)",   "avg_loc_error_m",     ".3f"),
        ("TV Avg Loc Err (m)",  "tv_avg_loc_error",    ".3f"),
        ("Loss Events",         "total_loss_events",   ".1f"),
        ("Reacq Rate",          "reacq_rate",          ".2f"),
        ("Handovers",           "total_handovers",     ".1f"),
        ("Sim FPS",             "sim_fps",             ".0f"),
    ]

    print(f"\n{'Scenario':<28}", end="")
    print(f"{'N':>2}  ", end="")
    for label, _, _ in cols:
        print(f"{label[:16]:>16}", end="")
    print()
    print("-" * 90)

    for scenario_name, rows in by_scenario.items():
        means = {}
        for _, key, _ in cols:
            vals = [r[key] for r in rows if key in r]
            means[key] = statistics.mean(vals) if vals else 0.0

        print(f"{scenario_name:<28}{len(rows):>2}  ", end="")
        for _, key, fmt in cols:
            val = means[key]
            try:
                formatted = format(val, fmt)
            except Exception:
                formatted = str(round(val, 3))
            print(f"{formatted:>16}", end="")
        print()

    print("=" * 90)
    print(f"Seeds: {SEEDS}   World: {WORLD_W}x{WORLD_H}m   UAVs: {N_UAVS}")
    print(f"Search phase: {SEARCH_STEPS * DT:.0f}s   dt: {DT}s")


if __name__ == "__main__":
    main()
