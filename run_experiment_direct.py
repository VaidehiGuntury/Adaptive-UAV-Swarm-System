"""
Direct experiment runner that skips slow BSA exploration.

Forces the search phase to start immediately after minimal exploration,
so we can generate real results from the search/tracking subsystem alone.

This is valid because:
1. The search module is independent of exploration duration.
2. We are testing search & tracking, not exploration coverage.
3. Exploration coverage at search start is reported as a metric.

Usage:
    python run_experiment_direct.py
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.config.search_config import MissionConfig
from src.environment.world import World
from src.evaluation.search_metrics import (
    SearchMetricsCollector,
    write_mission_metrics_csv,
    write_per_target_csv,
)
from src.search.mission_phase import MissionOrchestrator, MissionPhase
from src.simulation.simulation_engine import SimulationEngine

SEEDS = [42, 73, 137, 256, 999]
SPEEDS = [0.0, 0.5, 1.0, 1.5, 2.0]
NUM_UAVS = 10
SEARCH_DURATION = 300.0          # seconds of search phase
BASE_CFG = Path("configs/experiments/search_dynamic_10uav.yaml")
OUT_DIR = Path("experiments/results/moving_targets")
MOVING_COUNTS = dict(
    count_static=0, count_dynamic=4,
    count_time_varying=0, count_random_walk=2, count_waypoint_patrol=2,
)


def build_engine(seed: int, speed: float) -> SimulationEngine:
    config = load_config(BASE_CFG)

    # Force immediate search transition (threshold = 0.01)
    new_mission = MissionConfig(
        exploration_completion_threshold=0.01,
        min_coverage_before_search=0.01,
        search_timeout_s=SEARCH_DURATION + 60.0,
    )
    # Set target speed
    if speed == 0.0:
        from src.config.search_config import TargetSpawnConfig
        new_targets = replace(
            config.search.targets,
            count_static=8, count_dynamic=0, count_time_varying=0,
            count_random_walk=0, count_waypoint_patrol=0,
            seed=seed,
        )
    else:
        from src.config.search_config import TargetSpawnConfig
        new_targets = replace(
            config.search.targets,
            **MOVING_COUNTS,
            dynamic_speed_min=speed * 0.95,
            dynamic_speed_max=speed * 1.05,
            seed=seed,
        )
    new_search = replace(config.search, targets=new_targets, mission=new_mission)
    # Shorten total sim duration
    config = replace(config, search=new_search,
                     duration=SEARCH_DURATION + 30.0,
                     num_uavs=NUM_UAVS)

    world = World.from_config(config.environment, config.uav)
    spawn_center = np.array([config.spawn_center_x, config.spawn_center_y],
                             dtype=np.float64)
    agents = spawn_uavs(
        count=config.num_uavs,
        center=spawn_center,
        spread_radius=config.uav.initial_spread_radius,
        mission_radius=config.aggregation.mission_region_radius,
        max_speed=config.uav.max_speed,
        max_angular_velocity=config.uav.max_angular_velocity,
        seed=seed,
        spawn_mode=config.uav.spawn_mode,
        spawn_angular_noise=config.uav.spawn_angular_noise,
    )
    agg = SelfAggregationController(
        config=config.aggregation,
        uav_config=config.uav,
        rng=np.random.default_rng(seed),
    )
    engine = SimulationEngine(world, agents, agg, config)
    orch = MissionOrchestrator(
        config=config.search, world=world, agents=agents,
        rng=np.random.default_rng(seed + 1),
    )
    engine.mission_orchestrator = orch
    return engine


def run_one(seed: int, speed: float) -> dict:
    env_type = "static" if speed == 0.0 else "dynamic"
    print(f"  seed={seed}  speed={speed:.1f}m/s ...", end=" ", flush=True)
    t0 = time.perf_counter()

    engine = build_engine(seed, speed)
    orch = engine.mission_orchestrator
    collector = SearchMetricsCollector(
        num_uavs=NUM_UAVS, environment_type=env_type,
        seed=seed, target_speed_mps=speed,
    )
    search_started = False

    while engine.time_s < engine.config.duration:
        engine.step()
        collector.tick()
        phase = orch.phase

        if phase == MissionPhase.SEARCHING and not search_started:
            search_started = True
            collector.record_search_start(
                engine.time_s, engine.world.map.explored_fraction()
            )

        if phase in (MissionPhase.SEARCHING, MissionPhase.COMPLETED):
            collector.update(orch.target_manager, orch.tracker, engine.time_s)

        if phase == MissionPhase.COMPLETED:
            break

    collector.record_mission_end(
        engine.time_s, engine.world.map.explored_fraction()
    )
    mm = collector.compute_mission_metrics(orch.target_manager, orch.tracker)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_mission_metrics_csv(mm, OUT_DIR / "mission_summary.csv")
    write_per_target_csv(
        collector, orch.target_manager,
        OUT_DIR / f"seed{seed}_speed{speed:.1f}_per_target.csv",
    )

    elapsed = time.perf_counter() - t0
    print(
        f"det={mm.detection_rate:.0%}  "
        f"track={mm.search_success_rate:.0%}  "
        f"loc_err={mm.avg_localization_error_m:.2f}m  "
        f"ho={mm.successful_handovers}  "
        f"fps={mm.sim_fps:.0f}  "
        f"wall={elapsed:.1f}s"
    )
    return {
        "seed": seed, "speed": speed,
        "det_rate": mm.detection_rate,
        "track_success": mm.search_success_rate,
        "avg_det_time": mm.avg_first_detection_time,
        "avg_loc_err": mm.avg_localization_error_m,
        "loss": mm.total_loss_events,
        "reacq": mm.total_reacquisition_events,
        "handovers": mm.successful_handovers,
        "uav_util": mm.uav_utilization,
        "fps": mm.sim_fps,
        "wall_s": round(elapsed, 1),
        "completed": mm.completed_targets,
        "total": mm.total_targets,
    }


def main():
    summaries = []
    for speed in SPEEDS:
        print(f"\n=== Speed {speed:.1f} m/s ===")
        for seed in SEEDS:
            try:
                s = run_one(seed, speed)
                summaries.append(s)
            except Exception as exc:
                import traceback
                print(f"  ERROR: {exc}")
                traceback.print_exc()

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Speed':>6} {'Seed':>5} {'DetRate':>8} {'TrackSucc':>10} "
          f"{'AvgDetT':>8} {'LocErr':>7} {'Loss':>5} {'Reacq':>6} "
          f"{'HO':>4} {'Util':>6} {'FPS':>6} {'Wall':>6}")
    print("-" * 95)
    for s in sorted(summaries, key=lambda x: (x["speed"], x["seed"])):
        print(
            f"{s['speed']:>6.1f} {s['seed']:>5d} "
            f"{s['det_rate']:>8.1%} {s['track_success']:>10.1%} "
            f"{s['avg_det_time']:>8.1f} {s['avg_loc_err']:>7.3f} "
            f"{s['loss']:>5d} {s['reacq']:>6d} "
            f"{s['handovers']:>4d} {s['uav_util']:>6.1%} "
            f"{s['fps']:>6.0f} {s['wall_s']:>6.1f}s"
        )
    print("=" * 95)
    print(f"\nResults written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
