"""
Moving-target experiment runner.

Sweeps:
  seeds  : 5 deterministic seeds
  speeds : 0.0, 0.5, 1.0, 1.5, 2.0  m/s
  UAVs   : 10
  duration: 500 s (configurable)

Each combination runs as a fully self-contained simulation.
All results are written to experiments/results/moving_targets/.

Usage
-----
    python -m src.evaluation.run_moving_target_experiment
    python -m src.evaluation.run_moving_target_experiment --seeds 1 2 3 --speeds 0.5 1.5
    python -m src.evaluation.run_moving_target_experiment --quick   # single seed/speed smoke-test
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.config.search_config import SearchConfig, TargetSpawnConfig
from src.environment.world import World
from src.evaluation.search_metrics import (
    SearchMetricsCollector,
    write_mission_metrics_csv,
    write_per_target_csv,
)
from src.search.mission_phase import MissionOrchestrator, MissionPhase
from src.simulation.simulation_engine import SimulationEngine

# ── Experiment parameters ────────────────────────────────────────────────────

DEFAULT_SEEDS: list[int] = [42, 73, 137, 256, 999]
DEFAULT_SPEEDS: list[float] = [0.0, 0.5, 1.0, 1.5, 2.0]
NUM_UAVS = 10
DURATION_S = 500.0
BASE_CONFIG = Path("configs/experiments/search_dynamic_10uav.yaml")
RESULTS_DIR = Path("experiments/results/moving_targets")

# 8 moving targets per run: split across DYNAMIC + RANDOM_WALK + WAYPOINT_PATROL
MOVING_TARGET_COUNTS = {
    "count_static": 0,
    "count_dynamic": 4,
    "count_time_varying": 0,
    "count_random_walk": 2,
    "count_waypoint_patrol": 2,
}


# ── Engine builder ───────────────────────────────────────────────────────────

def _build_engine(
    base_cfg_path: Path,
    seed: int,
    speed_mps: float,
    duration_s: float = DURATION_S,
) -> SimulationEngine:
    """
    Build a SimulationEngine with target speed overridden to speed_mps.

    Uses a deterministic RNG derived from seed for every stochastic component.
    """
    config = load_config(base_cfg_path)

    # Override duration
    from dataclasses import replace
    sim_cfg = replace(config, duration=duration_s, num_uavs=NUM_UAVS)

    # Override search targets: all moving at the given speed
    if sim_cfg.search is None:
        raise ValueError(f"No search config in {base_cfg_path}")

    new_targets = replace(
        sim_cfg.search.targets,
        **MOVING_TARGET_COUNTS,
        dynamic_speed_min=max(speed_mps * 0.9, 1e-6) if speed_mps > 0 else 0.0,
        dynamic_speed_max=speed_mps * 1.1 if speed_mps > 0 else 1e-6,
        seed=seed,
    )
    # For 0 m/s: make all targets static instead
    if speed_mps == 0.0:
        new_targets = replace(
            sim_cfg.search.targets,
            count_static=8,
            count_dynamic=0,
            count_time_varying=0,
            count_random_walk=0,
            count_waypoint_patrol=0,
            seed=seed,
        )

    # Lower the exploration threshold so search starts quickly even at 500s
    from src.config.search_config import MissionConfig
    new_mission = replace(
        sim_cfg.search.mission,
        exploration_completion_threshold=0.20,
        min_coverage_before_search=0.15,
        search_timeout_s=duration_s,
    )
    new_search = replace(sim_cfg.search, targets=new_targets, mission=new_mission)
    sim_cfg = replace(sim_cfg, search=new_search)

    world = World.from_config(sim_cfg.environment, sim_cfg.uav)
    spawn_center = np.array(
        [sim_cfg.spawn_center_x, sim_cfg.spawn_center_y], dtype=np.float64
    )
    agents = spawn_uavs(
        count=sim_cfg.num_uavs,
        center=spawn_center,
        spread_radius=sim_cfg.uav.initial_spread_radius,
        mission_radius=sim_cfg.aggregation.mission_region_radius,
        max_speed=sim_cfg.uav.max_speed,
        max_angular_velocity=sim_cfg.uav.max_angular_velocity,
        seed=seed,
        spawn_mode=sim_cfg.uav.spawn_mode,  # type: ignore[arg-type]
        spawn_angular_noise=sim_cfg.uav.spawn_angular_noise,
    )
    aggregation = SelfAggregationController(
        config=sim_cfg.aggregation,
        uav_config=sim_cfg.uav,
        rng=np.random.default_rng(seed),
    )
    engine = SimulationEngine(world, agents, aggregation, sim_cfg)
    orchestrator = MissionOrchestrator(
        config=sim_cfg.search,
        world=world,
        agents=agents,
        rng=np.random.default_rng(seed + 1),
    )
    engine.mission_orchestrator = orchestrator
    return engine


# ── Single-run executor ───────────────────────────────────────────────────────

def run_one(
    seed: int,
    speed_mps: float,
    output_dir: Path,
    duration_s: float = DURATION_S,
    base_config: Path = BASE_CONFIG,
    verbose: bool = True,
) -> dict:
    """
    Execute one (seed, speed) combination and write CSVs.

    Returns a summary dict for the console table.
    """
    label = f"seed{seed}_speed{speed_mps:.1f}"
    if verbose:
        print(f"  [{label}] starting …", end=" ", flush=True)

    t_wall = time.perf_counter()
    engine = _build_engine(base_config, seed, speed_mps, duration_s)
    orch = engine.mission_orchestrator
    assert orch is not None

    env_type = "static" if speed_mps == 0.0 else "dynamic"
    collector = SearchMetricsCollector(
        num_uavs=NUM_UAVS,
        environment_type=env_type,
        seed=seed,
        target_speed_mps=speed_mps,
    )
    search_started = False

    while engine.time_s < engine.config.duration:
        engine.step()
        collector.tick()
        phase = orch.phase

        if phase == MissionPhase.SEARCHING and not search_started:
            search_started = True
            collector.record_search_start(
                engine.time_s,
                engine.world.map.explored_fraction(),
            )

        if phase in (MissionPhase.SEARCHING, MissionPhase.COMPLETED):
            collector.update(orch.target_manager, orch.tracker, engine.time_s)

        if phase == MissionPhase.COMPLETED:
            break

    collector.record_mission_end(
        engine.time_s, engine.world.map.explored_fraction()
    )

    mm = collector.compute_mission_metrics(orch.target_manager, orch.tracker)
    elapsed = time.perf_counter() - t_wall

    # Write CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    write_mission_metrics_csv(mm, output_dir / "mission_summary.csv")
    write_per_target_csv(collector, orch.target_manager,
                         output_dir / f"{label}_per_target.csv")

    summary = {
        "seed": seed,
        "speed": speed_mps,
        "det_rate": mm.detection_rate,
        "track_success": mm.search_success_rate,
        "avg_det_time": mm.avg_first_detection_time,
        "avg_track_dur": mm.avg_tracking_duration_s,
        "avg_loc_err": mm.avg_localization_error_m,
        "track_loss": mm.total_loss_events,
        "reacq_rate": mm.reacquisition_rate,
        "handovers": mm.successful_handovers,
        "uav_util": mm.uav_utilization,
        "fps": mm.sim_fps,
        "wall_s": round(elapsed, 1),
        "completed": mm.completed_targets,
        "total": mm.total_targets,
    }

    if verbose:
        print(
            f"done  det={mm.detection_rate:.0%}  "
            f"track={mm.search_success_rate:.0%}  "
            f"loc_err={mm.avg_localization_error_m:.2f}m  "
            f"ho={mm.successful_handovers}  "
            f"fps={mm.sim_fps:.0f}  "
            f"wall={elapsed:.1f}s"
        )
    return summary


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_sweep(
    seeds: list[int],
    speeds: list[float],
    output_dir: Path,
    duration_s: float = DURATION_S,
    base_config: Path = BASE_CONFIG,
) -> list[dict]:
    """Run full seed × speed grid. Returns list of summary dicts."""
    summaries: list[dict] = []
    total = len(seeds) * len(speeds)
    done = 0
    for speed in speeds:
        print(f"\n── Speed = {speed:.1f} m/s ──────────────────────────")
        for seed in seeds:
            try:
                s = run_one(seed, speed, output_dir, duration_s, base_config)
                summaries.append(s)
            except Exception as exc:
                print(f"  [FAIL seed={seed} speed={speed}] {exc}")
            done += 1
            print(f"  Progress: {done}/{total}", flush=True)
    return summaries


def _print_table(summaries: list[dict]) -> None:
    """Print a compact ASCII summary table."""
    print("\n" + "=" * 90)
    print(f"{'Speed':>6} {'Seed':>5} {'DetRate':>8} {'TrackSucc':>10} "
          f"{'AvgDetT':>8} {'LocErr':>7} {'Loss':>5} {'HO':>4} "
          f"{'Util':>5} {'FPS':>6} {'Wall':>6}")
    print("-" * 90)
    for s in sorted(summaries, key=lambda x: (x["speed"], x["seed"])):
        print(
            f"{s['speed']:>6.1f} {s['seed']:>5d} "
            f"{s['det_rate']:>8.1%} {s['track_success']:>10.1%} "
            f"{s['avg_det_time']:>8.1f} {s['avg_loc_err']:>7.3f} "
            f"{s['track_loss']:>5d} {s['handovers']:>4d} "
            f"{s['uav_util']:>5.1%} {s['fps']:>6.0f} {s['wall_s']:>6.1f}s"
        )
    print("=" * 90)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moving-target sweep experiment runner"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--speeds", nargs="+", type=float, default=DEFAULT_SPEEDS)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--quick", action="store_true",
        help="Single seed/speed smoke-test (seed=42, speed=1.0)"
    )
    parser.add_argument("--config", type=Path, default=BASE_CONFIG)
    args = parser.parse_args()

    if args.quick:
        print("Quick smoke-test: seed=42, speed=1.0 m/s, 10 UAVs")
        run_one(42, 1.0, args.output, args.duration, args.config)
        return

    print(
        f"Moving-target sweep: {len(args.seeds)} seeds × "
        f"{len(args.speeds)} speeds × {NUM_UAVS} UAVs × {args.duration:.0f}s"
    )
    summaries = run_sweep(args.seeds, args.speeds, args.output,
                          args.duration, args.config)
    _print_table(summaries)
    print(f"\nCSVs written to: {args.output}")


if __name__ == "__main__":
    main()
