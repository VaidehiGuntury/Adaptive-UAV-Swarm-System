"""
Batch experiment runner for the Target Search & Tracking extension.

Runs all 9 experiment configurations (3 fleet sizes × 3 environment types)
and writes per-mission and per-target CSV results.

Usage (from project root):
    python -m src.evaluation.run_search_experiment
    python -m src.evaluation.run_search_experiment --config configs/experiments/search_static_10uav.yaml
    python -m src.evaluation.run_search_experiment --all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.evaluation.search_metrics import (
    SearchMetricsCollector,
    write_mission_metrics_csv,
    write_per_target_csv,
)
from src.search.mission_phase import MissionOrchestrator, MissionPhase
from src.simulation.simulation_engine import SimulationEngine

# All 9 experiment configurations
EXPERIMENT_CONFIGS = [
    ("search_static_10uav",  "configs/experiments/search_static_10uav.yaml",  "static",  10),
    ("search_static_20uav",  "configs/experiments/search_static_20uav.yaml",  "static",  20),
    ("search_static_30uav",  "configs/experiments/search_static_30uav.yaml",  "static",  30),
    ("search_dynamic_10uav", "configs/experiments/search_dynamic_10uav.yaml", "dynamic", 10),
    ("search_dynamic_20uav", "configs/experiments/search_dynamic_20uav.yaml", "dynamic", 20),
    ("search_dynamic_30uav", "configs/experiments/search_dynamic_30uav.yaml", "dynamic", 30),
    ("search_mixed_10uav",   "configs/experiments/search_mixed_10uav.yaml",   "mixed",   10),
    ("search_mixed_20uav",   "configs/experiments/search_mixed_20uav.yaml",   "mixed",   20),
    ("search_mixed_30uav",   "configs/experiments/search_mixed_30uav.yaml",   "mixed",   30),
]

RESULTS_DIR = Path("experiments/results/search")


def build_engine_with_search(config_path: Path) -> SimulationEngine:
    """Construct simulation engine with search extension from YAML config."""
    config = load_config(config_path)
    if config.search is None or not config.search.enabled:
        raise ValueError(f"Search config not enabled in {config_path}")

    world = World.from_config(config.environment, config.uav)
    spawn_center = np.array(
        [config.spawn_center_x, config.spawn_center_y],
        dtype=np.float64,
    )
    agents = spawn_uavs(
        count=config.num_uavs,
        center=spawn_center,
        spread_radius=config.uav.initial_spread_radius,
        mission_radius=config.aggregation.mission_region_radius,
        max_speed=config.uav.max_speed,
        max_angular_velocity=config.uav.max_angular_velocity,
        seed=config.environment.obstacle_seed,
        spawn_mode=config.uav.spawn_mode,  # type: ignore[arg-type]
        spawn_angular_noise=config.uav.spawn_angular_noise,
    )
    aggregation = SelfAggregationController(
        config=config.aggregation,
        uav_config=config.uav,
        rng=np.random.default_rng(config.environment.obstacle_seed),
    )
    engine = SimulationEngine(world, agents, aggregation, config)
    orchestrator = MissionOrchestrator(
        config=config.search,
        world=world,
        agents=agents,
        rng=np.random.default_rng(config.environment.obstacle_seed + 1),
    )
    engine.mission_orchestrator = orchestrator
    return engine


def run_single_experiment(
    label: str,
    config_path: Path,
    env_type: str,
    num_uavs: int,
    output_dir: Path,
) -> dict:
    """
    Run one experiment end-to-end and write CSVs.

    Returns summary dict for console printing.
    """
    print(f"\n[{label}] Starting — {num_uavs} UAVs, {env_type} environment")
    t0 = time.perf_counter()

    engine = build_engine_with_search(config_path)
    orch = engine.mission_orchestrator
    assert orch is not None

    collector = SearchMetricsCollector(num_uavs=num_uavs, environment_type=env_type)
    search_started = False

    progress_interval_s = 20.0
    next_progress_log = progress_interval_s

    # Run simulation tick by tick so we can hook into phase transitions
    while engine.time_s < engine.config.duration:
        m = engine.step()
        phase = orch.phase

        if engine.time_s >= next_progress_log:
            elapsed_wall = time.perf_counter() - t0
            eta_s = (
                elapsed_wall * (engine.config.duration - engine.time_s) / engine.time_s
                if engine.time_s > 0
                else float("nan")
            )
            print(
                f"[{label}] t={engine.time_s:6.1f}s / {engine.config.duration:.0f}s  "
                f"coverage={m.explored_fraction*100:5.1f}%  phase={phase.name:<17} "
                f"wall={elapsed_wall:6.1f}s  ETA={eta_s:6.1f}s"
            )
            next_progress_log += progress_interval_s

        # Record search-phase start
        if phase == MissionPhase.SEARCHING and not search_started:
            search_started = True
            collector.record_search_start(
                engine.time_s,
                engine.world.map.explored_fraction(),
            )

        # Collect metrics each search tick
        if phase in (MissionPhase.SEARCHING, MissionPhase.COMPLETED):
            collector.update(orch.target_manager, orch.tracker, engine.time_s)

        # Stop early if mission complete
        if phase == MissionPhase.COMPLETED:
            break

    # Record final state
    collector.record_mission_end(
        engine.time_s,
        engine.world.map.explored_fraction(),
    )

    # Compute aggregated metrics
    mission_metrics = collector.compute_mission_metrics(orch.target_manager)

    # Write CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    mission_csv = output_dir / "mission_summary.csv"
    per_target_csv = output_dir / f"{label}_per_target.csv"

    write_mission_metrics_csv(mission_metrics, mission_csv)
    write_per_target_csv(collector, orch.target_manager, per_target_csv)

    elapsed = time.perf_counter() - t0
    summary = {
        "label": label,
        "num_uavs": num_uavs,
        "env_type": env_type,
        "coverage_start": f"{mission_metrics.coverage_at_search_start * 100:.1f}%",
        "coverage_end": f"{mission_metrics.coverage_at_mission_end * 100:.1f}%",
        "detected": mission_metrics.detected_count,
        "total": mission_metrics.total_targets,
        "completed": mission_metrics.completed_targets,
        "success_rate": f"{mission_metrics.search_success_rate * 100:.1f}%",
        "avg_tracking_s": f"{mission_metrics.avg_tracking_duration_s:.1f}",
        "mission_time_s": f"{mission_metrics.mission_completion_time or engine.time_s:.1f}",
        "wall_time_s": f"{elapsed:.1f}",
    }
    print(
        f"[{label}] Done | "
        f"coverage {summary['coverage_start']}→{summary['coverage_end']} | "
        f"detected {summary['detected']}/{summary['total']} | "
        f"completed {summary['completed']} | "
        f"success {summary['success_rate']} | "
        f"wall {summary['wall_time_s']}s"
    )
    return summary


def run_all_experiments(output_dir: Path) -> None:
    """Run all 9 experiments sequentially."""
    summaries = []
    for label, cfg_path, env_type, n_uavs in EXPERIMENT_CONFIGS:
        path = Path(cfg_path)
        if not path.exists():
            print(f"[SKIP] {label} — config not found: {path}")
            continue
        try:
            s = run_single_experiment(label, path, env_type, n_uavs, output_dir)
            summaries.append(s)
        except Exception as exc:
            print(f"[ERROR] {label}: {exc}")

    print("\n── Experiment Summary ──────────────────────────────────")
    for s in summaries:
        print(
            f"  {s['label']:<30} | UAVs={s['num_uavs']:<3} "
            f"| env={s['env_type']:<8} | success={s['success_rate']}"
        )
    print(f"\nCSVs written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search & Tracking experiment runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Run a single experiment config",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 9 experiments",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for CSV results",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="static",
        help="Environment type label (static/dynamic/mixed) for single-run",
    )
    parser.add_argument(
        "--num-uavs",
        type=int,
        default=10,
        help="Number of UAVs (for label in single-run)",
    )
    args = parser.parse_args()

    if args.all:
        run_all_experiments(args.output)
    elif args.config is not None:
        label = args.config.stem
        run_single_experiment(
            label, args.config, args.env_type, args.num_uavs, args.output
        )
    else:
        # Default: run all
        run_all_experiments(args.output)


if __name__ == "__main__":
    main()
