"""
Phase H — DEBS (BSA + IDE) vs. greedy-nearest-frontier baseline comparison.

Runs both allocators across the same seeds/fleet-sizes/duration, on an
identical World/SimulationEngine/metrics pipeline, so the two are
directly comparable. Follows the same structure as
experiments/scripts/run_multiseed.py (Phase E). No changes to
SimulationEngine, World, or metrics — GreedyFrontierAllocator
(src/algorithms/baselines/greedy_frontier.py) is a drop-in substitute
for SelfAggregationController; see that module's docstring for the
interface it implements.

This script does not modify any checked-in config file — each trial's
config is derived in-memory from configs/simulation.yaml via
dataclasses.replace (num_uavs, environment.obstacle_seed, and — for the
greedy baseline only — ide=None, since a naive allocator has no use for
IDE's fair-share region allocation).

Usage (from project root):
    python -m experiments.scripts.run_baseline_comparison
    python -m experiments.scripts.run_baseline_comparison --seeds 42 43 44 45 46 --fleet-sizes 10 20 30
    python -m experiments.scripts.run_baseline_comparison --help
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.algorithms.baselines.greedy_frontier import GreedyFrontierAllocator
from src.agents.uav import spawn_uavs
from src.config.loader import SimulationConfig, load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine

DEFAULT_CONFIG = Path("configs/simulation.yaml")
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_FLEET_SIZES = [10, 20, 30]
DEFAULT_DURATION = 120.0
DEFAULT_OUTPUT_DIR = Path("experiments/results/baseline_comparison")
DEFAULT_SUMMARY_CSV = Path("experiments/results/baseline_comparison_summary.csv")

AllocatorKind = Literal["debs", "greedy"]
ALLOCATOR_KINDS: tuple[AllocatorKind, ...] = ("debs", "greedy")

METRICS = [
    ("final_coverage", "Coverage %", 100.0),
    ("time_to_70", "Time-to-70% (s)", 1.0),
    ("time_to_85", "Time-to-85% (s)", 1.0),
    ("trajectory_efficiency", "Trajectory efficiency (m^2/m)", 1.0),
    ("mission_overlap_final", "Mission overlap", 1.0),
    ("path_distance_total", "Path distance (m, fleet total)", 1.0),
    ("travel_time_fraction", "Travel-time fraction %", 100.0),
    ("exploration_time_fraction", "Exploration-time fraction %", 100.0),
]


@dataclass
class RunResult:
    allocator: AllocatorKind
    label: str
    num_uavs: int
    seed: int
    final_coverage: float
    time_to_70: float | None
    time_to_85: float | None
    path_distance_total: float
    trajectory_efficiency: float
    mission_overlap_final: float
    travel_time_fraction: float
    exploration_time_fraction: float


def make_config(base_config: SimulationConfig, num_uavs: int, seed: int) -> SimulationConfig:
    """Derive a per-trial config in-memory — no YAML files written."""
    return replace(
        base_config,
        num_uavs=num_uavs,
        environment=replace(base_config.environment, obstacle_seed=seed),
    )


def build_engine(config: SimulationConfig, allocator_kind: AllocatorKind) -> SimulationEngine:
    """
    Build a SimulationEngine wired to either DEBS (BSA + IDE) or the
    naive GreedyFrontierAllocator, on identical World/agents/metrics.

    No dynamic obstacles, no search extension — exploration-layer
    comparison only (matches Phase E's scope), mirroring
    run_multiseed.py's build_engine().
    """
    world = World.from_config(config.environment, config.uav, dynamic_config=config.dynamic_environment)
    spawn_center = np.array(
        [config.spawn_center_x, config.spawn_center_y], dtype=np.float64
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

    if allocator_kind == "debs":
        aggregation = SelfAggregationController(
            config=config.aggregation,
            uav_config=config.uav,
            rng=np.random.default_rng(config.environment.obstacle_seed),
        )
        # SimulationEngine builds its own IDEAllocator internally from
        # config.ide (DEBS §4) — nothing extra to wire up here.
        return SimulationEngine(world, agents, aggregation, config)

    # Greedy baseline: no coordination layer, so IDE's fair-share
    # allocation must not run either. config.ide=None makes
    # SimulationEngine skip building its internal IDEAllocator (see
    # SimulationEngine.__init__) — no engine code change needed.
    greedy_config = replace(config, ide=None)
    aggregation = GreedyFrontierAllocator(config=config.aggregation, uav_config=config.uav)
    return SimulationEngine(world, agents, aggregation, greedy_config)


def in_transit(engine: SimulationEngine, agent) -> bool:
    """assigned, not yet arrived (dist > d_c), not yet timed out."""
    if agent.assigned_target is None:
        return False
    dist = float(np.linalg.norm(agent.position - agent.assigned_target))
    if dist <= engine.config.aggregation.d_c:
        return False
    timeout_map = getattr(engine.aggregation, "_target_timeout", None)
    if timeout_map is not None:
        deadline = timeout_map.get(agent.agent_id, float("inf"))
        if engine.time_s >= deadline:
            return False
    return True


def run_single_trial(
    base_config: SimulationConfig,
    num_uavs: int,
    seed: int,
    duration: float,
    label: str,
    allocator_kind: AllocatorKind,
) -> RunResult:
    config = make_config(base_config, num_uavs, seed)
    engine = build_engine(config, allocator_kind)
    dt = engine.config.dt

    time_to_70: float | None = None
    time_to_85: float | None = None
    transit_agent_seconds = 0.0
    total_agent_seconds = 0.0

    m = None
    while engine.time_s < duration:
        m = engine.step()

        n_transit = sum(1 for a in engine.agents if in_transit(engine, a))
        transit_agent_seconds += n_transit * dt
        total_agent_seconds += len(engine.agents) * dt

        if time_to_70 is None and m.explored_fraction >= 0.70:
            time_to_70 = m.time_s
        if time_to_85 is None and m.explored_fraction >= 0.85:
            time_to_85 = m.time_s

    assert m is not None
    final_coverage = m.explored_fraction

    path_distance_total = 0.0
    for positions in engine.agent_histories.values():
        for i in range(1, len(positions)):
            path_distance_total += float(np.linalg.norm(positions[i] - positions[i - 1]))

    explored_area = final_coverage * engine.world.width * engine.world.height
    trajectory_efficiency = (
        explored_area / path_distance_total if path_distance_total > 0 else 0.0
    )
    travel_time_fraction = (
        transit_agent_seconds / total_agent_seconds if total_agent_seconds > 0 else 0.0
    )

    return RunResult(
        allocator=allocator_kind,
        label=label,
        num_uavs=num_uavs,
        seed=seed,
        final_coverage=final_coverage,
        time_to_70=time_to_70,
        time_to_85=time_to_85,
        path_distance_total=path_distance_total,
        trajectory_efficiency=trajectory_efficiency,
        mission_overlap_final=m.mission_overlap_fraction,
        travel_time_fraction=travel_time_fraction,
        exploration_time_fraction=1.0 - travel_time_fraction,
    )


def write_summary_csv(results: list[RunResult], path: Path) -> None:
    """Raw per-run rows (one row per allocator x fleet-size x seed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "allocator", "label", "num_uavs", "seed", "final_coverage", "time_to_70", "time_to_85",
        "path_distance_total", "trajectory_efficiency", "mission_overlap_final",
        "travel_time_fraction", "exploration_time_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "allocator": r.allocator,
                    "label": r.label,
                    "num_uavs": r.num_uavs,
                    "seed": r.seed,
                    "final_coverage": r.final_coverage,
                    "time_to_70": r.time_to_70 if r.time_to_70 is not None else "",
                    "time_to_85": r.time_to_85 if r.time_to_85 is not None else "",
                    "path_distance_total": r.path_distance_total,
                    "trajectory_efficiency": r.trajectory_efficiency,
                    "mission_overlap_final": r.mission_overlap_final,
                    "travel_time_fraction": r.travel_time_fraction,
                    "exploration_time_fraction": r.exploration_time_fraction,
                }
            )


def _mean_std(values: list[float], scale: float) -> tuple[float, float]:
    arr = np.array(values) * scale
    return float(arr.mean()), float(arr.std())


def _format_mean_std(values: list[float], scale: float) -> str:
    if not values:
        return "-"
    mean, std = _mean_std(values, scale)
    return f"{mean:.2f} +/- {std:.2f}"


def print_and_save_comparison(results: list[RunResult], output_csv: Path) -> None:
    """
    Write the side-by-side DEBS-vs-greedy aggregate CSV and print the
    comparison table (mean +/- std per metric, per fleet size, per
    allocator) to stdout.
    """
    by_key: dict[tuple[str, AllocatorKind], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    never_reached: dict[tuple[str, AllocatorKind], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    n_runs: dict[tuple[str, AllocatorKind], int] = defaultdict(int)

    for r in results:
        key = (r.label, r.allocator)
        n_runs[key] += 1
        row = {
            "final_coverage": r.final_coverage,
            "time_to_70": r.time_to_70,
            "time_to_85": r.time_to_85,
            "trajectory_efficiency": r.trajectory_efficiency,
            "mission_overlap_final": r.mission_overlap_final,
            "path_distance_total": r.path_distance_total,
            "travel_time_fraction": r.travel_time_fraction,
            "exploration_time_fraction": r.exploration_time_fraction,
        }
        for metric_key, value in row.items():
            if value is None:
                never_reached[key][metric_key] += 1
            else:
                by_key[key][metric_key].append(value)

    agg_fields = ["label", "allocator", "n_seeds", "metric", "mean", "std", "n_reached", "never_reached"]
    agg_rows = []
    for key in sorted(by_key.keys()):
        label, allocator = key
        for metric_key, name, scale in METRICS:
            values = by_key[key][metric_key]
            missing = never_reached[key][metric_key]
            if values:
                mean, std = _mean_std(values, scale)
            else:
                mean, std = "", ""
            agg_rows.append(
                {
                    "label": label,
                    "allocator": allocator,
                    "n_seeds": n_runs[key],
                    "metric": name,
                    "mean": mean,
                    "std": std,
                    "n_reached": len(values),
                    "never_reached": missing,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\nSummary written to {output_csv}")

    print("\n=== Phase H: DEBS vs. greedy-frontier baseline comparison (mean +/- std) ===")
    labels = sorted({r.label for r in results})
    for label in labels:
        n_debs = n_runs.get((label, "debs"), 0)
        n_greedy = n_runs.get((label, "greedy"), 0)
        print(f"\n-- {label} (DEBS n={n_debs} seeds, Greedy n={n_greedy} seeds) --")
        print(f"  {'metric':<32} {'DEBS':>22} {'Greedy':>22}")
        for metric_key, name, scale in METRICS:
            debs_str = _format_mean_std(by_key[(label, "debs")][metric_key], scale)
            greedy_str = _format_mean_std(by_key[(label, "greedy")][metric_key], scale)
            print(f"  {name:<32} {debs_str:>22} {greedy_str:>22}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase H: DEBS vs. greedy-frontier baseline comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base config YAML")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build one engine per allocator/fleet size without stepping the simulation, then exit.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)

    if args.dry_run:
        for allocator_kind in ALLOCATOR_KINDS:
            for num_uavs in args.fleet_sizes:
                config = make_config(base_config, num_uavs, args.seeds[0])
                engine = build_engine(config, allocator_kind)
                print(
                    f"[dry-run] {allocator_kind} n{num_uavs}: built engine OK, "
                    f"{len(engine.agents)} agents spawned"
                )
        print("[dry-run] OK — no simulation steps executed.")
        return

    results: list[RunResult] = []
    total_runs = len(ALLOCATOR_KINDS) * len(args.fleet_sizes) * len(args.seeds)
    run_idx = 0
    for allocator_kind in ALLOCATOR_KINDS:
        for num_uavs in args.fleet_sizes:
            label = f"n{num_uavs}"
            for seed in args.seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] {allocator_kind} {label} seed={seed} running...")
                result = run_single_trial(base_config, num_uavs, seed, args.duration, label, allocator_kind)
                results.append(result)
                print(
                    f"[{run_idx}/{total_runs}] {allocator_kind} {label} seed={seed} "
                    f"coverage={result.final_coverage*100:.1f}% "
                    f"t70={result.time_to_70} t85={result.time_to_85}"
                )

    write_summary_csv(results, args.output_dir / "summary.csv")
    print_and_save_comparison(results, args.summary_csv)


if __name__ == "__main__":
    main()
