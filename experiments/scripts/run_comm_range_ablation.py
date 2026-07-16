"""
Communication-range ablation — replicates paper §7.2.3: DEBS robustness to
communication loss, run at 10m/20m/30m comm range vs. the 50m/unrestricted
default.

Sweeps four pre-built configs (configs/experiments/comm_range_{10,20,30}m.yaml
plus the unmodified configs/simulation.yaml for the 50m/default case) across
the same seeds and fleet size, on an identical World/SimulationEngine/metrics
pipeline for every run — only ide.communication_range differs between them.
Follows the same structure as experiments/scripts/run_baseline_comparison.py
(Phase H).

Requires the communication-range threading added to
SelfAggregationController / aggregation_utility_j_c (Paper 1 Eq. 6's J_C
dispersal term) — without it, ide.communication_range only gated IDE partner
eligibility and had no effect on BSA's own viewpoint selection; see
src/algorithms/aggregation/fitness_functions.py's aggregation_utility_j_c()
docstring.

This script does not modify any checked-in config file.

Usage (from project root):
    python -m experiments.scripts.run_comm_range_ablation
    python -m experiments.scripts.run_comm_range_ablation --seeds 42 43 44 45 46 --fleet-sizes 10
    python -m experiments.scripts.run_comm_range_ablation --help
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import SimulationConfig, load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine

COMM_RANGE_CONFIGS: dict[str, Path] = {
    "10m": Path("configs/experiments/comm_range_10m.yaml"),
    "20m": Path("configs/experiments/comm_range_20m.yaml"),
    "30m": Path("configs/experiments/comm_range_30m.yaml"),
    "50m": Path("configs/simulation.yaml"),
}
COMM_RANGE_LABELS: tuple[str, ...] = tuple(COMM_RANGE_CONFIGS.keys())

DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_FLEET_SIZES = [10]
DEFAULT_DURATION = 120.0
DEFAULT_OUTPUT_DIR = Path("experiments/results/comm_range_ablation")
DEFAULT_SUMMARY_CSV = Path("experiments/results/comm_range_ablation_summary.csv")

METRICS = [
    ("final_coverage", "Coverage %", 100.0),
    ("time_to_70", "Time-to-70% (s)", 1.0),
    ("time_to_85", "Time-to-85% (s)", 1.0),
    ("trajectory_efficiency", "Trajectory efficiency (m^2/m)", 1.0),
    ("mission_overlap_final", "Mission overlap", 1.0),
    ("path_distance_total", "Path distance (m, fleet total)", 1.0),
]


@dataclass
class RunResult:
    comm_range_label: str
    num_uavs: int
    seed: int
    final_coverage: float
    time_to_70: float | None
    time_to_85: float | None
    path_distance_total: float
    trajectory_efficiency: float
    mission_overlap_final: float


def make_config(base_config: SimulationConfig, num_uavs: int, seed: int) -> SimulationConfig:
    """Derive a per-trial config in-memory — no YAML files written."""
    return replace(
        base_config,
        num_uavs=num_uavs,
        environment=replace(base_config.environment, obstacle_seed=seed),
    )


def build_engine(config: SimulationConfig) -> SimulationEngine:
    """
    Build a DEBS (BSA + IDE) SimulationEngine on identical World/agents/metrics
    for every comm-range config — only config.ide.communication_range differs.

    No dynamic obstacles, no search extension — exploration-layer comparison
    only, mirroring run_multiseed.py's build_engine(). SimulationEngine builds
    its own IDEAllocator internally from config.ide (DEBS §4); after the J_C
    communication-range threading, BSA's dispersal term reads the same
    ide.communication_range value automatically (SimulationEngine.step()),
    so this single construction path is enough to exercise both.
    """
    world = World.from_config(config.environment, config.uav)
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
    aggregation = SelfAggregationController(
        config=config.aggregation,
        uav_config=config.uav,
        rng=np.random.default_rng(config.environment.obstacle_seed),
    )
    return SimulationEngine(world, agents, aggregation, config)


def run_single_trial(
    base_config: SimulationConfig,
    num_uavs: int,
    seed: int,
    duration: float,
    comm_range_label: str,
) -> RunResult:
    config = make_config(base_config, num_uavs, seed)
    engine = build_engine(config)

    time_to_70: float | None = None
    time_to_85: float | None = None

    m = None
    while engine.time_s < duration:
        m = engine.step()

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

    return RunResult(
        comm_range_label=comm_range_label,
        num_uavs=num_uavs,
        seed=seed,
        final_coverage=final_coverage,
        time_to_70=time_to_70,
        time_to_85=time_to_85,
        path_distance_total=path_distance_total,
        trajectory_efficiency=trajectory_efficiency,
        mission_overlap_final=m.mission_overlap_fraction,
    )


def write_summary_csv(results: list[RunResult], path: Path) -> None:
    """Raw per-run rows (one row per comm-range x fleet-size x seed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "comm_range", "num_uavs", "seed", "final_coverage", "time_to_70", "time_to_85",
        "path_distance_total", "trajectory_efficiency", "mission_overlap_final",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "comm_range": r.comm_range_label,
                    "num_uavs": r.num_uavs,
                    "seed": r.seed,
                    "final_coverage": r.final_coverage,
                    "time_to_70": r.time_to_70 if r.time_to_70 is not None else "",
                    "time_to_85": r.time_to_85 if r.time_to_85 is not None else "",
                    "path_distance_total": r.path_distance_total,
                    "trajectory_efficiency": r.trajectory_efficiency,
                    "mission_overlap_final": r.mission_overlap_final,
                }
            )


def _mean_std(values: list[float], scale: float) -> tuple[float, float]:
    arr = np.array(values) * scale
    return float(arr.mean()), float(arr.std())


def _format_mean_std(values: list[float], scale: float) -> str:
    if not values:
        return "-"
    mean, std = _mean_std(values, scale)
    return f"{mean:.2f}+/-{std:.2f}"


def print_and_save_comparison(results: list[RunResult], output_csv: Path) -> None:
    """
    Write the side-by-side comm-range aggregate CSV and print the comparison
    table (mean +/- std per metric, per fleet size, per comm-range) to stdout.
    """
    by_key: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    never_reached: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_runs: dict[tuple[str, str], int] = defaultdict(int)

    for r in results:
        key = (f"n{r.num_uavs}", r.comm_range_label)
        n_runs[key] += 1
        row = {
            "final_coverage": r.final_coverage,
            "time_to_70": r.time_to_70,
            "time_to_85": r.time_to_85,
            "trajectory_efficiency": r.trajectory_efficiency,
            "mission_overlap_final": r.mission_overlap_final,
            "path_distance_total": r.path_distance_total,
        }
        for metric_key, value in row.items():
            if value is None:
                never_reached[key][metric_key] += 1
            else:
                by_key[key][metric_key].append(value)

    agg_fields = ["fleet_label", "comm_range", "n_seeds", "metric", "mean", "std", "n_reached", "never_reached"]
    agg_rows = []
    for key in sorted(by_key.keys()):
        fleet_label, comm_range_label = key
        for metric_key, name, scale in METRICS:
            values = by_key[key][metric_key]
            missing = never_reached[key][metric_key]
            if values:
                mean, std = _mean_std(values, scale)
            else:
                mean, std = "", ""
            agg_rows.append(
                {
                    "fleet_label": fleet_label,
                    "comm_range": comm_range_label,
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

    print("\n=== Comm-range ablation (paper Sec 7.2.3): DEBS at 10m / 20m / 30m / 50m (mean +/- std) ===")
    fleet_labels = sorted({f"n{r.num_uavs}" for r in results})
    for fleet_label in fleet_labels:
        print(f"\n-- {fleet_label} --")
        header = f"  {'metric':<32}" + "".join(f"{label:>18}" for label in COMM_RANGE_LABELS)
        print(header)
        for metric_key, name, scale in METRICS:
            row_str = f"  {name:<32}"
            for label in COMM_RANGE_LABELS:
                values = by_key[(fleet_label, label)][metric_key]
                row_str += f"{_format_mean_std(values, scale):>18}"
            print(row_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comm-range ablation (paper Sec 7.2.3): DEBS at 10m/20m/30m/50m"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build one engine per comm-range/fleet size without stepping the simulation, then exit.",
    )
    args = parser.parse_args()

    base_configs = {label: load_config(path) for label, path in COMM_RANGE_CONFIGS.items()}

    if args.dry_run:
        for label, base_config in base_configs.items():
            for num_uavs in args.fleet_sizes:
                config = make_config(base_config, num_uavs, args.seeds[0])
                engine = build_engine(config)
                comm_range = config.ide.communication_range if config.ide is not None else None
                print(
                    f"[dry-run] comm_range={label} n{num_uavs}: built engine OK, "
                    f"{len(engine.agents)} agents spawned, "
                    f"config.ide.communication_range={comm_range}"
                )
        print("[dry-run] OK — no simulation steps executed.")
        return

    results: list[RunResult] = []
    total_runs = len(COMM_RANGE_CONFIGS) * len(args.fleet_sizes) * len(args.seeds)
    run_idx = 0
    for label, base_config in base_configs.items():
        for num_uavs in args.fleet_sizes:
            fleet_label = f"n{num_uavs}"
            for seed in args.seeds:
                run_idx += 1
                print(f"[{run_idx}/{total_runs}] comm_range={label} {fleet_label} seed={seed} running...")
                result = run_single_trial(base_config, num_uavs, seed, args.duration, label)
                results.append(result)
                print(
                    f"[{run_idx}/{total_runs}] comm_range={label} {fleet_label} seed={seed} "
                    f"coverage={result.final_coverage*100:.1f}% "
                    f"t70={result.time_to_70} t85={result.time_to_85}"
                )

    write_summary_csv(results, args.output_dir / "summary.csv")
    print_and_save_comparison(results, args.summary_csv)


if __name__ == "__main__":
    main()
