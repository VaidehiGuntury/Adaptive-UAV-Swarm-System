"""
Task 2 — multi-seed statistical validation across fleet sizes.

Runs N independent trials per fleet size (10/20/30 UAVs by default),
varying the master seed (obstacle placement, UAV spawn, and the
aggregation controller's RNG all derive from it), and reports:

  - per-run metrics CSV under experiments/results/multiseed/
  - a mean +/- std summary table printed to stdout AND saved to
    experiments/results/multiseed_summary.csv
  - a coverage-curve PNG per fleet size (mean line + shaded std band)
    under experiments/results/multiseed/plots/

This script does not modify any checked-in config file — each trial's
config is derived in-memory from configs/simulation.yaml via
dataclasses.replace (num_uavs and environment.obstacle_seed only).

Usage (from project root):
    python -m experiments.scripts.run_multiseed
    python -m experiments.scripts.run_multiseed --seeds 42 43 44 45 --fleet-sizes 10 20 30
    python -m experiments.scripts.run_multiseed --help
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

DEFAULT_CONFIG = Path("configs/simulation.yaml")
DEFAULT_SEEDS = [42, 43, 44, 45]
DEFAULT_FLEET_SIZES = [10, 20, 30]
DEFAULT_DURATION = 120.0
DEFAULT_OUTPUT_DIR = Path("experiments/results/multiseed")
DEFAULT_SUMMARY_CSV = Path("experiments/results/multiseed_summary.csv")

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
    timeseries: list[tuple[float, float]]  # (time_s, explored_fraction)


def make_config(base_config: SimulationConfig, num_uavs: int, seed: int) -> SimulationConfig:
    """Derive a per-trial config in-memory — no YAML files written."""
    return replace(
        base_config,
        num_uavs=num_uavs,
        environment=replace(base_config.environment, obstacle_seed=seed),
    )


def build_engine(config: SimulationConfig) -> SimulationEngine:
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
    base_config: SimulationConfig, num_uavs: int, seed: int, duration: float, label: str,
) -> RunResult:
    config = make_config(base_config, num_uavs, seed)
    engine = build_engine(config)
    dt = engine.config.dt

    time_to_70: float | None = None
    time_to_85: float | None = None
    transit_agent_seconds = 0.0
    total_agent_seconds = 0.0
    timeseries: list[tuple[float, float]] = [(0.0, 0.0)]

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

        timeseries.append((round(m.time_s, 2), m.explored_fraction))

    assert m is not None
    final_coverage = timeseries[-1][1]

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
        timeseries=timeseries,
    )


def write_timeseries_csv(result: RunResult, output_dir: Path) -> None:
    path = output_dir / f"{result.label}_seed{result.seed}_timeseries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "explored_fraction"])
        writer.writerows(result.timeseries)


def write_summary_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "num_uavs", "seed", "final_coverage", "time_to_70", "time_to_85",
        "path_distance_total", "trajectory_efficiency", "mission_overlap_final",
        "travel_time_fraction", "exploration_time_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
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


def print_and_save_aggregate(results: list[RunResult], output_csv: Path) -> None:
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    never_reached: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_runs: dict[str, int] = defaultdict(int)

    for r in results:
        n_runs[r.label] += 1
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
        for key, value in row.items():
            if value is None:
                never_reached[r.label][key] += 1
            else:
                by_label[r.label][key].append(value)

    agg_fields = ["label", "n_seeds", "metric", "mean", "std", "n_reached", "never_reached"]
    agg_rows = []

    print("\n=== Task 2 multi-seed validation summary (mean +/- std) ===")
    for label in sorted(by_label.keys()):
        print(f"\n-- {label} (n={n_runs[label]} seeds) --")
        for key, name, scale in METRICS:
            values = by_label[label][key]
            missing = never_reached[label][key]
            if values:
                arr = np.array(values) * scale
                mean, std = float(arr.mean()), float(arr.std())
                print(f"  {name:<32} {mean:>10.2f} +/- {std:<8.2f} (n={len(values)}, never_reached={missing})")
                agg_rows.append(
                    {
                        "label": label, "n_seeds": n_runs[label], "metric": name,
                        "mean": mean, "std": std, "n_reached": len(values), "never_reached": missing,
                    }
                )
            else:
                print(f"  {name:<32} {'—':>10}          (never_reached={missing})")
                agg_rows.append(
                    {
                        "label": label, "n_seeds": n_runs[label], "metric": name,
                        "mean": "", "std": "", "n_reached": 0, "never_reached": missing,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\nSummary written to {output_csv}")


def plot_coverage_curves(results: list[RunResult], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LINE_COLOR = "#2a78d6"
    BAND_ALPHA = 0.18
    PRIMARY_INK = "#0b0b0b"
    SECONDARY_INK = "#52514e"
    MUTED_INK = "#898781"
    GRIDLINE = "#e1e0d9"
    BASELINE = "#c3c2b7"
    SURFACE = "#fcfcfb"

    by_label: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_label[r.label].append(r)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for label, runs in by_label.items():
        common_times = np.array([t for t, _ in runs[0].timeseries])
        matrix = np.vstack([[v for _, v in r.timeseries] for r in runs])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)

        fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.fill_between(common_times, mean - std, mean + std, color=LINE_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(common_times, mean, color=LINE_COLOR, linewidth=2)
        ax.set_xlabel("Time (s)", color=SECONDARY_INK, fontsize=10)
        ax.set_ylabel("Explored fraction", color=SECONDARY_INK, fontsize=10)
        ax.set_title(f"Coverage over time — {label} ({len(runs)} seeds, mean ± std)", color=PRIMARY_INK, fontsize=12, loc="left")
        ax.set_ylim(0, 1.0)
        ax.grid(True, color=GRIDLINE, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED_INK, labelsize=9)
        fig.tight_layout()
        out_path = plots_dir / f"coverage_curve_{label}.png"
        fig.savefig(out_path, facecolor=SURFACE)
        plt.close(fig)
        print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 multi-seed validation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base config YAML")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=DEFAULT_FLEET_SIZES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse args and build one engine per fleet size without stepping the simulation, then exit.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)

    if args.dry_run:
        for num_uavs in args.fleet_sizes:
            config = make_config(base_config, num_uavs, args.seeds[0])
            engine = build_engine(config)
            print(f"[dry-run] n{num_uavs}: built engine OK, {len(engine.agents)} agents spawned")
        print("[dry-run] OK — no simulation steps executed.")
        return

    results: list[RunResult] = []
    total_runs = len(args.fleet_sizes) * len(args.seeds)
    run_idx = 0
    for num_uavs in args.fleet_sizes:
        label = f"n{num_uavs}"
        for seed in args.seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {label} seed={seed} running...")
            result = run_single_trial(base_config, num_uavs, seed, args.duration, label)
            write_timeseries_csv(result, args.output_dir)
            results.append(result)
            print(
                f"[{run_idx}/{total_runs}] {label} seed={seed} "
                f"coverage={result.final_coverage*100:.1f}% "
                f"t70={result.time_to_70} t85={result.time_to_85}"
            )

    write_summary_csv(results, args.output_dir / "summary.csv")
    print_and_save_aggregate(results, args.summary_csv)
    plot_coverage_curves(results, args.output_dir)


if __name__ == "__main__":
    main()
