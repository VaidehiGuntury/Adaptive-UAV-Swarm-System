"""
Batch experiment runner for Paper 1 baseline vs corrected behavioral comparison.

Usage (from project root):
    python -m src.evaluation.run_experiment --config configs/experiments/corrected.yaml
    python -m src.evaluation.run_experiment --config configs/experiments/baseline.yaml --label baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine, SimulationMetrics

CSV_FIELDS = [
    "time_s",
    "explored_fraction",
    "mean_target_separation",
    "frontier_reuse_frequency",
    "target_reassignment_count",
    "revisit_ratio",
    "active_frontier_count",
    "mean_speed",
    "mean_pairwise_distance",
]


def build_engine(config_path: Path) -> SimulationEngine:
    """Construct simulation engine from YAML configuration."""
    config = load_config(config_path)
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
    return SimulationEngine(world, agents, aggregation, config)


def write_metrics_csv(rows: list[SimulationMetrics], output_path: Path) -> None:
    """Write per-timestep metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for metrics in rows:
            writer.writerow(
                {
                    "time_s": metrics.time_s,
                    "explored_fraction": metrics.explored_fraction,
                    "mean_target_separation": metrics.mean_target_separation,
                    "frontier_reuse_frequency": metrics.frontier_reuse_frequency,
                    "target_reassignment_count": metrics.target_reassignment_count,
                    "revisit_ratio": metrics.revisit_ratio,
                    "active_frontier_count": metrics.active_frontier_count,
                    "mean_speed": metrics.mean_speed,
                    "mean_pairwise_distance": metrics.mean_pairwise_distance,
                }
            )


def run_experiment(config_path: Path, output_path: Path) -> SimulationMetrics:
    """Run one experiment and write CSV metrics log."""
    engine = build_engine(config_path)
    metrics_history = engine.run()
    write_metrics_csv(metrics_history, output_path)
    return metrics_history[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 exploration experiment runner")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path (default: experiments/results/<label>.csv)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Output filename label (defaults to config stem)",
    )
    args = parser.parse_args()

    label = args.label or args.config.stem
    output = args.output or Path("experiments/results") / f"{label}.csv"
    final = run_experiment(args.config, output)

    print(
        f"Experiment '{label}' complete: "
        f"coverage={final.explored_fraction * 100:.1f}%, "
        f"mean_target_sep={final.mean_target_separation:.1f}m, "
        f"frontier_reuse={final.frontier_reuse_frequency:.3f}, "
        f"revisit_ratio={final.revisit_ratio:.3f}"
    )
    print(f"Metrics written to {output}")


if __name__ == "__main__":
    main()
