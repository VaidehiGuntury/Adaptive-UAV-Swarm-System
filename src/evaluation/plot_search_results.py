"""
Plot search experiment results from CSV files.

Generates:
  - Search success rate vs fleet size (per environment type)
  - Average detection time vs fleet size
  - Average tracking duration vs fleet size
  - Mission completion time vs fleet size
  - Coverage before/after search
  - Per-target tracking accuracy box plots

Usage (from project root):
    python -m src.evaluation.plot_search_results
    python -m src.evaluation.plot_search_results --results experiments/results/search
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("experiments/results/search")
PLOTS_DIR = Path("experiments/results/search/plots")

ENV_TYPES = ["static", "dynamic", "mixed"]
FLEET_SIZES = [10, 20, 30]
ENV_COLORS = {"static": "#4285F4", "dynamic": "#EA4335", "mixed": "#34A853"}
MARKERS = {"static": "o", "dynamic": "s", "mixed": "^"}


def load_mission_summary(results_dir: Path) -> pd.DataFrame:
    """Load and return the aggregated mission summary CSV."""
    csv_path = results_dir / "mission_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Mission summary CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _save_or_show(fig: plt.Figure, path: Path | None, name: str) -> None:
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {path / name}.png")
    else:
        plt.show()
    plt.close(fig)


def plot_success_rate(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Search success rate vs fleet size, one line per environment type."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for env in ENV_TYPES:
        subset = df[df["environment_type"] == env].sort_values("num_uavs")
        if subset.empty:
            continue
        ax.plot(
            subset["num_uavs"],
            subset["search_success_rate"] * 100,
            marker=MARKERS[env],
            color=ENV_COLORS[env],
            label=env.capitalize(),
            linewidth=2,
        )
    ax.set_xlabel("Fleet size (UAVs)")
    ax.set_ylabel("Search success rate (%)")
    ax.set_title("Search Success Rate vs Fleet Size")
    ax.set_xticks(FLEET_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, plots_dir, "search_success_rate")


def plot_detection_time(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Average first detection time vs fleet size."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for env in ENV_TYPES:
        subset = df[df["environment_type"] == env].sort_values("num_uavs")
        if subset.empty:
            continue
        ax.plot(
            subset["num_uavs"],
            subset["avg_first_detection_time"],
            marker=MARKERS[env],
            color=ENV_COLORS[env],
            label=env.capitalize(),
            linewidth=2,
        )
    ax.set_xlabel("Fleet size (UAVs)")
    ax.set_ylabel("Average first detection time (s)")
    ax.set_title("Average First Detection Time vs Fleet Size")
    ax.set_xticks(FLEET_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, plots_dir, "avg_detection_time")


def plot_tracking_duration(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Average tracking duration vs fleet size."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for env in ENV_TYPES:
        subset = df[df["environment_type"] == env].sort_values("num_uavs")
        if subset.empty:
            continue
        ax.plot(
            subset["num_uavs"],
            subset["avg_tracking_duration_s"],
            marker=MARKERS[env],
            color=ENV_COLORS[env],
            label=env.capitalize(),
            linewidth=2,
        )
    ax.set_xlabel("Fleet size (UAVs)")
    ax.set_ylabel("Average tracking duration (s)")
    ax.set_title("Average Tracking Duration vs Fleet Size")
    ax.set_xticks(FLEET_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, plots_dir, "avg_tracking_duration")


def plot_mission_completion_time(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Mission completion time vs fleet size."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for env in ENV_TYPES:
        subset = df[df["environment_type"] == env].sort_values("num_uavs")
        if subset.empty:
            continue
        ax.plot(
            subset["num_uavs"],
            subset["mission_completion_time"],
            marker=MARKERS[env],
            color=ENV_COLORS[env],
            label=env.capitalize(),
            linewidth=2,
        )
    ax.set_xlabel("Fleet size (UAVs)")
    ax.set_ylabel("Mission completion time (s)")
    ax.set_title("Mission Completion Time vs Fleet Size")
    ax.set_xticks(FLEET_SIZES)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, plots_dir, "mission_completion_time")


def plot_coverage(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Coverage at search start and mission end vs fleet size."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in zip(
        axes,
        ["coverage_at_search_start", "coverage_at_mission_end"],
        ["Coverage at Search Start", "Coverage at Mission End"],
    ):
        for env in ENV_TYPES:
            subset = df[df["environment_type"] == env].sort_values("num_uavs")
            if subset.empty:
                continue
            ax.plot(
                subset["num_uavs"],
                subset[col] * 100,
                marker=MARKERS[env],
                color=ENV_COLORS[env],
                label=env.capitalize(),
                linewidth=2,
            )
        ax.set_xlabel("Fleet size (UAVs)")
        ax.set_ylabel("Coverage (%)")
        ax.set_title(title)
        ax.set_xticks(FLEET_SIZES)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, plots_dir, "coverage_comparison")


def plot_loss_reacquisition(df: pd.DataFrame, plots_dir: Path | None) -> None:
    """Total loss and reacquisition events vs fleet size."""
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = range(len(FLEET_SIZES))

    for i, env in enumerate(ENV_TYPES):
        subset = df[df["environment_type"] == env].sort_values("num_uavs")
        if subset.empty:
            continue
        offset = (i - 1) * width / len(ENV_TYPES)
        ax.bar(
            [xi + offset for xi in x],
            subset["total_loss_events"].values[:len(FLEET_SIZES)],
            width=width / len(ENV_TYPES),
            color=ENV_COLORS[env],
            alpha=0.7,
            label=f"{env} (loss)",
        )

    ax.set_xlabel("Fleet size (UAVs)")
    ax.set_ylabel("Total loss events")
    ax.set_title("Target Loss Events vs Fleet Size")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(n) for n in FLEET_SIZES])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save_or_show(fig, plots_dir, "loss_events")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot search experiment results")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing mission_summary.csv",
    )
    parser.add_argument(
        "--plots",
        type=Path,
        default=None,
        help="Output directory for plots (default: results/plots/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving",
    )
    args = parser.parse_args()

    plots_dir = None if args.show else (args.plots or args.results / "plots")

    try:
        df = load_mission_summary(args.results)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run experiments first: python -m src.evaluation.run_search_experiment --all")
        return

    print(f"Loaded {len(df)} experiment rows from {args.results}")

    plot_success_rate(df, plots_dir)
    plot_detection_time(df, plots_dir)
    plot_tracking_duration(df, plots_dir)
    plot_mission_completion_time(df, plots_dir)
    plot_coverage(df, plots_dir)
    plot_loss_reacquisition(df, plots_dir)

    print("Plotting complete.")


if __name__ == "__main__":
    main()
