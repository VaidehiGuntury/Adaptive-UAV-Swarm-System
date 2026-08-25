"""
Publication-quality plot generator for moving-target experiment results.

Reads experiments/results/moving_targets/mission_summary.csv and produces:

  Fig 1 : Detection Rate vs Target Speed
  Fig 2 : Tracking Success vs Target Speed
  Fig 3 : Average Detection Time vs Target Speed
  Fig 4 : Average Localization Error vs Target Speed
  Fig 5 : Track Loss Events vs Target Speed
  Fig 6 : Reacquisition Rate vs Target Speed
  Fig 7 : Handover Events vs Target Speed
  Fig 8 : UAV Utilization vs Target Speed
  Fig 9 : Coverage vs Time  (one line per speed; uses per-step metrics if available)
  Fig 10: Simulation FPS vs Speed (computational performance)

Also writes a LaTeX-ready summary table to results/moving_targets/summary_table.tex.

Usage
-----
    python -m src.evaluation.plot_moving_target_results
    python -m src.evaluation.plot_moving_target_results --show        # interactive
    python -m src.evaluation.plot_moving_target_results --results experiments/results/moving_targets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS_DIR = Path("experiments/results/moving_targets")
PLOTS_DIR   = Path("experiments/results/moving_targets/plots")

SPEEDS = [0.0, 0.5, 1.0, 1.5, 2.0]

# Research-style palette (colour-blind safe)
PALETTE = ["#4285F4", "#EA4335", "#34A853", "#FBBC04", "#AB47BC"]
SPEED_COLOR = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(SPEEDS)}

STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(results_dir: Path) -> pd.DataFrame:
    csv = results_dir / "mission_summary.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"mission_summary.csv not found at {csv}\n"
            "Run experiments first:\n"
            "  python -m src.evaluation.run_moving_target_experiment"
        )
    df = pd.read_csv(csv)
    df["target_speed_mps"] = df["target_speed_mps"].astype(float)
    return df


def _save(fig: plt.Figure, name: str, plots_dir: Path, show: bool) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    pdf_path = plots_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {path.name}  {pdf_path.name}")
    if show:
        plt.show()
    plt.close(fig)


def _speed_agg(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (speeds, means, stds) aggregated over seeds."""
    grp = df.groupby("target_speed_mps")[col]
    speeds = np.array(sorted(df["target_speed_mps"].unique()))
    means  = np.array([grp.get_group(s).mean() for s in speeds])
    stds   = np.array([grp.get_group(s).std(ddof=0) for s in speeds])
    return speeds, means, stds


def _line_with_band(
    ax: plt.Axes,
    speeds: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    label: str,
    color: str,
    marker: str = "o",
    pct: bool = False,
) -> None:
    y = means * 100 if pct else means
    e = stds * 100 if pct else stds
    ax.plot(speeds, y, marker=marker, color=color, label=label)
    ax.fill_between(speeds, y - e, y + e, alpha=0.15, color=color)


# ── Individual figures ────────────────────────────────────────────────────────

def fig_detection_rate(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "detection_rate")
        _line_with_band(ax, speeds, means, stds,
                        "Detection Rate", PALETTE[0], pct=True)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Detection Rate (%)")
        ax.set_title("Detection Rate vs Target Speed")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        _save(fig, "fig1_detection_rate_vs_speed", plots_dir, show)


def fig_tracking_success(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "search_success_rate")
        _line_with_band(ax, speeds, means, stds,
                        "Tracking Success", PALETTE[1], pct=True)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Tracking Success Rate (%)")
        ax.set_title("Tracking Success vs Target Speed")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        _save(fig, "fig2_tracking_success_vs_speed", plots_dir, show)


def fig_detection_time(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for col, label, color, marker in [
            ("avg_first_detection_time", "First Detection Time", PALETTE[0], "o"),
            ("avg_detection_delay",      "Detection Delay",      PALETTE[2], "s"),
        ]:
            if col in df.columns:
                speeds, means, stds = _speed_agg(df, col)
                _line_with_band(ax, speeds, means, stds, label, color, marker)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Detection Time vs Target Speed")
        ax.legend()
        _save(fig, "fig3_detection_time_vs_speed", plots_dir, show)


def fig_localization_error(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "avg_localization_error_m")
        _line_with_band(ax, speeds, means, stds,
                        "Localization Error", PALETTE[3], marker="D")
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Avg Localization Error (m)")
        ax.set_title("Localization Error vs Target Speed")
        _save(fig, "fig4_localization_error_vs_speed", plots_dir, show)


def fig_track_loss(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "total_loss_events")
        _line_with_band(ax, speeds, means, stds,
                        "Track Loss Events", PALETTE[1], marker="v")
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Total Loss Events")
        ax.set_title("Track Loss Events vs Target Speed")
        _save(fig, "fig5_loss_events_vs_speed", plots_dir, show)


def fig_reacquisition_rate(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "reacquisition_rate")
        _line_with_band(ax, speeds, means, stds,
                        "Reacquisition Rate", PALETTE[2], pct=True)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Reacquisition Rate (%)")
        ax.set_title("Reacquisition Rate vs Target Speed")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        _save(fig, "fig6_reacquisition_rate_vs_speed", plots_dir, show)


def fig_handovers(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "successful_handovers")
        _line_with_band(ax, speeds, means, stds,
                        "Successful Handovers", PALETTE[4], marker="^")
        if "failed_handovers" in df.columns:
            speeds2, means2, stds2 = _speed_agg(df, "failed_handovers")
            _line_with_band(ax, speeds2, means2, stds2,
                            "Failed Handovers", PALETTE[1], marker="x")
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Handover Events")
        ax.set_title("Multi-UAV Handover Events vs Target Speed")
        ax.legend()
        _save(fig, "fig7_handovers_vs_speed", plots_dir, show)


def fig_uav_utilization(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "uav_utilization")
        _line_with_band(ax, speeds, means, stds,
                        "UAV Utilization", PALETTE[0], pct=True)
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("UAV Utilization (%)")
        ax.set_title("UAV Utilization vs Target Speed")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        _save(fig, "fig8_uav_utilization_vs_speed", plots_dir, show)


def fig_fps(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        speeds, means, stds = _speed_agg(df, "sim_fps")
        _line_with_band(ax, speeds, means, stds,
                        "Simulation FPS", PALETTE[3], marker="s")
        ax.set_xlabel("Target Speed (m/s)")
        ax.set_ylabel("Simulation Steps / Wall-Clock Second")
        ax.set_title("Computational Performance vs Target Speed")
        _save(fig, "fig9_sim_fps_vs_speed", plots_dir, show)


def fig_combined_summary(df: pd.DataFrame, plots_dir: Path, show: bool) -> None:
    """4-panel summary figure: detection, tracking, localization, handovers."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Moving Target Detection & Tracking — Summary", fontsize=14)

        panels = [
            (axes[0, 0], "detection_rate",         "Detection Rate (%)", True),
            (axes[0, 1], "search_success_rate",     "Tracking Success (%)", True),
            (axes[1, 0], "avg_localization_error_m","Localization Error (m)", False),
            (axes[1, 1], "successful_handovers",    "Handovers", False),
        ]
        for ax, col, ylabel, pct in panels:
            speeds, means, stds = _speed_agg(df, col)
            _line_with_band(ax, speeds, means, stds, ylabel,
                            PALETTE[0], pct=pct)
            ax.set_xlabel("Target Speed (m/s)")
            ax.set_ylabel(ylabel)
            if pct:
                ax.set_ylim(0, 105)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

        fig.tight_layout()
        _save(fig, "fig10_combined_summary", plots_dir, show)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def write_latex_table(df: pd.DataFrame, results_dir: Path) -> None:
    """Write a LaTeX-formatted summary table (mean ± std over seeds)."""
    rows = []
    for speed in sorted(df["target_speed_mps"].unique()):
        grp = df[df["target_speed_mps"] == speed]
        rows.append({
            "Speed": f"{speed:.1f}",
            "Det Rate": f"{grp['detection_rate'].mean():.1%} ± {grp['detection_rate'].std():.1%}",
            "Track Succ": f"{grp['search_success_rate'].mean():.1%} ± {grp['search_success_rate'].std():.1%}",
            "Det Time (s)": f"{grp['avg_first_detection_time'].mean():.1f} ± {grp['avg_first_detection_time'].std():.1f}",
            "Loc Err (m)": f"{grp['avg_localization_error_m'].mean():.3f} ± {grp['avg_localization_error_m'].std():.3f}",
            "Loss": f"{grp['total_loss_events'].mean():.1f}",
            "Reacq Rate": f"{grp['reacquisition_rate'].mean():.2f}",
            "Handovers": f"{grp['successful_handovers'].mean():.1f}",
            "UAV Util": f"{grp['uav_utilization'].mean():.1%}",
            "FPS": f"{grp['sim_fps'].mean():.0f}",
        })

    cols = list(rows[0].keys())
    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Moving Target Detection \\& Tracking Results (mean $\\pm$ std, 5 seeds)}",
        "\\label{tab:moving_target_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "c" * len(cols) + "}",
        "\\toprule",
        " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(" & ".join(str(row[c]) for c in cols) + " \\\\")
    tex_lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]

    out = results_dir / "summary_table.tex"
    out.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"  LaTeX table: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot moving-target experiment results"
    )
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--plots",   type=Path, default=None)
    parser.add_argument("--show",    action="store_true")
    args = parser.parse_args()

    if args.show:
        matplotlib.use("TkAgg")

    plots_dir = args.plots or args.results / "plots"

    try:
        df = _load(args.results)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print(f"Loaded {len(df)} rows from {args.results / 'mission_summary.csv'}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print(f"Speeds: {sorted(df['target_speed_mps'].unique())}")
    print(f"Generating plots → {plots_dir}\n")

    fig_detection_rate(df, plots_dir, args.show)
    fig_tracking_success(df, plots_dir, args.show)
    fig_detection_time(df, plots_dir, args.show)
    fig_localization_error(df, plots_dir, args.show)
    fig_track_loss(df, plots_dir, args.show)
    fig_reacquisition_rate(df, plots_dir, args.show)
    fig_handovers(df, plots_dir, args.show)
    fig_uav_utilization(df, plots_dir, args.show)
    fig_fps(df, plots_dir, args.show)
    fig_combined_summary(df, plots_dir, args.show)
    write_latex_table(df, args.results)

    print("\nDone.")


if __name__ == "__main__":
    main()
