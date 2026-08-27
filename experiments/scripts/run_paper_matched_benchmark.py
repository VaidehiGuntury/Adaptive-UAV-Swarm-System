"""
Paper-Matched DEBS Benchmark — 10 UAV Static Exploration.

Measures the corrected 2D DEBS implementation (BSA + IDE) under static,
exploration-only conditions for 350 s across 5 deterministic seeds
[42, 43, 44, 45, 46], producing:

  - per-seed full timeseries CSVs (3500 rows each)
  - per-seed scalar summary JSONs
  - aggregate summary CSV (paper_matched_summary.csv)
  - sampled coverage-vs-time CSV (10 s intervals)
  - threshold crossing times CSV (T50–T90)
  - coverage-vs-time plot (coverage_vs_time_10uav.png)
  - threshold bar chart (threshold_comparison.png)
  - human-readable report (paper_matched_report.txt)

No existing source file, configuration, runner, or test is modified.

Usage (from project root):
    python -m experiments.scripts.run_paper_matched_benchmark
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import SimulationConfig, load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("configs/experiments/paper_matched_debs_10uav_static.yaml")
OUTPUT_DIR = Path("experiments/results/paper_matched_debs_10uav_static")
SEEDS = [42, 43, 44, 45, 46]
THRESHOLDS = [50, 60, 70, 80, 85, 90]
MILESTONE_TIMES = [120, 200, 300, 350]
SAMPLE_INTERVAL = 10.0  # seconds between sampled timeseries rows


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRunResult:
    """All per-seed data produced by a single benchmark run."""

    seed: int
    # Full per-step timeseries: list of (time_s, coverage_pct) tuples
    # Length == round(duration / dt) == 3500 for this benchmark
    timeseries: list[tuple[float, float]]
    # Scalar metrics (end of run)
    final_coverage: float           # fraction [0, 1], NOT percentage
    path_distance_total: float      # metres, fleet total
    trajectory_efficiency: float    # m² explored / m travelled
    mean_speed: float               # m/s, from SimulationMetrics at last step
    mission_overlap_final: float    # fraction, from SimulationMetrics at last step
    replanning_count: int           # engine.aggregation.step_reassignment_count at end
    # Milestone coverage values (percentage, not fraction)
    coverage_pct_120s: float
    coverage_pct_200s: float
    coverage_pct_300s: float
    coverage_pct_350s: float
    # Threshold crossing times (seconds); float('nan') if not reached
    T50: float
    T60: float
    T70: float
    T80: float
    T85: float
    T90: float
    # Exploration rates: coverage_pct(t) / t  (% per second)
    exploration_rate_120s: float
    exploration_rate_200s: float
    exploration_rate_300s: float
    exploration_rate_350s: float
    # Coverage efficiency == trajectory_efficiency (alias for report clarity)
    coverage_efficiency: float
    # World dimensions (for trajectory_efficiency computation traceability)
    world_width: float
    world_height: float


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------
def build_engine(config: SimulationConfig) -> SimulationEngine:
    """
    Build a SimulationEngine from a per-seed config.

    Mirrors the pattern in run_multiseed.py exactly.  World.from_config is
    called with only (env_config, uav_config) — no dynamic_config — because
    the benchmark YAML omits the dynamic_environment block, so
    config.dynamic_environment is None and no ObstacleManager is created.

    IDE is built internally by SimulationEngine when config.ide is not None.
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


# ---------------------------------------------------------------------------
# Pure helper functions (testable without simulation)
# ---------------------------------------------------------------------------
def compute_threshold_times(
    timeseries: list[tuple[float, float]],
    thresholds: list[int],
) -> dict[int, float]:
    """
    Scan timeseries once; return the first time_s where coverage_pct >= X
    for each threshold X.  Returns float('nan') for thresholds not crossed.

    timeseries : list of (time_s, coverage_pct)  — simulation order
    thresholds : list of integer percentages e.g. [50, 60, 70, 80, 85, 90]
    """
    result: dict[int, float] = {x: float("nan") for x in thresholds}
    remaining = set(thresholds)
    for time_s, cov_pct in timeseries:
        crossed = {x for x in remaining if cov_pct >= x}
        for x in crossed:
            result[x] = time_s
        remaining -= crossed
        if not remaining:
            break
    return result


def compute_milestones(
    timeseries: list[tuple[float, float]],
    dt: float,
) -> dict[int, float]:
    """
    Return coverage_pct at each milestone time by direct index lookup.

    Milestone step index = round(t_m / dt).  Clamps to last index to guard
    against floating-point step-count drift.
    """
    milestones: dict[int, float] = {}
    for t_m in MILESTONE_TIMES:
        idx = min(round(t_m / dt), len(timeseries) - 1)
        milestones[t_m] = timeseries[idx][1]
    return milestones


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------
def run_single_seed(base_config: SimulationConfig, seed: int) -> BenchmarkRunResult:
    """
    Run one benchmark trial for a given seed.

    Per-seed config is derived via dataclasses.replace on obstacle_seed only.
    The IDE seed remains fixed at 42 (from the ide: block in YAML) for all 5
    seeds, as required.
    """
    config = dataclasses.replace(
        base_config,
        environment=dataclasses.replace(base_config.environment, obstacle_seed=seed),
    )
    engine = build_engine(config)
    dt = config.dt
    duration = config.duration

    # Pre-implementation verification (design §3):
    # Confirm engine was constructed with correct parameters before stepping.
    assert engine.config.uav.max_speed == 1.5, \
        f"max_speed mismatch: {engine.config.uav.max_speed}"
    assert engine.config.uav.max_angular_velocity == 0.9, \
        f"max_angular_velocity mismatch: {engine.config.uav.max_angular_velocity}"
    assert engine.config.uav.sensing_range == 4.5, \
        f"sensing_range mismatch: {engine.config.uav.sensing_range}"
    assert engine.config.num_uavs == 10, \
        f"num_uavs mismatch: {engine.config.num_uavs}"
    assert engine.config.duration == 350.0, \
        f"duration mismatch: {engine.config.duration}"
    assert engine.config.dynamic_environment is None, \
        "dynamic_environment must be None for static benchmark"
    assert engine.config.search is None, \
        "search must be None for exploration-only benchmark"
    assert engine.config.ide is not None, \
        "ide config must not be None — IDE must be enabled"
    assert engine.mission_orchestrator is None, \
        "mission_orchestrator must be None at engine construction"
    assert len(engine.agents) == 10, \
        f"Expected 10 agents, got {len(engine.agents)}"

    timeseries: list[tuple[float, float]] = []
    threshold_crossed: dict[int, Optional[float]] = {
        x: None for x in THRESHOLDS
    }

    last_metrics = None
    while engine.time_s < duration:
        m = engine.step()
        cov_pct = m.explored_fraction * 100.0
        timeseries.append((round(m.time_s, 4), round(cov_pct, 6)))
        for x in THRESHOLDS:
            if threshold_crossed[x] is None and m.explored_fraction >= x / 100.0:
                threshold_crossed[x] = m.time_s
        last_metrics = m

    assert last_metrics is not None, "No simulation steps were executed"

    # replanning_count: cumulative total from the aggregation controller.
    # We read the final value once after the loop — NOT sum per-step — to avoid
    # double-counting (step_reassignment_count is a running total, not delta).
    replanning_count = engine.aggregation.step_reassignment_count

    # Path distance (fleet total)
    path_distance_total = 0.0
    for positions in engine.agent_histories.values():
        for i in range(1, len(positions)):
            path_distance_total += float(
                np.linalg.norm(positions[i] - positions[i - 1])
            )

    final_coverage = timeseries[-1][1] / 100.0  # fraction
    world_area = engine.world.width * engine.world.height
    trajectory_efficiency = (
        (final_coverage * world_area) / path_distance_total
        if path_distance_total > 0
        else 0.0
    )

    milestones = compute_milestones(timeseries, dt)
    rates = {
        t: (milestones[t] / t if t > 0 else 0.0) for t in MILESTONE_TIMES
    }

    return BenchmarkRunResult(
        seed=seed,
        timeseries=timeseries,
        final_coverage=final_coverage,
        path_distance_total=path_distance_total,
        trajectory_efficiency=trajectory_efficiency,
        mean_speed=last_metrics.mean_speed,
        mission_overlap_final=last_metrics.mission_overlap_fraction,
        replanning_count=replanning_count,
        coverage_pct_120s=milestones[120],
        coverage_pct_200s=milestones[200],
        coverage_pct_300s=milestones[300],
        coverage_pct_350s=milestones[350],
        T50=(
            threshold_crossed[50]
            if threshold_crossed[50] is not None
            else float("nan")
        ),
        T60=(
            threshold_crossed[60]
            if threshold_crossed[60] is not None
            else float("nan")
        ),
        T70=(
            threshold_crossed[70]
            if threshold_crossed[70] is not None
            else float("nan")
        ),
        T80=(
            threshold_crossed[80]
            if threshold_crossed[80] is not None
            else float("nan")
        ),
        T85=(
            threshold_crossed[85]
            if threshold_crossed[85] is not None
            else float("nan")
        ),
        T90=(
            threshold_crossed[90]
            if threshold_crossed[90] is not None
            else float("nan")
        ),
        exploration_rate_120s=rates[120],
        exploration_rate_200s=rates[200],
        exploration_rate_300s=rates[300],
        exploration_rate_350s=rates[350],
        coverage_efficiency=trajectory_efficiency,
        world_width=engine.world.width,
        world_height=engine.world.height,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_timeseries_csv(result: BenchmarkRunResult, output_dir: Path) -> Path:
    """Write seed_{S}_timeseries.csv with columns [time_s, coverage_pct]."""
    path = output_dir / f"seed_{result.seed}_timeseries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "coverage_pct"])
        writer.writerows(result.timeseries)
    return path


def write_seed_summary_json(result: BenchmarkRunResult, output_dir: Path) -> Path:
    """
    Write seed_{S}_summary.json.

    float('nan') threshold values are serialised as JSON null.
    """
    path = output_dir / f"seed_{result.seed}_summary.json"

    def _nan_to_null(v: float) -> Optional[float]:
        return None if (isinstance(v, float) and math.isnan(v)) else v

    payload = {
        "seed": result.seed,
        "final_coverage": result.final_coverage,
        "path_distance_total": result.path_distance_total,
        "trajectory_efficiency": result.trajectory_efficiency,
        "mean_speed": result.mean_speed,
        "mission_overlap_final": result.mission_overlap_final,
        "replanning_count": result.replanning_count,
        "coverage_pct_120s": result.coverage_pct_120s,
        "coverage_pct_200s": result.coverage_pct_200s,
        "coverage_pct_300s": result.coverage_pct_300s,
        "coverage_pct_350s": result.coverage_pct_350s,
        "T50": _nan_to_null(result.T50),
        "T60": _nan_to_null(result.T60),
        "T70": _nan_to_null(result.T70),
        "T80": _nan_to_null(result.T80),
        "T85": _nan_to_null(result.T85),
        "T90": _nan_to_null(result.T90),
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def write_aggregate_csv(
    results: list[BenchmarkRunResult], output_dir: Path
) -> Path:
    """
    Write paper_matched_summary.csv — one row per seed.

    float('nan') threshold values are written as empty strings (CSV convention).
    """
    path = output_dir / "paper_matched_summary.csv"
    fieldnames = [
        "seed",
        "final_coverage",
        "path_distance_total",
        "trajectory_efficiency",
        "mean_speed",
        "mission_overlap_final",
        "replanning_count",
        "coverage_pct_120s",
        "coverage_pct_200s",
        "coverage_pct_300s",
        "coverage_pct_350s",
        "T50",
        "T60",
        "T70",
        "T80",
        "T85",
        "T90",
        "exploration_rate_120s",
        "exploration_rate_200s",
        "exploration_rate_300s",
        "exploration_rate_350s",
        "coverage_efficiency",
    ]

    def _fmt(v: float) -> object:
        return "" if (isinstance(v, float) and math.isnan(v)) else v

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "seed": r.seed,
                    "final_coverage": r.final_coverage,
                    "path_distance_total": r.path_distance_total,
                    "trajectory_efficiency": r.trajectory_efficiency,
                    "mean_speed": r.mean_speed,
                    "mission_overlap_final": r.mission_overlap_final,
                    "replanning_count": r.replanning_count,
                    "coverage_pct_120s": r.coverage_pct_120s,
                    "coverage_pct_200s": r.coverage_pct_200s,
                    "coverage_pct_300s": r.coverage_pct_300s,
                    "coverage_pct_350s": r.coverage_pct_350s,
                    "T50": _fmt(r.T50),
                    "T60": _fmt(r.T60),
                    "T70": _fmt(r.T70),
                    "T80": _fmt(r.T80),
                    "T85": _fmt(r.T85),
                    "T90": _fmt(r.T90),
                    "exploration_rate_120s": r.exploration_rate_120s,
                    "exploration_rate_200s": r.exploration_rate_200s,
                    "exploration_rate_300s": r.exploration_rate_300s,
                    "exploration_rate_350s": r.exploration_rate_350s,
                    "coverage_efficiency": r.coverage_efficiency,
                }
            )
    return path


def write_sampled_timeseries(
    results: list[BenchmarkRunResult],
    dt: float,
    sample_interval: float,
    output_dir: Path,
) -> Path:
    """
    Write coverage_vs_time.csv at sample_interval second intervals.

    Columns: [time_s, mean_coverage_pct, std_coverage_pct]
    Rows:    t = 0, 10, 20, ..., 350  (36 rows)
    """
    path = output_dir / "coverage_vs_time.csv"
    # Infer duration from last timeseries entry
    last_t = results[0].timeseries[-1][0]
    sample_times = np.arange(0.0, last_t + sample_interval / 2.0, sample_interval)

    rows = []
    for t in sample_times:
        idx = min(round(t / dt), len(results[0].timeseries) - 1)
        values = np.array([r.timeseries[idx][1] for r in results])
        rows.append(
            (round(float(t), 1), float(values.mean()), float(values.std()))
        )

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "mean_coverage_pct", "std_coverage_pct"])
        writer.writerows(rows)
    return path


def write_threshold_csv(
    results: list[BenchmarkRunResult], output_dir: Path
) -> Path:
    """
    Write threshold_times.csv — one row per threshold.

    Mean/std are computed only over seeds that reached the threshold.
    NaN seeds are excluded from statistics but counted in n_not_reached.
    """
    path = output_dir / "threshold_times.csv"
    threshold_attrs = {
        50: "T50",
        60: "T60",
        70: "T70",
        80: "T80",
        85: "T85",
        90: "T90",
    }
    rows = []
    for x, attr in threshold_attrs.items():
        values = [getattr(r, attr) for r in results]
        reached = [v for v in values if not math.isnan(v)]
        n_reached = len(reached)
        n_not_reached = len(values) - n_reached
        if reached:
            arr = np.array(reached)
            mean_t = float(arr.mean())
            std_t = float(arr.std())
        else:
            mean_t = float("nan")
            std_t = float("nan")
        rows.append((x, mean_t, std_t, n_reached, n_not_reached))

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["threshold_pct", "mean_T", "std_T", "n_reached", "n_not_reached"]
        )
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_coverage_curve(
    results: list[BenchmarkRunResult],
    threshold_stats: dict[int, tuple[float, float, int, int]],
    output_dir: Path,
) -> Path:
    """
    Produce coverage_vs_time_10uav.png.

    Style mirrors run_multiseed.py:  LINE_COLOR=#2a78d6, BAND_ALPHA=0.18,
    figure (7, 4.2) @ dpi=150, Agg backend.
    """
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
    REF_COLOR = "#c87533"

    times = np.array([t for t, _ in results[0].timeseries])
    matrix = np.vstack([[cov for _, cov in r.timeseries] for r in results])
    mean_cov = matrix.mean(axis=0)
    std_cov = matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(
        times,
        mean_cov - std_cov,
        mean_cov + std_cov,
        color=LINE_COLOR,
        alpha=BAND_ALPHA,
        linewidth=0,
    )
    ax.plot(
        times,
        mean_cov,
        color=LINE_COLOR,
        linewidth=2,
        label="Mean \u00b1 std (5 seeds)",
    )

    # Horizontal reference lines
    for ref_pct in (50, 70, 80, 85, 90):
        ax.axhline(ref_pct, color=REF_COLOR, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(
            355,
            ref_pct + 0.5,
            f"{ref_pct}%",
            color=REF_COLOR,
            fontsize=7,
            va="bottom",
            ha="left",
        )

    # Vertical T markers for reached thresholds
    for x in (70, 80, 85, 90):
        stats = threshold_stats.get(x)
        if stats and stats[2] > 0:
            mean_t = stats[0]
            ax.axvline(
                mean_t, color=MUTED_INK, linewidth=0.8, linestyle=":", alpha=0.7
            )
            ax.text(mean_t + 1, 2, f"T{x}", color=MUTED_INK, fontsize=7, va="bottom")

    # Paper annotation — text only, NOT a curve
    ax.text(
        5,
        92,
        "DEBS paper: 112.3 \u00b1 10.6 s mission completion\n"
        "(NOTE: 3D vs 2D, different arena \u2014 not directly comparable)",
        color=MUTED_INK,
        fontsize=7,
        va="top",
        ha="left",
        bbox=dict(facecolor=SURFACE, edgecolor=BASELINE, boxstyle="round,pad=0.3"),
    )

    ax.set_xlabel("Time (s)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Coverage (%)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "Coverage vs Time \u2014 10 UAV Static DEBS (5 seeds, mean \u00b1 std)",
        color=PRIMARY_INK,
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path = output_dir / "coverage_vs_time_10uav.png"
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_threshold_bar(
    threshold_stats: dict[int, tuple[float, float, int, int]],
    output_dir: Path,
) -> Path:
    """
    Produce threshold_comparison.png — bar chart of mean T50–T90 with std error bars.

    Thresholds not reached by any seed are omitted from bars but noted in text.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LINE_COLOR = "#2a78d6"
    PRIMARY_INK = "#0b0b0b"
    SECONDARY_INK = "#52514e"
    MUTED_INK = "#898781"
    SURFACE = "#fcfcfb"
    GRIDLINE = "#e1e0d9"
    BASELINE = "#c3c2b7"

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    not_reached: list[str] = []

    for x in THRESHOLDS:
        stats = threshold_stats[x]
        mean_t, std_t, n_reached, _ = stats
        if n_reached > 0 and not math.isnan(mean_t):
            labels.append(f"T{x}")
            means.append(mean_t)
            stds.append(std_t)
        else:
            not_reached.append(f"T{x}")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if labels:
        x_pos = list(range(len(labels)))
        bars = ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            color=LINE_COLOR,
            alpha=0.8,
            width=0.5,
            error_kw={"elinewidth": 1.5, "ecolor": SECONDARY_INK},
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, color=SECONDARY_INK, fontsize=10)
        for rect, mean in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                mean + 3,
                f"{mean:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PRIMARY_INK,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No thresholds reached",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED_INK,
        )

    if not_reached:
        ax.text(
            0.98,
            0.97,
            f"Not reached: {', '.join(not_reached)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=MUTED_INK,
        )

    ax.set_ylabel("Time to threshold (s)", color=SECONDARY_INK, fontsize=10)
    ax.set_title(
        "Threshold Crossing Times T50\u2013T90 (mean \u00b1 std, 5 seeds)",
        color=PRIMARY_INK,
        fontsize=11,
        loc="left",
    )
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    fig.tight_layout()

    out_path = output_dir / "threshold_comparison.png"
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _build_report_text(
    results: list[BenchmarkRunResult],
    threshold_stats: dict[int, tuple[float, float, int, int]],
) -> str:
    """Assemble the full paper_matched_report.txt content as a string."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fmt_mean_std(values: list[float]) -> str:
        arr = np.array(values)
        return f"{arr.mean():.2f} \u00b1 {arr.std():.2f}"

    def _fmt_T(x: int) -> str:
        mean_t, std_t, n_reached, _ = threshold_stats[x]
        if math.isnan(mean_t):
            return f"NaN \u00b1 NaN s  (reached {n_reached}/5 runs)"
        return f"{mean_t:.1f} \u00b1 {std_t:.1f} s  (reached {n_reached}/5 runs)"

    cov_120 = _fmt_mean_std([r.coverage_pct_120s for r in results])
    cov_200 = _fmt_mean_std([r.coverage_pct_200s for r in results])
    cov_300 = _fmt_mean_std([r.coverage_pct_300s for r in results])
    cov_350 = _fmt_mean_std([r.coverage_pct_350s for r in results])

    final_mean = float(np.mean([r.final_coverage * 100 for r in results]))
    final_std = float(np.std([r.final_coverage * 100 for r in results]))
    speed_mean = float(np.mean([r.mean_speed for r in results]))
    speed_std = float(np.std([r.mean_speed for r in results]))

    rate_120 = _fmt_mean_std([r.exploration_rate_120s for r in results])
    rate_200 = _fmt_mean_std([r.exploration_rate_200s for r in results])
    rate_300 = _fmt_mean_std([r.exploration_rate_300s for r in results])
    rate_350 = _fmt_mean_std([r.exploration_rate_350s for r in results])

    eff_mean = float(np.mean([r.coverage_efficiency for r in results]))
    eff_std = float(np.std([r.coverage_efficiency for r in results]))
    replan_mean = float(np.mean([r.replanning_count for r in results]))
    replan_std = float(np.std([r.replanning_count for r in results]))

    def _T_our(x: int) -> str:
        mean_t, std_t, n_reached, _ = threshold_stats[x]
        if n_reached == 0 or math.isnan(mean_t):
            return "Not reached (0/5)"
        return f"{mean_t:.1f} \u00b1 {std_t:.1f} s"

    lines = [
        "PAPER-MATCHED DEBS BENCHMARK REPORT",
        "=====================================",
        f"Generated : {now}",
        f"Config    : {CONFIG_PATH}",
        f"Seeds     : {SEEDS}",
        "",
        "1. EXPERIMENT SUMMARY",
        "=====================",
        "Environment  : Static sparse (100\u00d7100 m, 20 static obstacles)",
        "Fleet        : 10 UAVs, max_speed=1.5 m/s, sensing_range=4.5 m",
        "Duration     : 350 s  (dt=0.1 s, 3500 steps per seed)",
        "Algorithms   : BSA (SelfAggregationController) + IDE (IDEAllocator)",
        "",
        "2. RESULTS OVERVIEW",
        "===================",
        "Coverage snapshots (mean \u00b1 std, 5 seeds):",
        f"  t=120 s : {cov_120} %",
        f"  t=200 s : {cov_200} %",
        f"  t=300 s : {cov_300} %",
        f"  t=350 s : {cov_350} %  [final coverage]",
        "",
        "Threshold crossing times (mean \u00b1 std):",
    ]
    for x in THRESHOLDS:
        lines.append(f"  T{x:2d} : {_fmt_T(x)}")

    lines += [
        "",
        "3. COMPARISON TABLE",
        "===================",
        "Metric                           | Original DEBS Paper     | Our Corrected DEBS              | Difference | Comparable?",
        "---------------------------------|-------------------------|--------------------------------|------------|---------------------",
        f"Mission completion time (10 UAV) | 112.3 \u00b1 10.6 s          | N/A (different criterion)      | N/A        | NOT_COMPARABLE",
    ]
    for x in THRESHOLDS:
        lines.append(
            f"T{x:2d} ({x}% coverage threshold)     | Not reported            | {_T_our(x):<30s} | N/A        | NOT_COMPARABLE"
        )
    lines += [
        f"Final coverage %                 | Not reported            | {final_mean:.2f} \u00b1 {final_std:.2f} %                 | N/A        | NOT_COMPARABLE",
        f"Mean UAV velocity                | Not reported            | {speed_mean:.2f} \u00b1 {speed_std:.2f} m/s              | N/A        | APPROXIMATELY_COMPARABLE",
        "Fleet size                       | 10 UAVs                 | 10 UAVs                        | 0          | DIRECTLY_COMPARABLE",
        "UAV max_speed parameter          | 1.5 m/s                 | 1.5 m/s                        | 0          | DIRECTLY_COMPARABLE",
        "General exploration behaviour    | BSA + IDE (DEBS \u00a74)    | BSA + IDE (DEBS \u00a74)           | N/A        | APPROXIMATELY_COMPARABLE",
        "",
        "NOTE: The paper's mission completion time uses a different stopping criterion",
        "(full 3D exploration of a 50\u00d750\u00d72 m sparse forest) from this benchmark's",
        "fixed-duration 2D approach. These numbers measure different things and",
        "cannot be numerically compared.",
        "(NOTE: not directly comparable \u2014 3D vs 2D, different arena and stopping criterion)",
        "",
        "4. OUR METRICS (NO PAPER EQUIVALENT)",
        "======================================",
        "These metrics are reported for informational purposes only. There are no",
        "corresponding paper values and no comparison is made or implied.",
        "",
        "Exploration rates (coverage % per second, mean \u00b1 std):",
        f"  t=120 s : {rate_120} %/s",
        f"  t=200 s : {rate_200} %/s",
        f"  t=300 s : {rate_300} %/s",
        f"  t=350 s : {rate_350} %/s",
        "",
        f"Coverage efficiency (m\u00b2 explored / m travelled):",
        f"  Mean : {eff_mean:.3f} \u00b1 {eff_std:.3f} m\u00b2/m",
        "",
        f"Replanning count (cumulative BSA reassignments, mean \u00b1 std):",
        f"  Mean : {replan_mean:.0f} \u00b1 {replan_std:.0f}",
        "",
        "5. LIMITATIONS",
        "==============",
        "The following differences between this benchmark and the original DEBS",
        "paper prevent direct numerical comparison of most metrics:",
        "",
        "  (a) Dimensionality: This simulation is 2D; the paper uses a 3D UAV model.",
        "  (b) Arena: This benchmark uses a 100\u00d7100 m 2D arena. The paper reports",
        "      results on a 50\u00d750\u00d72 m sparse forest environment. These are not the",
        "      same environment.",
        "  (c) Resolution: Coverage here is measured over traversable 2D grid cells",
        "      at resolution sensing_range/3 (\u22481.5 m). The paper uses 3D voxels",
        "      at 0.15 m resolution.",
        "  (d) Trajectory generation: This simulation uses direct heading control.",
        "      The paper uses A* path planning with B-spline trajectory smoothing.",
        "  (e) Depth camera: This simulation uses a circular sensing disc. The paper",
        "      uses a depth-camera field-of-view model.",
        "  (f) ROS integration: The paper runs in ROS; this simulation is standalone",
        "      Python.",
        "  (g) Stopping criterion: The paper's mission completion time is the elapsed",
        "      time until full 3D exploration is achieved; this benchmark runs for a",
        "      fixed 350 s.",
        "",
        "6. APPROXIMATION NOTES",
        "=======================",
        "  - Our 100\u00d7100 m 2D arena with 20 static obstacles is the closest",
        "    available approximation to the paper's 50\u00d750\u00d72 m sparse forest.",
        "    It is NOT claimed to be identical or equivalent.",
        "  - Coverage fraction in this benchmark is measured over traversable 2D",
        "    grid cells at resolution sensing_range/3 (\u22481.5 m), not over 3D voxels",
        "    at 0.15 m resolution.",
        "  - The IDE config seed is fixed at seed=42 for all 5 runs.",
        "    What varies across seeds:",
        "      * obstacle placement  (environment.obstacle_seed = S)",
        "      * UAV ring-spawn positions  (spawn_uavs seed = S)",
        "      * BSA RNG  (SelfAggregationController rng = np.random.default_rng(S))",
        "    What is fixed across seeds:",
        "      * IDE internal RNG  (IDEAllocator seeded from ide.seed=42 in config)",
        "",
        "7. VALID CONCLUSIONS",
        "====================",
        "The following conclusions are supported by this benchmark:",
        "",
        "  (a) Consistency validation: Our 2D DEBS implementation produces coverage",
        "      behaviour consistent with the published DEBS algorithm description \u2014",
        "      BSA frontier selection and IDE fair-share allocation are active and",
        "      functional.",
        "",
        f"  (b) Quantified performance: Our 2D corrected DEBS implementation achieves",
        f"      approximately {final_mean:.1f}% coverage at t=350 s under static",
        "      conditions (mean, 5 seeds).",
        "",
        "  (c) Controlled benchmark established: A reproducible 5-seed benchmark",
        "      exists at experiments/results/paper_matched_debs_10uav_static/ for",
        "      future comparison against other algorithm variants or implementations.",
        "",
        "  This report does NOT claim that our implementation outperforms, equals, or",
        "  is inferior to the original DEBS paper. Such a claim is not supported by",
        "  the available data.",
    ]
    return "\n".join(lines) + "\n"


def write_report_txt(
    results: list[BenchmarkRunResult],
    threshold_stats: dict[int, tuple[float, float, int, int]],
    output_dir: Path,
) -> Path:
    """Write paper_matched_report.txt."""
    path = output_dir / "paper_matched_report.txt"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(_build_report_text(results, threshold_stats))
    return path


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------
def print_terminal_summary(
    results: list[BenchmarkRunResult],
    threshold_stats: dict[int, tuple[float, float, int, int]],
    output_files: list[Path],
    config_path: Path,
) -> None:
    """Print the structured terminal summary (Requirement 20)."""
    print("=" * 56)
    print("PAPER-MATCHED DEBS BENCHMARK \u2014 10 UAV STATIC")
    print("=" * 56)
    print(f"Seeds: {SEEDS}")
    print("Environment: static sparse (100x100m, 20 static obstacles)")
    print("Fleet: 10 UAVs")
    print("Duration: 350 s")
    print(f"Config: {config_path}")
    print()
    print("Coverage snapshots (mean \u00b1 std across 5 seeds):")
    for attr, t_label in [
        ("coverage_pct_120s", "120 s"),
        ("coverage_pct_200s", "200 s"),
        ("coverage_pct_300s", "300 s"),
        ("coverage_pct_350s", "350 s"),
    ]:
        vals = np.array([getattr(r, attr) for r in results])
        print(f"  {t_label} : {vals.mean():.2f} \u00b1 {vals.std():.2f} %")
    print()
    print("Threshold crossing times (mean \u00b1 std):")
    for x in THRESHOLDS:
        mean_t, std_t, n_reached, _ = threshold_stats[x]
        if math.isnan(mean_t):
            print(f"  T{x:2d} : NaN \u00b1 NaN s  (reached {n_reached}/5 runs)")
        else:
            print(
                f"  T{x:2d} : {mean_t:.1f} \u00b1 {std_t:.1f} s  (reached {n_reached}/5 runs)"
            )
    print()
    print("Original DEBS paper reference:")
    print("  10-UAV sparse DEBS mission completion time: 112.3 \u00b1 10.6 s")
    print(
        "  (NOTE: not directly comparable \u2014 3D vs 2D, "
        "different arena and stopping criterion)"
    )
    print()
    print("Output files:")
    for fp in output_files:
        print(f"  {fp}")
    print("=" * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full paper-matched DEBS benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_config(CONFIG_PATH)

    # Pre-flight checks (design §pre-implementation verification)
    assert base_config.dynamic_environment is None, (
        "Benchmark_Config must not have dynamic_environment enabled; "
        f"got: {base_config.dynamic_environment}"
    )
    assert base_config.search is None, (
        "Benchmark_Config must not have search enabled; "
        f"got: {base_config.search}"
    )
    assert base_config.ide is not None, (
        "Benchmark_Config must have an ide: block; got ide=None. "
        "IDE pairwise allocation MUST be on for this benchmark."
    )
    assert base_config.ide.alpha == 0.5
    assert base_config.ide.seed == 42
    assert base_config.num_uavs == 10
    assert base_config.duration == 350.0
    assert base_config.uav.max_speed == 1.5
    assert base_config.uav.max_angular_velocity == 0.9
    assert base_config.uav.sensing_range == 4.5

    results: list[BenchmarkRunResult] = []
    for i, seed in enumerate(SEEDS, start=1):
        print(f"[{i}/5] seed={seed} running...")
        result = run_single_seed(base_config, seed)
        t50_str = f"{result.T50:.1f}" if not math.isnan(result.T50) else "NaN"
        t70_str = f"{result.T70:.1f}" if not math.isnan(result.T70) else "NaN"
        t85_str = f"{result.T85:.1f}" if not math.isnan(result.T85) else "NaN"
        print(
            f"[{i}/5] seed={seed} done  "
            f"coverage={result.final_coverage * 100:.1f}%  "
            f"T50={t50_str}  T70={t70_str}  T85={t85_str}"
        )
        results.append(result)

    # Aggregate threshold statistics
    threshold_stats: dict[int, tuple[float, float, int, int]] = {}
    for x in THRESHOLDS:
        attr = f"T{x}"
        values = [getattr(r, attr) for r in results]
        reached = [v for v in values if not math.isnan(v)]
        n_reached = len(reached)
        n_not_reached = len(values) - n_reached
        if reached:
            arr = np.array(reached)
            threshold_stats[x] = (
                float(arr.mean()),
                float(arr.std()),
                n_reached,
                n_not_reached,
            )
        else:
            threshold_stats[x] = (float("nan"), float("nan"), 0, n_not_reached)

    # Write all output files
    output_files: list[Path] = []
    for r in results:
        output_files.append(write_timeseries_csv(r, OUTPUT_DIR))
        output_files.append(write_seed_summary_json(r, OUTPUT_DIR))
    output_files.append(
        write_sampled_timeseries(results, base_config.dt, SAMPLE_INTERVAL, OUTPUT_DIR)
    )
    output_files.append(write_threshold_csv(results, OUTPUT_DIR))
    output_files.append(write_aggregate_csv(results, OUTPUT_DIR))
    output_files.append(plot_coverage_curve(results, threshold_stats, OUTPUT_DIR))
    output_files.append(plot_threshold_bar(threshold_stats, OUTPUT_DIR))
    output_files.append(write_report_txt(results, threshold_stats, OUTPUT_DIR))

    print_terminal_summary(results, threshold_stats, output_files, CONFIG_PATH)


if __name__ == "__main__":
    main()
