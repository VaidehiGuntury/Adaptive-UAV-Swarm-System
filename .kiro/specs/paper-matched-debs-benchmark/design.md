# Design Document — Paper-Matched DEBS Benchmark

## Overview

This design covers two new files that together form the **Paper-Matched DEBS Benchmark**:

1. `configs/experiments/paper_matched_debs_10uav_static.yaml` — the **Benchmark_Config**
2. `experiments/scripts/run_paper_matched_benchmark.py` — the **Benchmark_Runner**

The benchmark runs the corrected 2D DEBS implementation (BSA + IDE) under static, exploration-only conditions for 350 s across 5 deterministic seeds (`[42, 43, 44, 45, 46]`), producing coverage timeseries, threshold crossing times, scalar metrics, plots, and a scientific report. No file in `src/`, `tests/`, or any existing config is modified.

The design deliberately mirrors the patterns established in `experiments/scripts/run_multiseed.py` to keep the codebase consistent: `load_config` → `dataclasses.replace` → `build_engine` → step loop → metric extraction → output writing.

---

## Architecture

### Data Flow Diagram

```mermaid
flowchart TD
    A["paper_matched_debs_10uav_static.yaml<br/>(Benchmark_Config)"] -->|load_config| B["SimulationConfig\n(search=None, dynamic_environment=None)"]
    B -->|dataclasses.replace per seed| C["Per-seed SimulationConfig\n(obstacle_seed=S)"]
    C -->|build_engine| D["SimulationEngine\n(World + Agents + BSA + IDE)"]
    D -->|engine.step() × 3500| E["SimulationMetrics list\n+ agent_histories"]
    E -->|run_single_seed| F["BenchmarkRunResult\n(per-seed dataclass)"]
    F -->|write_timeseries_csv| G["seed_S_timeseries.csv"]
    F -->|write_seed_summary_json| H["seed_S_summary.json"]
    F -->|compute_threshold_times| I["threshold data"]
    F -->|compute_milestones| J["milestone data"]
    
    subgraph "Aggregate (all 5 seeds)"
        K["List[BenchmarkRunResult]"]
    end
    
    F --> K
    K -->|write_sampled_timeseries| L["coverage_vs_time.csv"]
    K -->|write_threshold_csv| M["threshold_times.csv"]
    K -->|write_aggregate_csv| N["paper_matched_summary.csv"]
    K -->|plot_coverage_curve| O["coverage_vs_time_10uav.png"]
    K -->|plot_threshold_bar| P["threshold_comparison.png"]
    K -->|write_report_txt| Q["paper_matched_report.txt"]
    K -->|print_terminal_summary| R["stdout terminal summary"]
```

### Module Structure

```
experiments/scripts/run_paper_matched_benchmark.py
├── Imports and constants
├── BenchmarkRunResult  (dataclass)
├── build_engine()
├── run_single_seed()
├── compute_threshold_times()
├── compute_milestones()
├── write_timeseries_csv()
├── write_seed_summary_json()
├── write_aggregate_csv()
├── write_sampled_timeseries()
├── write_threshold_csv()
├── plot_coverage_curve()
├── plot_threshold_bar()
├── write_report_txt()
├── print_terminal_summary()
└── main()
```

---

## Components and Interfaces

### 1. Benchmark_Config (YAML)

**File:** `configs/experiments/paper_matched_debs_10uav_static.yaml`

The config is intentionally minimal: it omits `dynamic_environment` and `search` blocks entirely, causing `load_config` to return `SimulationConfig(dynamic_environment=None, search=None)`. The `obstacle_seed` in the YAML acts as the default; at runtime `dataclasses.replace` overrides it per seed.

```yaml
# Paper-Matched DEBS Benchmark — 10 UAV, static environment, exploration only
# Loaded by: experiments/scripts/run_paper_matched_benchmark.py
# Purpose: Reproduce DEBS paper conditions as closely as possible in 2D.
#
# INTENTIONALLY OMITTED:
#   dynamic_environment — causes World.from_config to create no ObstacleManager
#   search              — causes _load_search_config to return None
#
# Seeds are overridden at runtime via dataclasses.replace(obstacle_seed=S).

environment:
  width: 100.0
  height: 100.0
  obstacle_count: 20
  obstacle_min_radius: 1.5
  obstacle_max_radius: 4.0
  obstacle_seed: 42        # default; overridden per run

uav:
  max_speed: 1.5
  max_angular_velocity: 0.9
  sensing_range: 4.5
  initial_spread_radius: 20.0
  spawn_mode: ring
  spawn_angular_noise: 0.15

aggregation:
  d_c: 0.5
  d_0: 17.8
  k_a: 1.0
  turn_cost_weight: 1.0
  trail_penalty: 8.0
  candidates_per_frontier: 6
  replan_interval: 2.0
  mission_region_radius: 17.8
  cluster_penalty_weight: 2.0
  trail_penalty_weight: 1.0
  turn_penalty_weight: 0.5

simulation:
  num_uavs: 10
  dt: 0.1
  duration: 350.0          # extended from simulation.yaml's 120.0 s
  spawn_center_x: 50.0
  spawn_center_y: 50.0
  animation_interval_ms: 50

ide:
  alpha: 0.5
  population_size: 20
  fe_max: 200
  t_att: 2.0
  d_star: 30.0
  communication_range: 50.0
  bounds_padding: 0.5
  seed: 42                 # fixed for all 5 seeds; only obstacle_seed varies
```

**Design decision**: The `dynamic_environment` and `search` blocks are omitted rather than set to `enabled: false`. Both `_load_dynamic_environment_config` and `_load_search_config` in `loader.py` return `None` when their respective YAML keys are absent, so omission is the cleanest signal. This also makes the YAML self-documenting (absence = disabled).

---

### 2. BenchmarkRunResult Dataclass

```python
@dataclass
class BenchmarkRunResult:
    seed: int
    # Full per-step timeseries: list of (time_s, coverage_pct) tuples
    # Length == round(duration / dt) == 3500
    timeseries: list[tuple[float, float]]
    # Scalar metrics at end of run
    final_coverage: float           # fraction [0,1], NOT percentage
    path_distance_total: float      # metres, fleet total
    trajectory_efficiency: float    # m² explored / m travelled
    mean_speed: float               # m/s, from SimulationMetrics at last step
    mission_overlap_final: float    # fraction, from SimulationMetrics at last step
    replanning_count: int           # cumulative step_reassignment_count across all steps
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
```

**Design decision**: `final_coverage` is stored as a fraction [0, 1] to match the convention in `run_multiseed.py`. All `coverage_pct_*` and `exploration_rate_*` fields use percentages so that CSV columns match the requirement names directly.

---

### 3. Function Signatures

#### `build_engine(config: SimulationConfig) -> SimulationEngine`

Identical in structure to `run_multiseed.py`'s `build_engine`. Constructs `World`, `spawn_uavs`, `SelfAggregationController`, and `SimulationEngine`. The `seed` argument to `spawn_uavs` and `np.random.default_rng` both use `config.environment.obstacle_seed`.

```python
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
        spawn_mode=config.uav.spawn_mode,
        spawn_angular_noise=config.uav.spawn_angular_noise,
    )
    aggregation = SelfAggregationController(
        config=config.aggregation,
        uav_config=config.uav,
        rng=np.random.default_rng(config.environment.obstacle_seed),
    )
    return SimulationEngine(world, agents, aggregation, config)
```

**Note on `World.from_config` signature**: When `dynamic_environment=None` in the config, `World.from_config` is called without a `dynamic_config` argument (matching the call pattern in `run_multiseed.py`). No `ObstacleManager` is created.

---

#### `run_single_seed(base_config: SimulationConfig, seed: int) -> BenchmarkRunResult`

Derives the per-seed config, builds the engine, steps 3500 times, collects all required data, and returns a `BenchmarkRunResult`.

```python
def run_single_seed(base_config: SimulationConfig, seed: int) -> BenchmarkRunResult:
    config = dataclasses.replace(
        base_config,
        environment=dataclasses.replace(base_config.environment, obstacle_seed=seed),
    )
    engine = build_engine(config)
    dt = config.dt
    duration = config.duration

    timeseries: list[tuple[float, float]] = []
    thresholds = {50: None, 60: None, 70: None, 80: None, 85: None, 90: None}
    replanning_count = 0
    last_metrics = None

    while engine.time_s < duration:
        m = engine.step()
        cov_pct = m.explored_fraction * 100.0
        timeseries.append((round(m.time_s, 4), round(cov_pct, 6)))
        for x in thresholds:
            if thresholds[x] is None and m.explored_fraction >= x / 100.0:
                thresholds[x] = m.time_s
        replanning_count += m.target_reassignment_count
        last_metrics = m

    # --- path distance ---
    path_distance_total = 0.0
    for positions in engine.agent_histories.values():
        for i in range(1, len(positions)):
            path_distance_total += float(np.linalg.norm(positions[i] - positions[i - 1]))

    final_coverage = timeseries[-1][1] / 100.0  # fraction
    world_area = engine.world.width * engine.world.height
    trajectory_efficiency = (
        (final_coverage * world_area) / path_distance_total
        if path_distance_total > 0 else 0.0
    )

    milestones = compute_milestones(timeseries, dt)
    rates = {t: (milestones[t] / t if t > 0 else 0.0)
             for t in (120, 200, 300, 350)}

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
        T50=thresholds[50] if thresholds[50] is not None else float('nan'),
        T60=thresholds[60] if thresholds[60] is not None else float('nan'),
        T70=thresholds[70] if thresholds[70] is not None else float('nan'),
        T80=thresholds[80] if thresholds[80] is not None else float('nan'),
        T85=thresholds[85] if thresholds[85] is not None else float('nan'),
        T90=thresholds[90] if thresholds[90] is not None else float('nan'),
        exploration_rate_120s=rates[120],
        exploration_rate_200s=rates[200],
        exploration_rate_300s=rates[300],
        exploration_rate_350s=rates[350],
        coverage_efficiency=trajectory_efficiency,
        world_width=engine.world.width,
        world_height=engine.world.height,
    )
```

**Design decision on `replanning_count`**: `SimulationMetrics.target_reassignment_count` is `engine.aggregation.step_reassignment_count` at each step. Since `step_reassignment_count` is the cumulative total in the aggregation controller and incremented each step it fires, summing `m.target_reassignment_count` over all steps would double-count. Instead, we take the final value from `engine.aggregation.step_reassignment_count` directly after the loop. This matches the semantics in the requirements ("cumulative `engine.aggregation.step_reassignment_count` across all steps" means the running total, not a sum of per-step snapshots).

**Revised implementation note for `replanning_count`:**

```python
    # After the step loop completes:
    replanning_count = engine.aggregation.step_reassignment_count
```

---

#### `compute_threshold_times(timeseries: list[tuple[float, float]], thresholds: list[int]) -> dict[int, float]`

Scans the timeseries once and returns the first time at which coverage_pct crosses each threshold. Returns `float('nan')` for thresholds never crossed.

```python
def compute_threshold_times(
    timeseries: list[tuple[float, float]],
    thresholds: list[int],
) -> dict[int, float]:
    """
    For each threshold X in thresholds, return the first time_s where
    coverage_pct >= X. Returns float('nan') if not crossed by end of run.

    timeseries: list of (time_s, coverage_pct) in simulation order.
    thresholds: list of integer coverage percentages e.g. [50,60,70,80,85,90].
    """
    result = {x: float('nan') for x in thresholds}
    remaining = set(thresholds)
    for time_s, cov_pct in timeseries:
        crossed = {x for x in remaining if cov_pct >= x}
        for x in crossed:
            result[x] = time_s
        remaining -= crossed
        if not remaining:
            break
    return result
```

---

#### `compute_milestones(timeseries: list[tuple[float, float]], dt: float) -> dict[int, float]`

Returns coverage_pct at milestone times 120, 200, 300, 350 by indexing the timeseries at `round(t / dt)`. Uses the last available step if the index is out of range (guards against floating-point step count drift).

```python
def compute_milestones(
    timeseries: list[tuple[float, float]],
    dt: float,
) -> dict[int, float]:
    """
    Return coverage_pct at each milestone time by indexing the timeseries.
    Milestone step index = round(t / dt).
    """
    milestones = {}
    for t in (120, 200, 300, 350):
        idx = min(round(t / dt), len(timeseries) - 1)
        milestones[t] = timeseries[idx][1]
    return milestones
```

---

#### `write_timeseries_csv(result: BenchmarkRunResult, output_dir: Path) -> Path`

Writes `seed_{S}_timeseries.csv` with header `[time_s, coverage_pct]` and 3500 data rows.

```python
def write_timeseries_csv(result: BenchmarkRunResult, output_dir: Path) -> Path:
    path = output_dir / f"seed_{result.seed}_timeseries.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "coverage_pct"])
        writer.writerows(result.timeseries)
    return path
```

---

#### `write_seed_summary_json(result: BenchmarkRunResult, output_dir: Path) -> Path`

Writes `seed_{S}_summary.json`. Non-reached thresholds (stored as `float('nan')`) are serialised as JSON `null` via a custom encoder.

```python
def write_seed_summary_json(result: BenchmarkRunResult, output_dir: Path) -> Path:
    path = output_dir / f"seed_{result.seed}_summary.json"

    def _nan_to_null(v):
        """Convert float nan to None for JSON serialisation."""
        import math
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
```

---

#### `write_aggregate_csv(results: list[BenchmarkRunResult], output_dir: Path) -> Path`

Writes `paper_matched_summary.csv` with one row per seed. `float('nan')` threshold values are written as empty strings in the CSV (matching the convention in `run_multiseed.py`).

```python
def write_aggregate_csv(
    results: list[BenchmarkRunResult], output_dir: Path
) -> Path:
    path = output_dir / "paper_matched_summary.csv"
    fieldnames = [
        "seed", "final_coverage", "path_distance_total", "trajectory_efficiency",
        "mean_speed", "mission_overlap_final", "replanning_count",
        "coverage_pct_120s", "coverage_pct_200s", "coverage_pct_300s", "coverage_pct_350s",
        "T50", "T60", "T70", "T80", "T85", "T90",
        "exploration_rate_120s", "exploration_rate_200s",
        "exploration_rate_300s", "exploration_rate_350s",
        "coverage_efficiency",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            def _fmt(v):
                import math
                return "" if (isinstance(v, float) and math.isnan(v)) else v
            writer.writerow({
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
                "T50": _fmt(r.T50), "T60": _fmt(r.T60),
                "T70": _fmt(r.T70), "T80": _fmt(r.T80),
                "T85": _fmt(r.T85), "T90": _fmt(r.T90),
                "exploration_rate_120s": r.exploration_rate_120s,
                "exploration_rate_200s": r.exploration_rate_200s,
                "exploration_rate_300s": r.exploration_rate_300s,
                "exploration_rate_350s": r.exploration_rate_350s,
                "coverage_efficiency": r.coverage_efficiency,
            })
    return path
```

---

#### `write_sampled_timeseries(results: list[BenchmarkRunResult], dt: float, sample_interval: float, output_dir: Path) -> Path`

Writes `coverage_vs_time.csv` sampled at 10 s intervals. For each sample time `t` in `[0, 10, 20, ..., 350]`, reads coverage from each seed's timeseries at `round(t / dt)`, then computes mean and std across seeds.

```python
def write_sampled_timeseries(
    results: list[BenchmarkRunResult],
    dt: float,
    sample_interval: float,
    output_dir: Path,
) -> Path:
    """
    Writes coverage_vs_time.csv with columns [time_s, mean_coverage_pct, std_coverage_pct]
    at sample_interval second intervals (default 10 s).
    """
    path = output_dir / "coverage_vs_time.csv"
    duration = results[0].timeseries[-1][0]  # inferred from last data point
    sample_times = np.arange(0.0, duration + sample_interval / 2.0, sample_interval)

    rows = []
    for t in sample_times:
        idx = min(round(t / dt), len(results[0].timeseries) - 1)
        values = np.array([r.timeseries[idx][1] for r in results])
        rows.append((round(float(t), 1), float(values.mean()), float(values.std())))

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_s", "mean_coverage_pct", "std_coverage_pct"])
        writer.writerows(rows)
    return path
```

---

#### `write_threshold_csv(results: list[BenchmarkRunResult], output_dir: Path) -> Path`

Writes `threshold_times.csv` with one row per threshold. Mean and std are computed only over seeds that reached the threshold; NaN seeds are excluded from statistics but counted.

```python
def write_threshold_csv(
    results: list[BenchmarkRunResult], output_dir: Path
) -> Path:
    import math
    path = output_dir / "threshold_times.csv"
    threshold_attrs = {
        50: "T50", 60: "T60", 70: "T70", 80: "T80", 85: "T85", 90: "T90"
    }
    rows = []
    for x, attr in threshold_attrs.items():
        values = [getattr(r, attr) for r in results]
        reached = [v for v in values if not math.isnan(v)]
        n_reached = len(reached)
        n_not_reached = len(values) - n_reached
        if reached:
            arr = np.array(reached)
            mean_T = float(arr.mean())
            std_T = float(arr.std())
        else:
            mean_T = float('nan')
            std_T = float('nan')
        rows.append((x, mean_T, std_T, n_reached, n_not_reached))

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["threshold_pct", "mean_T", "std_T", "n_reached", "n_not_reached"])
        writer.writerows(rows)
    return path
```

---

#### `plot_coverage_curve(results: list[BenchmarkRunResult], threshold_stats: dict, output_dir: Path) -> Path`

Produces `coverage_vs_time_10uav.png`. Uses the same visual style as `run_multiseed.py`.

```python
def plot_coverage_curve(
    results: list[BenchmarkRunResult],
    threshold_stats: dict,   # {int: (mean_T, std_T, n_reached, n_not_reached)}
    output_dir: Path,
) -> Path:
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
    REF_COLOR = "#c87533"   # muted amber for reference lines

    # Build common time axis from first result's timeseries
    times = np.array([t for t, _ in results[0].timeseries])
    matrix = np.vstack([[cov for _, cov in r.timeseries] for r in results])
    mean_cov = matrix.mean(axis=0)
    std_cov = matrix.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Mean line + std band
    ax.fill_between(times, mean_cov - std_cov, mean_cov + std_cov,
                    color=LINE_COLOR, alpha=BAND_ALPHA, linewidth=0)
    ax.plot(times, mean_cov, color=LINE_COLOR, linewidth=2, label="Mean ± std (5 seeds)")

    # Horizontal reference lines at 50, 70, 80, 85, 90 %
    for ref_pct in (50, 70, 80, 85, 90):
        ax.axhline(ref_pct, color=REF_COLOR, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(355, ref_pct + 0.5, f"{ref_pct}%", color=REF_COLOR,
                fontsize=7, va="bottom", ha="left")

    # Vertical T markers for reached thresholds (T70, T80, T85, T90)
    for x in (70, 80, 85, 90):
        stats = threshold_stats.get(x)
        if stats and stats[2] > 0:   # n_reached > 0
            mean_T = stats[0]
            ax.axvline(mean_T, color=MUTED_INK, linewidth=0.8, linestyle=":", alpha=0.7)
            ax.text(mean_T + 1, 2, f"T{x}", color=MUTED_INK, fontsize=7, va="bottom")

    # Paper annotation — text only, not a curve
    ax.text(5, 92,
            "DEBS paper: 112.3 ± 10.6 s mission completion\n"
            "(NOTE: 3D vs 2D, different arena — not directly comparable)",
            color=MUTED_INK, fontsize=7, va="top", ha="left",
            bbox=dict(facecolor=SURFACE, edgecolor=BASELINE, boxstyle="round,pad=0.3"))

    ax.set_xlabel("Time (s)", color=SECONDARY_INK, fontsize=10)
    ax.set_ylabel("Coverage (%)", color=SECONDARY_INK, fontsize=10)
    ax.set_title("Coverage vs Time — 10 UAV Static DEBS (5 seeds, mean ± std)",
                 color=PRIMARY_INK, fontsize=11, loc="left")
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
```

---

#### `plot_threshold_bar(threshold_stats: dict, output_dir: Path) -> Path`

Produces `threshold_comparison.png` — a bar chart of mean T50–T90 with std error bars. Thresholds not reached by any seed are skipped.

```python
def plot_threshold_bar(
    threshold_stats: dict,   # {int: (mean_T, std_T, n_reached, n_not_reached)}
    output_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    LINE_COLOR = "#2a78d6"
    PRIMARY_INK = "#0b0b0b"
    SECONDARY_INK = "#52514e"
    MUTED_INK = "#898781"
    SURFACE = "#fcfcfb"
    GRIDLINE = "#e1e0d9"
    BASELINE = "#c3c2b7"

    labels, means, stds = [], [], []
    not_reached = []
    for x in (50, 60, 70, 80, 85, 90):
        stats = threshold_stats[x]
        mean_T, std_T, n_reached, _ = stats
        if n_reached > 0 and not math.isnan(mean_T):
            labels.append(f"T{x}")
            means.append(mean_T)
            stds.append(std_T)
        else:
            not_reached.append(f"T{x}")

    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if labels:
        x_pos = range(len(labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                      color=LINE_COLOR, alpha=0.8, width=0.5,
                      error_kw={"elinewidth": 1.5, "ecolor": SECONDARY_INK})
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels, color=SECONDARY_INK, fontsize=10)

        # Annotate mean values on bars
        for rect, mean in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width() / 2.0, mean + 3,
                    f"{mean:.1f}s", ha="center", va="bottom",
                    fontsize=8, color=PRIMARY_INK)

    if not_reached:
        ax.text(0.98, 0.97, f"Not reached: {', '.join(not_reached)}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=MUTED_INK)

    ax.set_ylabel("Time to threshold (s)", color=SECONDARY_INK, fontsize=10)
    ax.set_title("Threshold Crossing Times T50–T90 (mean ± std, 5 seeds)",
                 color=PRIMARY_INK, fontsize=11, loc="left")
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
```

---

#### `write_report_txt(results: list[BenchmarkRunResult], threshold_stats: dict, output_dir: Path) -> Path`

Writes `paper_matched_report.txt`. The report is structured in sections defined below (see "Report Structure" section).

```python
def write_report_txt(
    results: list[BenchmarkRunResult],
    threshold_stats: dict,
    output_dir: Path,
) -> Path:
    path = output_dir / "paper_matched_report.txt"
    report = _build_report_text(results, threshold_stats)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(report)
    return path
```

The `_build_report_text` helper assembles all sections as a multi-line string (no external templating library needed).

---

#### `print_terminal_summary(results: list[BenchmarkRunResult], threshold_stats: dict, output_files: list[Path], config_path: Path) -> None`

Prints the terminal summary in the exact format specified in Requirement 20.

```python
def print_terminal_summary(
    results: list[BenchmarkRunResult],
    threshold_stats: dict,
    output_files: list[Path],
    config_path: Path,
) -> None:
    import math
    print("=" * 56)
    print("PAPER-MATCHED DEBS BENCHMARK — 10 UAV STATIC")
    print("=" * 56)
    print(f"Seeds: [42, 43, 44, 45, 46]")
    print("Environment: static sparse (100x100m, 20 static obstacles)")
    print("Fleet: 10 UAVs")
    print("Duration: 350 s")
    print(f"Config: {config_path}")
    print()
    print("Coverage snapshots (mean ± std across 5 seeds):")
    for attr, t_label in [
        ("coverage_pct_120s", "120 s"),
        ("coverage_pct_200s", "200 s"),
        ("coverage_pct_300s", "300 s"),
        ("coverage_pct_350s", "350 s"),
    ]:
        vals = np.array([getattr(r, attr) for r in results])
        print(f"  {t_label} : {vals.mean():.2f} ± {vals.std():.2f} %")
    print()
    print("Threshold crossing times (mean ± std):")
    for x in (50, 60, 70, 80, 85, 90):
        mean_T, std_T, n_reached, _ = threshold_stats[x]
        if math.isnan(mean_T):
            print(f"  T{x} : NaN ± NaN s  (reached {n_reached}/5 runs)")
        else:
            print(f"  T{x} : {mean_T:.1f} ± {std_T:.1f} s  (reached {n_reached}/5 runs)")
    print()
    print("Original DEBS paper reference:")
    print("  10-UAV sparse DEBS mission completion time: 112.3 ± 10.6 s")
    print("  (NOTE: not directly comparable — 3D vs 2D, different arena and stopping criterion)")
    print()
    print("Output files:")
    for fp in output_files:
        print(f"  {fp}")
    print("=" * 56)
```

---

#### `main() -> None`

Orchestrates the full benchmark run. Guarded by `if __name__ == "__main__":`.

```python
def main() -> None:
    CONFIG_PATH = Path("configs/experiments/paper_matched_debs_10uav_static.yaml")
    OUTPUT_DIR = Path("experiments/results/paper_matched_debs_10uav_static")
    SEEDS = [42, 43, 44, 45, 46]
    SAMPLE_INTERVAL = 10.0
    THRESHOLDS = [50, 60, 70, 80, 85, 90]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_config(CONFIG_PATH)

    results: list[BenchmarkRunResult] = []
    for i, seed in enumerate(SEEDS, start=1):
        print(f"[{i}/5] seed={seed} running...")
        result = run_single_seed(base_config, seed)
        # Per-seed progress line
        t50_str = f"{result.T50:.1f}" if not math.isnan(result.T50) else "NaN"
        t70_str = f"{result.T70:.1f}" if not math.isnan(result.T70) else "NaN"
        t85_str = f"{result.T85:.1f}" if not math.isnan(result.T85) else "NaN"
        print(
            f"[{i}/5] seed={seed} done "
            f"coverage={result.final_coverage * 100:.1f}% "
            f"T50={t50_str} T70={t70_str} T85={t85_str}"
        )
        results.append(result)

    # Compute aggregate threshold statistics
    threshold_stats = {}
    for x in THRESHOLDS:
        attr = f"T{x}"
        values = [getattr(r, attr) for r in results]
        reached = [v for v in values if not math.isnan(v)]
        n_reached = len(reached)
        n_not_reached = len(values) - n_reached
        if reached:
            arr = np.array(reached)
            threshold_stats[x] = (float(arr.mean()), float(arr.std()), n_reached, n_not_reached)
        else:
            threshold_stats[x] = (float('nan'), float('nan'), 0, n_not_reached)

    # Write all outputs
    output_files: list[Path] = []
    for r in results:
        output_files.append(write_timeseries_csv(r, OUTPUT_DIR))
        output_files.append(write_seed_summary_json(r, OUTPUT_DIR))
    output_files.append(write_sampled_timeseries(results, base_config.dt, SAMPLE_INTERVAL, OUTPUT_DIR))
    output_files.append(write_threshold_csv(results, OUTPUT_DIR))
    output_files.append(write_aggregate_csv(results, OUTPUT_DIR))
    output_files.append(plot_coverage_curve(results, threshold_stats, OUTPUT_DIR))
    output_files.append(plot_threshold_bar(threshold_stats, OUTPUT_DIR))
    output_files.append(write_report_txt(results, threshold_stats, OUTPUT_DIR))

    print_terminal_summary(results, threshold_stats, output_files, CONFIG_PATH)


if __name__ == "__main__":
    main()
```

---

## Data Models

### Output File Summary

| File | Columns / Content | Notes |
|---|---|---|
| `seed_{S}_timeseries.csv` | `time_s, coverage_pct` | 3500 data rows; full resolution |
| `seed_{S}_summary.json` | 17 scalar fields | `null` for non-reached T_X |
| `coverage_vs_time.csv` | `time_s, mean_coverage_pct, std_coverage_pct` | 36 rows (0–350 at 10s intervals) |
| `threshold_times.csv` | `threshold_pct, mean_T, std_T, n_reached, n_not_reached` | 6 rows; `nan` if 0 reached |
| `paper_matched_summary.csv` | All per-seed scalars | 5 rows; empty string for non-reached T_X |
| `coverage_vs_time_10uav.png` | Publication-ready coverage curve | (7, 4.2) @ 150 dpi |
| `threshold_comparison.png` | Bar chart T50–T90 | (7, 4.2) @ 150 dpi |
| `paper_matched_report.txt` | Human-readable report | Sections detailed below |

### Seed-to-Config Mapping

| Seed | obstacle_seed | UAV spawn seed | BSA RNG seed | IDE seed |
|---|---|---|---|---|
| 42 | 42 | 42 | 42 | 42 (fixed) |
| 43 | 43 | 43 | 43 | 42 (fixed) |
| 44 | 44 | 44 | 44 | 42 (fixed) |
| 45 | 45 | 45 | 45 | 42 (fixed) |
| 46 | 46 | 46 | 46 | 42 (fixed) |

The IDE allocator's internal RNG is seeded from `ide.seed=42` and does not vary across benchmark seeds. This is intentional: the requirements fix `ide.seed=42` in the YAML and specify it is NOT overridden per seed.

### Metric Computation Summary

| Metric | Formula | Source |
|---|---|---|
| `coverage_pct` | `explored_fraction × 100` | `engine.world.map.explored_fraction()` |
| `final_coverage` | `timeseries[-1][1] / 100` | last timeseries entry (fraction) |
| `path_distance_total` | `Σ ‖pos[i] - pos[i-1]‖₂` | `engine.agent_histories` |
| `trajectory_efficiency` | `(final_coverage × W × H) / path_distance_total` | derived |
| `coverage_efficiency` | same as `trajectory_efficiency` | alias |
| `replanning_count` | `engine.aggregation.step_reassignment_count` | cumulative total at end |
| `T_X` | first `time_s` where `coverage_pct ≥ X` | scanned per step |
| `coverage_pct_{t}s` | `timeseries[round(t/dt)][1]` | index into timeseries |
| `exploration_rate_{t}s` | `coverage_pct_{t}s / t` | derived (% per second) |
| `mean_coverage_pct` (sampled) | `mean([r.timeseries[idx][1] for r in results])` | across 5 seeds |
| `std_coverage_pct` (sampled) | `std([r.timeseries[idx][1] for r in results])` | across 5 seeds |

---

## Report Structure (`paper_matched_report.txt`)

The report is structured as plain text with clearly delimited sections. All section headers use `===` underline style for readability without markdown.

```
PAPER-MATCHED DEBS BENCHMARK REPORT
=====================================
Generated: <ISO timestamp>
Config: configs/experiments/paper_matched_debs_10uav_static.yaml
Seeds: [42, 43, 44, 45, 46]

1. EXPERIMENT SUMMARY
=====================
Environment  : Static sparse (100×100 m, 20 static obstacles)
Fleet        : 10 UAVs, max_speed=1.5 m/s, sensing_range=4.5 m
Duration     : 350 s  (dt=0.1 s, 3500 steps per seed)
Algorithms   : BSA (SelfAggregationController) + IDE (IDEAllocator)

2. RESULTS OVERVIEW
===================
Coverage snapshots (mean ± std, 5 seeds):
  t=120 s : XX.XX ± X.XX %
  t=200 s : XX.XX ± X.XX %
  t=300 s : XX.XX ± X.XX %
  t=350 s : XX.XX ± X.XX %  [final coverage]

Threshold crossing times (mean ± std):
  T50 : XXX.X ± XX.X s  (reached N/5 runs)
  T60 : XXX.X ± XX.X s  (reached N/5 runs)
  T70 : XXX.X ± XX.X s  (reached N/5 runs)
  T80 : XXX.X ± XX.X s  (reached N/5 runs)
  T85 : XXX.X ± XX.X s  (reached N/5 runs)
  T90 : XXX.X ± XX.X s  (reached N/5 runs)

3. COMPARISON TABLE
===================
Metric                         | Original DEBS Paper   | Our Corrected DEBS        | Difference | Comparable?
-------------------------------|----------------------|---------------------------|------------|--------------------
Mission completion time (10 UAV)| 112.3 ± 10.6 s       | N/A (different criterion) | N/A        | NOT_COMPARABLE
T50 (50% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
T60 (60% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
T70 (70% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
T80 (80% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
T85 (85% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
T90 (90% coverage threshold)   | Not reported          | XXX.X ± XX.X s            | N/A        | NOT_COMPARABLE
Final coverage %               | Not reported          | XX.XX ± X.XX %            | N/A        | NOT_COMPARABLE
Mean UAV velocity              | Not reported          | X.XX ± X.XX m/s           | N/A        | APPROXIMATELY_COMPARABLE
Fleet size                     | 10 UAVs               | 10 UAVs                   | 0          | DIRECTLY_COMPARABLE
UAV max_speed parameter        | 1.5 m/s               | 1.5 m/s                   | 0          | DIRECTLY_COMPARABLE
General exploration behaviour  | BSA + IDE (DEBS §4)   | BSA + IDE (DEBS §4)       | N/A        | APPROXIMATELY_COMPARABLE

NOTE: The paper's mission completion time uses a different stopping criterion (full 3D
exploration of a 50×50×2 m sparse forest) from this benchmark's fixed-duration 2D
approach. These numbers measure different things and cannot be numerically compared.

4. OUR METRICS (NO PAPER EQUIVALENT)
======================================
These metrics are reported for informational purposes only. There are no corresponding
paper values and no comparison is made or implied.

Exploration rates (coverage % per second, mean ± std):
  t=120 s : X.XXX ± X.XXX %/s
  t=200 s : X.XXX ± X.XXX %/s
  t=300 s : X.XXX ± X.XXX %/s
  t=350 s : X.XXX ± X.XXX %/s

Coverage efficiency (m² explored / m travelled):
  Mean : X.XX ± X.XX m²/m

Replanning count (cumulative BSA reassignments, mean ± std):
  Mean : XXX ± XX

5. LIMITATIONS
==============
The following differences between this benchmark and the original DEBS paper prevent
direct numerical comparison of most metrics:

  (a) Dimensionality: This simulation is 2D; the paper uses a 3D UAV model.
  (b) Arena: This benchmark uses a 100×100 m 2D arena. The paper reports results on
      a 50×50×2 m sparse forest environment. These are not the same environment.
  (c) Resolution: Coverage in this benchmark is measured over traversable 2D grid
      cells at resolution sensing_range/3. The paper uses 3D voxels at 0.15 m.
  (d) Trajectory generation: This simulation uses direct heading control. The paper
      uses A* path planning with B-spline trajectory smoothing.
  (e) Depth camera: This simulation uses a circular sensing disc. The paper uses a
      depth-camera field-of-view model.
  (f) ROS integration: The paper runs in ROS; this simulation is standalone Python.
  (g) Stopping criterion: The paper's "mission completion time" is the elapsed time
      until full 3D exploration is achieved; this benchmark runs for a fixed 350 s.

6. APPROXIMATION NOTES
=======================
  - Our 100×100 m 2D arena with 20 static obstacles is the closest available
    approximation to the paper's 50×50×2 m sparse forest. It is NOT claimed to be
    identical or equivalent.
  - Coverage fraction in this benchmark is measured over traversable 2D grid cells at
    resolution sensing_range/3 (≈1.5 m), not over 3D voxels at 0.15 m resolution.
  - The IDE config seed is fixed at seed=42 for all 5 runs. What varies across seeds:
      * obstacle placement (environment.obstacle_seed = S)
      * UAV ring-spawn positions (spawn_uavs seed = S)
      * BSA RNG (SelfAggregationController rng = np.random.default_rng(S))
    What is fixed across seeds:
      * IDE internal RNG (IDEAllocator seeded from ide.seed=42 in the config)

7. VALID CONCLUSIONS
====================
The following conclusions are supported by this benchmark:

  (a) Consistency validation: Our 2D DEBS implementation produces coverage behaviour
      consistent with the published DEBS algorithm description — BSA frontier selection
      and IDE fair-share allocation are active and functional.

  (b) Quantified performance: Our 2D corrected DEBS implementation achieves
      approximately XX.X% coverage at t=350 s under static conditions (mean, 5 seeds).

  (c) Controlled benchmark established: A reproducible 5-seed benchmark exists at
      experiments/results/paper_matched_debs_10uav_static/ for future comparison
      against other algorithm variants or implementations.

  This report does NOT claim that our implementation outperforms, equals, or is
  inferior to the original DEBS paper. Such a claim is not supported by the available
  data.
```

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Config Flags Invariant

*For any* `SimulationConfig` loaded from the Benchmark_Config YAML, `config.dynamic_environment` SHALL be `None` or have `enabled = False`, and `config.search` SHALL be `None`.

**Validates: Requirements 1.1, 3.1, 3.2, 3.3**

### Property 2: Step Count Invariant

*For any* valid `(duration, dt)` pair where `duration` and `dt` are positive and `duration` is an integer multiple of `dt`, the number of simulation steps produced by the step loop SHALL equal `round(duration / dt)`.

**Validates: Requirements 6.3, 8.4**

### Property 3: Seed Determinism

*For any* seed `S` from `{42, 43, 44, 45, 46}`, running `run_single_seed(base_config, S)` twice on the same machine SHALL produce identical `timeseries` lists element-by-element.

**Validates: Requirements 7.1, 7.7**

### Property 4: Timeseries Shape and Coverage Monotonicity

*For any* `BenchmarkRunResult` produced by `run_single_seed`, the `timeseries` list SHALL have exactly `round(config.duration / config.dt)` entries (3500 for the benchmark config), all `coverage_pct` values SHALL be in `[0.0, 100.0]`, and coverage SHALL be non-decreasing (because explored cells are never un-marked).

**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

### Property 5: Sampled Timeseries Row Count

*For any* full timeseries and sample interval of 10 s over a duration of 350 s, `write_sampled_timeseries` SHALL produce exactly 36 rows (time values `0, 10, 20, ..., 350`), each with a valid `mean_coverage_pct` in `[0.0, 100.0]`.

**Validates: Requirements 8.5**

### Property 6: Milestone Index Correctness and Exploration Rate Formula

*For any* timeseries and milestone time `t_m` in `{120, 200, 300, 350}`, the value returned by `compute_milestones(timeseries, dt)[t_m]` SHALL equal `timeseries[round(t_m / dt)][1]` exactly, and the corresponding exploration rate SHALL equal `compute_milestones(timeseries, dt)[t_m] / t_m` (% per second).

**Validates: Requirements 9.1, 9.2, 15.1**

### Property 7: Threshold Crossing Soundness

*For any* timeseries and threshold `X` in `{50, 60, 70, 80, 85, 90}`, if `compute_threshold_times` returns a finite value `T_X`, then `timeseries[round(T_X / dt)][1] >= X` AND for all steps before `T_X`, `coverage_pct < X`. If it returns `NaN`, no step in the timeseries has `coverage_pct >= X`.

**Validates: Requirements 10.1, 10.2, 10.3**

### Property 8: Trajectory Efficiency Invariant

*For any* `BenchmarkRunResult` where `path_distance_total > 0`, the following SHALL hold to within floating-point tolerance: `trajectory_efficiency × path_distance_total ≈ final_coverage × world_width × world_height`.

**Validates: Requirements 11.3**

### Property 9: Summary JSON Completeness and Null Encoding

*For any* `BenchmarkRunResult`, `write_seed_summary_json` SHALL produce a valid JSON file that contains exactly the 17 required keys, and any `T_X` field stored as `float('nan')` in the dataclass SHALL be serialised as JSON `null` (not `0`, not `"NaN"`, not the empty string).

**Validates: Requirements 12.1, 12.2, 12.3**

---

## Error Handling

### Config Loading Errors

`load_config` raises `KeyError` for missing required fields and `ValueError` for invalid field types. The runner does not catch these — a misconfigured YAML should fail loudly at startup before any simulation steps are run.

### Output Directory Creation

`OUTPUT_DIR.mkdir(parents=True, exist_ok=True)` is called at the start of `main()` before any simulation runs. If the directory cannot be created (e.g. permission error), a clear `OSError` is raised before any computation is performed.

### Threshold Non-Crossing

The design explicitly uses `float('nan')` (not `None`, `0`, or `""`) throughout the runtime data structures to represent thresholds not crossed. Conversion to other representations (`null` in JSON, `""` in CSV) occurs only at the output-writing boundary.

### Step Loop Termination

The step loop uses `while engine.time_s < duration` (matching `run_multiseed.py`). Due to floating-point accumulation `engine.time_s` may fall marginally short of `350.0` on the last step; the final `timeseries` entry is taken from the actual last step, which is functionally equivalent to 350 s.

### Matplotlib Backend

`matplotlib.use("Agg")` is called before any `import matplotlib.pyplot` in both plotting functions. This is safe to call multiple times and prevents display errors in headless environments.

---

## Testing Strategy

### Unit Tests

Unit tests should verify specific examples and edge cases, not simulate a full 3500-step run. Recommended structure:

- **Config correctness**: Load `paper_matched_debs_10uav_static.yaml`, assert all parameter values match requirements.
- **Milestone indexing**: Call `compute_milestones` with a synthetic timeseries of known length and verify exact index-based retrieval.
- **Threshold computation**: Call `compute_threshold_times` with a synthetic timeseries where threshold crossings are at known positions; assert correct return values including `float('nan')` for never-crossed thresholds.
- **Trajectory efficiency formula**: Construct a `BenchmarkRunResult` with known values and assert `trajectory_efficiency * path_distance_total ≈ final_coverage * world_area`.
- **JSON serialisation**: Call `write_seed_summary_json` with a result containing `NaN` thresholds and assert the written JSON has `null` for those fields.
- **Sampled timeseries row count**: Call `write_sampled_timeseries` with 5 synthetic results of length 3500 and assert 36 rows in the output CSV.

### Property-Based Tests

The property-based testing library used is **Hypothesis** (Python). Each test targets a specific correctness property from the design. All property tests run a minimum of 100 iterations.

**Feature tag format**: `# Feature: paper-matched-debs-benchmark, Property N: <property text>`

#### PBT 1 — Config Flags Invariant (`Property 1`)

```python
# Feature: paper-matched-debs-benchmark, Property 1: Config flags invariant
@given(...)   # no input variation needed; load from fixed YAML
def test_config_flags_invariant():
    config = load_config("configs/experiments/paper_matched_debs_10uav_static.yaml")
    assert config.dynamic_environment is None or not config.dynamic_environment.enabled
    assert config.search is None
```

This property is deterministic (one config file), but the "for any config loaded from this YAML" framing means it should remain passing for any future YAML edits that preserve semantics.

#### PBT 2 — Timeseries Shape and Monotonicity (`Property 4`)

```python
# Feature: paper-matched-debs-benchmark, Property 4: Timeseries shape and coverage monotonicity
@given(st.integers(min_value=0, max_value=4))  # seed index into [42..46]
@settings(max_examples=5)
def test_timeseries_shape_and_monotonicity(seed_idx):
    seed = [42, 43, 44, 45, 46][seed_idx]
    config = load_config("configs/experiments/paper_matched_debs_10uav_static.yaml")
    result = run_single_seed(config, seed)
    assert len(result.timeseries) == round(config.duration / config.dt)
    for _, cov in result.timeseries:
        assert 0.0 <= cov <= 100.0
    coverages = [cov for _, cov in result.timeseries]
    for i in range(1, len(coverages)):
        assert coverages[i] >= coverages[i - 1] - 1e-9   # non-decreasing within float tolerance
```

#### PBT 3 — Threshold Crossing Soundness (`Property 7`)

```python
# Feature: paper-matched-debs-benchmark, Property 7: Threshold crossing soundness
@given(
    st.lists(st.floats(min_value=0.0, max_value=100.0), min_size=1, max_size=500),
    st.integers(min_value=1, max_value=99),
)
def test_threshold_crossing_soundness(coverage_values, threshold):
    import math
    dt = 0.1
    timeseries = [(round(i * dt, 4), v) for i, v in enumerate(coverage_values)]
    result = compute_threshold_times(timeseries, [threshold])
    T = result[threshold]
    if math.isnan(T):
        # Verify no step crossed the threshold
        assert all(v < threshold for _, v in timeseries)
    else:
        # Verify the crossing step has coverage >= threshold
        crossing_idx = round(T / dt)
        assert timeseries[crossing_idx][1] >= threshold
        # Verify all prior steps were below threshold
        assert all(timeseries[j][1] < threshold for j in range(crossing_idx))
```

#### PBT 4 — Trajectory Efficiency Invariant (`Property 8`)

```python
# Feature: paper-matched-debs-benchmark, Property 8: Trajectory efficiency invariant
@given(
    st.floats(min_value=0.0, max_value=1.0),   # final_coverage fraction
    st.floats(min_value=1.0, max_value=1e6),   # path_distance_total
    st.floats(min_value=10.0, max_value=1000.0),  # world_width
    st.floats(min_value=10.0, max_value=1000.0),  # world_height
)
def test_trajectory_efficiency_invariant(final_coverage, path_distance, width, height):
    efficiency = (final_coverage * width * height) / path_distance
    assert abs(efficiency * path_distance - final_coverage * width * height) < 1e-6
```

#### PBT 5 — Milestone Index Correctness (`Property 6`)

```python
# Feature: paper-matched-debs-benchmark, Property 6: Milestone index correctness
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=100.0),
        min_size=3501, max_size=3501,
    )
)
def test_milestone_index_correctness(coverage_values):
    dt = 0.1
    timeseries = [(round(i * dt, 4), v) for i, v in enumerate(coverage_values)]
    milestones = compute_milestones(timeseries, dt)
    for t_m in (120, 200, 300, 350):
        idx = round(t_m / dt)
        assert milestones[t_m] == timeseries[idx][1]
```

#### PBT 6 — Summary JSON Null Encoding (`Property 9`)

```python
# Feature: paper-matched-debs-benchmark, Property 9: Summary JSON completeness and null encoding
@given(
    st.floats(allow_nan=True),   # T50 — may be nan
    st.floats(allow_nan=True),   # T85 — may be nan
)
def test_summary_json_null_encoding(t50, t85, tmp_path):
    import math, json
    # Build a minimal BenchmarkRunResult with these threshold values
    result = _make_minimal_result(T50=t50, T85=t85)
    write_seed_summary_json(result, tmp_path)
    payload = json.loads((tmp_path / f"seed_{result.seed}_summary.json").read_text())
    for key in ("seed", "final_coverage", "path_distance_total", "trajectory_efficiency",
                "mean_speed", "mission_overlap_final", "replanning_count",
                "coverage_pct_120s", "coverage_pct_200s", "coverage_pct_300s",
                "coverage_pct_350s", "T50", "T60", "T70", "T80", "T85", "T90"):
        assert key in payload
    if math.isnan(t50):
        assert payload["T50"] is None
    if math.isnan(t85):
        assert payload["T85"] is None
```

### Integration / Smoke Tests

- **Dry-run integration**: Call `build_engine(config)` for each of the 5 seeds and assert the engine has 10 agents and `mission_orchestrator is None`.
- **Output file set**: After a full run, assert all 13 expected output files exist in `OUTPUT_DIR`.
- **Report content**: Parse `paper_matched_report.txt`, assert it contains `"112.3 ± 10.6"`, `"Not reported"`, `"NOT_COMPARABLE"`, `"DIRECTLY_COMPARABLE"`, `"APPROXIMATELY_COMPARABLE"`, and `"Approximation Notes"`.
- **No existing file modification**: Hash-check `configs/simulation.yaml` and the `src/` directory before and after a full benchmark run.

---
