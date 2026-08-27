# Requirements Document

## Introduction

This document specifies the requirements for the **Paper-Matched DEBS Benchmark** experiment.
The benchmark establishes a controlled, reproducible measurement of our corrected 2D DEBS
(Decentralised Exploration by a Biologically-inspired Swarm) implementation against the results
reported in the original DEBS paper, under the closest available equivalent conditions.

The experiment is **purely additive**: it introduces one new YAML configuration file and one new
runner script. It does not modify any existing source file, test, or configuration. All existing
tests must continue to pass unchanged.

The scope is **exploration phase only** — 10 UAVs, static obstacles, BSA + IDE enabled,
no target search, no dynamic obstacles, fixed 350 s duration, 5 deterministic seeds.

---

## Glossary

- **BSA**: Bio-inspired Self-Aggregation controller (`SelfAggregationController`).
  The primary frontier-selection and aggregation algorithm from DEBS Paper 1.
- **IDE**: Iterative Differential Evolution allocator (`IDEAllocator`).
  The fair-share region-allocation algorithm from DEBS Paper 1 §4.
- **SimulationEngine**: The discrete-time simulator at `src/simulation/simulation_engine.py`.
  Steps BSA, IDE, kinematics, collision resolution, map update, and metric collection.
- **SimulationConfig**: The top-level dataclass at `src/config/loader.py` that holds all
  configuration. `search=None` disables `MissionOrchestrator`; `dynamic_environment=None`
  (or `enabled: false` / `scenario: static`) disables `ObstacleManager`.
- **World**: The environment container at `src/environment/world.py`. Built via
  `World.from_config(env_config, uav_config, dynamic_config)`. When `dynamic_config=None`
  or `dynamic_config.enabled=False`, no `ObstacleManager` is created.
- **ExplorationMap**: Grid-based coverage map owned by `World`. Provides
  `explored_fraction()` — the fraction of traversable cells marked explored.
- **obstacle_seed**: The per-run integer that seeds obstacle placement, UAV ring spawn, and
  the `SelfAggregationController` RNG. Derived at runtime via `dataclasses.replace` on the
  loaded config; the YAML file is never written during a run.
- **T_X**: The threshold crossing time — the earliest simulation time `t` at which
  `explored_fraction(t) >= X/100`. Checked at every simulation step (dt = 0.1 s).
- **coverage_pct**: `explored_fraction * 100`, expressed as a percentage.
- **path_distance_total**: Sum of Euclidean step distances across all UAV agent histories
  for a single run (fleet total, metres).
- **trajectory_efficiency**: `explored_area_m2 / path_distance_total`. Numerator is
  `final_coverage * world_width * world_height`.
- **Benchmark_Runner**: The new script `experiments/scripts/run_paper_matched_benchmark.py`.
- **Benchmark_Config**: The new YAML `configs/experiments/paper_matched_debs_10uav_static.yaml`.
- **Output_Dir**: `experiments/results/paper_matched_debs_10uav_static/`.
- **DIRECTLY_COMPARABLE**: Same metric definition, same environment type, same evaluation
  protocol. A numerical difference is meaningful.
- **APPROXIMATELY_COMPARABLE**: Similar metric definition but environment or protocol differs.
  A numerical difference requires contextual interpretation.
- **NOT_COMPARABLE**: Fundamentally different environments, dimensionality, or stopping
  criteria. No valid numerical comparison can be made.

---

## Requirements

---

### Requirement 1: Experiment Purpose and Scope

**User Story:** As a researcher, I want a controlled paper-matched benchmark experiment, so
that I can quantify how our corrected 2D DEBS implementation performs under conditions as
close as possible to the original DEBS paper's 10-UAV sparse-forest scenario.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL measure exploration performance of the corrected 2D DEBS
   implementation (BSA + IDE) under static-obstacle, exploration-only conditions, where
   "static-obstacle" means `dynamic_environment.enabled: false` in the loaded config and
   "exploration-only" means `search.enabled: false` (or absent) in the loaded config.
2. THE Benchmark_Runner SHALL be a new, isolated script located under `experiments/scripts/`
   that does not modify any existing source file, existing experiment script, existing
   configuration file, or existing test.
3. THE Benchmark_Runner SHALL leave all files in `src/`, `tests/`, and
   `configs/experiments/` (except the new Benchmark_Config, which is a new file that does
   not collide with any existing filename in that directory) unchanged.
4. WHEN `pytest tests/` is executed before and after adding the two new files, THE test
   suite SHALL produce no newly-failing tests and no change in the count of passing tests.
5. THE Benchmark_Runner SHALL produce only additive output: new files written under
   `Output_Dir` (which is a subdirectory of `experiments/results/`) and the new
   Benchmark_Config.

---

### Requirement 2: Algorithm Configuration

**User Story:** As a researcher, I want the benchmark to use BSA and IDE in their unmodified
corrected forms, so that the results faithfully reflect the published algorithm behaviour.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL instantiate `SelfAggregationController` with the aggregation
   parameters from Benchmark_Config without modification.
2. THE Benchmark_Runner SHALL enable IDE allocation by loading a Benchmark_Config that
   includes an `ide:` block, causing `SimulationEngine` to build its internal `IDEAllocator`
   from `config.ide`.
3. THE Benchmark_Runner SHALL NOT pass `ide=None` to `SimulationConfig`; IDE must remain
   active for all 5 seeds.
4. THE Benchmark_Config SHALL include the following IDE parameters, identical to
   `configs/simulation.yaml`: `alpha=0.5`, `population_size=20`, `fe_max=200`, `t_att=2.0`,
   `d_star=30.0`, `communication_range=50.0`, `bounds_padding=0.5`, `seed=42`.
5. THE Benchmark_Runner SHALL NOT alter BSA equations, IDE equations, repulsion sign,
   interaction radius, or hysteresis/commitment logic in any source module.
6. THE Benchmark_Runner SHALL NOT introduce new control behaviour or new algorithm
   parameters beyond those already present in the loaded Benchmark_Config.

---

### Requirement 3: Static-Only Environment

**User Story:** As a researcher, I want the benchmark to use a static obstacle environment,
so that the results isolate exploration behaviour from dynamic-obstacle effects.

#### Acceptance Criteria

1. THE Benchmark_Config SHALL NOT include an `enabled: true` dynamic environment block that
   would cause `World.from_config` to create an `ObstacleManager`.
2. WHEN the Benchmark_Config is loaded by `load_config`, THE resulting `SimulationConfig`
   SHALL have `dynamic_environment` equal to `None` OR have
   `dynamic_environment.enabled = False` OR have `dynamic_environment.scenario = "static"`,
   ensuring no `ObstacleManager` is created.
3. THE Benchmark_Config SHALL NOT include a `search:` block with `enabled: true`;
   `_load_search_config` SHALL return `None` for the loaded config, ensuring no
   `MissionOrchestrator` is ever attached to `SimulationEngine`.
4. WHILE a benchmark run is executing, THE SimulationEngine SHALL have
   `mission_orchestrator = None` for all 5 seeds.
5. THE Benchmark_Runner SHALL NOT invoke any target-search, target-tracking,
   target-prioritisation, target-assignment, lawnmower-sweep, or detection logic.
6. THE Benchmark_Config SHALL specify `environment.obstacle_count = 20`, `width = 100.0`,
   `height = 100.0`, `obstacle_min_radius = 1.5`, `obstacle_max_radius = 4.0` — the same
   static obstacle field used in `configs/simulation.yaml`.

---

### Requirement 4: Fleet and UAV Parameters

**User Story:** As a researcher, I want the UAV parameters to match the DEBS paper values,
so that the kinematics layer is directly comparable.

#### Acceptance Criteria

1. THE Benchmark_Config SHALL specify `simulation.num_uavs = 10`.
2. THE Benchmark_Config SHALL specify `uav.max_speed = 1.5` m/s.
3. THE Benchmark_Config SHALL specify `uav.max_angular_velocity = 0.9` rad/s.
4. THE Benchmark_Config SHALL specify `uav.sensing_range = 4.5` m.
5. THE Benchmark_Config SHALL specify `uav.initial_spread_radius = 20.0` m,
   `uav.spawn_mode = ring`, and `uav.spawn_angular_noise = 0.15`.
6. THE Benchmark_Config SHALL specify `simulation.spawn_center_x = 50.0` and
   `simulation.spawn_center_y = 50.0`.

---

### Requirement 5: BSA Aggregation Parameters

**User Story:** As a researcher, I want the BSA parameters to match the corrected baseline
from `configs/simulation.yaml`, so that the benchmark reflects the state of the corrected
implementation.

#### Acceptance Criteria

1. THE Benchmark_Config SHALL specify the following aggregation parameters, identical to
   `configs/simulation.yaml`: `d_c=0.5`, `d_0=17.8`, `k_a=1.0`, `turn_cost_weight=1.0`,
   `trail_penalty=8.0`, `candidates_per_frontier=6`, `replan_interval=2.0`,
   `mission_region_radius=17.8`, `cluster_penalty_weight=2.0`, `trail_penalty_weight=1.0`,
   `turn_penalty_weight=0.5`.

---

### Requirement 6: Simulation Duration and Timestep

**User Story:** As a researcher, I want the simulation to run for 350 s with the existing
dt, so that I can measure coverage behaviour well beyond the paper's 112.3 s reference point.

#### Acceptance Criteria

1. THE Benchmark_Config SHALL specify `simulation.duration = 350.0` s.
2. THE Benchmark_Config SHALL specify `simulation.dt = 0.1` s, preserving the existing
   timestep used by all other experiments.
3. WHEN a benchmark run executes, THE SimulationEngine SHALL step until `engine.time_s`
   reaches `350.0` s (i.e., 3500 simulation steps per seed).
4. THE Benchmark_Runner SHALL NOT override the duration with a shorter value at runtime;
   the full 350 s MUST be simulated for every seed.

---

### Requirement 7: Seed Reproducibility

**User Story:** As a researcher, I want exactly 5 deterministic seeds, so that results are
reproducible across machines and re-runs.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL execute exactly 5 independent runs using seeds
   `[42, 43, 44, 45, 46]`.
2. FOR EACH seed `S` in `[42, 43, 44, 45, 46]`, THE Benchmark_Runner SHALL derive the
   per-run config via `dataclasses.replace(base_config, environment=dataclasses.replace(base_config.environment, obstacle_seed=S))`,
   matching the convention in `run_multiseed.py`.
3. FOR EACH seed `S`, THE Benchmark_Runner SHALL pass `obstacle_seed=S` to `spawn_uavs`
   (as the `seed` argument), so that UAV ring-spawn positions are seeded from `S`.
4. FOR EACH seed `S`, THE Benchmark_Runner SHALL construct `SelfAggregationController` with
   `rng=np.random.default_rng(S)`, so that the BSA RNG is seeded from `S`.
5. THE IDE config `seed` field SHALL remain fixed at `42` (from the `ide:` block in
   Benchmark_Config) for all 5 seeds; it is NOT overridden per seed.
6. THE Benchmark_Runner SHALL print each seed value before its run begins, in the format:
   `[N/5] seed=S running...` where N is the 1-based run index.
7. WHEN the same seed is supplied on two separate executions of THE Benchmark_Runner on the
   same machine, THE simulation SHALL produce identical output CSV values.

---

### Requirement 8: Coverage Timeseries Recording

**User Story:** As a researcher, I want full per-step coverage data saved per seed, so that
I can perform post-hoc analysis at any time resolution.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL record `(time_s, coverage_pct)` at every simulation step
   (every `dt = 0.1` s) for each seed.
2. THE coverage_pct value at each step SHALL be `engine.world.map.explored_fraction() * 100`.
3. THE Benchmark_Runner SHALL write one file `seed_{S}_timeseries.csv` per seed under
   `Output_Dir`, with columns `[time_s, coverage_pct]`.
4. THE `seed_{S}_timeseries.csv` file SHALL include a header row and SHALL contain
   one data row per simulation step (3500 rows for a 350 s run at dt=0.1 s).
5. THE Benchmark_Runner SHALL also produce a sampled timeseries `coverage_vs_time.csv`
   under `Output_Dir` with columns `[time_s, mean_coverage_pct, std_coverage_pct]`,
   at 10 s intervals (t = 0, 10, 20, ..., 350), computed across all 5 seeds.

---

### Requirement 9: Milestone Coverage Values

**User Story:** As a researcher, I want exact coverage values at specific timestamps, so that
I can compare performance at the paper's reported mission-completion reference time and beyond.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL record `coverage_pct` at exactly `t = 120 s`, `t = 200 s`,
   `t = 300 s`, and `t = 350 s` for each seed.
2. THE milestone values SHALL be read from the per-step timeseries at the step whose
   `time_s` is closest to the milestone time (using the step at `round(t / dt) * dt`).
3. THE terminal summary output SHALL include mean ± std across 5 seeds for each of the
   four milestone times, in the format: `  120 s : XX.XX ± X.XX %`.
4. THE `paper_matched_summary.csv` SHALL include columns for each milestone coverage value
   (`coverage_pct_120s`, `coverage_pct_200s`, `coverage_pct_300s`, `coverage_pct_350s`).

---

### Requirement 10: Threshold Crossing Time Metrics (T50–T90)

**User Story:** As a researcher, I want the time at which specific coverage thresholds are
first crossed, so that I have comparable milestone metrics for the benchmark.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL compute `T_X` for X in `{50, 60, 70, 80, 85, 90}` for each
   seed, defined as the earliest simulation time `t` (in seconds) at which
   `explored_fraction(t) >= X / 100`.
2. THE threshold check SHALL be performed at every simulation step (every `dt = 0.1` s);
   no interpolation between steps is required or permitted.
3. IF `explored_fraction` does not reach `X / 100` by `t = 350 s`, THEN THE Benchmark_Runner
   SHALL record `NaN` (not zero, not the empty string) for that seed's `T_X`.
4. THE statistics over 5 seeds for each threshold SHALL be computed as follows:
   mean and std are computed only over seeds that reached the threshold; seeds with
   `NaN` are excluded from the mean/std calculation but are counted separately.
5. THE Benchmark_Runner SHALL record and report `n_reached` (number of seeds that reached
   the threshold) and `n_not_reached` (number of seeds that did not) for each threshold.
6. THE Benchmark_Runner SHALL NOT silently drop, zero-fill, or substitute threshold
   non-crossings; `n_not_reached > 0` is a valid and expected outcome that must be
   preserved in output files and terminal summary.
7. THE Benchmark_Runner SHALL write `threshold_times.csv` under `Output_Dir` with columns
   `[threshold_pct, mean_T, std_T, n_reached, n_not_reached]`, one row per threshold.
8. THE terminal summary SHALL print all six threshold rows regardless of whether every
   seed reached the threshold, using `NaN` for mean/std when `n_reached = 0`.

---

### Requirement 11: Additional Scalar Metrics

**User Story:** As a researcher, I want a standard set of scalar metrics per run, so that
the benchmark integrates consistently with the existing experiment pipeline.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL compute `final_coverage` as `explored_fraction` at `t = 350 s`
   (the last step), expressed as a fraction (not percentage) in internal storage.
2. THE Benchmark_Runner SHALL compute `path_distance_total` as the sum of Euclidean
   step-distances over `engine.agent_histories` for all agents in a single run, matching
   the calculation in `run_multiseed.py`.
3. THE Benchmark_Runner SHALL compute `trajectory_efficiency` as
   `(final_coverage * world_width * world_height) / path_distance_total`,
   using `engine.world.width` and `engine.world.height`.
4. THE Benchmark_Runner SHALL record `mean_speed` from `SimulationMetrics.mean_speed`
   at the final step.
5. THE Benchmark_Runner SHALL record `mission_overlap_final` from
   `SimulationMetrics.mission_overlap_fraction` at the final step.
6. THE Benchmark_Runner SHALL record `replanning_count` as the cumulative count of
   `engine.aggregation.step_reassignment_count` across all steps for a run.
7. THE Benchmark_Runner SHALL NOT add new instrumentation, new fields, or new methods
   to any file in `src/` to collect these metrics; only the existing `SimulationMetrics`
   fields and `engine.agent_histories` may be used.
8. THE Benchmark_Runner SHALL write `paper_matched_summary.csv` under `Output_Dir`
   with all per-seed scalar metrics in one file, with columns:
   `[seed, final_coverage, path_distance_total, trajectory_efficiency, mean_speed,
   mission_overlap_final, replanning_count, coverage_pct_120s, coverage_pct_200s,
   coverage_pct_300s, coverage_pct_350s, T50, T60, T70, T80, T85, T90]`.

---

### Requirement 12: Per-Seed Summary File

**User Story:** As a researcher, I want a per-seed scalar summary file, so that individual
run results are inspectable without parsing the full timeseries.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL write `seed_{S}_summary.json` (or `seed_{S}_summary.csv`)
   under `Output_Dir` for each seed `S`.
2. THE per-seed file SHALL include: `seed`, `final_coverage`, `path_distance_total`,
   `trajectory_efficiency`, `mean_speed`, `mission_overlap_final`, `replanning_count`,
   `coverage_pct_120s`, `coverage_pct_200s`, `coverage_pct_300s`, `coverage_pct_350s`,
   `T50`, `T60`, `T70`, `T80`, `T85`, `T90`.
3. THE per-seed file SHALL represent non-reached threshold values as `null` (JSON) or as
   an empty string (CSV), not as `0`.

---

### Requirement 13: Coverage-vs-Time Plot

**User Story:** As a researcher, I want a publication-ready coverage curve, so that I can
include the benchmark results in a paper or report.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL produce a coverage-vs-time plot saved as
   `coverage_vs_time_10uav.png` under `Output_Dir`.
2. THE plot x-axis SHALL span 0 to 350 s, labelled "Time (s)".
3. THE plot y-axis SHALL span 0 to 100%, labelled "Coverage (%)".
4. THE plot SHALL include a mean coverage line across all 5 seeds.
5. THE plot SHALL include a shaded band of ±1 std around the mean line, using
   `matplotlib.axes.Axes.fill_between` with the style from `run_multiseed.py`
   (line color `#2a78d6`, band alpha `0.18`).
6. THE plot SHALL include horizontal reference lines at 50%, 70%, 80%, 85%, and 90%
   coverage, rendered as dashed or dotted lines in a muted colour.
7. THE plot SHALL include vertical markers at the mean values of `T70`, `T80`, `T85`,
   and `T90`; markers for thresholds not reached by any seed SHALL be omitted.
8. THE plot SHALL include a text annotation: `"DEBS paper: 112.3 ± 10.6 s mission completion"`
   visible in the figure, clearly labelled as a paper reference and NOT rendered as a curve.
9. THE plot SHALL NOT include any paper coverage curve (paper timeseries data are not
   available and must not be fabricated).
10. THE plot figure size SHALL be `(7, 4.2)` inches at `dpi=150`, matching the style of
    existing plots in `run_multiseed.py`.
11. THE plot SHALL be rendered with `matplotlib.use("Agg")` (non-interactive backend).

---

### Requirement 14: Threshold Bar/Table Plot

**User Story:** As a researcher, I want a visual summary of threshold crossing times, so
that the T50–T90 results are easy to inspect at a glance.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL produce a threshold summary plot saved as
   `threshold_comparison.png` under `Output_Dir`.
2. THE plot SHALL display mean ± std for each of T50, T60, T70, T80, T85, T90 as a
   bar chart or grouped point plot with error bars.
3. IF a threshold was not reached by any seed, THE Benchmark_Runner SHALL omit that
   threshold bar/point from the plot or mark it explicitly as "not reached".
4. THE plot SHALL clearly label each threshold on the x-axis and time in seconds on
   the y-axis.
5. THE plot SHALL be rendered with `matplotlib.use("Agg")` (non-interactive backend).

---

### Requirement 15: Normalised Efficiency Metrics

**User Story:** As a researcher, I want normalised exploration efficiency metrics, so that
I can assess how efficiently the fleet covers space over time.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL compute `exploration_rate` at `t = 120 s`, `200 s`,
   `300 s`, and `350 s`, defined as `coverage_pct(t) / t` (% per second).
2. THE Benchmark_Runner SHALL compute `coverage_efficiency` per run, defined as
   `trajectory_efficiency` (explored area in m² divided by total fleet path in m).
3. THE `paper_matched_summary.csv` SHALL include columns for
   `exploration_rate_120s`, `exploration_rate_200s`, `exploration_rate_300s`,
   `exploration_rate_350s`, and `coverage_efficiency`.
4. THE `paper_matched_report.txt` SHALL label these metrics explicitly as "our metrics
   (no paper equivalent)" to prevent incorrect comparison.

---

### Requirement 16: Paper Data Integrity

**User Story:** As a researcher, I want the benchmark report to preserve the integrity of
the original paper's data, so that no fabricated or estimated paper-side values appear in
any output.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL use exactly one paper-reported number: the 10-UAV sparse
   forest mission completion time `112.3 ± 10.6 s`.
2. THE Benchmark_Runner SHALL NOT fabricate, interpolate, digitize, or estimate any
   other paper-side value (T50, T60, T70, T80, T85, T90, final coverage %, mean velocity).
3. THE `paper_matched_report.txt` and any comparison table SHALL mark T50, T60, T70,
   T80, T85, T90, final coverage %, and mean velocity in the "Original DEBS Paper" column
   as the string `"Not reported"`.
4. THE Benchmark_Runner SHALL NOT include any statement equating the paper's mission
   completion time with any specific threshold metric (T85 or otherwise) without explicit
   written justification in the report text.
5. THE `paper_matched_report.txt` SHALL include an explicit note that the paper's
   mission completion time uses a different stopping criterion (full exploration in 3D)
   from this benchmark's fixed-duration approach.

---

### Requirement 17: Comparability Classification

**User Story:** As a researcher, I want every reported metric clearly labelled with its
comparability level, so that readers do not draw invalid conclusions from numerical differences.

#### Acceptance Criteria

1. THE `paper_matched_report.txt` and comparison table SHALL assign one of three labels
   to each metric: `DIRECTLY_COMPARABLE`, `APPROXIMATELY_COMPARABLE`, or `NOT_COMPARABLE`.
2. THE mission completion time vs any threshold metric (T50–T90) SHALL be labelled
   `NOT_COMPARABLE` (different stopping criteria, 3D vs 2D, different arena size).
3. THE mean UAV speed SHALL be labelled `APPROXIMATELY_COMPARABLE` (same configured
   parameter value, but trajectory generation and physics differ).
4. THE fleet size (10 UAVs) SHALL be labelled `DIRECTLY_COMPARABLE`.
5. THE UAV max_speed parameter (1.5 m/s) SHALL be labelled `DIRECTLY_COMPARABLE`.
6. THE coverage percentage at specific timestamps SHALL be labelled `NOT_COMPARABLE`
   (3D vs 2D, arena size differs).
7. THE general exploration behaviour SHALL be labelled `APPROXIMATELY_COMPARABLE`
   (same algorithm family, different physics layer).

---

### Requirement 18: Comparison Table Output

**User Story:** As a researcher, I want a structured comparison table in the report, so
that all results and their comparability status are presented in a single, scannable view.

#### Acceptance Criteria

1. THE `paper_matched_report.txt` SHALL include a comparison table with columns:
   `Metric | Original DEBS Paper | Our Corrected DEBS | Difference | Comparable?`.
2. THE table SHALL include rows for: mission completion time (10-UAV sparse),
   T50, T60, T70, T80, T85, T90, final coverage %, and mean UAV velocity.
3. THE "Original DEBS Paper" column SHALL contain `112.3 ± 10.6 s` for mission
   completion time and `Not reported` for all other rows.
4. THE "Our Corrected DEBS" column SHALL contain actual measured mean ± std values
   for all rows.
5. THE "Difference" column SHALL contain a computed difference only where both paper
   and our values are present; all other cells SHALL contain `N/A`.
6. THE "Comparable?" column SHALL use the classification from Requirement 17.

---

### Requirement 19: Scientific Reporting Constraints

**User Story:** As a researcher, I want the benchmark outputs to make only scientifically
defensible claims, so that the report can be included in academic work without overstating
the comparison.

#### Acceptance Criteria

1. THE `paper_matched_report.txt` SHALL NOT include any statement claiming that our
   implementation outperforms, is superior to, or is better than the original DEBS paper.
2. THE `paper_matched_report.txt` SHALL NOT include any comparative superiority claim
   based solely on a numerical difference between our results and the paper value.
3. THE `paper_matched_report.txt` SHALL include a section explicitly acknowledging
   the following limitations: 2D vs 3D dimensionality, 100×100 m vs 50×50×2 m arena,
   absence of voxel resolution, absence of ROS / A* / B-spline trajectory generation,
   absence of a depth camera model, and different stopping criterion.
4. THE valid conclusions that THE `paper_matched_report.txt` MAY state are:
   (a) consistency validation — whether our implementation produces coverage behaviour
   consistent with the published DEBS algorithm description;
   (b) quantified performance — our 2D implementation achieves X% coverage at T seconds
   under static conditions;
   (c) controlled benchmark established for future comparison.
5. WHEN the paper reference value is cited, THE `paper_matched_report.txt` SHALL
   accompany it with the note: "(NOTE: not directly comparable — 3D vs 2D, different
   arena and stopping criterion)".

---

### Requirement 20: Terminal Summary Output

**User Story:** As a researcher, I want a structured terminal summary printed at the end
of the run, so that I can read results immediately without opening any output file.

#### Acceptance Criteria

1. WHEN all 5 seeds have completed, THE Benchmark_Runner SHALL print a terminal summary
   in the following exact format:

   ```
   ========================================================
   PAPER-MATCHED DEBS BENCHMARK — 10 UAV STATIC
   ========================================================
   Seeds: [42, 43, 44, 45, 46]
   Environment: static sparse (100x100m, 20 static obstacles)
   Fleet: 10 UAVs
   Duration: 350 s
   Config: configs/experiments/paper_matched_debs_10uav_static.yaml

   Coverage snapshots (mean ± std across 5 seeds):
     120 s : XX.XX ± X.XX %
     200 s : XX.XX ± X.XX %
     300 s : XX.XX ± X.XX %
     350 s : XX.XX ± X.XX %

   Threshold crossing times (mean ± std):
     T50 : XXX.X ± XX.X s  (reached N/5 runs)
     T60 : XXX.X ± XX.X s  (reached N/5 runs)
     T70 : XXX.X ± XX.X s  (reached N/5 runs)
     T80 : XXX.X ± XX.X s  (reached N/5 runs)
     T85 : XXX.X ± XX.X s  (reached N/5 runs)
     T90 : XXX.X ± XX.X s  (reached N/5 runs)

   Original DEBS paper reference:
     10-UAV sparse DEBS mission completion time: 112.3 ± 10.6 s
     (NOTE: not directly comparable — 3D vs 2D, different arena and stopping criterion)

   Output files:
     [list all generated files with their full paths]
   ========================================================
   ```

2. THE terminal summary SHALL use `NaN` for mean and std values when a threshold was not
   reached by any of the 5 seeds.
3. THE Benchmark_Runner SHALL print per-seed progress in the format
   `[N/5] seed=S running... coverage=XX.X% T50=... T70=... T85=...` after each seed
   completes, to allow monitoring of long-running experiments.

---

### Requirement 21: Output File Set

**User Story:** As a researcher, I want all outputs written to a dedicated directory, so
that the results are isolated from other experiments and easy to archive.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL create the directory
   `experiments/results/paper_matched_debs_10uav_static/` if it does not already exist.
2. THE Benchmark_Runner SHALL write the following files under `Output_Dir`:
   - `seed_42_timeseries.csv`, `seed_43_timeseries.csv`, `seed_44_timeseries.csv`,
     `seed_45_timeseries.csv`, `seed_46_timeseries.csv` — per-seed full timeseries
     with columns `[time_s, coverage_pct]`.
   - `seed_42_summary.json` (or `.csv`), ..., `seed_46_summary.json` — per-seed
     scalar metrics.
   - `coverage_vs_time.csv` — sampled at 10 s intervals, columns
     `[time_s, mean_coverage_pct, std_coverage_pct]`.
   - `threshold_times.csv` — columns
     `[threshold_pct, mean_T, std_T, n_reached, n_not_reached]`.
   - `paper_matched_summary.csv` — all per-seed scalar metrics in one file.
   - `coverage_vs_time_10uav.png` — publication-ready coverage curve.
   - `threshold_comparison.png` — bar/table plot of T50–T90.
   - `paper_matched_report.txt` — human-readable report with results and
     comparison table.
3. THE Benchmark_Runner SHALL NOT overwrite any file in
   `experiments/results/multiseed/`, `experiments/results/baseline_comparison/`,
   `experiments/results/comm_range_ablation/`, or any other pre-existing results
   directory.

---

### Requirement 22: New Files Created

**User Story:** As a developer, I want the complete list of new files to be unambiguous,
so that code review and regression checks are straightforward.

#### Acceptance Criteria

1. THE implementation SHALL create exactly two new non-results files:
   - `configs/experiments/paper_matched_debs_10uav_static.yaml` — the Benchmark_Config.
   - `experiments/scripts/run_paper_matched_benchmark.py` — the Benchmark_Runner.
2. THE Benchmark_Config (`paper_matched_debs_10uav_static.yaml`) SHALL be a valid YAML
   file loadable by `src/config/loader.load_config` without error.
3. THE Benchmark_Config SHALL NOT include a `dynamic_environment: enabled: true` block
   with a non-static scenario.
4. THE Benchmark_Config SHALL NOT include a `search: enabled: true` block.
5. THE Benchmark_Runner SHALL be importable as a Python module without error (i.e., the
   `main()` function is guarded by `if __name__ == "__main__":`).
6. IF `experiments/results/paper_matched_debs_10uav_static/` does not exist, THE
   Benchmark_Runner SHALL create it at runtime before writing any output file.

---

### Requirement 23: Approximation Disclaimer

**User Story:** As a researcher, I want the report to explicitly document the simulation
approximations relative to the paper, so that readers understand the boundary conditions
of any comparison.

#### Acceptance Criteria

1. THE `paper_matched_report.txt` SHALL include an "Approximation Notes" section that
   states: our 100×100 m 2D arena with 20 static obstacles is the closest available
   approximation to the paper's 50×50×2 m sparse forest, and is NOT claimed to be
   identical.
2. THE `paper_matched_report.txt` SHALL state that coverage fraction in this benchmark
   is measured over traversable 2D grid cells at resolution `sensing_range / 3`, not
   over 3D voxels at 0.15 m resolution.
3. THE `paper_matched_report.txt` SHALL state that the IDE config `seed=42` is fixed
   across all 5 runs, and explain which seeds vary (obstacle placement, UAV spawn,
   BSA RNG) and which are fixed (IDE internal RNG via `ide.seed=42`).

---

### Requirement 24: Regression Safety

**User Story:** As a developer, I want the new feature to be fully isolated from existing
code, so that no existing experiment or test is broken.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL import only from `src/` modules, `configs/`, standard
   library, `numpy`, `matplotlib`, and `csv`/`json`/`pathlib` — the same dependency
   set used by existing runners.
2. THE Benchmark_Runner SHALL NOT monkey-patch, subclass, or wrap any class in `src/`.
3. WHEN `pytest tests/` is executed after adding the two new files, THE test results
   SHALL be identical to the results before adding the files.
4. THE Benchmark_Runner SHALL NOT write to `stdout` in a way that interferes with the
   existing test suite (i.e., terminal output is only produced during `__main__` execution).
5. THE Benchmark_Config SHALL NOT shadow, override, or delete any key in
   `configs/simulation.yaml` or any other existing config file.
