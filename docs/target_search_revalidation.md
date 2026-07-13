# Target Search & Tracking — Revalidation Status

## Starting state

Read through the full search/tracking subsystem (`src/search/`, `src/evaluation/search_metrics.py`,
`src/config/search_config.py`). Confirmed genuinely wired and internally
consistent: target representation and lifecycle (`src/search/target.py`),
`TargetManager` spawning/status transitions, `DetectionSystem` scanning and
confidence fusion, `PriorityScorer`, `SearchAssigner`, `TargetTracker`,
`SearchController`'s per-agent FSM, and `MissionOrchestrator`'s phase
dispatch — all called from `SimulationEngine.step()` via
`MissionOrchestrator.step()`, not dead/decorative code (unlike the prior
dynamic-obstacles finding).

Two open items identified at that point:
- The SEARCHING phase had never actually been reached by any run in the
  repo — all prior runs (`experiments/results/task0_n{10,20,30}.csv`, 120s
  each) capped out at 60.7-74.2% coverage, below the 0.85
  `exploration_completion_threshold`.
- `prioritization.py`'s `_type_score()` unconditionally routed every
  `STATIC` target to `score_static_object` (the lowest priority tier),
  even though `Target`'s own docstring lists "injured human" as a STATIC
  example and `PriorityConfig` defines a separate, higher
  `score_static_human` tier that STATIC targets could never reach.

## Priority-scoring fix (commit `63e81db`)

Three options were considered for how to distinguish an injured-human
STATIC target from a generic object, since no such distinguishing field
existed anywhere in `Target`, `TargetSpawnConfig`, or the loader:

1. Add a `Target.is_human: bool` field (STATIC-only), plus an additive
   `TargetSpawnConfig.count_static_human` spawn count.
2. Treat all STATIC targets as `static_human` unconditionally (simplest,
   but makes `score_static_object` permanently dead config).
3. Defer the fix and leave it as a documented gap.

Option 1 was chosen (user decision, via explicit question) as the most
correct long-term fix, despite widening scope beyond `prioritization.py`
alone.

Implementation across 5 files:
- `src/search/target.py` — added `is_human: bool = False` field to
  `Target` (default preserves prior behavior for anything not opting in).
- `src/search/prioritization.py` — `_type_score()`'s STATIC branch now
  returns `cfg.score_static_human if target.is_human else
  cfg.score_static_object`.
- `src/config/search_config.py` — `TargetSpawnConfig` gained
  `count_static_human: int = 0`, additive to `count_static` (total STATIC
  count = `count_static + count_static_human`), not a partition.
- `src/config/loader.py` — parses `count_static_human` from YAML
  (defaulting to 0).
- `src/search/target_manager.py` — `spawn_targets()` spawns
  `count_static_human` additional STATIC targets with `is_human=True`;
  `_make_target()` gained an `is_human` passthrough param.

Verified via `tests/test_search_prioritization.py` (4 tests): `_type_score()`
returns the correct tier for both `is_human=True/False`, the full composite
`score()` ranks a human static above an equivalent object static, and
`TargetManager.spawn_targets()` produces the correct additive counts.
Full suite green at time of commit (58/58).

## Search-phase reachability validation (commit `a13b0db`)

Config: `configs/experiments/search_reachability_validation_10uav.yaml`,
derived from `configs/simulation.yaml` (the Task 0 120s-benchmark base
config) with exactly two values changed:
- `simulation.duration`: 120.0 → 420.0.
- `dynamic_environment.enabled`: true → false.

Two isolation choices were made deliberately (flagged as decisions, not
assumed):
- **Obstacles disabled** — isolates the search-phase-reachability question
  from the separate, still-deferred dynamic-obstacle validation gap
  (`docs/dynamic_obstacles_revalidation.md`).
- **10-UAV fleet** (the base config's default, left unchanged) — keeps the
  known, unresolved 30-UAV regression (commit `9879a12`, "spatial
  clustering hypothesis [rejected]") from confounding this result.

Duration was set to 420s (not the initially-proposed 330s) after Task 0's
10-UAV coverage curve was found to be bursty/non-monotonic rather than
smoothly decelerating (a 40-70s near-stall followed by a sharp burst at
70-90s in the 120s baseline data) — 330s sat too close to a conservative
linear extrapolation (~325s) to be a genuine safety margin.

Result: coverage reached the 0.85 `exploration_completion_threshold` and
the SEARCHING phase was entered at **t=303.0s** (derived from
`mission_summary.csv`: `mission_completion_time` 420.0 −
`search_phase_duration_s` 117.0 = 303.0; independently confirmed by the
first `SEARCHING`-phase row in the agent-trace CSV, timestamped 303.0).
The phase held through the remainder of the fixed 420s run (117s of
searching) — this is the first run in the project's history to ever
exercise the SEARCHING phase.

## Performance fix (commit `d06a39e`)

`SimulationEngine._collect_metrics()` called
`revisit_ratio(self.agent_histories, self.world.map)`
(`src/evaluation/exploration_metrics.py`) every tick, which rescans the
*entire* history of every agent's positions from t=0 on every call —
O(agents × current-step-count) per tick, O(agents × N²) total over an
N-step run. Confirmed by reading `revisit_ratio()` directly, not inferred
from timing alone.

Fix: `SimulationEngine.__init__()` and `step()` now maintain two
incremental counters (`_revisit_total_visits`, `_revisit_unique_cells`),
updated in the same per-agent loop that already appends to
`agent_histories`, seeded at construction to match the pre-seeded initial
position. `_collect_metrics()` computes `revisit_ratio` from these
counters instead of calling the O(N²) function. `revisit_ratio()` itself
is untouched (still directly tested by `tests/test_exploration_metrics.py`).
Equivalence is exact (set membership is order-independent, so incremental
insertion of the same elements produces an identical final set to a
from-scratch scan), proven by `tests/test_revisit_ratio_incremental.py`,
which asserts the engine's incremental value matches a fresh call to
`revisit_ratio()` on a snapshot of `agent_histories` at two separate
checkpoints (step 5 and step 12) during a live run, not just at the end.

Performance validation — three wall-clock (EXPLORING-phase-only) timings
were taken across three runs of the same config/seed:
- 42.2s — pre-Thread-A/B baseline (the `a13b0db` run above; revisit_ratio
  fix already applied).
- 96.9s (2.3x slower) — the re-run immediately after Thread A (commit
  `107e5c6`) and Thread B (commit `24c1eca`) were applied. Investigated:
  neither diff touches any code path reachable during EXPLORING
  (`MissionOrchestrator._step_exploring()` never calls into
  `SearchController`; Thread B's trace-logging block is gated behind
  `if phase == MissionPhase.SEARCHING:`, confirmed by direct inspection of
  the applied file, not just the diff). Flagged as a system-load outlier
  rather than a reintroduced regression, since the delta's shape (rising
  t=60-140s, then declining toward t=300s) is inconsistent with a
  reintroduced O(N²) driver, which would climb monotonically rather than
  decline.
- 44.7s — a clean repeat of the same config/seed with no other load on the
  machine, confirming the 96.9s run was a system-load outlier: timing
  returned to within ~6% of the original 42.2s baseline.

## Detection/completion gap investigation

All three runs (same seed, same config) produced identical results:
5/8 targets detected, 4/8 completed, 1 lost, 3 undetected —
`detection_rate`/`search_success_rate` both 0.625 (62.5%).

### Targets 2 and 7 — never approached within effective detection range

Reconstructed exact spawn positions (target 2, static,
`pos=(20.83, 78.09)`) and full kinematic trajectories from t=303.0 (targets
4 and 7 move; reconstructed via `Target.update_position()` /
`apply_state_schedule()` using the same seed, config, and `current_time`
the real run used). Cross-referenced against every UAV's logged position
in the agent-trace CSV.

The nominal `detection_radius` (4.5m, from `uav.sensing_range`) is not the
true detection threshold — `DetectionSystem._compute_confidence()` also
gates on `min_detection_confidence` (0.3 in this config), which shrinks
the *effective* detection range to approximately 2.91m (static), 2.74m
(dynamic), 2.83m (time_varying), given `base_confidence=0.85` and the
per-type confidence multipliers.

- Target 2: closest logged approach 5.698m (t=348.0, agent 2) — beyond
  both the nominal and effective range.
- Target 7: closest logged approach 4.704m (t=338.0, agent 9) — beyond
  both the nominal and effective range.

Both are **expected misses given limited search-phase coverage, not bugs**.

### Target 4 — one data point explained, one unconfirmed

Closest *raw, non-interpolated* logged approach: 3.208m (t=303.0, agent 4
— the very first search-phase tick). Within the nominal 4.5m radius, but
running the actual confidence formula gives `confidence ≈ 0.2196`, below
the 0.3 threshold — **this miss is fully explained by the sensor model
working as configured, not a bug**.

A separate, linearly-*interpolated* estimate (between the 5s-apart
t=303.0/t=308.0 samples) suggested a possible closer approach of ~1.77m,
which would have cleared the confidence threshold (≈0.46) had it been
real. This is **not confirmed** — agent 4 was `IDLE` during that window,
and `SearchController`'s IDLE behavior replans a new random waypoint every
1.5s (`replan_interval_s`), so the true path within a 5s sampling window
can have 2-3 direction changes that a straight-line interpolation between
endpoints cannot capture. Documented as a known instrumentation-resolution
limitation, not a resolved bug, pending finer-grained (≤1.5s) position
logging.

### Target 5 — lost; original hypothesis investigated and disproven

Initial hypothesis (from the first investigation pass): target 5's loss
was caused by `SearchAgentState.assigned_target_id` being a single scalar
overwritten when UAV 5 was also assigned target 6 simultaneously,
interrupting active tracking.

**Disproven by direct trace data.** Tracing UAV 5 through the agent-trace
CSV: target 6 was assigned at t≈348, completed cleanly at t=358. Target 5
was not even detected until t=403.0 — **45 seconds after target 6 had
already completed and cleared out**. There was no moment where UAV 5 held
two simultaneous live assignments; the two assignments were fully
sequential.

Actual mechanism, per the trace: target 5 (dynamic, 0.3-0.8 m/s, slower
than the UAV) was detected at t=403.0 with only 17s left in the fixed 420s
run, entered `TRACKING` immediately, but per the per-target CSV
accumulated only 2.3s of continuous tracking (`tracking_duration_s=2.3`)
before visibility was lost (`loss_events=1`), consistent with the FSM
transitioning to `SEARCHING` between the t=413 and t=418 trace samples —
matching an 8.0s `loss_timeout_s` measured from the ~t=405.3 loss-of-sight
point. This looks like ordinary pursuit/relative-motion drift interacting
with the 1.5s replan cadence on a moving target, not an assignment bug.
The run ended (t=420) before recovery (`RecoverySearchBehaviour`) could
reacquire it — 17s of remaining time was enough for the 10.0s tracking
requirement in principle, but not enough once the loss event and its 8.0s
timeout consumed most of that window.

## Fixes applied as a result

- **Thread A** (commit `107e5c6`) — `SearchAgentState` gained
  `assigned_target_ids: list[int]` as a proper multi-assignment queue.
  `assign_target()` no longer overwrites the active target while the agent
  is in `TRACKING`; a new `_advance_to_next_target()` hands off to the
  next queued target when the current one completes or is permanently
  lost, instead of going inert in `COMPLETED` forever. Verified by
  `tests/test_search_controller_multi_target.py` (3 tests). **Not
  exercised in practice by the validation run above** — traced via the
  agent-trace CSV and confirmed no UAV ever held two overlapping
  assignments in that run (target 6 and target 5 on UAV 5 were 45s apart,
  fully sequential). The fix remains correct-by-test but untested against
  a real scenario that actually triggers the interruption condition it was
  built for.
- **Thread B** (commit `24c1eca`) — per-UAV position + assigned-target-id
  + phase snapshot, logged every 5s during `SEARCHING` to
  `{label}_agent_trace.csv`, written incrementally (not buffered) to
  `src/evaluation/run_search_experiment.py`. This is the instrumentation
  that made the target 2/4/5/7 investigations above possible.

## Known limitations / open items

- **5s logging resolution vs. 1.5s replan cadence.** The agent-trace log
  samples every 5s, but `SearchController`'s IDLE and active-pursuit
  behaviors replan every 1.5s. This is coarse enough to leave target 4's
  closest-approach question genuinely unresolved (see above) — the true
  path between samples cannot be reconstructed from straight-line
  interpolation when 2-3 replans could have occurred in that window.
- **Only 5 of 10 UAVs ever received a target assignment** in this run
  (UAVs 0, 1, 5 handled the 5 detected targets between them; UAVs 2, 3, 4,
  6, 7, 8, 9 spent the entire 117s search phase in `SearchController`'s
  IDLE behavior, sampling random waypoints uniformly across the whole
  world with no bias toward regions more likely to contain undetected
  targets). Noted as a possible avenue for a future investigation into
  IDLE-sweep efficiency, if detection rates need improving.
