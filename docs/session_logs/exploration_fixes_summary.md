# Session Log Index — Exploration, Obstacle, and Target-Search Work

Top-level index for all fix/validation work across the three subsystem
phases tracked in this repo (branch `exploration-phase-e`): Phase E
(exploration/allocation layer), Phase F (dynamic obstacles), Phase G
(target search & tracking). This file was referenced as existing at the
start of recent work but did not — `docs/session_logs/` was present but
empty; this is the first file in it.

## 1. Exploration layer fixes (Phase E, pre-existing — before recent sessions)

Three fixes, all bundled into a single commit:
**`3ea924a` — "WIP: hysteresis commitment + d0/J_C fixes,
pre-merge-reconciliation backup"** (verified directly via `git show`,
not just the commit message):

- **J_C sign fix** — `aggregation_utility_j_c()` in
  `src/algorithms/aggregation/fitness_functions.py` had
  `utility += utility_u_a(vp_c, other.position, ...)`; the diff in
  `3ea924a` changes this to `utility -=`. Confirmed still live at current
  `HEAD` (`grep` on the file shows the `-=` form). The bug: J_C is meant
  to *repel* a candidate viewpoint from other UAVs' current positions
  (dispersal); the `+=` form attracted toward them instead.
- **d0 fix** — `configs/simulation.yaml`'s `aggregation.d_0` raised from
  `12.0` to `17.8` in the same commit, with an inline comment explaining
  it now matches `mission_region_radius` (`√(10000/10/π)` for 10 UAVs on
  a 100×100m map) so BSA's Eq. 7 attraction reaches the full extent of a
  UAV's own fair-share disk rather than a smaller inner radius.
- **Hysteresis-based commitment fix** — `SelfAggregationController` in
  `src/algorithms/aggregation/self_aggregation.py` reworked so a UAV's
  current BSA target is only replaced on arrival (within `d_c`), a
  safety-net timeout (time to cross the world diagonal at `max_speed`),
  or a *measurably better* re-scored candidate — not unconditionally
  every `replan_interval`. The commit's own docstring documents two
  rejected simpler alternatives with measured coverage: unconditional
  per-cycle switching (~14% of cycles reached `sensing_range` of the
  target before reassignment) and unconditional commit-until-arrival
  (coverage dropped to ~41%).

This is a single WIP/backup commit bundling all three changes together —
not three separately attributable commits. Fully verified via direct
`git show`/`git diff`/`grep` against current `HEAD` (see
`docs/ablation_summary.md`, Task 4, for the full ablation table with
per-change verification sources).

Some earlier, pre-fix intermediate numbers ("Original: 49.34%" at 10
UAVs; "after d0/J_C fix only: 57.79%/68.28%/80.99%" at 10/20/30 UAVs)
were reported in the original briefing for this phase but describe a
state — d0+J_C fixed, hysteresis not yet added — that was never
committed on its own (`3ea924a` bundles all three together). **These
numbers are reported, not independently verified in this repo** — no
artifact exists to check them against (see `docs/ablation_summary.md`'s
closing section for the same caveat).

## 2. 30-UAV regression investigation (Phase E)

Status: **closed as unresolved** — a known, isolated limitation, not
fixed. Coverage at 30 UAVs dropped from 80.99% (d0/J_C fix only, before
hysteresis) to 74.19% (after the hysteresis fix). Three hypotheses were
investigated:

- **Frontier starvation** and **density-aware relaxation of the
  hysteresis improvement threshold** — both listed as ruled out in
  `experiments/results/task1_spatial_clustering_diagnosis.md`'s Status
  section, and density-aware relaxation additionally appears as its own
  row in `docs/ablation_summary.md` ("Made all fleet sizes worse...
  Regressed 10/20 UAV coverage as a side effect while not fixing the
  30-UAV case it targeted"). **Both are marked in `ablation_summary.md`
  as "session log (pre-repo experiment, not committed — reverted before
  commit)"** — reported to have been run on an earlier, different local
  checkout before this repo/branch existed, with no git commit, data
  file, or other artifact in this repository to verify them against. I
  could not find a dedicated commit, script, or results file for either
  hypothesis anywhere in this repo's history. Carried over from prior
  work, not independently verified in this repo.
- **Spatial clustering / reduced spatial parallelism** — investigated
  with full methodology and data this session: commit `9879a12`
  ("Diagnose 30-UAV regression: spatial clustering hypothesis
  [rejected]"), data in
  `experiments/results/{before,after}_hysteresis_n30_d120_targets.csv`,
  write-up in `experiments/results/task1_spatial_clustering_diagnosis.md`.
  **Rejected**: mean target-pairwise distance was *higher* after the
  hysteresis fix (44.7m vs 39.0m before) — the opposite of what the
  clustering hypothesis predicts — and UAVs-in-transit settled to 0/30
  by t=100s after the fix vs. a perpetual 30/30 before it.

No fix is proposed for the 30-UAV regression pending new evidence — this
matches `task1_spatial_clustering_diagnosis.md`'s own conclusion.

## 3. Dynamic obstacles bug fix (Phase F)

**Commit `2bde77a` — "Fix: wire dynamic obstacles into collision
resolution (were previously decorative)"**. Dynamic obstacles were fully
tracked (position, motion, rendering, metrics) but never checked in
`World.resolve_collisions()` — a UAV could fully overlap a moving
obstacle with zero effect. Fixed by adding
`ObstacleManager.nearest_free_point()` (mirroring the static
`ObstacleField`'s push-out-then-clip algorithm) as a second collision
pass. Verified via a forced-collision test showing correct push-out
(0.0m → 3.3m) instead of full overlap sustained for 30s. Full
paper-style scenario validation across obstacle-speed categories remains
deferred (see `docs/dynamic_obstacles_revalidation.md`).

## 4. Detailed subsystem docs

- **`docs/target_search_revalidation.md`** — target-search subsystem
  (Phase G): the priority-scoring fix (`63e81db`), the search-phase
  reachability validation config and result (`a13b0db`), the
  `revisit_ratio` O(N²) performance fix and its validation (`d06a39e`),
  and the full detection/completion-gap investigation (targets 2/4/5/7).
- **`docs/dynamic_obstacles_revalidation.md`** — obstacle collision fix
  (`2bde77a`) plus the deferred full-scenario validation matrix
  (`experiments/scripts/run_obstacle_scenarios.py`, not yet executed to
  completion).

## 5. Consolidated status across Phase E / F / G

| Phase | Item | Status |
|---|---|---|
| E — Exploration | J_C sign fix | ✅ done, verified live at HEAD |
| E — Exploration | d0 fix (12.0m → 17.8m) | ✅ done, verified live at HEAD |
| E — Exploration | Hysteresis-based commitment fix | ✅ done, verified live at HEAD |
| E — Exploration | 30-UAV regression | 🔲 unresolved, closed without a fix pending new evidence |
| E — Exploration | Multi-seed validation | ⏭️ harness exists (`experiments/scripts/run_multiseed.py`, commit `e30dfdb`) but never executed — no results artifact found |
| F — Dynamic obstacles | Collision-resolution wiring fix | ✅ done, verified via forced-collision test |
| F — Dynamic obstacles | Full obstacle-scenario validation matrix (static/slow/equal_speed/fast/mixed × 10/20 UAVs) | ⏭️ deferred — script exists (`run_obstacle_scenarios.py`) but not yet run to completion |
| F — Dynamic obstacles | Hysteresis/obstacle-avoidance interaction check | 🔲 not started |
| G — Target search | Detection/prioritization/assignment/FSM/tracking wiring | ✅ confirmed genuinely wired (not decorative) |
| G — Target search | Priority-scoring bug (STATIC targets) | ✅ fixed (`63e81db`) |
| G — Target search | Search-phase reachability | ✅ validated — first reached at t=303.0s (`a13b0db`) |
| G — Target search | `revisit_ratio` O(N²) performance bug | ✅ fixed (`d06a39e`), validated performance-neutral via clean repeat |
| G — Target search | Multi-target assignment/actuation queue fix | ✅ fixed (`107e5c6`) but **not exercised** by the validated scenario (no overlapping assignment occurred) — correct-by-test, untested-in-practice |
| G — Target search | Detection-gap investigation (targets 2, 4, 5, 7) | ✅ investigated — 2 expected misses, 1 sensor-model-explained + 1 instrumentation-limited miss, 1 lost-target mechanism identified |

## 6. Remaining open items for the next contributor

- **Multi-seed validation (Phase E)** — `experiments/scripts/run_multiseed.py`
  exists but has never been executed; no results artifact found in
  `experiments/results/`.
- **Obstacle scenario validation matrix (Phase F)** — full
  static/slow/equal_speed/fast/mixed × 10/20-UAV matrix
  (`experiments/scripts/run_obstacle_scenarios.py`) not yet run to
  completion; runtime cost estimated at ~1hr+ for the full batch.
- **Hysteresis/obstacle-avoidance interaction check (Phase F)** — the
  hysteresis-based commitment fix (Phase E) and the obstacle
  collision-resolution fix (Phase F) have not been validated together;
  whether obstacle avoidance interacts with the hysteresis
  arrival/timeout/measurable-improvement logic in unexpected ways is
  unchecked.
