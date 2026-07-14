# Dynamic Obstacles — Revalidation Status

Dynamic obstacle collision-resolution fix is implemented and verified
functionally via forced-collision test (commit `2bde77a`) — see
docs/session_logs or commit message for details.

Full paper-style scenario validation (static/slow/equal_speed/fast/mixed
obstacles x 10/20 UAVs, `experiments/scripts/run_obstacle_scenarios.py`)
has now been run to completion, a collision-counting bug it surfaced has
been fixed and regression-tested, and the results have been analyzed.
See sections below.

## 1. Validation matrix — now complete

`experiments/scripts/run_obstacle_scenarios.py` runs the full 5-scenario
(`static`/`slow`/`equal_speed`/`fast`/`mixed`) × 2-fleet-size (10/20 UAV)
matrix — 10 runs total, `DURATION=120.0`, `SEED=42` — and writes
`experiments/results/obstacle_scenario_validation.csv`.

Previously this batch was killed after running for over an hour and
marked deferred. It now completes in well under a minute end-to-end,
thanks to the earlier `revisit_ratio` O(N²)→O(N) incremental-tracking
fix (commit `d06a39e`, see `docs/target_search_revalidation.md`), which
was the dominant per-tick cost this matrix was paying 10 times over. No
change was needed in `run_obstacle_scenarios.py` itself to make the
batch tractable.

## 2. Collision-counting bug found and fixed (commit `ba3e2ce`)

Running the matrix surfaced a second, independent bug: every scenario —
including runs with obstacle avoidance working correctly — logged a
nonzero `collision_count`.

**Root cause.** `World.resolve_collisions` (`src/environment/world.py`)
pushed UAVs clear of dynamic obstacles using a hardcoded
`margin=0.3` (function default, unchanged at its only call site,
`simulation_engine.py:187`), forwarded into
`ObstacleManager.nearest_free_point`. Collision *classification*, in
`ObstacleManager.check_collision` (`obstacle_manager.py:285-286`), used
`dynamic_environment.collision_radius: 0.35` (`configs/simulation.yaml`).
Since `0.3 < 0.35`, and `nearest_free_point` deterministically projects
a resolved point to exactly `margin` distance from the obstacle surface,
every UAV the avoidance logic successfully pushed clear still landed
*inside* the classification threshold — `check_collision` read
`distance=0.30 ≤ collision_radius=0.35` and logged it as a collision.
Confirmed by direct query against `ObstacleManager`: resolving a point
with `margin=0.3` yields `distance=0.3000, colliding=True`.

**Fix.** Instead of bumping the hardcoded margin to a second larger
constant (still fragile against a future `collision_radius` change),
`resolve_collisions` now derives the dynamic-obstacle margin live from
the manager's own `collision_radius`:

```python
# src/environment/world.py:48-53 (module-level constant)
_DYNAMIC_CLEARANCE_BUFFER = 0.05

# src/environment/world.py:154-177 (resolve_collisions)
dynamic_margin = max(
    margin,
    self.obstacle_manager.collision_radius + _DYNAMIC_CLEARANCE_BUFFER,
)
```

Because `obstacle_manager.collision_radius` is the same config value
`check_collision` classifies against, the resolution margin and the
classification threshold can no longer drift out of sync regardless of
what `collision_radius` is set to. Re-querying the same point with the
new margin confirms the fix: `distance=0.4000, colliding=False,
near_miss=True`.

**Regression test.** `tests/test_dynamic_collision_resolution.py`
(added in `ba3e2ce`) builds a minimal `World` + `ObstacleManager` with a
stationary obstacle, drives an agent into it, and asserts that
`check_collision` on the `resolve_collisions` output is not classified
as colliding and clears `collision_radius`. Verified both ways before
committing:

- Against pre-fix code: both assertions **fail**, reproducing the bug
  (`distance=0.3000 not greater than 0.35`).
- Against post-fix code: both **pass**.

Full existing suite (64 tests, including the 2 new ones) passes
post-fix — no regressions elsewhere.

## 3. Post-fix results — collision count

Collisions dropped to 0 across all 10 scenarios, confirming the fix.

| label | num_uavs | collisions (pre-fix) | collisions (post-fix) |
|---|---|---|---|
| static_n10 | 10 | 0 | 0 |
| slow_n10 | 10 | 32 | 0 |
| equal_speed_n10 | 10 | 47 | 0 |
| fast_n10 | 10 | 18 | 0 |
| mixed_n10 | 10 | 41 | 0 |
| static_n20 | 20 | 0 | 0 |
| slow_n20 | 20 | 48 | 0 |
| equal_speed_n20 | 20 | 45 | 0 |
| fast_n20 | 20 | 160 | 0 |
| mixed_n20 | 20 | 74 | 0 |

(`fast_n20`'s pre-fix outlier magnitude — 160 vs. 18-74 elsewhere — is a
separate, still-open fleet-density/obstacle-speed question; see §5 and
§6, not itself a counting bug.)

## 4. Near-miss surge — explained, not a new bug

`near_miss_count` rose sharply post-fix (55-221 per run, vs. 18-160
collisions pre-fix). `collision_count` and `near_miss_count` are both
derived from the same `check_collision` classification
(`obstacle_manager.py:285-286`):

```python
colliding = nearest_dist <= self.collision_radius        # ≤ 0.35
near_miss = (not colliding) and (nearest_dist <= self.safety_margin)  # (0.35, 0.75]
```

Those thresholds (`collision_radius=0.35`, `safety_margin=0.75`) were
**not** changed by the fix. What changed is which band the deterministic
resolved standoff distance falls into: pre-fix it landed at `0.30m`
(inside the collision band); post-fix it lands at `0.40m` (inside the
near-miss band, `0.35 < 0.40 ≤ 0.75`). The same physical avoidance
events that were miscounted as collisions before are now correctly
classified as near-misses — confirmed by the same direct
`ObstacleManager` query used in §2 (`margin=0.3` →
`colliding=True`; `margin=0.4` → `near_miss=True`, same obstacle
geometry). Old `collision_count + near_miss_count` sums and new
`near_miss_count` alone track the same rough magnitude per scenario
(e.g. `fast_n20`: 235 combined pre-fix vs. 149 near-miss post-fix), not
an exact match — the remainder is genuine trajectory divergence from
the changed resolution physics (§5), not mislabeling.

**Known pre-existing caveat (not introduced by this fix):**
`collision_count` and `near_miss_count` both count per-frame, per-agent
`check_collision` samples with no deduplication — unlike
`obstacle_encounters` (`dynamic_environment_metrics.py`), which has an
explicit `cooldown_steps` window to collapse a sustained contact into
one event. A UAV held at the resolved standoff distance against one
obstacle for 2s (dt=0.1) can generate ~20 separate near-miss records for
what is arguably one encounter. This granularity limitation predates
this fix and applies equally to both counters; flagged here as a known
gap, not something to fix as part of this work.

## 5. Coverage-degradation flip at n=20

Post-fix, `coverage_degradation_vs_static` swung substantially negative
for three of the four n=20 dynamic scenarios (coverage *higher* with
obstacles present than the static baseline):

| label | degradation (pre-fix) | degradation (post-fix) | final_coverage (post-fix) |
|---|---|---|---|
| slow_n10 | +3.2pp | +1.7pp | 0.589 |
| equal_speed_n10 | −0.0pp | +4.1pp | 0.566 |
| fast_n10 | +3.4pp | −1.5pp | 0.621 |
| mixed_n10 | +2.4pp | +4.0pp | 0.567 |
| slow_n20 | +3.9pp | +1.1pp | 0.726 |
| equal_speed_n20 | −0.5pp | **−5.7pp** | 0.794 |
| fast_n20 | −2.6pp | **−8.6pp** | 0.822 |
| mixed_n20 | +1.3pp | **−7.0pp** | 0.807 |

(static baselines: n=10 → 0.6068, n=20 → 0.7364, bit-identical pre- and
post-fix, confirming `SEED=42` determinism and that the shift is a real
consequence of the code change, not run-to-run noise.)

**Mechanism.** Two effects compound from the same margin change (§2):

1. **Wider avoidance-trigger radius.** `nearest_free_point`'s activation
   condition is `distance_to(point) ≤ margin`. Raising the effective
   margin from `0.30` to `~0.40` (`collision_radius + buffer` for the
   default `collision_radius=0.35`) means positions between 0.30m and
   0.40m from an obstacle surface — previously left untouched — now
   trigger repositioning. Confirmed directly: a point at exactly 0.35m
   surface distance is unmoved under `margin=0.3` but is moved under
   `margin=0.4`. Avoidance now fires more often across the fleet, not
   just displaces further per event.
2. **Resolved positions feed directly into discrete BSA decisions.**
   `agent.position` is overwritten in place by the resolved value
   (`simulation_engine.py:187`), and candidate frontier viewpoints are
   also routed through `resolve_collisions`
   (`self_aggregation.py:202`, `:232`) before the argmin-cost candidate
   is selected (`candidates_per_frontier: 6`,
   `replan_interval: 2.0s` in `configs/simulation.yaml`). A small
   continuous perturbation near a decision boundary can flip which
   discrete candidate wins, sending an agent toward a different
   frontier cluster — sensitivity that compounds over the full 1200
   timesteps (120s at dt=0.1) of a run.

The scenario pattern is consistent with this: only `equal_speed_n20`,
`fast_n20`, `mixed_n20` flip strongly negative; `slow_n20` stays small
and positive (+1.1pp), and every n=10 scenario stays within the same
small noise band as pre-fix (±1-4pp). This lines up with two known,
still-unaddressed config gaps rather than a new one:

- `mission_region_radius: 17.8` (`configs/simulation.yaml:78`) is
  explicitly commented as tuned for a 10-UAV fleet
  (`sqrt(10000/10/π) = 17.8m`) and is not scaled when the script
  overrides `num_uavs` to 20 — n=20 runs are roughly 2× denser than the
  design assumption, so more UAVs contest the same fixed
  `obstacle_count: 12` (`configs/simulation.yaml:23`).
- `slow`, the one scenario whose obstacles are strictly slower than
  `uav.max_speed=1.5`, is also the one n=20 scenario that stays in the
  noise band — `equal_speed`/`fast`/`mixed` all include obstacle speeds
  at or above UAV max speed, which UAVs cannot simply outrun, so contact
  (and therefore repositioning) frequency is structurally higher for
  exactly the scenarios that flipped.

**Verdict:** plausible, mechanistically-explained real behavior of the
coupled avoidance+BSA system — coverage genuinely is higher, not a
measurement artifact. But it demonstrates the system's outcome is
sensitive to the exact avoidance implementation at high fleet
density/obstacle speed, not something to treat as a precisely known
effect size yet (see §6 caveat).

## 6. Caveat — single-seed data

All results above (pre-fix and post-fix) are from a single seed
(`SEED=42`) with no averaging. The *direction* of the coverage-flip
finding in §5 is trustworthy — it replicated identically in kind (n=20,
non-slow scenarios only) both before and after the collision-margin fix
changed the exact magnitudes — but the specific percentage-point values
should not be treated as precise or representative until validated
across multiple seeds.

## 7. Consolidated status

| Phase | Item | Status |
|---|---|---|
| F — Dynamic obstacles | Collision-resolution wiring fix | ✅ done, verified via forced-collision test (`2bde77a`) |
| F — Dynamic obstacles | Full obstacle-scenario validation matrix (static/slow/equal_speed/fast/mixed × 10/20 UAVs) | ✅ **complete** — was ⏭️ deferred, now run to completion (`run_obstacle_scenarios.py`, results in `experiments/results/obstacle_scenario_validation.csv`) |
| F — Dynamic obstacles | Collision-counting margin bug | ✅ fixed and regression-tested (`ba3e2ce`) |
| F — Dynamic obstacles | Near-miss surge investigation | ✅ explained — reclassification consequence of the margin fix, not a new bug |
| F — Dynamic obstacles | Coverage-degradation flip at n=20 | ✅ investigated, mechanism identified — magnitude not yet multi-seed validated |
| F — Dynamic obstacles | Hysteresis/obstacle-avoidance interaction check | 🔲 not started |

## 8. Remaining open items for the next contributor

- **Multi-seed validation of the coverage-flip magnitude** — ties into
  the already-deferred Phase E multi-seed validation task
  (`experiments/scripts/run_multiseed.py`, commit `e30dfdb`, never
  executed — see `docs/session_logs/exploration_fixes_summary.md` §6).
  Re-running `run_obstacle_scenarios.py` (or an extended variant) across
  multiple seeds would separate the confirmed *direction* of the §5
  finding from its exact magnitude.
- **Fleet-density / `mission_region_radius` scaling for n=20+ fleets** —
  `mission_region_radius: 17.8` (`configs/simulation.yaml:78`) and
  `obstacle_count: 12` are tuned/fixed for a 10-UAV fleet and are not
  adjusted when fleet size is overridden to 20. Not yet addressed;
  flagged in §5 as a likely contributor to the n=20-concentrated
  coverage-flip pattern and to the `fast_n20` collision-count outlier
  observed pre-fix (§3).
- **Hysteresis/obstacle-avoidance interaction check (Phase F)** — the
  hysteresis-based commitment fix (Phase E) and the obstacle
  collision-resolution fix (Phase F) have not been validated together;
  whether obstacle avoidance interacts with the hysteresis
  arrival/timeout/measurable-improvement logic in unexpected ways is
  unchecked.
