# Dynamic Environment Extension — Full Summary

Repo: `Adaptive-UAV-Swarm-System`, branch `exploration-phase-e` (built on
`feature/dynamic-environment`, merged into `main` via `b9b7300`).

---

## Step 1 — Where the work lives

Dynamic-environment work happened as its own feature branch, in four
phase commits, later merged into `main` and then patched once more on
`exploration-phase-e`:

```
fb15010  Complete Phase 1: Dynamic obstacle framework and obstacle manager
b1bc73f  Complete Phase 2: integrate dynamic environment into simulation pipeline
0311481  Complete Phase 3: Dynamic environment metrics and visualization
a192de6  Complete Phase 4: Scenario-aware dynamic obstacle generation
   ...   (merged into main at b9b7300, then main gained IDE/target-search work)
2bde77a  Fix: wire dynamic obstacles into collision resolution (were previously decorative)
53ab8a5  Add dynamic-obstacles revalidation doc
```

**Files that actually implement it** (diff of `8a949f9..a192de6`, the
pre-Phase-1 commit vs. end of Phase 4):

| File | Role |
|---|---|
| `src/environment/dynamic_obstacles.py` | Obstacle representation + 3 motion models |
| `src/environment/obstacle_manager.py` | Lifecycle, per-tick update, collision queries, push-out resolution |
| `src/environment/world.py` | Builds obstacles from config; wires manager into `resolve_collisions()` |
| `src/config/loader.py` | `DynamicEnvironmentConfig`, `LinearMotionConfig`, `WaypointMotionConfig`, `RandomWalkMotionConfig` |
| `src/simulation/simulation_engine.py` | Calls `obstacle_manager.update(dt)` once per tick |
| `src/evaluation/dynamic_environment_metrics.py` | Post-hoc metrics (collisions, near-misses, encounters, etc.) |
| `src/visualization/renderer.py`, `pygame_renderer.py`, `render_palette.py`, `layer_toggles.py` | Drawing moving obstacles |
| `docs/dynamic_environment_design.md` | 2500-line up-front design spec (SDS §22–37) |
| `docs/dynamic_obstacles_revalidation.md`, `docs/member2_revalidation_notes.md` | Post-hoc bug diagnosis and validation status |

Nothing in `src/environment/map.py` (the occupancy grid used by BSA),
`src/environment/belief_map.py`, `src/agents/uav.py`,
`src/algorithms/allocation/ide_allocator.py`, or
`src/algorithms/aggregation/self_aggregation.py` references
`obstacle_manager` at all — confirmed by repo-wide grep. This absence is
itself the most important finding and drives most of the answers below.

---

## Step 2 — How it works, file by file

### a) Representation

`DynamicObstacle` (abstract base, `dynamic_obstacles.py`) stores:
`obstacle_id`, `position` (2-D, metres), `velocity` (2-D, m/s), `radius`
(collision radius, metres), `active` flag, and a `motion_type` tag.
Obstacles are circles — same geometric primitive as the pre-existing
static `CircularObstacle`, just with a `velocity` field added and a
per-tick `update(dt)` instead of a fixed position.

Three concrete subclasses:

- **`LinearObstacle`** — constant-velocity motion, elastic bounce off
  world boundaries (velocity component flips at the wall).
- **`WaypointObstacle`** — patrols a closed loop of fixed waypoints at
  constant speed, advancing to the next waypoint once within an
  arrival tolerance (defaults to its own radius).
- **`RandomWalkObstacle`** — Brownian-like: heading perturbed each tick
  by Gaussian noise scaled by `√dt` (correct Itô scaling so the
  diffusion rate doesn't change if the timestep changes), speed held
  constant, boundary reflection applied to the heading.

### b) How they move

All three are **scripted/parametric motion models**, not physics
(no forces, no obstacle-obstacle collision, no acceleration limits) and
not path-planned (no obstacle ever "sees" a UAV or reacts to one).
Motion is **continuous-in-space but discretely integrated** — each
`update(dt)` does a first-order Euler position step
(`position += velocity * dt`); it is called once per fixed simulation
tick, not on a continuous clock or as an event.

### c) Update cadence

`ObstacleManager.update(dt)` is called from
`SimulationEngine.step()` — **once every simulation tick**, i.e. every
`dt = 0.1s` (from `configs/simulation.yaml`), unconditionally, as step
1 of the 9-step per-tick pipeline (see `simulation_engine.py:127-217`).
It runs *before* IDE allocation and BSA aggregation each tick, so those
two subsystems (if they read obstacle state, which they don't — see
(f)) would see the latest position.

### d) Mapping / stale data

**There is no stale-data problem, because dynamic obstacles are never
written into the map at all.** `ExplorationMap` builds its
`_obstacle_mask` exactly once, at `World.__init__` time, purely from
the static `ObstacleField` (`map.py:87`, `self.obstacles.is_collision(point)`).
`obstacle_manager` (the dynamic set) is not passed to `ExplorationMap`
and is never consulted when marking cells explored/occupied. So the
voxel/occupancy grid that BSA reads for frontier/coverage decisions is
**100% static-obstacle-only**, exactly as in the original paper — it
neither gains nor needs to lose knowledge about moving obstacles,
because it never had any to begin with.

### e) Collision avoidance — the "decorative" bug and its fix

This is the key implementation bug found and fixed on this branch.

**The bug:** `World.resolve_collisions()` — called every tick, right
after each UAV's kinematic update — only ever queried the **static**
`ObstacleField`. `world.obstacle_manager` (dynamic obstacles) was
referenced in exactly three other places (obstacle motion update,
rendering, and post-hoc metrics) but nowhere in collision resolution.
Concretely: a UAV could fly straight through a moving obstacle's
centre and nothing would happen — no push-out, no repulsion, nothing —
while the obstacle kept moving and rendering normally, and the metrics
module would (separately) *report* the collision as having happened,
even though the UAV's actual trajectory was completely unaffected.
This was diagnosed and written up in `docs/member2_revalidation_notes.md`
before the fix, including the observation that `explored_fraction`
was bit-for-bit identical with dynamic obstacles enabled vs. disabled —
not a seed coincidence, but structurally guaranteed by the missing
reference.

**The fix (`2bde77a`):** added `ObstacleManager.nearest_free_point()`,
a direct port of the static `ObstacleField`'s existing push-out-then-clip
algorithm (project the point away from any obstacle whose surface
distance is less than the margin, iterate up to 8 passes, clip to world
bounds) but applied to `DynamicObstacle.position`/`.radius` instead of
the static obstacle's. `World.resolve_collisions()` now runs this as a
**second pass**, after the static-obstacle pass, every tick. Verified
by a forced-collision test showing correct push-out (0.0 m separation
→ 3.3 m) instead of 30 s of full overlap, and by re-running the
10-UAV/120 s exploration-only benchmark to confirm identical results
(60.7%) when `obstacle_manager` is absent — i.e. the fix is additive
and doesn't perturb the static-only path.

**What this collision avoidance actually is:** a reactive, purely
kinematic **position correction** (teleport-style projection out of the
nearest obstacle), applied after the UAV has already moved. It is not
velocity-based avoidance, not a potential field the UAV steers by in
advance, and not path re-routing — a UAV only "reacts" to a dynamic
obstacle at the instant of overlap, by having its position clipped
sideways.

### f) Task allocation (IDE) / target pose selection (BSA) — does NOT re-trigger

**This is the most important finding.** Neither the IDE pairwise
allocator (`src/algorithms/allocation/ide_allocator.py`) nor the BSA
self-aggregation controller (`src/algorithms/aggregation/self_aggregation.py`,
`fitness_functions.py`) contains a single reference to `obstacle_manager`
or to any dynamic-obstacle type. Grepping both files for "obstacle"
turns up only one unrelated docstring comment. IDE re-runs on a fixed
wall-clock cadence (`replan_interval = 2.0s`, config-driven, checked
against elapsed sim time — nothing to do with obstacle motion). BSA
runs every tick as it always did in the base algorithm. **Neither is
triggered, gated, or perturbed by anything the dynamic-obstacle system
does.** A UAV can be mid-flight toward a BSA-selected viewpoint that a
moving obstacle has since parked on top of, and neither the viewpoint
choice nor the region assignment will change because of that — the UAV
will keep heading there and simply get kinematically shoved sideways
by `resolve_collisions()` if/when it actually touches the obstacle.

This is by explicit design, not oversight: the design document states
up front (`docs/dynamic_environment_design.md`, §4 "Scope of Work")
that the module "will not modify BSA viewpoint selection, IDE
optimization, target discovery, formation control, communication
architecture" — the extension was scoped as an environment-layer-only
addition from the start.

### g) Replanning logic — none

Following directly from (f): there is no mechanism anywhere that
detects "my target viewpoint is now a bad choice because an obstacle
moved" and triggers a UAV to abandon or recompute it. The only thing
that happens is the low-level positional swerve described in (e).
UAVs never re-evaluate or re-request a target because of obstacle
motion — only because of the pre-existing, obstacle-unaware
hysteresis logic (arrival, timeout, or a measurably better re-scored
BSA candidate on the next tick), all of which is orthogonal to
whether a dynamic obstacle happens to be involved.

---

## Step 3 — Comparison against the original DEBS paper's assumptions

The paper assumes static obstacle positions, fixed-density synthetic
forests, and a voxel/occupancy map that only ever grows/fills in.

| Point | New capability, or fix for a broken static assumption? |
|---|---|
| **(a) Representation** | **Genuinely new.** The paper has no notion of obstacle velocity or motion type at all; this is an addition, not a repair. |
| **(b) Motion models** | **Genuinely new.** Three motion models (linear/waypoint/random-walk) are a capability the paper never needed or specified. |
| **(c) Update cadence** | **Genuinely new**, but reuses the paper's existing fixed-tick simulation loop unchanged — no new timing concept, just a new thing ticked. |
| **(d) Mapping / stale data** | **Not a fix — the static assumption was never actually challenged.** The paper's "map only ever grows" assumption survives completely intact in this codebase, because dynamic obstacles were deliberately kept out of the map layer. There is no un-knowing logic because there was never anything to un-know. This is worth stating precisely to an instructor: the extension sidesteps the hardest part of "true" dynamic-environment mapping (stale occupancy data) rather than solving it. |
| **(e) Collision avoidance** | **Both.** The underlying push-out algorithm is a direct reuse of the paper-era static-obstacle mechanism (genuinely new only in that it's applied a second time to a moving target). But the *fact that it was needed at all* is a direct consequence of the static assumption breaking — a UAV that never re-checks its surroundings against a position that changed since last checked will drive straight through it. The `2bde77a` fix is best framed as: **the static-era "check once, push out" logic was silently insufficient the moment obstacles could move, and had to be re-applied every tick to stay correct** — a real gap the paper's assumptions caused, not a stylistic addition. |
| **(f) IDE / BSA re-trigger** | **N/A — deliberately out of scope.** The paper's task-allocation and viewpoint-selection math is untouched and, by design, obstacle-agnostic. This is neither a new capability nor a fix; it is an explicit non-goal documented in the design doc. Worth flagging to an instructor as a **known limitation**, not a hidden one. |
| **(g) Replanning** | **Absent, consistent with (f).** Since allocation/pose-selection never learns about dynamic obstacles, there is nothing to replan. This is the direct downstream consequence of the (f) scoping decision. |

---

## Step 4 — What the revalidation docs found and what's still open

`docs/dynamic_obstacles_revalidation.md`, `docs/member2_revalidation_notes.md`,
and the session-log index (`docs/session_logs/exploration_fixes_summary.md`,
Phase F section) together document:

**Found and fixed:**
- Dynamic obstacles were tracked, animated, rendered, and measured, but
  had **zero causal effect on any UAV** — a structural wiring gap, not a
  probabilistic/seed-dependent one, confirmed by code inspection
  (`member2_revalidation_notes.md`) before being patched.
- Fixed in `2bde77a` by adding a second `nearest_free_point()` pass
  against the dynamic obstacle set inside `World.resolve_collisions()`.
  Verified with a forced-collision unit test (0.0 m sustained overlap →
  correct 3.3 m push-out) and a regression check confirming the
  10-UAV/120s static-only exploration benchmark is bit-for-bit unchanged
  (60.7%) when no dynamic obstacles are configured. All 54 tests pass.

**Still open / deferred (explicitly marked, not hidden):**
- **Full paper-style scenario validation matrix** — static / slow /
  equal-speed / fast / mixed obstacle scenarios × 10/20/30-UAV fleet
  sizes — was designed and scripted
  (`experiments/scripts/run_obstacle_scenarios.py`) but **never executed
  to completion**, due to ~1hr+ estimated runtime for the full batch.
  So while the collision mechanism is known to work in isolation, there
  is no quantitative before/after evidence (coverage %, collision
  counts, near-miss counts) across the paper's own scenario categories.
- **Hysteresis × dynamic-obstacle interaction** — the Phase E
  hysteresis-based BSA commitment fix and the Phase F collision fix have
  never been validated running together; whether obstacle-triggered
  position pushes interfere with the hysteresis arrival/timeout logic is
  explicitly flagged as unchecked ("not started" in the status table).
- The revalidation doc frames this candidly: "Member 2 subsystem is
  functionally fixed but not yet quantitatively validated across
  obstacle-speed categories" — i.e. correctness of the mechanism is
  established, its measured *impact* is not.

---

## Step 5 — 3 key things to tell an instructor

1. **The extension adds a genuinely new capability the paper never
   modeled — moving obstacles with three distinct motion models
   (deterministic linear, scripted patrol, stochastic random-walk) —
   integrated as a self-contained environment-layer module with its own
   config surface, metrics, and visualization, while leaving the paper's
   core BSA/IDE algorithms completely untouched.** This is a clean,
   explicitly-scoped extension, not a patch scattered through the
   existing algorithm.

2. **Making that extension actually matter required discovering and
   fixing a real integration bug: the "decorative obstacle" gap.**
   Dynamic obstacles moved, rendered, and were logged in metrics for an
   entire development phase before anyone noticed they had zero effect
   on UAV trajectories — because the collision-resolution code path
   still only ever consulted the static obstacle set inherited from the
   paper's original assumptions. This is a good concrete illustration
   of *why* static-world assumptions are dangerous to relax carelessly:
   the bug was invisible in code review and only surfaced via a
   targeted forced-collision test.

3. **The extension is honest about — and this codebase explicitly
   documents — what it deliberately does *not* solve.** Two things the
   paper's static assumption quietly depended on are still true here by
   design, not by accident: (i) the occupancy map used for exploration
   decisions never has to un-know anything, because dynamic obstacles
   are kept out of it entirely; and (ii) task allocation (IDE) and
   target/viewpoint selection (BSA) never replan in response to
   obstacle motion — a UAV only ever reacts kinematically, at the
   instant of near-contact, never strategically. Both limitations are
   named as such in the design doc and the revalidation notes rather
   than being silently absorbed — a stronger research narrative than
   claiming full dynamic-environment support, and a natural direction
   for future work (obstacle-aware replanning, occupancy decay/staleness
   modeling) that the current, deferred obstacle-scenario validation
   matrix would be the first step toward measuring.
