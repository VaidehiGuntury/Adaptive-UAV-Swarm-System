# Notes for Member 2 (dynamic obstacles) re-validation — draft

Running notes captured during exploration-phase-e validation, to be
finalized in Task 8's sequencing directive.

## Dynamic obstacles are decorative, not collision-active (2026-07-13)

`world.resolve_collisions()` (`src/environment/world.py:147`) resolves
UAV positions against `self.obstacles` (the **static** obstacle set)
only, via `nearest_free_point`. `world.obstacle_manager` (dynamic
obstacles) is referenced in exactly three places repo-wide:

- `simulation_engine.py:137-138` — advances obstacle motion (`.update(dt)`)
- `visualization/renderer.py` / `pygame_renderer.py` — drawing only
- `evaluation/dynamic_environment_metrics.py` — post-hoc collision
  *measurement* (imports `CollisionResult`)

No UAV kinematics, collision-resolution, or BSA fitness/candidate code
reads `world.obstacle_manager`. Dynamic obstacles move and are measured
for near-misses/collisions after the fact, but they do not influence
UAV trajectories, velocities, or target selection at all.

Confirmed by code inspection (not empirically re-verified, since the
missing reference makes an effect structurally impossible regardless
of seed). This explains the earlier smoke-test observation of
bit-for-bit identical `explored_fraction` with and without dynamic
obstacles wired up — it isn't a seed coincidence, obstacles cannot
affect this metric under any seed.

**Action needed from Member 2**: confirm whether this is intentional
(metrics-only instrumentation phase) or a wiring gap against their own
design intent, before their re-validation pass is considered meaningful
against the fixed exploration layer.
