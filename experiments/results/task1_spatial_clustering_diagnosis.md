# Task 1 — 30-UAV regression: spatial-clustering hypothesis

**Verdict: REJECTED.**

## Method

30 UAVs, 120s, `experiments/scratch/simulation_n30.yaml` (num_uavs=30
variant of `configs/simulation.yaml`, d0/J_C fixes in place throughout).
Checkpoints every 10s. "Before" = `self_aggregation.py` reverted to the
pre-hysteresis-fix version at commit `4b7d156` (unconditional replan
every `replan_interval`), J_C and d0 fixes left untouched. "After" =
current `exploration-phase-e` HEAD (`3ea924a`).

Raw data:
- `before_hysteresis_n30_d120_targets.csv`
- `after_hysteresis_n30_d120_targets.csv`

Script: `src/evaluation/spatial_clustering_diagnostic.py`.

## Results (mean over the 13 checkpoints, t=0..120s)

| Metric | Before | After |
|---|---|---|
| mean target-pairwise distance | 38.98 m | 44.69 m |
| min target-pairwise distance | 0.205 m | 0.192 m |
| UAVs in transit | 30/30 at every checkpoint after t=0 | 30/30 through t=90, then **0/30** at t=100/110/120 |

## Interpretation

If the hysteresis fix caused targets to cluster together (reduced
spatial parallelism), we'd expect the after-fix mean pairwise target
distance to be *lower* than before, and/or more UAVs perpetually "in
transit" toward closely-packed targets. We see the opposite:

- Mean target separation is **higher** after the fix (44.7m vs 39.0m) —
  targets are more spread out, not more clustered.
- Min pairwise distance is statistically indistinguishable between the
  two conditions (~0.2m in both) — in both cases there is essentially
  always at least one near-coincident target pair at any instant, which
  is a property of the ring-spawn/frontier-assignment process itself,
  not something introduced by the hysteresis fix.
- The before-fix condition shows all 30 UAVs perpetually "in transit"
  (never arrived) for the entire run — consistent with the original bug
  (targets abandoned before being reached). The after-fix condition
  shows UAVs transitioning to 0-in-transit by t=100s, i.e. UAVs are
  arriving and holding position (exploration largely complete/converged
  by that point), not clustering.

This directly contradicts the "reduced spatial parallelism" hypothesis.
Per Task 1's decision tree (rejected branch): no fix is proposed here.

Since the hypothesis was rejected at 30 UAVs, no clustering effect
exists to check for at 10/20 UAVs — item 3 of Task 1 (whether the
effect is visible at smaller fleet sizes) doesn't apply.

## Status

The 30-UAV regression (80.99% -> 74.19% coverage after the hysteresis
fix) remains a **known, isolated, unresolved limitation**. Ruled out so
far: frontier starvation, density-aware relaxation, and now spatial
clustering / reduced parallelism. No further fix is proposed without
new evidence.
