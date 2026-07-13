# Task 7 — Exploration/allocation layer architecture snapshot

Frozen as of `exploration-phase-e` @ commit `8eb0887` (branched from
`main` @ `3ea924a`). This documents the validated state of the
exploration/allocation layer so future work (Member 2/3
re-validation, further phases) has a fixed reference point.

## Files in scope

| File | Role |
|---|---|
| `src/algorithms/aggregation/self_aggregation.py` | BSA viewpoint decision loop: candidate generation, hysteresis-based commitment, target assignment |
| `src/algorithms/aggregation/fitness_functions.py` | Paper 1 Eqs. 6-10 cost terms (J_C, U_a, J_V, J_L) and the composite viewpoint score |
| `src/algorithms/allocation/ide_allocator.py` | DEBS §4 IDE allocator — pairwise DE-based region reallocation (Algorithms 1 & 2) |
| `configs/simulation.yaml` | Validated parameter values for the above (see below) |

## Key functions

### `fitness_functions.py`
- `utility_u_a(point_a, point_b, config, lambda_beta)` — Eq. 7 piecewise attraction/repulsion between two points, shaped by `d_c`/`d_0`/`k_a`.
- `compute_lambda_beta(config)` — Eq. 8 continuity scaling factor; requires `d_0 > d_c`.
- `aggregation_utility_j_c(candidate, agent, all_agents, config, lambda_beta)` — Eq. 6. Attracts toward the agent's own `allocation_center` (assigned region, falling back to current target), **repels** from every other agent's allocation center *and* current position (both terms are subtracted — this is the J_C sign fix, see `ablation_summary.md`).
- `turning_cost_j_v(candidate, agent, config)` — Eq. 9, `turn_cost_weight · (1 − cos θ)` between current velocity heading and candidate direction.
- `trail_penalty_j_l(candidate, config)` — Eq. 10, flat `trail_penalty` if the candidate's region is marked as a trail (already-visited) region.
- `evaluate_viewpoint_cost(...)` — composite score `w_C·J_C − w_V·J_V − w_L·J_L` used for argmax candidate selection. Weights are `cluster_penalty_weight` (w_C), `turn_penalty_weight` (w_V), `trail_penalty_weight` (w_L).

### `self_aggregation.py`
- `SelfAggregationController.update(agent, all_agents, world, dt)` — the per-agent, per-step decision entry point. Re-evaluates candidates every `replan_interval`, but only **replaces** the current target on: no target yet (bootstrap), arrival (`dist ≤ d_c`), a safety-net timeout (`world_diagonal / max_speed`), or a strictly-better `evaluate_viewpoint_cost` score for the best new candidate vs. continuing toward the current target. This is the hysteresis fix (Task 4, Design C).
- `_generate_candidates(agent, clusters, world)` — builds `ViewpointCandidate` list from frontier clusters.
- `_fallback_target(agent, world)` — used when no frontier candidates exist.
- Internal state: `_target_timeout: dict[agent_id, float]` (safety-net deadline), `_target_region_key: dict[agent_id, RegionKey]` (for re-scoring "keep current target" against new candidates).

### `ide_allocator.py`
- `IDEAllocator.allocate(agents, current_time)` — top-level entry, called once per `replan_interval` from `SimulationEngine.step()` (before BSA reads `p̃*`, per DEBS §4 Algorithm 2 ordering).
- `_pick_partner(uav_i_id, uav_positions, current_time)` — selects a pairwise interaction partner within `communication_range`, gated by `t_att` (min inter-interaction gap).
- `_run_ide(...)` — the DE loop itself: `_lhs_init` (local Latin Hypercube init within `bounds_padding` of centre), `_mutate`/`_crossover` (DE/rand/1/bin-style), `_adaptive_F`/`_adaptive_CR` (paper Eq. 2-3 schedule — currently **not** the cosine variant, see ablation table), `_pairwise_objective` (desired-separation cost using `d_star`).

## Validated parameters (`configs/simulation.yaml`)

| Parameter | Value | Note |
|---|---|---|
| `aggregation.d_c` | 0.5 | Eq. 7 "very close" threshold |
| `aggregation.d_0` | 17.8 | = `mission_region_radius`; raised from 12.0 (Task 4) |
| `aggregation.k_a` | 1.0 | |
| `aggregation.mission_region_radius` | 17.8 | fair-share radius, 10 UAVs / 100×100m: `sqrt(10000/10/π)` |
| `aggregation.cluster_penalty_weight` (w_C) | 2.0 | raised from 1.0 alongside the d0 fix |
| `aggregation.trail_penalty_weight` (w_L) | 1.0 | unchanged |
| `aggregation.turn_penalty_weight` (w_V) | 0.5 | unchanged |
| `aggregation.trail_penalty` | 8.0 | flat penalty magnitude |
| `aggregation.replan_interval` | 2.0s | re-evaluation cadence (unchanged by hysteresis fix — only the switch condition changed) |
| `ide.d_star` | 30.0 | revised from paper's 9.0m; `2 × 17.8 = 35.6`, rounded down for headroom |
| `ide.fe_max` | 200 | paper value |
| `ide.t_att` | 2.0s | |
| `ide.communication_range` | 50.0 | paper `Rcomm` |
| `ide.bounds_padding` (Qs) | 0.5 | local LHS step size, paper value |

## DO-NOT-MODIFY list

The following files are frozen pending separately-approved changes —
do not modify without explicit sign-off, since they carry today's
validated fixes and any edit invalidates the Task 0-2 benchmark
baseline:

- `src/algorithms/aggregation/self_aggregation.py`
- `src/algorithms/aggregation/fitness_functions.py`
- `src/algorithms/allocation/ide_allocator.py`

`configs/simulation.yaml`'s `aggregation:` and `ide:` blocks carry the
validated parameter values above and should also be treated as frozen
for the same reason, even though the file as a whole is shared with
Member 2/3's config blocks (`dynamic_environment:`, `search:`).

## Known open issues (carried forward, not fixed by this phase)

- 30-UAV coverage regression after the hysteresis fix (80.99% →
  74.19%) remains unresolved; spatial-clustering hypothesis rejected
  in Task 1 (`experiments/results/task1_spatial_clustering_diagnosis.md`).
- Dynamic obstacles are decorative — not consulted by collision
  resolution or BSA (`docs/member2_revalidation_notes.md`).
