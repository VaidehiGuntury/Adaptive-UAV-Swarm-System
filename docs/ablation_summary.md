# Task 4 — Full ablation summary

Every change tested on the exploration/allocation layer, accepted and
reverted, with provenance made explicit. See footnote for why some
rows are sourced differently than others.

| Change | Hypothesis | Result (quantified) | Kept/Reverted | Reasoning | Source |
|---|---|---|---|---|---|
| J_C sign correction (`fitness_functions.py`) | `aggregation_utility_j_c` should *repel* from other UAVs' positions, not attract | Sign was `utility += ...`; fixed to `utility -= ...` (diff: `4b7d156`→`3ea924a`) | **Kept** | Attraction toward other UAVs directly contradicts the paper's dispersal intent — repulsion is required for coverage spread | this repo — verified via `git diff` |
| d0 raised 12.0m → 17.8m (`configs/simulation.yaml`) | BSA attraction (Eq. 7) should reach the full extent of a UAV's own fair-share disk (mission_region_radius), not a smaller inner radius | `d_0` set equal to `mission_region_radius` (17.8m = √(10000/10/π) for 10 UAVs on 100×100m map) | **Kept** | Prevents BSA from losing discrimination across the UAV's own assigned area | this repo — verified via config diff + inline comment |
| cluster_penalty_weight 1.0 → 2.0 (`configs/simulation.yaml`) | J_C needed more influence to compete with frontier utility once d0 was widened | (not independently re-run this session) | **Kept** | Companion tuning change alongside d0 fix, same commit | this repo — verified via inline comment, numeric result not re-run |
| d_star (IDE) 9.0m (paper) → 30.0m | Paper's desired-separation constant undersized for a 100×100m arena at the derived fair-share radius | `d_star = 2 × 17.8m = 35.6m`, rounded down to 30.0m for headroom | **Kept** | Paper value assumed a different arena scale; rescaled to this project's world size | this repo — verified via inline comment |
| Execution-commitment — Design A: unconditional switch every `replan_interval` (original/pre-fix behavior) | N/A (baseline being replaced) | Only ~14% of replan cycles got a UAV within `sensing_range` of its assigned target before being reassigned | **Reverted** | `replan_interval × max_speed` = 3.0m/cycle, so viewpoints tens of metres away were abandoned before ever being reached | this repo — verified via docstring in `self_aggregation.py` diff |
| Execution-commitment — Design B: commit unconditionally until arrival/timeout | Removing all mid-flight re-evaluation would stop premature abandonment | Coverage dropped further, to ~41% | **Reverted** | Over-corrected — ignored closer/better frontiers that appeared mid-journey since BSA stopped re-evaluating until arrival | this repo — verified via docstring in `self_aggregation.py` diff |
| Execution-commitment — Design C: hysteresis (re-evaluate every cycle, switch only on arrival/timeout/measurable improvement) | Keep decisions current without discarding in-progress trajectories for merely-different ones | 10/20/30 UAV coverage: 60.68/73.64/74.19% (independently re-measured this session: 60.7/73.6/74.2%) | **Kept** | Best of the three; only design avoiding both premature abandonment and stale over-commitment | this repo — verified via `git diff` + independently re-run in Task 0 |
| Task 1 — spatial-clustering / reduced-parallelism hypothesis for the 30-UAV regression | Hysteresis fix causes UAV targets to cluster together at 30 UAVs, explaining the 80.99%→74.19% regression | Mean target-pairwise distance was *higher* after the fix (44.7m vs 39.0m before); min distance statistically indistinguishable (~0.2m both); in-transit counts settled to 0 by t=100s after the fix vs. perpetual 30/30 before | **Rejected** | Opposite of predicted direction — targets are more spread out after the fix, not less. 30-UAV regression remains open/unresolved | this repo — verified this session, see `experiments/results/task1_spatial_clustering_diagnosis.md` |
| IDE LHS search-domain expansion (flat d_star and residual-scaled variants) | Expanding the LHS local-search domain would improve IDE allocation quality | Regressed coverage; direction-agnostic objective caused angular drift | **Reverted** | — | session log (pre-repo experiment, not committed — reverted before commit)¹ |
| Cosine adaptive F/CR schedule (paper Eq. 2-3) | Adaptive mutation/crossover schedule would improve DE convergence within the allocation step | Correctly implemented but too few effective DE generations at `fe_max=200` | **Reverted** | — | session log (pre-repo experiment, not committed — reverted before commit)¹ |
| Deterministic per-cluster candidate viewpoint sampling | Deterministic candidate placement would reduce noise vs. random sampling | Marginal regression | **Reverted** | — | session log (pre-repo experiment, not committed — reverted before commit)¹ |
| Density-aware relaxation of the hysteresis improvement threshold | Relaxing the improvement threshold in dense (30-UAV) fleets would fix the 30-UAV regression | Made all fleet sizes worse | **Reverted** | Regressed 10/20 UAV coverage as a side effect while not fixing the 30-UAV case it targeted | session log (pre-repo experiment, not committed — reverted before commit)¹ |

¹ **Provenance note:** the four rows marked "session log" were reported
to have been run in an earlier session on a different local checkout,
before this repo/branch existed, and reverted prior to any commit — so
by construction they leave no git trace here. The cited source file,
`docs/session_logs/exploration_fixes_summary.md`, does not exist in
this repository as of this writing (the `docs/session_logs/` directory
is present but empty). These four rows are therefore recorded as
**reported, not independently verified** — distinct from every other
row in this table, which is either verified directly against this
repo's `git diff`/`git log`/docstrings, or independently re-run this
session. If the source log is located, these rows should be updated
with its actual content and this note removed.

## Numbers reported but not independently re-verified this session

The original briefing for this phase also cited pre-hysteresis-fix
coverage figures — "Original: 49.34%" (10 UAVs, pre any fix) and
"after d0/J_C fix only: 57.79%/68.28%/80.99%" (10/20/30 UAVs). These
describe an intermediate state (d0+J_C fixed, hysteresis not yet
added) that was never committed on its own — commit `3ea924a` bundles
d0, J_C, and the hysteresis fix together — so, like the four rows
above, there is no artifact in this repo to check them against. They
are plausible and consistent in direction with the fully-verified
before/after hysteresis numbers, but are reported here for completeness
without independent verification.
