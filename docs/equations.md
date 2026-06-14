# Equation Index

Maps paper equations to `src/` modules and implementation status.

**Status:** `implemented` | `stub` | `deferred`

---

## Paper 1 — DEBS (Layer 1)

| Eq. | Description | Module | Status |
|-----|-------------|--------|--------|
| (1) | Pairwise allocation objective f(p̃ᵢ, p̃ⱼ) | `algorithms/aggregation/` (future IDE) | deferred |
| (2–3) | IDE adaptive F, CR | `algorithms/aggregation/` (future) | deferred |
| (4–5) | IDE mutation, crossover | `algorithms/aggregation/` (future) | deferred |
| (6) | BSA cost J_C | `algorithms/aggregation/fitness_functions.py` | implemented |
| (7) | Utility U_a | `algorithms/aggregation/fitness_functions.py` | implemented |
| (8) | Continuity λ_β | `algorithms/aggregation/fitness_functions.py` | implemented |
| (9) | Turning cost J_V | `algorithms/aggregation/fitness_functions.py` | implemented |
| (10) | Trail penalty J_L | `algorithms/aggregation/fitness_functions.py` | implemented |
| — | Viewpoint selection (Sec. 5) | `algorithms/aggregation/self_aggregation.py` | implemented |
| — | Voxel mapping / frontiers | `environment/map.py` | implemented (2D grid) |
| — | Trajectory execution | `agents/uav.py` | deferred (simplified steering) |

---

## Paper 2 — Formation Tracking (Layer 2)

| Eq. | Description | Module | Status |
|-----|-------------|--------|--------|
| (1) | Second-order agent model | `agents/base_agent.py` (future) | deferred |
| (2) | Augmented state T_i | `algorithms/formation/` | deferred |
| (4) | Formation tracking objective | `algorithms/formation/` | deferred |
| (5) | Grouping attractive field F_g | `algorithms/formation/` | deferred |
| (6) | UGV control U_g | `algorithms/formation/formation_controller.py` | stub (leader only) |
| (7) | UAV control U_a | `algorithms/formation/formation_controller.py` | implemented (P + consensus placeholder) |
| (8) | General follower control | `algorithms/formation/consensus_controller.py` | implemented (lightweight) |
| (21–23) | APF repulsive forces | `algorithms/formation/` | deferred |
| (25–26) | Rotational traction, piecewise U^total | `algorithms/formation/` | deferred |
| (27–29) | Desired formations M_i(t) | `environment/formation_spec.py` | stub |
| — | Directed graph G = (Q, E, W) | `environment/communication.py` | stub |

---

## Paper 3 — TLDNNA (Layer 3)

| Eq. | Description | Module | Status |
|-----|-------------|--------|--------|
| (1) | Belief normalization | `environment/belief_map.py` | stub |
| (3–6) | Bayesian belief predict/update | `algorithms/search/` | deferred |
| (7–10) | Detection probabilities P^k, P_C^t | `algorithms/search/` | deferred |
| (11) | Objective F(O^L) | `algorithms/search/` | deferred |
| (12–18) | Baseline NNA | `algorithms/search/` | deferred |
| (19–27) | TLDNNA extensions | `algorithms/search/` | deferred |
| — | Search area Φ | `environment/target_region.py` | stub |

---

## Simulation Metrics (Paper 1 Evaluation)

| Metric | Module | Status |
|--------|--------|--------|
| Explored fraction vs time | `environment/map.py`, `simulation/simulation_engine.py` | implemented |
| Mean velocity | `simulation/simulation_engine.py` | implemented |
| Mean pairwise distance | `simulation/simulation_engine.py` | implemented |
| Mission time | `simulation/simulation_engine.py` | implemented |
| Mean target separation (p̃* pairwise) | `evaluation/exploration_metrics.py` | implemented |
| Frontier reuse frequency | `evaluation/exploration_metrics.py` | implemented |
| Target reassignment count | `algorithms/aggregation/self_aggregation.py` | implemented |
| Revisit ratio | `evaluation/exploration_metrics.py` | implemented |
| Active frontier count | `simulation/simulation_engine.py` | implemented |
