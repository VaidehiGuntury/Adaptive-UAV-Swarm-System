# Simulation Assumptions

Cross-paper assumptions for the unified swarm simulator. Status markers:

- **Implemented** — active in current code
- **Stub** — data structure only, no algorithm logic
- **Deferred** — planned for a later development week

Sources: `papers/notes/paper1_notes.md`, `paper2_notes.md`, `paper3_notes.md`.

---

## Global Simulation

| Assumption | Status | Notes |
|------------|--------|-------|
| Python 3.11+, OOP, NumPy for vectors | Implemented | |
| Configuration via YAML | Implemented | `configs/simulation.yaml` |
| 2D horizontal-plane dynamics for agents | Implemented | Altitude stored for visualization only |
| Discrete-time simulation with fixed `dt` | Implemented | |
| No central server / fully decentralized agents | Implemented | Paper 1 model |
| Pygame primary renderer | Stub | `PygameRenderer` skeleton; Matplotlib still default in `main.py` |

---

## Layer 1 — Paper 1 (DEBS / BSA)

| Assumption | Status | Notes |
|------------|--------|-------|
| Homogeneous quadrotor UAV fleet | Implemented | `UAV` agent |
| Max linear velocity 1.5 m/s, max angular velocity 0.9 rad/s | Implemented | `UAVConfig` |
| Disk sensing range 4.5 m | Implemented | Not full 80°×60° FOV camera model |
| Voxel map resolution 0.15 m | Deferred | Using 2D grid at `sensing_range / 3` |
| Pairwise IDE task allocation (Eq. 1, Algs. 1–2) | Deferred | Ring spawn (`spawn_mode: ring`) sets p̃* at spawn position |
| Limited-range communication | Stub | `CommunicationGraph` exists; not used in BSA loop |
| BSA viewpoint utility U_a, costs J_C, J_V, J_L (Eqs. 6–10) | Implemented | `algorithms/aggregation/fitness_functions.py` |
| Hybrid A* + B-spline trajectories | Deferred | Direct velocity steering in `UAV.move` |
| Frontier / trail cluster marking | Implemented | Geographic `region_key` (grid cell of centroid); trails persist across cluster IDs |
| Mission region = circle at allocated target | Implemented | `UAVRegion`; `set_region` separate from BSA `set_target` (vp_c) |
| Trail penalty J_L weight | Implemented | `trail_penalty: 8.0` in YAML after geographic trail fix |
| Spawn geometry | Implemented | `ring` (default) with `spawn_angular_noise`; `legacy` for baseline experiments |

---

## Layer 2 — Paper 2 (Formation Tracking)

| Assumption | Status | Notes |
|------------|--------|-------|
| 1 master leader + n UGV leaders + m UAV followers | Stub | `MasterLeader`, `UGV` agent classes |
| Directed communication graph with spanning tree | Stub | `CommunicationGraph.from_adjacency_matrix` |
| 2D second-order dynamics, bounded control | Deferred | |
| Velocity hierarchy λ₀ < λ₁ < λ₂ | Deferred | |
| Weight-based grouping (Alg. 1, Eq. 5) | Deferred | |
| Consensus formation control (Eqs. 6–8) | Deferred | |
| APF collision / obstacle avoidance (Eqs. 21–26) | Deferred | |
| Time-varying desired formations (Eqs. 27–29) | Stub | `FormationSpec` render-only data |
| UGVs and UAVs at different altitudes (Remark 4) | Stub | `altitude` field on agents |

---

## Layer 3 — Paper 3 (TLDNNA Search)

| Assumption | Status | Notes |
|------------|--------|-------|
| Search area Φ as 2D grid | Stub | `BeliefMap` grid container |
| Target motion as discrete-time Markov process | Deferred | |
| Recursive Bayesian belief update (Eqs. 3–6) | Deferred | |
| Sensor: detected / not-detected observations | Deferred | |
| Flight path objective F(O^L) = Σ P^k (Eq. 11) | Deferred | |
| TLDNNA optimizer (Algs. 1–2) | Deferred | |
| Target / interference prior regions | Stub | `TargetRegion` dataclasses |
| 40×40 grid experiments | Deferred | Configurable via future `SearchConfig` |

---

## Layer 4 — Cross-Cutting (Future)

| Assumption | Status | Notes |
|------------|--------|-------|
| Dynamic obstacles | Deferred | Static `CircularObstacle` only |
| Communication delays and packet loss | Deferred | |
| Obstacle occlusion on comm links | Deferred | |

---

## Simplifications (Explicit)

1. **2D not 3D dynamics** — formation and exploration logic operates on `p_xy`; altitude is visual only.
2. **Circular obstacles** — proxy for forest trees; not tree-density scenes from Paper 1 experiments.
3. **No ROS** — standalone Python simulator vs. Paper 1 ROS setup.
4. **Single Paper 1 controller** — `SimulationEngine` runs BSA only; no multi-controller orchestration yet.
5. **Trail penalty raised to 8.0** — after fixing geographic trail identity, a stronger J_L discourages frontier re-selection without adding new heuristics.
6. **p̃* initialised at spawn position** — ring spawn only; IDE allocation remains deferred.
