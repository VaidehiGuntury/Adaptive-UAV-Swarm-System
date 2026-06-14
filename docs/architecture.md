# Unified Swarm Architecture

Source of truth for module layout and layer boundaries.

---

## Research Layers

```
Layer 1: Self Aggregation and Task Allocation     (Paper 1 — DEBS)
Layer 2: Formation Tracking Control               (Paper 2)
Layer 3: Lost Target Search                       (Paper 3 — TLDNNA)
Layer 4: Dynamic Obstacles, Communication Delays  (future)
```

**Data flow:**

```
Aggregation → Formation Control → Search & Recovery
```

Each layer reads agent state and environment data from the shared `World`; layers do not call each other directly. A future `MissionOrchestrator` will activate controllers per mission phase.

---

## Package Layout

```
src/
├── agents/                    # Agent models (UAV, UGV, MasterLeader)
│   ├── base_agent.py          # Abstract contract + AgentRole
│   ├── uav.py                 # Paper 1 quadrotor
│   ├── ugv.py                 # Paper 2 ground leader (stub)
│   └── master_leader.py       # Paper 2 global reference (stub)
├── algorithms/
│   ├── aggregation/           # Paper 1 BSA + future IDE
│   ├── formation/             # Paper 2 definitions, state, slot assignment
│   │   ├── formation_types.py # FormationType, FormationDefinition, transforms
│   │   ├── formation_state.py # FormationState runtime container
│   │   ├── leader_agent.py    # LeaderAgent reference frame
│   │   ├── follower_state.py  # FollowerSlotState snapshots
│   │   ├── formation_metrics.py # Observational metrics
│   │   ├── formation_controller.py # Acquisition + consensus composition
│   │   ├── consensus_controller.py # Neighbor consensus correction
│   │   ├── consensus_metrics.py    # Distributed observability metrics
│   │   ├── comm_refresh.py         # CommunicationGraph refresh
│   │   ├── velocity_control.py     # P-control + clamp helpers
│   │   ├── group_snapshot.py       # FormationGroupSnapshot assembler
│   │   └── slot_assignment.py      # Geometric slot assignment utilities
│   └── search/                # Paper 3 (placeholder)
├── environment/               # World, maps, obstacles, comm graph
│   ├── world.py               # Top-level environment container
│   ├── map.py                 # Exploration / frontier grid (Paper 1)
│   ├── obstacles.py           # Static circular obstacles
│   ├── communication.py       # NetworkX directed graph (Paper 1/2)
│   ├── belief_map.py          # Belief grid container (Paper 3)
│   ├── target_region.py       # Search zones (Paper 3)
│   └── formation_spec.py      # Immutable FormationSpec (Paper 2 desired shapes)
├── simulation/
│   └── simulation_engine.py   # Discrete-time loop, metrics, history
├── visualization/
│   ├── base.py                # RendererProtocol
│   ├── coordinate_transform.py
│   ├── formation_renderer.py  # Paper 2 slot/edge overlay (Pygame)
│   ├── renderer.py            # Matplotlib backend (SimulationRenderer)
│   └── pygame_renderer.py     # Pygame backend (primary, Week 2+)
├── config/
│   └── loader.py              # YAML → typed dataclasses
└── main.py                    # CLI entry point
```

**Backward compatibility:** `src/aggregation/` re-exports from `src/algorithms/aggregation/` (deprecated).

---

## Core Data Contracts

### `World`

Bundles all environment state:

| Field | Paper | Purpose |
|-------|-------|---------|
| `map` | 1 | Exploration grid, frontier clusters |
| `obstacles` | 1, 2 | Collision geometry |
| `communication_graph` | 1, 2 | Neighbor / leader-follower topology |
| `belief_map` | 3 | Target probability grid |
| `target_regions` | 3 | Search / prior-knowledge zones |
| `formation_specs` | 2 | Immutable desired group shapes (`FormationSpec`) |
| `formation_states` | 2 | Mutable runtime slot assignments (`FormationState`) |

### `SimulationEngine`

- **Input:** `World`, `list[UAV]`, `SelfAggregationController`, `SimulationConfig`
- **Per step:** BSA update → agent move → explore → metrics
- **Output:** `SimulationState` snapshots, `agent_histories`, `metrics_history`

### `SimulationState`

```python
timestep: int
time_s: float
agents: list[UAV]
metrics: SimulationMetrics
```

Extended fields (comm edges, belief snapshot) planned for Week 2.

### `RendererProtocol`

Backends implement:

- `run_and_record() -> list[SimulationState]`
- Live display loop (Pygame) or `animate()` (Matplotlib)

Renderers are visualization-only; they never modify simulation state.

---

## Layer 2 — Formation Infrastructure (Paper 2)

Data-only foundation for future tracking controllers. No consensus, APF, or
motion control in this phase.

### Formation coordinate system

All slot offsets are defined in the **leader-local frame**:

- Origin: leader position **p**ₗ
- Rotation: leader heading θ (`leader_heading`)
- World mapping: **p** = **p**ₗ + **R**(θ) **m**ᵢ

Shared transform utilities in `algorithms/formation/formation_types.py`:

| Function | Purpose |
|----------|---------|
| `rotate_offset(offset, theta)` | Rotate a local offset |
| `transform_to_world(local, leader_pos, theta)` | Local → world |
| `transform_to_local(world, leader_pos, theta)` | World → local |

Initial demos use θ = 0; renderer and utilities already support rotation.

### `FormationDefinition` and `FormationType`

Extensible template catalog (`formation_types.py`):

| Field | Role |
|-------|------|
| `base_slots` | Canonical preview geometry at default spacing |
| `generator_fn(n, spacing)` | Procedural slot layout for N agents |
| `edge_fn(n)` | Slot-index edge pairs for rendering / future graph control |

Supported types: `LINE`, `TRIANGLE`, `DIAMOND`, `WEDGE` (default showcase).

`register_formation_definition()` allows future perimeter / adaptive types
without changing downstream APIs.

### `FormationState` (mutable runtime)

| Field | Role |
|-------|------|
| `leader_id` | Leader agent (always slot 0) |
| `formation_type` | Active template |
| `leader_heading` | θ for local → world transform |
| `slot_assignments` | `agent_id → slot_index` (slot-space) |
| `slot_offsets` | `slot_index → local offset` (geometry-space) |
| `desired_offsets` | `agent_id → local offset` (convenience mirror) |
| `active_members` | Assigned agent IDs |
| `slot_tolerance` | Occupancy threshold (default 0.5 m) |

**Centroid** is never stored. Use `FormationState.centroid(positions)` or
`compute_formation_centroid()` with current member world positions.

### `FormationSpec` (immutable)

Frozen dataclass for planners, renderers, and future network sync.
Built via `FormationSpec.from_state(group_id, state)`. Includes
`slot_assignments`, `slot_offsets`, `desired_offsets`, and precomputed `edges`.

### Slot assignment strategy

`assign_formation_slots(agent_ids, leader_id, ...)` in `slot_assignment.py`:

1. Leader → slot 0, local offset (0, 0)
2. Remaining agents sorted by `agent_id` → slots 1 … N−1
3. Offsets from `FormationDefinition.generator_fn`
4. No agent motion

Default formation: **WEDGE**, spacing **2.0 m**.

Weight-based grouping (Paper 2 Alg. 1) is deferred; this geometric assignment
is a deterministic placeholder.

### Deferred (Paper 2 controllers)

- Consensus tracking (Eqs. 6–8)
- APF collision avoidance (Eqs. 21–26)
- Weight-based grouping (Alg. 1)
- `SimulationEngine` formation updates
- YAML `formation:` config
- Autonomous regrouping

### Leader–follower architecture

Read-only snapshot layer built on demand from `FormationState` + agent poses.
No motion control; error vectors are representational only.

```
FormationState + agent poses
        → build_formation_group_snapshot(timestep, time_s, ...)
                ├─ LeaderAgent          (composition, not UAV inheritance)
                ├─ tuple[FollowerSlotState]
                └─ FormationMetricsSnapshot
```

#### `LeaderAgent`

| Field / method | Role |
|----------------|------|
| `agent_id`, `position`, `heading` | Formation reference frame origin |
| `formation_target` | Optional global formation goal; **not** auto-filled from BSA `assigned_target` |
| `forward_direction`, `left_direction` | Leader-local +x / +y unit vectors |
| `to_world(local)`, `to_local(world)` | Wrappers over shared transform utilities |

Built via `leader_agent_from_pose()` from any agent pose.

#### `FollowerSlotState`

Per-follower snapshot (followers only; leader slot 0 excluded):

| Field | Role |
|-------|------|
| `slot_index`, `local_offset` | Slot-space assignment **m**ᵢ |
| `world_slot_position` | **p**ᵢ* = **p**ₗ + **R**(θ) **m**ᵢ |
| `formation_error` | **p**ᵢ − **p**ᵢ* (world frame, not used for control) |
| `is_occupied` | `‖error‖ ≤ slot_tolerance` (**inclusive** at boundary) |

#### `FormationGroupSnapshot`

Synchronized group view with `timestep` and `time_s` for replay, metrics sync,
and future stability analysis.

#### `FormationMetricsSnapshot`

| Metric | Definition |
|--------|------------|
| `mean_formation_error` | Mean ‖error‖ over **followers only** |
| `slot_occupancy_completeness` | occupied slots / total assigned slots |
| `occupied_slot_count` | Includes leader slot when present |
| `vacant_slot_count` | total − occupied |

#### Visualization (formation overlay)

- Leader: distinct ring + fill
- Reference frame: +x forward (red), +y left (blue) axes from leader origin
- Error vectors: **agent → desired slot** (future correction direction)
- Slot occupancy: filled (occupied) vs hollow (vacant) markers

#### Deferred controller stages

- Full second-order consensus dynamics (Paper 2 Eqs. 6–8)
- APF collision avoidance (Eqs. 21–26)
- Using `formation_error` as a control input
- Controller stability metrics
- MasterLeader → UGV → UAV multi-tier wiring
- Mission orchestration linking Paper 1 targets to Paper 2 motion

### Formation acquisition (centralized slot tracking)

Mission flow:

```
BSA aggregation → mean_pairwise_distance < threshold → WEDGE acquisition
```

`FormationAcquisitionController` (`formation_controller.py`):

| Component | Role |
|-----------|------|
| `FormationAcquisitionConfig` | `k_p`, dead-zone, activation threshold, WEDGE spacing |
| Slot command | `u_slot = k_p · (p* − p)` with dead-zone and speed clamp |
| Leader | Continues BSA; excluded from formation commands |
| Followers | Proportional slot tracking after activation |

### Distributed consensus layer

Extends slot tracking — does **not** replace it:

```
u_total = u_slot + k_consensus · Σ_{j∈N_f(i)} (p_j − p_i)
```

| Module | Role |
|--------|------|
| `consensus_controller.py` | Raw neighbor sum + gain scaling |
| `comm_refresh.py` | Rebuilds `CommunicationGraph` each active step |
| `consensus_metrics.py` | Neighbor count, connectivity, residual, spacing variance |

Communication range: `uav.sensing_range` (no delay, no YAML). Isolated followers
(`|N_f(i)| = 0`) receive zero consensus correction.

Visualization: comm links (`C`), consensus vectors, isolation warnings.

#### Still deferred

- APF / collision avoidance beyond obstacle projection
- Dynamic formation morphing and autonomous regrouping
- Communication delay and Paper 2 fixed adjacency matrices
- Spectral stability analysis
- Paper 3 search

---

## Configuration

`configs/simulation.yaml` drives:

- `environment` — world size, obstacles
- `uav` — motion and sensing limits
- `aggregation` — BSA parameters (Paper 1 Eqs. 6–10)
- `simulation` — fleet size, dt, duration

Future sections: `communication`, `formation`, `search`, `visualization`.

---

## Testing Strategy

| Module | Test file |
|--------|-----------|
| BSA / agents | `tests/test_aggregation.py` |
| Environment | `tests/test_environment.py` |
| Communication graph | `tests/test_communication.py` |
| Formation | `tests/test_formation.py`, `tests/test_leader_follower.py`, `tests/test_consensus_formation.py` |
| Pygame layers | `tests/test_pygame_renderer.py`, `tests/test_layer_toggles.py` |

Gate: `python -m unittest discover -s tests -v`

---

## Visualization Layers

Pygame `PygameRenderer` draws research layers in bottom-to-top order. Toggle keys apply during `run_live()` and `playback()`.

| Layer | Module / data source | Toggle | Default |
|-------|----------------------|--------|---------|
| **Grid** | `ExplorationMap.explored_mask()`, `obstacle_mask()` | `G` | off |
| **Frontier** | `frontier_mask()`, `extract_frontier_clusters()` centroids | `F` | on |
| **Obstacles** | `ObstacleField` | — | always on |
| **Sensor radius** | `config.uav.sensing_range`, `agent.position` | `S` | on |
| **Trails** | `SimulationEngine.agent_histories` (max 150 pts, faded) | `T` | on |
| **Targets** | `UAV.assigned_target` (BSA viewpoint markers + lines) | `Y` | on |
| **Velocity** | `UAV.velocity` arrows | `V` | on |
| **Formations** | `World.formation_specs` / `formation_states` | `M` | on if data exists |
| **Communication** | `World.communication_graph`, consensus overlay | `C` | on if controller present |
| **Agents** | `SimulationState.agents` (markers + IDs) | — | always on |
| **Dashboard** | `SimulationState.metrics`, frontier count at render time | — | always on |

Supporting modules:

- `visualization/formation_renderer.py` — Paper 2 slot markers, edges, occupancy
- `visualization/layer_toggles.py` — keyboard toggle state
- `visualization/render_palette.py` — research colour palette
- `visualization/coordinate_transform.py` — world ↔ screen (viewport excludes dashboard width)

The right-side **research dashboard** displays mission time, step, fleet size, coverage %, mean speed, mean pairwise distance, and frontier cluster count. Frontier IDs on the map are not shown (centroids only).

**Demo scope label:** “DEBS Stage 2 — BSA Viewpoint Selection”. IDE allocation and A*/B-spline planning are deferred.

---

## Visualization Stack

| Backend | Class | Status |
|---------|-------|--------|
| Matplotlib | `SimulationRenderer` | Trajectories, targets, basic metrics |
| Pygame | `PygameRenderer` | Paper 1 research demo (layers above) |

`.cursorrules` designates Pygame as the primary real-time engine. Matplotlib remains for offline analysis plots.
