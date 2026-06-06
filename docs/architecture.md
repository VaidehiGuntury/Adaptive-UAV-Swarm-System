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
│   ├── formation/             # Paper 2 (placeholder)
│   └── search/                # Paper 3 (placeholder)
├── environment/               # World, maps, obstacles, comm graph
│   ├── world.py               # Top-level environment container
│   ├── map.py                 # Exploration / frontier grid (Paper 1)
│   ├── obstacles.py           # Static circular obstacles
│   ├── communication.py       # NetworkX directed graph (Paper 1/2)
│   ├── belief_map.py          # Belief grid container (Paper 3)
│   ├── target_region.py       # Search zones (Paper 3)
│   └── formation_spec.py      # Desired offsets (Paper 2, render-only)
├── simulation/
│   └── simulation_engine.py   # Discrete-time loop, metrics, history
├── visualization/
│   ├── base.py                # RendererProtocol
│   ├── coordinate_transform.py
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
| `formation_specs` | 2 | Render-only desired group shapes |

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
| Belief map | `tests/test_belief_map.py` |

Gate: `python -m unittest discover -s tests -v`

---

## Visualization Stack

| Backend | Class | Status |
|---------|-------|--------|
| Matplotlib | `SimulationRenderer` | Full (trajectories, targets, metrics) |
| Pygame | `PygameRenderer` | Skeleton (obstacles, UAVs, IDs, HUD) |

`.cursorrules` designates Pygame as the primary real-time engine. Matplotlib remains for offline analysis plots.
