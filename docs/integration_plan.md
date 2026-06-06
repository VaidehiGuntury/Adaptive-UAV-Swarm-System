# Integration Plan

Week-by-week roadmap for combining Papers 1–3 into a unified simulator.

Architecture data flow (from `architecture.md`):

```
Aggregation (Paper 1)
    ↓
Formation Control (Paper 2)
    ↓
Search & Recovery (Paper 3)
```

---

## Week 1 — Audit & Environment Foundation

- [x] Codebase audit
- [x] Week 1.5 architecture refactor plan

**Deliverable:** Audit report, refactor proposal.

---

## Week 1.5 — Architecture Refactor (Current)

**PR-1:** Documentation (`assumptions.md`, `equations.md`, `integration_plan.md`, `architecture.md`)

**PR-2:** Environment data layer
- `CommunicationGraph`, `BeliefMap`, `TargetRegion`, `FormationSpec`
- Extended `World` container

**PR-3:** Agent stubs
- `UGV`, `MasterLeader`, `AgentRole`, UAV `altitude`

**PR-4:** Package move
- `src/aggregation/` → `src/algorithms/aggregation/` with backward-compat shims

**Parallel:** Pygame renderer skeleton (obstacles, UAVs, IDs, HUD)

**Not in scope:** Formation control, search logic, Bayesian updates, IDE allocation.

---

## Week 2 — Pygame Renderer & Mission Builder

- Complete `PygameRenderer` layers (trajectories, comm links, belief overlay)
- `MissionEnvironment` factory from YAML
- Generalize `SimulationEngine` with `Controller` protocol
- Config extensions (`communication`, `mission`, `visualization`)
- `--renderer pygame` CLI flag

**Integration point:** Renderer consumes `SimulationState` + `World` optional overlays.

---

## Week 3 — Paper 2 Formation (Layer 2)

1. **Grouping** — Algorithm 1, weight-based UAV→UGV assignment
2. **Graph topology** — Load adjacency matrices W₃, W₄, W₆, W₈ from config
3. **Formation controllers** — Eqs. (6–8) for UGV and UAV agents
4. **APF safety** — Eqs. (21–26) piecewise switching
5. **Visualization** — Formation ghost offsets from `FormationSpec`

**Inputs from Layer 1:** Agent positions, obstacle field, optional mission regions.

**Outputs to Layer 3:** Group assignments, stabilized formations for search dispatch.

---

## Week 4 — Paper 1 Completion (Layer 1 Gaps)

1. **IDE pairwise allocation** — Eqs. (1–5), Algs. 1–2
2. **Communication protocol** — Range-limited neighbor negotiation via `CommunicationGraph`
3. **Improved mapping** — Finer grid / optional voxel abstraction
4. **Path planning stub** — Kinodynamic feasibility check before target assignment

**Dependency:** Layer 2 can run independently; Layer 1 IDE enhances exploration missions.

---

## Week 5 — Paper 3 Search (Layer 3)

1. **Belief map update** — Eqs. (3–6) along candidate paths
2. **Objective evaluator** — F(O^L) for flight path candidates
3. **TLDNNA optimizer** — `algorithms/search/tldnna.py`
4. **Scenario configs** — Nine Paper 3 scenario layouts on `TargetRegion` grid
5. **Belief heatmap overlay** — `BeliefMap` → Pygame layer

**Inputs from Layer 2:** UAV position within assigned group; optional formation hold.

---

## Week 6 — Unified Mission & Evaluation

- Multi-phase mission: explore → form → search
- `evaluation/` metrics scripts (BCP, tracking error, explored fraction curves)
- Experiment configs in `experiments/`
- README and reproduction guide

---

## Interface Contracts (Cross-Layer)

| Contract | Producer | Consumer |
|----------|----------|----------|
| `World` | `environment/world.py` | Engine, all controllers, renderers |
| `list[BaseAgent]` | Agent factories | Engine, `CommunicationGraph` |
| `SimulationState` | `SimulationEngine.get_state()` | All renderers |
| `FormationSpec` | Paper 2 grouping (future) | Pygame formation layer |
| `BeliefMap` | Paper 3 update (future) | Pygame belief layer |
| `CommunicationGraph` | Engine refresh (future) | Paper 1 IDE, Paper 2 controllers, renderer |

---

## Risk Mitigation

1. **Preserve Paper 1 tests** after every PR — `python -m unittest discover -s tests`
2. **Shim deprecated import paths** until Week 3 (`src/aggregation/` → `src/algorithms/aggregation/`)
3. **Optional YAML sections** — new config keys must not break existing `simulation.yaml`
4. **No algorithm logic in environment modules** — data containers only until controller week
