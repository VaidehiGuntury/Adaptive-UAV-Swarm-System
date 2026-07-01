# Dynamic Environment Extension Design Document

**Project:** Adaptive UAV Swarm System  
**Repository:** DEBS (Distributed Exploration using Bio-inspired Self-Aggregation) Extension  
**Document Type:** Software Design Specification (SDS)  
**Version:** 2.0  
**Status:** Design Phase (Pre-Implementation)  
**Module Owner:** Member 2 – Dynamic Environment Extension  
**Target Language:** Python 3.11+  
**Simulation Framework:** NumPy + Pygame  
**Last Updated:** July 2026

---

# 1. Purpose

This document defines the software architecture, implementation strategy, and integration plan for extending the existing DEBS exploration framework from a **static environment** to a **dynamic environment**.

Unlike the original DEBS implementation, which assumes stationary obstacles throughout the mission, this extension introduces moving obstacles with multiple motion models while preserving the existing exploration pipeline.

This document serves as:

- the architectural reference for the Dynamic Environment module,
- the implementation guide for Member 2,
- the integration reference for Members 1 and 3,
- the primary design document for future maintenance and research.

---

# 2. Background

The current repository implements the exploration layer presented in the DEBS paper.

Implemented capabilities include:

- Environment generation
- Obstacle generation
- Exploration map
- Frontier extraction
- Bio-inspired Self Aggregation (BSA)
- UAV motion
- Exploration metrics
- Real-time visualization

The implementation intentionally leaves IDE optimization, dynamic environments, and target search for future work.

The objective of this work is **not to modify the DEBS algorithm**, but to extend the environment in which the algorithm operates.

---

# 3. Problem Statement

The original DEBS paper evaluates exploration in environments containing only static obstacles.

This assumption simplifies the planning problem because obstacle positions never change during mission execution.

However, practical exploration missions involve dynamic environments where obstacles may move unpredictably.

Examples include:

- pedestrians
- vehicles
- construction equipment
- wildlife
- rescue personnel
- collaborative robots
- autonomous ground vehicles

Static assumptions become insufficient for such environments.

Consequently, the repository requires a modular extension capable of introducing dynamic obstacles without disrupting the existing exploration framework.

---

# 4. Scope of Work

This module is responsible only for extending the **environment layer**.

The exploration algorithm itself remains unchanged.

The Dynamic Environment module will introduce:

- moving obstacles
- multiple obstacle motion models
- obstacle lifecycle management
- dynamic collision prediction
- environment-dependent evaluation metrics
- visualization support
- experiment scenarios

This module will not modify:

- BSA viewpoint selection
- IDE optimization
- target discovery
- formation control
- communication architecture

---

# 5. Existing Repository Architecture

The current repository follows a modular layered architecture.

```
main.py
│
├── Configuration Loader
│
├── World Construction
│
├── UAV Initialization
│
├── Self Aggregation Controller
│
├── Simulation Engine
│
└── Renderer
```

Internally the execution flow becomes

```
Configuration

↓

World

↓

Simulation

↓

Visualization
```

The architecture already separates environment, simulation, visualization, agents and algorithms into independent packages.

This separation enables the Dynamic Environment module to be integrated with minimal architectural changes.

---

# 6. Repository Structure Analysis

Current repository structure

```
src
│
├── agents
│     ├── base_agent.py
│     ├── uav.py
│     ├── ugv.py
│     └── master_leader.py
│
├── algorithms
│     ├── aggregation
│     ├── formation
│     └── search
│
├── config
│
├── environment
│     ├── belief_map.py
│     ├── communication.py
│     ├── formation_spec.py
│     ├── map.py
│     ├── obstacles.py
│     ├── target_region.py
│     └── world.py
│
├── evaluation
│
├── simulation
│
└── visualization
```

The repository exhibits a clean separation of responsibilities.

Each package performs a single high-level responsibility.

This architecture will be preserved.

---

# 7. Existing Execution Pipeline

Current simulation execution follows the sequence

```
Load Configuration

↓

Construct World

↓

Spawn UAVs

↓

Initialize Aggregation

↓

Simulation Loop

↓

Collect Metrics

↓

Render Frame
```

Within each simulation step

```
Aggregation Update

↓

UAV Motion Update

↓

Collision Resolution

↓

Exploration Update

↓

Metrics Collection
```

Obstacle positions remain fixed during the entire simulation.

---

# 8. Existing Environment Layer

The Environment package currently consists of the following logical components.

## World

Responsibilities

- stores world dimensions
- owns exploration map
- owns obstacle field
- performs collision resolution

---

## ObstacleField

Responsibilities

- maintains circular obstacles
- performs collision queries
- nearest free point projection

Current obstacle representation

```
CircularObstacle
```

Properties

- center
- radius

Behavior

- contains(point)
- distance_to(point)

The obstacle representation is immutable and static.

---

## ExplorationMap

Responsibilities

- occupancy representation
- explored region tracking
- frontier extraction
- frontier clustering
- trail management

The map assumes obstacle occupancy never changes after initialization.

---

## SimulationEngine

Responsibilities

- execute simulation timestep
- update aggregation
- update UAV kinematics
- resolve collisions
- update exploration map
- collect metrics

No environment update stage currently exists.

---

## UAV

Responsibilities

- maintain UAV state
- execute motion
- follow assigned viewpoint

The UAV currently performs kinematic motion only.

Obstacle avoidance is handled externally by the World.

---

# 9. Current Limitations

The repository currently assumes:

- static obstacles
- immutable obstacle geometry
- static occupancy grid
- fixed collision environment
- time-invariant obstacle mask

As a result,

- no moving objects can exist
- collision prediction is impossible
- dynamic obstacle avoidance cannot be evaluated
- environment complexity remains constant throughout simulation

These assumptions limit applicability to real-world exploration scenarios.

---

# 10. Motivation for Extension

The Dynamic Environment Extension transforms the repository from a static exploration simulator into a time-varying environment simulator.

The proposed extension enables investigation of:

- exploration under environmental uncertainty
- moving obstacle interaction
- obstacle encounter frequency
- collision prediction
- mission degradation
- robustness of decentralized exploration

This significantly broadens the research scope while preserving compatibility with the original DEBS implementation.

---

# 11. Design Objectives

The Dynamic Environment Extension is designed according to the following objectives.

## Primary Objectives

- Preserve compatibility with the current repository.
- Preserve Paper 1 exploration behavior.
- Introduce dynamic obstacle support.
- Maintain modular architecture.
- Minimize code coupling.
- Support future research extensions.

---

## Secondary Objectives

- Support multiple obstacle motion models.
- Support configurable environments.
- Support quantitative evaluation.
- Support visualization.
- Support reproducible experiments.

---

# 12. Functional Requirements

The implementation shall:

- support static and dynamic obstacles simultaneously
- support multiple obstacle motion models
- update obstacle states every simulation step
- provide collision prediction
- expose obstacle queries to the simulation engine
- generate dynamic-environment metrics
- integrate with the existing renderer
- preserve current static simulation behavior

---

# 13. Non-Functional Requirements

The implementation shall satisfy:

### Modularity

Each component shall have a single responsibility.

---

### Backward Compatibility

Disabling the Dynamic Environment module shall produce identical behavior to the current repository.

---

### Maintainability

The implementation shall follow the existing project coding style.

---

### Extensibility

New obstacle motion models shall be addable without modifying existing implementations.

---

### Performance

Obstacle updates shall execute once per simulation timestep with minimal overhead.

---

### Testability

Each motion model shall be independently unit testable.

---

# 14. Architectural Design Principles

The Dynamic Environment Extension follows the following architectural principles.

## Single Responsibility Principle

Each class performs exactly one responsibility.

Examples

- DynamicObstacle → obstacle motion
- ObstacleManager → obstacle lifecycle
- World → environment ownership
- SimulationEngine → simulation execution

---

## Open–Closed Principle

The framework shall be open for new obstacle types without modifying existing classes.

Future obstacle models can inherit from DynamicObstacle.

---

## Composition over Inheritance

The World owns an ObstacleManager rather than inheriting its behavior.

---

## Backward Compatibility

The original DEBS implementation remains the default execution path whenever dynamic environments are disabled.

---

## Separation of Concerns

Environment logic shall never be embedded directly inside:

- UAV
- Aggregation
- ExplorationMap

Instead, the Environment layer owns all obstacle behavior.

---

# 15. Summary

The current repository already provides a strong modular foundation for decentralized swarm exploration.

The proposed Dynamic Environment Extension builds upon this architecture by introducing a dedicated environment layer for moving obstacles, predictive collision handling, and dynamic evaluation while preserving the integrity of the existing DEBS exploration framework.

Subsequent sections of this document describe the proposed architecture, implementation details, integration strategy, testing methodology, and experimental design.

---

# 16. Proposed Software Architecture

The Dynamic Environment Extension introduces a dedicated environment management layer while preserving the existing DEBS exploration pipeline.

Unlike the current implementation, where the environment remains immutable after initialization, the proposed architecture models the environment as a time-varying system whose state evolves continuously throughout mission execution.

The architecture follows a layered design to isolate environment behavior from swarm behavior.

---

# 17. High-Level System Architecture

The proposed architecture extends the existing repository as follows.

```
                          +----------------------+
                          |      main.py         |
                          +----------+-----------+
                                     |
                                     |
                          Load Configuration
                                     |
                                     |
                                     ▼
                         +----------------------+
                         |      World           |
                         +----------+-----------+
                                    |
             +----------------------+----------------------+
             |                                             |
             ▼                                             ▼
   ExplorationMap                              ObstacleManager
                                                      |
                             +------------------------+---------------------+
                             |                                              |
                             ▼                                              ▼
                     ObstacleField                               DynamicObstacle
                    (Static Obstacles)                                |
                                                         +-------------+-------------+
                                                         |             |             |
                                                         ▼             ▼             ▼
                                               LinearObstacle   WaypointObstacle  RandomWalkObstacle

                                    |
                                    ▼

                          +----------------------+
                          | Simulation Engine    |
                          +----------+-----------+
                                     |
                                     ▼
                           Aggregation Controller
                                     |
                                     ▼
                                   UAV Fleet
                                     |
                                     ▼
                                 Visualization
```

This architecture introduces a single new subsystem without changing the ownership hierarchy of the repository.

---

# 18. Design Philosophy

The extension follows one fundamental principle.

> The environment owns obstacle behavior.
> UAVs only react to the environment.

Therefore,

DynamicObstacle objects are **not** owned by the UAV.

They are **not** owned by the SimulationEngine.

They are **not** owned by the Aggregation algorithm.

Instead,

```
World
    ↓
ObstacleManager
    ↓
Dynamic Obstacles
```

becomes the sole ownership chain.

This greatly reduces coupling between research modules.

---

# 19. Repository Integration Map

Existing repository

```
src/
│
├── agents/
├── algorithms/
├── config/
├── environment/
├── evaluation/
├── simulation/
└── visualization/
```

Proposed repository

```
src/
│
├── agents/
│
├── algorithms/
│
├── config/
│
├── environment/
│   │
│   ├── obstacles.py
│   ├── dynamic_obstacles.py
│   ├── obstacle_manager.py
│   ├── world.py
│   ├── map.py
│   └── ...
│
├── evaluation/
│   ├── exploration_metrics.py
│   └── dynamic_environment_metrics.py
│
├── simulation/
│
└── visualization/
```

Only a small number of files require modification.

The majority of the extension is implemented through newly introduced modules.

---

# 20. File Modification Strategy

The following table summarizes every repository file.

| File | Status | Purpose |
|------|---------|----------|
| obstacles.py | Keep | Static obstacle implementation |
| world.py | Modify | Own ObstacleManager |
| map.py | Minimal/None | Preserve current exploration logic |
| simulation_engine.py | Modify | Update dynamic obstacles each timestep |
| loader.py | Modify | Load DynamicEnvironmentConfig |
| simulation.yaml | Modify | Dynamic environment parameters |
| pygame_renderer.py | Modify | Draw dynamic obstacles |
| exploration_metrics.py | Keep | Preserve existing metrics |
| dynamic_environment_metrics.py | New | Dynamic environment metrics |
| dynamic_obstacles.py | New | Motion models |
| obstacle_manager.py | New | Obstacle lifecycle |

This minimizes merge conflicts.

---

# 21. New Environment Layer

The Environment package will be extended with two new modules.

```
environment
│
├── obstacles.py
│
├── dynamic_obstacles.py
│
└── obstacle_manager.py
```

Their responsibilities are intentionally separated.

---

# 22. DynamicObstacle Hierarchy

The proposed object-oriented hierarchy is

```
DynamicObstacle
│
├── LinearObstacle
│
├── WaypointObstacle
│
└── RandomWalkObstacle
```

Future extensions may include

```
HumanObstacle

VehicleObstacle

DroneObstacle

AnimalObstacle
```

without modifying any existing implementation.

This follows the Open–Closed Principle.

---

# 23. DynamicObstacle Base Class

The base class represents any moving obstacle.

State

```
obstacle_id

position

velocity

radius

active

motion_type
```

Responsibilities

- maintain obstacle state
- predict future position
- perform collision queries
- expose motion interface

Interface

```
update(dt)

predict_position()

distance_to()

collides()

bounding_circle()

serialize()
```

Every motion model implements

```
update(dt)
```

independently.

---

# 24. Motion Models

## Linear Motion

Mathematical model

```
p(t+Δt)

=

p(t)

+

vΔt
```

Characteristics

- deterministic
- constant velocity
- lowest computational cost

Applications

- vehicles
- conveyor robots
- ground robots

---

## Waypoint Motion

State

```
waypoints

current waypoint

speed
```

Algorithm

```
Current Waypoint

↓

Move

↓

Reached?

↓

Next Waypoint
```

Applications

- patrol vehicles

- autonomous forklifts

- inspection robots

---

## Random Walk

State

```
heading

speed

turn_noise
```

Algorithm

```
heading

↓

random perturbation

↓

move
```

Applications

- pedestrians

- wildlife

- uncertain moving objects

---

# 25. ObstacleManager Design

ObstacleManager becomes the central coordinator for every obstacle.

```
ObstacleManager
│
├── Static Obstacles
│
├── Dynamic Obstacles
│
├── Update
│
├── Collision Queries
│
├── Spawn
│
└── Remove
```

Responsibilities

- maintain obstacle lists
- update dynamic obstacles
- answer collision queries
- expose nearest obstacle
- expose obstacle lookup
- provide prediction services

ObstacleManager never performs rendering.

ObstacleManager never performs UAV planning.

Its responsibility is environment management only.

---

# 26. Configuration Design

The existing configuration hierarchy

```
Environment

UAV

Aggregation

Simulation
```

will be extended.

```
Environment

↓

DynamicEnvironment

↓

UAV

↓

Aggregation

↓

Simulation
```

New configuration object

```
DynamicEnvironmentConfig
```

contains

```
enabled

scenario

dynamic obstacle count

collision radius

near miss radius

safety margin

motion model parameters

random seed
```

---

# 27. YAML Configuration

simulation.yaml gains

```yaml
dynamic_environment:

  enabled: true

  scenario: mixed

  obstacle_count: 12

  safety_margin: 0.75

  collision_radius: 0.35

  random_seed: 42

  linear:

    speed: 0.5

  waypoint:

    speed: 1.0

  random_walk:

    speed: 0.8

    turn_noise: 0.15
```

When

```
enabled = false
```

the simulator behaves exactly like the current repository.

---

# 28. World Integration

Current ownership

```
World

↓

ObstacleField
```

Proposed ownership

```
World

│

├── ObstacleField

│

└── ObstacleManager
```

World remains the owner of all environment state.

SimulationEngine should never own obstacles directly.

---

# 29. Simulation Lifecycle

Current

```
Aggregation

↓

UAV Motion

↓

Collision

↓

Metrics
```

Proposed

```
Obstacle Update

↓

Aggregation

↓

UAV Motion

↓

Collision Prediction

↓

Collision Resolution

↓

Exploration Update

↓

Metrics

↓

Rendering
```

This ordering ensures UAVs react to the latest obstacle state.

---

# 30. Sequence Diagram

Each simulation step becomes

```
SimulationEngine

↓

ObstacleManager.update(dt)

↓

DynamicObstacle.update(dt)

↓

Aggregation.update()

↓

UAV.update()

↓

CollisionPrediction()

↓

World.resolveCollisions()

↓

ExplorationMap.markExplored()

↓

Dynamic Metrics

↓

Renderer.drawFrame()
```

Every object performs one clearly defined responsibility.

---

# 31. Dependency Rules

To prevent future architectural drift, the following dependency rules shall be enforced.

DynamicObstacle

MUST NOT import

- UAV
- Aggregation
- SimulationEngine

ObstacleManager

MUST NOT import

- Renderer
- Metrics

SimulationEngine

MUST NOT implement

- obstacle motion

World

MUST NOT implement

- motion models

Renderer

MUST NOT modify simulation state

These rules preserve clean architecture boundaries.

---

# 32. Architecture Summary

The proposed architecture introduces a dedicated Dynamic Environment subsystem while preserving the existing DEBS implementation.

The extension is intentionally modular, backward compatible, and open to future research extensions such as predictive collision avoidance, adaptive path planning, cooperative obstacle negotiation, and heterogeneous environments.

By introducing DynamicObstacle and ObstacleManager as independent components owned by the World, the repository evolves from a static exploration simulator into a scalable dynamic environment research platform without disrupting the existing exploration algorithms.

---

# 33. Dynamic Collision Prediction

## Overview

The current DEBS implementation resolves collisions only after a UAV reaches an obstacle boundary. This reactive approach is sufficient for static environments but is inadequate when obstacles move over time.

The Dynamic Environment Extension introduces a predictive collision model that estimates future interactions between UAVs and dynamic obstacles before physical overlap occurs.

This prediction layer provides the foundation for future collision avoidance algorithms while remaining independent of the current BSA exploration strategy.

---

## Collision Prediction Model

For every simulation timestep, each UAV is evaluated against every dynamic obstacle.

Given:

- UAV position: \(p_u(t)\)
- UAV velocity: \(v_u(t)\)
- Obstacle position: \(p_o(t)\)
- Obstacle velocity: \(v_o(t)\)

Predict future positions over a prediction horizon \(\tau\):

\[
p_u(t+\tau)=p_u(t)+v_u\tau
\]

\[
p_o(t+\tau)=p_o(t)+v_o\tau
\]

Compute the predicted separation distance:

\[
d=\|p_u(t+\tau)-p_o(t+\tau)\|
\]

Three conditions are defined:

### Safe

\[
d > r_{safe}
\]

No interaction occurs.

---

### Near Miss

\[
r_{collision}<d<r_{safe}
\]

A potentially hazardous interaction occurs without physical contact.

---

### Collision

\[
d\le r_{collision}
\]

The UAV and obstacle occupy overlapping safety regions.

---

# 34. Dynamic Environment Metrics

The original repository measures exploration quality.

The Dynamic Environment Extension introduces metrics that evaluate environmental complexity and swarm robustness.

These metrics are intentionally separated from exploration metrics to preserve modularity.

Implementation file:

```
src/evaluation/dynamic_environment_metrics.py
```

---

## 34.1 Collision Count

Definition

Total number of physical collisions occurring during mission execution.

Mathematical definition

\[
CollisionCount=\sum_{t=0}^{T}I(d(t)\le r_c)
\]

where

- \(I(\cdot)\) is the indicator function
- \(r_c\) is the collision radius

Purpose

Measures navigation robustness.

---

## 34.2 Near Miss Count

Definition

Number of interactions entering the safety margin without collision.

Condition

\[
r_c<d<r_s
\]

where

- \(r_s\) = safety radius

Purpose

Measures collision risk.

---

## 34.3 Obstacle Encounter Count

Definition

Number of unique obstacle interactions experienced by the UAV fleet.

Repeated interaction with the same obstacle within a configurable cooldown interval counts as a single encounter.

Purpose

Measures environmental complexity.

---

## 34.4 Coverage Degradation

Definition

Reduction in exploration efficiency caused by dynamic obstacles.

\[
CoverageLoss=
Coverage_{Static}
-
Coverage_{Dynamic}
\]

Purpose

Quantifies the impact of environmental motion.

---

## 34.5 Mission Completion Time

Definition

Elapsed simulation time until exploration reaches the termination criterion.

Purpose

Measures mission efficiency.

---

## 34.6 Blocked Path Events

Definition

Number of planned UAV trajectories interrupted by moving obstacles.

Purpose

Quantifies environmental interference.

---

## 34.7 Average Obstacle Speed

Definition

Mean speed of all dynamic obstacles.

\[
\bar v
=
\frac1N
\sum_{i=1}^{N}v_i
\]

Purpose

Used for experiment comparison.

---

# 35. Experiment Design

The Dynamic Environment Extension introduces five standardized experiment scenarios.

All experiments use identical UAV parameters.

Only the environment changes.

---

## Scenario A — Static Environment

Purpose

Baseline comparison.

Obstacle behavior

```
No motion
```

Expected outcome

Highest coverage

Lowest mission time

Zero dynamic interactions

---

## Scenario B — Slow Obstacles

Condition

\[
v_{obstacle}<v_{uav}
\]

Example

```
Obstacle speed = 0.5 m/s

UAV speed = 1.5 m/s
```

Expected behavior

Minor trajectory interruptions.

---

## Scenario C — Equal-Speed Obstacles

Condition

\[
v_{obstacle}\approx v_{uav}
\]

Expected behavior

Maximum interaction frequency.

Expected to produce the largest number of near misses.

---

## Scenario D — Fast Obstacles

Condition

\[
v_{obstacle}>v_{uav}
\]

Expected behavior

Frequent path blocking.

Longer mission completion time.

---

## Scenario E — Mixed Environment

Contains

- slow obstacles
- equal-speed obstacles
- fast obstacles

Purpose

Approximate realistic environments.

---

# 36. Expected Outputs

Each experiment produces

```
CSV

Plots

Simulation screenshots

Performance summary
```

Generated files

```
experiments/results/

coverage.csv

dynamic_metrics.csv

collision_summary.csv

mission_statistics.csv
```

Plots

```
Coverage vs Time

Mission Time

Collision Count

Near Miss Count

Coverage Degradation

Obstacle Encounters
```

---

# 37. Visualization Design

The current renderer supports

- UAVs
- trails
- sensor range
- frontiers
- obstacles

The Dynamic Environment Extension expands visualization while preserving all existing layers.

---

## Static Obstacles

Color

```
Gray
```

---

## Dynamic Obstacles

### Slow

```
Green
```

---

### Equal Speed

```
Orange
```

---

### Fast

```
Red
```

---

## Velocity Vectors

Every moving obstacle displays

```
Position

↓

Velocity Arrow
```

Purpose

Visual verification of motion models.

---

## Optional Visualization Layers

The renderer should support optional overlays.

Examples

- predicted trajectory
- collision radius
- safety radius
- obstacle identifiers
- obstacle paths

These layers are intended for debugging and research demonstrations.

---

# 38. Testing Strategy

Testing is divided into three levels.

---

## Level 1 — Unit Tests

Test each motion model independently.

### LinearObstacle

Verify

```
position

=

position

+

velocity × dt
```

---

### WaypointObstacle

Verify

- waypoint switching
- path continuity

---

### RandomWalkObstacle

Verify

- heading update
- bounded motion

---

## Level 2 — Integration Tests

Verify

ObstacleManager

↓

World

↓

SimulationEngine

interaction.

Expected

No regression in existing exploration.

---

## Level 3 — Regression Tests

Run existing Paper 1 experiments.

With

```
Dynamic Environment Disabled
```

Expected result

Outputs must remain identical to the original repository.

This verifies backward compatibility.

---

# 39. Performance Requirements

The Dynamic Environment Extension shall satisfy the following constraints.

Obstacle update

```
O(number of dynamic obstacles)
```

Collision prediction

```
O(number of UAVs × number of obstacles)
```

Memory usage

Linear in obstacle count.

The extension shall not significantly affect the runtime of existing static simulations.

---

# 40. Failure Modes

Potential failure cases include

- obstacle overlap
- obstacle leaving world bounds
- waypoint oscillation
- infinite random walk drift
- excessive collision prediction cost
- invalid configuration parameters

Each failure mode shall be handled gracefully.

---

# 41. Backward Compatibility

Backward compatibility is a mandatory design requirement.

When

```
dynamic_environment.enabled = false
```

the simulator shall

- generate only static obstacles,
- bypass ObstacleManager updates,
- bypass dynamic metrics,
- preserve existing exploration metrics,
- produce behavior equivalent to the original implementation.

No existing experiments shall require modification.

---

# 42. Research Contributions

The Dynamic Environment Extension introduces the following research contributions.

- Dynamic obstacle framework for DEBS.
- Multiple obstacle motion models.
- Time-varying environment simulation.
- Predictive collision analysis.
- Dynamic exploration metrics.
- Standardized experimental scenarios.
- Visual validation framework.

These additions extend the original DEBS repository while maintaining compatibility with its decentralized exploration algorithm.

---

# 43. Summary

The Dynamic Environment Extension transforms the repository from a static exploration simulator into a flexible research platform capable of evaluating decentralized UAV exploration in dynamic environments.

By separating obstacle behavior, environment management, metrics, and visualization into dedicated modules, the extension remains modular, scalable, and suitable for future integration with IDE optimization, target search, and adaptive path planning.

---

# 44. Implementation Roadmap

The Dynamic Environment Extension shall be implemented incrementally to ensure repository stability, maintain backward compatibility, and minimize merge conflicts with concurrent development.

Implementation is divided into independent phases. Each phase should compile and execute successfully before proceeding to the next.

---

# Phase 1 — Dynamic Obstacle Framework

## Objective

Establish the object-oriented hierarchy for dynamic obstacles.

### New File

```
src/environment/dynamic_obstacles.py
```

### Classes

```
DynamicObstacle

↓

LinearObstacle

↓

WaypointObstacle

↓

RandomWalkObstacle
```

### Deliverables

- Base class
- Motion model implementations
- Unit tests for motion models
- Complete type hints
- Documentation

### Expected Outcome

A reusable framework for representing moving obstacles without affecting the existing simulation.

---

# Phase 2 — Obstacle Manager

## Objective

Centralize all obstacle lifecycle management.

### New File

```
src/environment/obstacle_manager.py
```

### Responsibilities

- Store dynamic obstacles
- Update obstacle states
- Spawn obstacles
- Remove obstacles
- Collision queries
- Future position prediction

### Deliverables

- ObstacleManager implementation
- Independent unit tests

---

# Phase 3 — Configuration Integration

## Objective

Introduce configurable dynamic environments.

### Files

```
configs/simulation.yaml

src/config/loader.py
```

### Deliverables

DynamicEnvironmentConfig

Configuration parsing

Scenario definitions

Validation

---

# Phase 4 — World Integration

## Objective

Integrate ObstacleManager into World ownership.

### Modified File

```
src/environment/world.py
```

### Deliverables

```
World

├── ObstacleField

└── ObstacleManager
```

The existing static obstacle implementation shall remain unchanged.

---

# Phase 5 — Simulation Engine Integration

## Objective

Introduce dynamic obstacle updates into the simulation lifecycle.

### Modified File

```
src/simulation/simulation_engine.py
```

Simulation order

```
Obstacle Update

↓

Aggregation

↓

UAV Update

↓

Collision Prediction

↓

Collision Resolution

↓

Map Update

↓

Metrics
```

---

# Phase 6 — Dynamic Metrics

## Objective

Implement quantitative evaluation of dynamic environments.

### New File

```
src/evaluation/dynamic_environment_metrics.py
```

Metrics

- Collision Count
- Near Miss Count
- Obstacle Encounters
- Coverage Degradation
- Mission Completion Time
- Blocked Path Events
- Average Obstacle Speed

---

# Phase 7 — Visualization

## Objective

Visualize moving obstacles.

### Modified File

```
src/visualization/pygame_renderer.py
```

Features

- Dynamic obstacle rendering
- Velocity vectors
- Motion visualization
- Color-coded obstacle classes

---

# Phase 8 — Experimental Evaluation

## Objective

Evaluate repository performance under dynamic environments.

Scenarios

- Static
- Slow
- Equal-Speed
- Fast
- Mixed

Outputs

CSV

Graphs

Screenshots

Research statistics

---

# 45. Git Branching Strategy

The Dynamic Environment Extension shall be developed exclusively on a dedicated feature branch.

```
main

│

└── feature/dynamic-environment
```

Direct commits to

```
main
```

are prohibited.

---

# 46. Commit Strategy

Each commit shall represent one logical implementation step.

Recommended sequence

```
Commit 1

Add dynamic obstacle hierarchy

--------------------------------

Commit 2

Implement obstacle motion models

--------------------------------

Commit 3

Implement obstacle manager

--------------------------------

Commit 4

Add dynamic environment configuration

--------------------------------

Commit 5

Integrate ObstacleManager into World

--------------------------------

Commit 6

Integrate simulation update loop

--------------------------------

Commit 7

Implement dynamic metrics

--------------------------------

Commit 8

Extend renderer

--------------------------------

Commit 9

Add experiment scenarios

--------------------------------

Commit 10

Documentation and cleanup
```

Every commit shall compile successfully.

---

# 47. Code Review Checklist

Before every commit verify

## Architecture

- Single Responsibility Principle
- No circular dependencies
- No duplicated logic
- Environment owns obstacle behavior

---

## Style

- Repository naming conventions
- Type hints
- Docstrings
- Black formatting
- Existing coding style

---

## Performance

- Minimal allocations
- No unnecessary loops
- Efficient NumPy usage

---

## Testing

Unit tests pass.

Static simulation remains unchanged.

Dynamic scenarios execute correctly.

---

# 48. Assignment Verification

The project assignment requires four questions to be answered before implementation.

---

## Question 1

### Identify environment classes.

Existing classes

```
World

ObstacleField

CircularObstacle

ExplorationMap
```

New classes

```
DynamicObstacle

LinearObstacle

WaypointObstacle

RandomWalkObstacle

ObstacleManager
```

---

## Question 2

### Explain where dynamic obstacles belong.

Dynamic obstacles belong exclusively to the Environment Layer.

Ownership

```
World

↓

ObstacleManager

↓

DynamicObstacle
```

No other module owns obstacle state.

---

## Question 3

### Explain how they affect UAV exploration.

Dynamic obstacles modify the environment over time.

Consequences

- blocked trajectories
- increased replanning
- delayed exploration
- collision risk
- reduced coverage
- longer missions

The BSA exploration algorithm itself remains unchanged.

---

## Question 4

### Wait for approval.

The implementation order defined in this document shall be reviewed before development begins.

Coding should proceed only after architectural approval.

---

# 49. Risk Assessment

Potential implementation risks

| Risk | Impact | Mitigation |
|------|----------|------------|
| Merge conflicts | Medium | Independent feature branch |
| Performance degradation | Medium | Efficient update loops |
| Regression | High | Static simulation regression tests |
| Excessive coupling | High | Strict ownership rules |
| Configuration errors | Low | YAML validation |
| Visualization slowdown | Medium | Optional rendering layers |

---

# 50. Future Research Opportunities

The Dynamic Environment Extension creates the foundation for several future research directions.

Examples include

- Dynamic path planning
- Online obstacle avoidance
- Predictive trajectory optimization
- Cooperative obstacle negotiation
- Multi-agent collision avoidance
- Learning-based obstacle prediction
- Dynamic frontier selection
- Adaptive sensing
- Risk-aware exploration
- Human–robot interaction

These topics are intentionally outside the scope of the current implementation.

---

# 51. Coding Standards

All implementation shall satisfy the following requirements.

Python

- Python 3.11+

Typing

- Complete type hints

Documentation

- Google-style or repository-style docstrings

Architecture

- Single Responsibility Principle
- Open–Closed Principle
- Composition over inheritance

Formatting

- Follow existing repository style

No placeholder implementations.

No unfinished TODO sections.

Every public method shall be documented.

---

# 52. Claude Code Development Workflow

Each implementation phase shall follow the same workflow.

```
Read Design Document

↓

Inspect Repository

↓

Explain Implementation Plan

↓

Generate Code

↓

Run Tests

↓

Review Architecture

↓

Commit

↓

Proceed to Next Phase
```

Claude Code shall never implement multiple phases simultaneously.

Each phase must be independently reviewable.

---

# 53. Definition of Done

The Dynamic Environment Extension shall be considered complete when the following conditions are satisfied.

### Functional

- Dynamic obstacles implemented
- Motion models implemented
- ObstacleManager operational
- Simulation integration complete
- Metrics implemented
- Visualization complete
- Experiment scenarios implemented

### Quality

- Unit tests pass
- Static simulation unchanged
- Dynamic scenarios execute successfully
- Documentation complete

### Research

- CSV results generated
- Experimental figures generated
- Quantitative comparison performed

---

# 54. Conclusion

This Software Design Specification defines a modular, scalable, and backward-compatible extension of the DEBS exploration repository from static environments to dynamic environments.

The proposed architecture preserves the integrity of the original exploration framework while introducing dynamic obstacle modeling, predictive collision analysis, environment-specific evaluation metrics, configurable experiment scenarios, and enhanced visualization capabilities.

By isolating all new functionality within the Environment Layer and integrating it through well-defined ownership boundaries, the extension minimizes architectural risk, simplifies future maintenance, and enables seamless collaboration with parallel development efforts on IDE optimization and Target Search.

The resulting system forms a robust research platform capable of supporting future investigations into autonomous exploration under time-varying environmental conditions.

---

# Appendix A — Repository Ownership

| Module | Owner |
|---------|-------|
| IDE Optimization | Member 1 |
| Dynamic Environment | Member 2 |
| Target Search | Member 3 |

---

# Appendix B — Repository Files Modified

Modified

```
world.py

simulation_engine.py

loader.py

simulation.yaml

pygame_renderer.py
```

New

```
dynamic_obstacles.py

obstacle_manager.py

dynamic_environment_metrics.py
```

---

# Appendix C — Development Principles

1. Preserve existing functionality.
2. Extend rather than rewrite.
3. Minimize modifications to existing modules.
4. Prefer composition over inheritance.
5. Maintain backward compatibility.
6. Keep research modules independent.
7. Ensure every phase is independently testable.
8. Document all public interfaces.
9. Validate through experiments.
10. Keep the repository maintainable for future contributors.