"""
MissionPhase and MissionOrchestrator — top-level mission lifecycle manager.

MissionPhase
------------
EXPLORING       — BSA exploration is active; DetectionSystem runs passively
SEARCH_TRANSITION — exploration done; spawning targets, computing assignments
SEARCHING       — SearchController is active; BSA is disabled
COMPLETED       — all targets resolved; mission over

MissionOrchestrator
-------------------
Monitors exploration completion condition and drives phase transitions.
Called by SimulationEngine.step() each tick.

Integration
-----------
- SimulationEngine holds a MissionOrchestrator instance
- Engine calls orchestrator.step() after each exploration tick
- When phase transitions, engine switches from BSA update to SearchController update
- BSA and SearchController are NEVER active in the same tick
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from src.agents.uav import UAV
from src.config.search_config import SearchConfig
from src.environment.world import World
from src.search.assignment import SearchAssigner
from src.search.detection import DetectionSystem
from src.search.prioritization import PriorityScorer
from src.search.search_controller import SearchController
from src.search.target_manager import TargetManager
from src.search.tracker import TargetTracker


class MissionPhase(Enum):
    """Top-level mission phase discriminator."""

    EXPLORING = auto()
    SEARCH_TRANSITION = auto()
    SEARCHING = auto()
    COMPLETED = auto()


class MissionOrchestrator:
    """
    Drives mission phase transitions and coordinates all search subsystems.

    Parameters
    ----------
    config : SearchConfig
        Full search configuration.
    world : World
        Shared world container.
    agents : list[UAV]
        The UAV fleet.
    rng : np.random.Generator
    """

    def __init__(
        self,
        config: SearchConfig,
        world: World,
        agents: list[UAV],
        rng: np.random.Generator | None = None,
    ) -> None:
        self._config = config
        self._world = world
        self._agents = agents
        self._rng = rng or np.random.default_rng()
        self.phase: MissionPhase = MissionPhase.EXPLORING

        # Subsystems — instantiated here, used by engine
        self.target_manager = TargetManager(
            config=config,
            world_width=world.width,
            world_height=world.height,
        )
        self.detection_system = DetectionSystem(
            config=config.detection,
            uav_sensing_range=self._infer_sensing_range(agents),
        )
        self.priority_scorer = PriorityScorer(config.priority)
        self.assigner = SearchAssigner(config.assignment)
        self.tracker = TargetTracker(config.tracking, config.behaviour)
        self.search_controller = SearchController(
            config=config.behaviour,
            tracker=self.tracker,
            rng=self._rng,
            world_width=world.width,
            world_height=world.height,
            num_uavs=len(agents),
        )

        # Attach target_manager to world so renderers can access it
        world.target_manager = self.target_manager  # type: ignore[attr-defined]

        # Metrics
        self.search_start_time: float | None = None
        self.mission_complete_time: float | None = None
        self.phase_transition_coverage: float = 0.0

    # ------------------------------------------------------------------
    # Main tick interface (called by SimulationEngine.step())
    # ------------------------------------------------------------------

    def step(
        self,
        current_time: float,
        dt: float,
    ) -> MissionPhase:
        """
        Advance the mission orchestrator by one timestep.

        Returns the current MissionPhase after processing.
        Called AFTER agent kinematics have been updated.
        """
        if self.phase == MissionPhase.EXPLORING:
            self._step_exploring(current_time, dt)

        elif self.phase == MissionPhase.SEARCH_TRANSITION:
            self._step_transition(current_time, dt)

        elif self.phase == MissionPhase.SEARCHING:
            self._step_searching(current_time, dt)

        elif self.phase == MissionPhase.COMPLETED:
            pass

        return self.phase

    # ------------------------------------------------------------------
    # Phase step handlers
    # ------------------------------------------------------------------

    def _step_exploring(self, current_time: float, dt: float) -> None:
        """
        During exploration: run passive detection scan only.
        Check if exploration completion threshold is met.
        """
        # Passive detection during exploration (targets not yet spawned,
        # so this is a no-op until targets are spawned in transition)
        explored = self._world.map.explored_fraction()
        threshold = self._config.mission.exploration_completion_threshold
        min_coverage = self._config.mission.min_coverage_before_search

        if explored >= threshold and explored >= min_coverage:
            self.phase_transition_coverage = explored
            self.phase = MissionPhase.SEARCH_TRANSITION

    def _step_transition(self, current_time: float, dt: float) -> None:
        """
        One-time setup: spawn targets, score priorities, assign to UAVs.

        This runs for exactly one tick then transitions to SEARCHING.
        """
        self.search_start_time = current_time

        # 1. Spawn targets
        self.target_manager.spawn_targets(current_time)

        # 2. Run initial detection scan (UAVs may already be on top of targets)
        self.detection_system.scan_all(self._agents, self.target_manager, current_time)

        # 3. Score priorities for all detected targets
        fleet_centroid = np.mean(
            [a.position for a in self._agents], axis=0
        ).astype(np.float64)
        for target in self.target_manager.detected_targets():
            self.priority_scorer.score(
                target, fleet_centroid, current_time,
                max_distance=max(self._world.width, self._world.height),
            )

        # 4. Assign targets to UAVs
        assignments = self.assigner.assign(
            self._agents, self.target_manager, current_time
        )

        # 5. Initialize SearchController for each agent
        for agent in self._agents:
            self.search_controller.initialize_agent(agent)

        # 6. Assign targets to SearchController
        for uav_id, target_ids in assignments.items():
            agent = next((a for a in self._agents if a.agent_id == uav_id), None)
            if agent is None:
                continue
            for tid in target_ids:
                target = self.target_manager.get_target(tid)
                if target is not None:
                    self.search_controller.assign_target(agent, target, self._world)
                    self.target_manager.begin_searching(tid)

        self.phase = MissionPhase.SEARCHING

    def _step_searching(self, current_time: float, dt: float) -> None:
        """
        Main search phase tick:
        1. Update target positions (dynamic targets move)
        2. Scan visibility (detection system)
        3. Update tracker
        4. Update search controllers for each agent
        5. Check mission completion
        6. Check search timeout
        """
        # 1. Move dynamic targets
        self.target_manager.update(dt, current_time)

        # 2. Full detection scan (finds UNDISCOVERED targets + refreshes known ones)
        self.detection_system.scan_all(
            self._agents, self.target_manager, current_time
        )

        # 3. Scan visibility for tracking updates
        visibility_map = self.detection_system.update_tracking_observations(
            self._agents, self.target_manager, current_time
        )

        # 3. Update tracker
        tracker_events = self.tracker.update(
            self.target_manager, visibility_map, current_time
        )

        # 4. Re-score priorities periodically and handle new detections
        fleet_centroid = np.mean(
            [a.position for a in self._agents], axis=0
        ).astype(np.float64)

        # Assign newly detected (but unassigned) targets
        newly_detected = self.target_manager.detected_targets()
        if newly_detected:
            for target in newly_detected:
                self.priority_scorer.score(
                    target, fleet_centroid, current_time,
                    max_distance=max(self._world.width, self._world.height),
                )
            new_assignments = self.assigner.assign(
                self._agents, self.target_manager, current_time
            )
            for uav_id, target_ids in new_assignments.items():
                agent = next((a for a in self._agents if a.agent_id == uav_id), None)
                if agent is None:
                    continue
                for tid in target_ids:
                    target = self.target_manager.get_target(tid)
                    if target is not None:
                        self.search_controller.assign_target(agent, target, self._world)
                        self.target_manager.begin_searching(tid)

        # 5. Check for UAV failures and rebalance
        for agent in self._agents:
            self.assigner.update_agent_position(
                agent.agent_id, agent.position, current_time
            )
        self.assigner.rebalance(self._agents, self.target_manager, current_time)

        # 6. Update each agent's search controller
        for agent in self._agents:
            self.search_controller.update(
                agent=agent,
                target_manager=self.target_manager,
                tracker_events=tracker_events,
                visibility_map=visibility_map,
                world=self._world,
                dt=dt,
            )

        # 7. Release completed targets from assigner
        for target in self.target_manager.completed_targets():
            self.assigner.release_target(target.target_id, self.target_manager)

        # 8. Check mission completion
        if self.target_manager.all_missions_complete():
            self.phase = MissionPhase.COMPLETED
            self.mission_complete_time = current_time
            return

        # 9. Check search timeout
        if self.search_start_time is not None:
            elapsed = current_time - self.search_start_time
            if elapsed >= self._config.mission.search_timeout_s:
                self.phase = MissionPhase.COMPLETED
                self.mission_complete_time = current_time

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_sensing_range(agents: list[UAV]) -> float:
        """Infer sensing range from agent max_speed as fallback."""
        if agents:
            return getattr(agents[0], "max_speed", 4.5) * 3.0
        return 4.5
