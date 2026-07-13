"""Regression test for the multi-target assignment/actuation fix.

A UAV assigned two targets (max_targets_per_uav > 1) must not have its
active TRACKING interrupted by the second assignment, and must pick up
the queued target once the first is done (completed or permanently
lost) rather than going inert forever.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.config.search_config import SearchBehaviourConfig
from src.environment.world import World
from src.search.search_controller import AgentSearchPhase, SearchController
from src.search.target import Target, TargetStatus, TargetType
from src.search.target_manager import TargetManager


class TestMultiTargetAssignment(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        config = load_config(config_path)

        self.world = World.from_config(config.environment, config.uav)
        self.agent = spawn_uavs(
            count=1,
            center=np.array([50.0, 50.0]),
            spread_radius=5.0,
            mission_radius=config.aggregation.mission_region_radius,
            max_speed=config.uav.max_speed,
            max_angular_velocity=config.uav.max_angular_velocity,
            seed=1,
        )[0]

        self.target_manager = TargetManager(
            config=config.search, world_width=self.world.width, world_height=self.world.height
        )
        self.target_a = Target(
            target_id=0,
            target_type=TargetType.STATIC,
            position=np.array([52.0, 50.0]),
            status=TargetStatus.ASSIGNED,
            assigned_uav=self.agent.agent_id,
        )
        self.target_b = Target(
            target_id=1,
            target_type=TargetType.STATIC,
            position=np.array([48.0, 50.0]),
            status=TargetStatus.ASSIGNED,
            assigned_uav=self.agent.agent_id,
        )
        self.target_manager._targets[0] = self.target_a
        self.target_manager._targets[1] = self.target_b

        self.controller = SearchController(
            config=SearchBehaviourConfig(), tracker=None, rng=np.random.default_rng(0)
        )
        self.controller.initialize_agent(self.agent)

    def test_second_assignment_does_not_interrupt_active_tracking(self) -> None:
        self.controller.assign_target(self.agent, self.target_a, self.world)
        state = self.agent.search_state
        self.assertEqual(state.assigned_target_id, self.target_a.target_id)

        # Simulate target_a having become actively tracked.
        state.phase = AgentSearchPhase.TRACKING

        self.controller.assign_target(self.agent, self.target_b, self.world)

        self.assertEqual(state.assigned_target_id, self.target_a.target_id)
        self.assertEqual(state.phase, AgentSearchPhase.TRACKING)
        self.assertIn(self.target_b.target_id, state.assigned_target_ids)
        self.assertIn(self.target_a.target_id, state.assigned_target_ids)

    def test_picks_up_queued_target_after_first_completes(self) -> None:
        self.controller.assign_target(self.agent, self.target_a, self.world)
        state = self.agent.search_state
        state.phase = AgentSearchPhase.TRACKING
        self.controller.assign_target(self.agent, self.target_b, self.world)

        self.controller.update(
            agent=self.agent,
            target_manager=self.target_manager,
            tracker_events={self.target_a.target_id: "completed"},
            visibility_map={},
            world=self.world,
            dt=0.1,
        )

        self.assertEqual(state.assigned_target_id, self.target_b.target_id)
        self.assertEqual(state.phase, AgentSearchPhase.NAVIGATING_TO_SEARCH)
        self.assertNotIn(self.target_a.target_id, state.assigned_target_ids)
        self.assertIn(self.target_b.target_id, state.assigned_target_ids)

    def test_goes_completed_when_queue_empties(self) -> None:
        self.controller.assign_target(self.agent, self.target_a, self.world)
        state = self.agent.search_state
        state.phase = AgentSearchPhase.TRACKING

        self.controller.update(
            agent=self.agent,
            target_manager=self.target_manager,
            tracker_events={self.target_a.target_id: "completed"},
            visibility_map={},
            world=self.world,
            dt=0.1,
        )

        self.assertEqual(state.phase, AgentSearchPhase.COMPLETED)
        self.assertIsNone(state.assigned_target_id)
        self.assertEqual(state.assigned_target_ids, [])


if __name__ == "__main__":
    unittest.main()
