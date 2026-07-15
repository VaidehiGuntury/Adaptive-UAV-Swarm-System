"""Unit tests for GreedyFrontierAllocator (Phase H baseline)."""

from __future__ import annotations

import unittest

import numpy as np

from src.agents.uav import UAV
from src.algorithms.baselines.greedy_frontier import GreedyFrontierAllocator
from src.config.loader import AggregationConfig, UAVConfig
from src.environment.obstacles import ObstacleField
from src.environment.world import World


def _make_allocator() -> GreedyFrontierAllocator:
    aggregation_config = AggregationConfig(
        d_c=0.5,
        d_0=17.8,
        k_a=1.0,
        turn_cost_weight=1.0,
        trail_penalty=8.0,
        cluster_penalty_weight=2.0,
        turn_penalty_weight=0.5,
        trail_penalty_weight=1.0,
        candidates_per_frontier=6,
        mission_region_radius=17.8,
        replan_interval=2.0,
    )
    uav_config = UAVConfig(
        max_speed=1.5,
        max_angular_velocity=0.9,
        sensing_range=4.5,
        initial_spread_radius=20.0,
        spawn_mode="ring",
        spawn_angular_noise=0.15,
    )
    return GreedyFrontierAllocator(config=aggregation_config, uav_config=uav_config)


class TestGreedyFrontierAllocatorIndependentAssignment(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            width=30.0, height=30.0, obstacles=ObstacleField([]), map_resolution=1.0
        )
        # Two disjoint explored blobs, far apart, each with its own
        # frontier ring — no coordination should make either UAV
        # aware of the other blob's frontier.
        self.blob_a_center = np.array([7.5, 7.5])
        self.blob_b_center = np.array([22.5, 22.5])
        self.world.map.mark_explored(self.blob_a_center, radius=3.0)
        self.world.map.mark_explored(self.blob_b_center, radius=3.0)

        self.agent_a = UAV(agent_id=0, position=self.blob_a_center.copy())
        self.agent_b = UAV(agent_id=1, position=self.blob_b_center.copy())
        self.allocator = _make_allocator()

    def test_each_agent_gets_its_own_nearest_frontier(self) -> None:
        self.allocator.update(self.agent_a, [self.agent_a, self.agent_b], self.world, dt=0.1)
        self.allocator.update(self.agent_b, [self.agent_a, self.agent_b], self.world, dt=0.1)

        self.assertIsNotNone(self.agent_a.assigned_target)
        self.assertIsNotNone(self.agent_b.assigned_target)

        dist_a_to_blob_a = float(np.linalg.norm(self.agent_a.assigned_target - self.blob_a_center))
        dist_a_to_blob_b = float(np.linalg.norm(self.agent_a.assigned_target - self.blob_b_center))
        self.assertLess(dist_a_to_blob_a, dist_a_to_blob_b)

        dist_b_to_blob_b = float(np.linalg.norm(self.agent_b.assigned_target - self.blob_b_center))
        dist_b_to_blob_a = float(np.linalg.norm(self.agent_b.assigned_target - self.blob_a_center))
        self.assertLess(dist_b_to_blob_b, dist_b_to_blob_a)

        # No coordination/deduplication: independently nearest picks land
        # in different places given the disjoint geometry above.
        self.assertFalse(np.array_equal(self.agent_a.assigned_target, self.agent_b.assigned_target))

    def test_stub_interface_members(self) -> None:
        """replan_region_history / step_reassignment_count are documented
        stubs (no region/replan-count concept for a naive allocator)."""
        self.allocator.begin_step()  # must not raise
        self.assertEqual(self.allocator.replan_region_history, [])
        self.assertEqual(self.allocator.step_reassignment_count, 0)


class TestGreedyFrontierAllocatorFullyExplored(unittest.TestCase):
    def setUp(self) -> None:
        self.world = World(
            width=5.0, height=5.0, obstacles=ObstacleField([]), map_resolution=1.0
        )
        # Radius far exceeds the world diagonal, so every free cell
        # becomes explored and no frontier cells remain.
        self.world.map.mark_explored(np.array([2.5, 2.5]), radius=20.0)
        self.assertFalse(
            self.world.map.frontier_mask().any(),
            "test fixture is not actually fully explored",
        )
        self.agent = UAV(agent_id=0, position=np.array([2.5, 2.5]))
        self.allocator = _make_allocator()

    def test_holds_position_instead_of_crashing(self) -> None:
        self.allocator.update(self.agent, [self.agent], self.world, dt=0.1)

        self.assertIsNotNone(self.agent.assigned_target)
        np.testing.assert_allclose(self.agent.assigned_target, self.agent.position)

    def test_stays_stable_across_repeated_replans(self) -> None:
        # Step past several replan cycles — must keep holding, not raise.
        for _ in range(50):
            self.allocator.update(self.agent, [self.agent], self.world, dt=0.1)
        np.testing.assert_allclose(self.agent.assigned_target, self.agent.position)


if __name__ == "__main__":
    unittest.main()
