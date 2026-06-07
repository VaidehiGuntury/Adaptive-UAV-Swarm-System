"""Unit tests for observational exploration metrics."""

from __future__ import annotations

import unittest

import numpy as np

from src.agents.uav import UAV, spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.evaluation.exploration_metrics import (
    frontier_reuse_frequency,
    mean_target_separation,
    revisit_ratio,
)


class TestExplorationMetrics(unittest.TestCase):
    def test_mean_target_separation_ring_spawn(self) -> None:
        config_path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "configs"
            / "simulation.yaml"
        )
        config = load_config(config_path)
        center = np.array([50.0, 50.0], dtype=np.float64)
        agents = spawn_uavs(
            count=10,
            center=center,
            spread_radius=20.0,
            mission_radius=4.5,
            max_speed=1.5,
            max_angular_velocity=0.9,
            seed=0,
            spawn_mode="ring",
        )
        separation = mean_target_separation(agents)
        self.assertGreater(separation, 15.0)

    def test_frontier_reuse_frequency(self) -> None:
        self.assertEqual(frontier_reuse_frequency([]), 0.0)
        self.assertEqual(frontier_reuse_frequency([(1, 2), (3, 4)]), 0.0)
        self.assertAlmostEqual(
            frontier_reuse_frequency([(1, 2), (1, 2), (3, 4)]),
            1.0 / 3.0,
        )
        self.assertAlmostEqual(
            frontier_reuse_frequency([(1, 2), (3, 4), (1, 2)]),
            1.0 / 3.0,
        )

    def test_revisit_ratio(self) -> None:
        config_path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "configs"
            / "simulation.yaml"
        )
        config = load_config(config_path)
        world = World.from_config(config.environment, config.uav)
        p0 = np.array([10.0, 10.0], dtype=np.float64)
        p1 = np.array([11.0, 10.0], dtype=np.float64)
        histories = {0: [p0, p1, p0]}
        self.assertGreater(revisit_ratio(histories, world.map), 0.0)

    def test_set_target_does_not_move_allocation(self) -> None:
        agent = UAV(agent_id=0, position=np.array([0.0, 0.0]))
        allocation = np.array([5.0, 0.0])
        agent.set_region(allocation, radius=4.5)
        agent.set_target(np.array([20.0, 0.0]))
        np.testing.assert_array_equal(agent.assigned_region.center, allocation)


if __name__ == "__main__":
    unittest.main()
