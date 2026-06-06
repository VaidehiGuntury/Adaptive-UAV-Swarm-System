"""Unit tests for environment module."""

from __future__ import annotations

import unittest

import numpy as np

from src.config.loader import load_config
from src.environment.obstacles import CircularObstacle, ObstacleField, generate_obstacles
from src.environment.world import World


class TestObstacleGeneration(unittest.TestCase):
    def test_generates_requested_count_with_seed(self) -> None:
        field = generate_obstacles(
            count=5,
            width=50.0,
            height=50.0,
            min_radius=1.0,
            max_radius=2.0,
            seed=7,
        )
        self.assertEqual(len(field.obstacles), 5)

    def test_obstacles_within_bounds(self) -> None:
        width, height = 40.0, 30.0
        field = generate_obstacles(
            count=10,
            width=width,
            height=height,
            min_radius=0.5,
            max_radius=1.5,
            seed=1,
        )
        for obstacle in field.obstacles:
            self.assertGreaterEqual(obstacle.center[0], 0.0)
            self.assertGreaterEqual(obstacle.center[1], 0.0)
            self.assertLessEqual(obstacle.center[0], width)
            self.assertLessEqual(obstacle.center[1], height)

    def test_collision_detection(self) -> None:
        obstacle = CircularObstacle(center=np.array([5.0, 5.0]), radius=2.0)
        field = ObstacleField([obstacle])
        inside = np.array([5.0, 5.0])
        outside = np.array([10.0, 10.0])
        self.assertTrue(field.is_collision(inside))
        self.assertFalse(field.is_collision(outside))


class TestWorldAndMap(unittest.TestCase):
    def setUp(self) -> None:
        config_path = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "configs"
            / "simulation.yaml"
        )
        self.config = load_config(config_path)
        self.world = World.from_config(self.config.environment, self.config.uav)

    def test_world_dimensions(self) -> None:
        self.assertEqual(self.world.width, self.config.environment.width)
        self.assertEqual(self.world.height, self.config.environment.height)

    def test_explored_fraction_increases(self) -> None:
        start = self.world.map.explored_fraction()
        self.world.map.mark_explored(
            np.array([self.world.width / 2, self.world.height / 2]),
            radius=self.config.uav.sensing_range,
        )
        end = self.world.map.explored_fraction()
        self.assertGreater(end, start)

    def test_clip_and_resolve(self) -> None:
        point = np.array([-5.0, self.world.height + 5.0])
        clipped = self.world.clip_position(point)
        self.assertGreaterEqual(clipped[0], 0.0)
        self.assertLessEqual(clipped[1], self.world.height)

    def test_world_optional_fields_default_empty(self) -> None:
        self.assertIsNone(self.world.communication_graph)
        self.assertIsNone(self.world.belief_map)
        self.assertEqual(self.world.target_regions, [])
        self.assertEqual(self.world.formation_specs, [])


if __name__ == "__main__":
    unittest.main()
