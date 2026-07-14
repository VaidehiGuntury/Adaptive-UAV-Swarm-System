"""
Regression test for the resolve_collisions / check_collision margin mismatch.

Bug: World.resolve_collisions used a hardcoded margin (0.3) to push agents
clear of dynamic obstacles, while ObstacleManager classified collisions at
distance <= collision_radius (0.35 by default, SDS §33). Since 0.3 < 0.35,
every "successfully avoided" position still registered as colliding.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.environment.dynamic_obstacles import LinearObstacle
from src.environment.obstacle_manager import ObstacleManager
from src.environment.obstacles import ObstacleField
from src.environment.world import World


class TestDynamicCollisionResolutionMargin(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ObstacleManager(collision_radius=0.35, safety_margin=0.75)
        self.manager.add_obstacle(
            LinearObstacle(
                obstacle_id="blocker",
                position=np.array([50.0, 50.0]),
                velocity=np.array([0.0, 0.0]),
                radius=0.5,
                world_bounds=(100.0, 100.0),
            )
        )
        self.world = World(
            width=100.0,
            height=100.0,
            obstacles=ObstacleField([]),
            obstacle_manager=self.manager,
        )

    def test_resolved_position_is_not_colliding(self) -> None:
        # Deep inside the obstacle, forcing resolve_collisions to act.
        agent_position = np.array([50.05, 50.0])
        resolved = self.world.resolve_collisions(agent_position)

        result = self.manager.check_collision(resolved)

        self.assertFalse(
            result.colliding,
            f"resolve_collisions produced a position still classified as "
            f"colliding (distance={result.distance:.4f}, "
            f"collision_radius={self.manager.collision_radius})",
        )

    def test_resolved_position_clears_collision_radius(self) -> None:
        agent_position = np.array([50.0, 50.0])
        resolved = self.world.resolve_collisions(agent_position)

        result = self.manager.check_collision(resolved)

        self.assertGreater(result.distance, self.manager.collision_radius)


if __name__ == "__main__":
    unittest.main()
