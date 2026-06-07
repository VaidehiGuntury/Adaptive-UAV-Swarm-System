"""Smoke tests for Pygame renderer (headless when pygame is available)."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

# Headless SDL driver for CI / environments without a display.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine

PYGAME_AVAILABLE = pygame is not None


@unittest.skipUnless(PYGAME_AVAILABLE, "pygame is not installed")
class TestPygameRenderer(unittest.TestCase):
    def setUp(self) -> None:
        from src.visualization.pygame_renderer import PygameRenderer

        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        self.config = replace(load_config(config_path), duration=0.5, num_uavs=3)
        self.world = World.from_config(self.config.environment, self.config.uav)
        center = np.array([self.config.spawn_center_x, self.config.spawn_center_y])
        agents = spawn_uavs(
            count=self.config.num_uavs,
            center=center,
            spread_radius=self.config.uav.initial_spread_radius,
            mission_radius=self.config.aggregation.mission_region_radius,
            max_speed=self.config.uav.max_speed,
            max_angular_velocity=self.config.uav.max_angular_velocity,
            seed=0,
        )
        aggregation = SelfAggregationController(
            config=self.config.aggregation,
            uav_config=self.config.uav,
            rng=np.random.default_rng(0),
        )
        self.engine = SimulationEngine(self.world, agents, aggregation, self.config)
        self.PygameRenderer = PygameRenderer
        self.renderer = PygameRenderer(self.world, self.engine, screen_width=400, screen_height=400)

    def tearDown(self) -> None:
        if pygame is not None:
            pygame.quit()

    def test_draw_frame_headless(self) -> None:
        pygame.init()
        surface = pygame.Surface((400, 400))
        self.engine.step()
        state = self.engine.get_state()
        self.renderer.draw_frame(state, surface=surface)
        self.assertEqual(surface.get_size(), (400, 400))

    def test_run_and_record_preserves_history(self) -> None:
        history = self.renderer.run_and_record()
        self.assertGreater(len(history), 1)

    def test_draw_frame_all_layers_enabled(self) -> None:
        pygame.init()
        surface = pygame.Surface((800, 800))
        self.renderer.toggles.show_grid = True
        for _ in range(3):
            self.engine.step()
        state = self.engine.get_state()
        self.renderer.draw_frame(state, surface=surface, frame_index=3)

    def test_draw_frame_all_layers_disabled(self) -> None:
        pygame.init()
        surface = pygame.Surface((800, 800))
        self.renderer.toggles.show_grid = False
        self.renderer.toggles.show_frontiers = False
        self.renderer.toggles.show_trails = False
        self.renderer.toggles.show_velocity = False
        self.renderer.toggles.show_targets = False
        self.renderer.toggles.show_sensor_radius = False
        self.engine.step()
        state = self.engine.get_state()
        self.renderer.draw_frame(state, surface=surface)


if __name__ == "__main__":
    unittest.main()
