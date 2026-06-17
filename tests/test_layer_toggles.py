"""Unit tests for visualization layer toggle state."""

from __future__ import annotations

import unittest

try:
    import pygame
except ImportError:
    pygame = None  # type: ignore[assignment]

from src.visualization.layer_toggles import LayerToggles

PYGAME_AVAILABLE = pygame is not None


class TestLayerToggles(unittest.TestCase):
    def test_default_states(self) -> None:
        toggles = LayerToggles()
        self.assertFalse(toggles.show_grid)
        self.assertTrue(toggles.show_frontiers)
        self.assertTrue(toggles.show_trails)
        self.assertTrue(toggles.show_velocity)
        self.assertTrue(toggles.show_targets)
        self.assertTrue(toggles.show_sensor_radius)


@unittest.skipUnless(PYGAME_AVAILABLE, "pygame is not installed")
class TestLayerToggleKeys(unittest.TestCase):
    def test_toggle_keys_flip_flags(self) -> None:
        toggles = LayerToggles()
        self.assertTrue(toggles.handle_key(pygame.K_g, pygame))
        self.assertTrue(toggles.show_grid)
        self.assertTrue(toggles.handle_key(pygame.K_g, pygame))
        self.assertFalse(toggles.show_grid)

    def test_unknown_key_not_handled(self) -> None:
        toggles = LayerToggles()
        self.assertFalse(toggles.handle_key(pygame.K_z, pygame))


if __name__ == "__main__":
    unittest.main()
