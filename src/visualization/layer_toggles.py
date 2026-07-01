"""
Keyboard-toggle state for Pygame visualization layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LayerToggles:
    """
    Per-layer visibility flags for the Paper 1 research demo.

    Defaults favour BSA storytelling: grid off, exploration overlays on.
    """

    show_grid: bool = False
    show_frontiers: bool = True
    show_trails: bool = True
    show_velocity: bool = True
    show_targets: bool = True
    show_sensor_radius: bool = True
    # Dynamic environment overlays (SDS §37)
    show_dynamic_predictions: bool = False  # predicted obstacle trajectory (D key)
    show_dynamic_safety: bool = False        # safety-radius ring per obstacle (R key)

    def handle_key(self, key: int, pg: Any) -> bool:
        """
        Toggle a layer from a key press.

        Returns True if the key was handled.
        """
        key_map = {
            pg.K_g: "show_grid",
            pg.K_f: "show_frontiers",
            pg.K_t: "show_trails",
            pg.K_v: "show_velocity",
            pg.K_y: "show_targets",
            pg.K_s: "show_sensor_radius",
            pg.K_d: "show_dynamic_predictions",
            pg.K_r: "show_dynamic_safety",
        }
        attr = key_map.get(key)
        if attr is None:
            return False
        current = getattr(self, attr)
        setattr(self, attr, not current)
        return True
