"""
Search behaviour strategies for decentralized target search.

Each behaviour is a stateless or minimally-stateful object that computes
the next navigation waypoint given the current agent and world state.
The waypoint is returned to SearchController which calls agent.set_target().

Behaviours
----------
DirectNavBehaviour     — navigate directly to last known target position
SpiralSearchBehaviour  — outward Archimedean spiral around a center point
ExpandingSearchBehaviour — expanding grid/box search from a center point
RecoverySearchBehaviour  — random walk + spiral after prolonged LOST state

All behaviours use world.obstacles and world.clip_position / resolve_collisions
to produce collision-free waypoints.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.config.search_config import SearchBehaviourConfig
from src.environment.world import World
from src.search.target import Target


class SearchBehaviour(Enum):
    """Enumeration of available search behaviour modes."""

    DIRECT_NAV = auto()
    SPIRAL_SEARCH = auto()
    EXPANDING_SEARCH = auto()
    RECOVERY = auto()


class BehaviourStrategy(Protocol):
    """Protocol for all search behaviour strategies."""

    def next_waypoint(
        self,
        agent_position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64]:
        """Return the next navigation waypoint [m]."""
        ...

    def reset(self) -> None:
        """Reset internal state (called when behaviour is re-activated)."""
        ...


# ------------------------------------------------------------------
# DirectNavBehaviour
# ------------------------------------------------------------------

class DirectNavBehaviour:
    """
    Navigate directly to the target's last known position.

    Used when last_seen age < config.direct_nav_staleness_s and
    confidence is above threshold.
    """

    def __init__(self, target: Target) -> None:
        self._target = target

    def reset(self) -> None:
        pass  # stateless

    def next_waypoint(
        self,
        agent_position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64]:
        """Return last known target position, collision-resolved."""
        if self._target.last_seen_position is not None:
            goal = self._target.last_seen_position.copy()
        else:
            goal = self._target.position.copy()

        goal = world.clip_position(goal)
        goal = world.resolve_collisions(goal)
        return goal


# ------------------------------------------------------------------
# SpiralSearchBehaviour
# ------------------------------------------------------------------

class SpiralSearchBehaviour:
    """
    Outward Archimedean spiral search around a center point.

    Generates waypoints along a spiral path expanding from center outward.
    Wraps around when max_radius is exceeded.
    """

    def __init__(
        self,
        center: NDArray[np.float64],
        config: SearchBehaviourConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._center = center.copy()
        self._cfg = config
        self._rng = rng or np.random.default_rng()
        self._angle: float = 0.0
        self._radius: float = config.spiral_initial_radius
        self._angular_step: float = np.pi / 4.0  # 8 points per loop

    def reset(self) -> None:
        self._angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        self._radius = self._cfg.spiral_initial_radius

    def next_waypoint(
        self,
        agent_position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64]:
        """Advance spiral and return next waypoint."""
        # Advance spiral parameters
        self._angle += self._angular_step
        if self._angle >= 2.0 * np.pi:
            self._angle -= 2.0 * np.pi
            self._radius += self._cfg.spiral_step

        # Wrap radius back to start when max is exceeded
        if self._radius > self._cfg.spiral_max_radius:
            self._radius = self._cfg.spiral_initial_radius

        offset = self._radius * np.array(
            [np.cos(self._angle), np.sin(self._angle)],
            dtype=np.float64,
        )
        waypoint = self._center + offset
        waypoint = world.clip_position(waypoint)
        waypoint = world.resolve_collisions(waypoint)
        return waypoint

    def update_center(self, new_center: NDArray[np.float64]) -> None:
        """Recentre the spiral (e.g. when new information arrives)."""
        self._center = new_center.copy()
        self.reset()


# ------------------------------------------------------------------
# ExpandingSearchBehaviour
# ------------------------------------------------------------------

class ExpandingSearchBehaviour:
    """
    Expanding box / grid search from a center point.

    Generates waypoints on a growing bounding box around the last
    known position. Steps outward by expanding_step_m each loop.
    """

    def __init__(
        self,
        center: NDArray[np.float64],
        config: SearchBehaviourConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._center = center.copy()
        self._cfg = config
        self._rng = rng or np.random.default_rng()
        self._step_size: float = config.expanding_step_m
        self._current_radius: float = config.expanding_step_m
        self._corner_index: int = 0
        self._corners: list[NDArray[np.float64]] = []
        self._build_corners()

    def reset(self) -> None:
        self._current_radius = self._cfg.expanding_step_m
        self._corner_index = 0
        self._build_corners()

    def next_waypoint(
        self,
        agent_position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64]:
        """Return next corner of the expanding box."""
        if not self._corners:
            self._build_corners()

        waypoint = self._corners[self._corner_index].copy()
        self._corner_index += 1

        if self._corner_index >= len(self._corners):
            # Expand to next ring
            self._current_radius += self._step_size
            if self._current_radius > self._cfg.expanding_max_radius:
                self._current_radius = self._step_size
            self._build_corners()
            self._corner_index = 0

        waypoint = world.clip_position(waypoint)
        waypoint = world.resolve_collisions(waypoint)
        return waypoint

    def update_center(self, new_center: NDArray[np.float64]) -> None:
        """Update the expansion center."""
        self._center = new_center.copy()
        self.reset()

    def _build_corners(self) -> None:
        """Build 8-point perimeter around current radius."""
        r = self._current_radius
        c = self._center
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        self._corners = [
            c + r * np.array([np.cos(a), np.sin(a)], dtype=np.float64)
            for a in angles
        ]


# ------------------------------------------------------------------
# RecoverySearchBehaviour
# ------------------------------------------------------------------

class RecoverySearchBehaviour:
    """
    Recovery search used after prolonged LOST state.

    Combines a random initial displacement with a small spiral to
    probe the area around the last known position.
    """

    def __init__(
        self,
        last_known: NDArray[np.float64],
        config: SearchBehaviourConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._last_known = last_known.copy()
        self._cfg = config
        self._rng = rng or np.random.default_rng()
        self._spiral = SpiralSearchBehaviour(last_known, config, rng)
        self._phase = 0  # 0 = random displacement, 1 = spiral

    def reset(self) -> None:
        self._phase = 0
        self._spiral.reset()

    def next_waypoint(
        self,
        agent_position: NDArray[np.float64],
        world: World,
    ) -> NDArray[np.float64]:
        """
        Phase 0: random displacement toward last_known + noise.
        Phase 1: spiral search from last_known.
        """
        if self._phase == 0:
            # Move toward last known with random offset
            angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
            noise_r = float(self._rng.uniform(0.0, self._cfg.spiral_initial_radius))
            noise = noise_r * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
            waypoint = self._last_known + noise
            self._phase = 1  # switch to spiral next call
        else:
            waypoint = self._spiral.next_waypoint(agent_position, world)

        waypoint = world.clip_position(waypoint)
        waypoint = world.resolve_collisions(waypoint)
        return waypoint

    def update_last_known(self, position: NDArray[np.float64]) -> None:
        """Update the recovery center position."""
        self._last_known = position.copy()
        self._spiral.update_center(position)
        self.reset()
