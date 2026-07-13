"""
PriorityScorer — configurable multi-factor target priority scoring.

Priority factors
----------------
1. Target type class score (moving human > vehicle > static object)
2. Detection confidence (higher confidence = higher priority)
3. Movement bonus (dynamic targets are higher priority)
4. Distance score (closer targets are slightly more tractable)
5. Urgency (time since detection — older detections are more urgent)
6. Freshness (time since last_seen — fresher = more reliable)

Returns a single float priority score. Higher = more urgent.
The scorer is stateless — call score() freely.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.config.search_config import PriorityConfig
from src.search.target import Target, TargetType


class PriorityScorer:
    """
    Computes a numerical priority score for a target.

    Parameters
    ----------
    config : PriorityConfig
        Weight configuration for each scoring factor.
    """

    def __init__(self, config: PriorityConfig) -> None:
        self._cfg = config

    def score(
        self,
        target: Target,
        observer_position: NDArray[np.float64],
        current_time: float,
        max_distance: float = 100.0,
    ) -> float:
        """
        Compute and return the priority score for ``target``.

        Also sets target.priority in place for downstream consumers.

        Parameters
        ----------
        target : Target
            The target to score.
        observer_position : NDArray
            Position used for distance calculation (e.g. fleet centroid or UAV pos).
        current_time : float
            Current simulation time [s].
        max_distance : float
            World diagonal or max range for distance normalization [m].
        """
        cfg = self._cfg

        # Factor 1: Type-based base score
        type_score = self._type_score(target)

        # Factor 2: Confidence [0, 1]
        conf_score = float(np.clip(target.confidence, 0.0, 1.0))

        # Factor 3: Movement bonus — dynamic targets are harder to miss, higher urgency
        movement_bonus = 1.0 if target.is_moving else 0.0

        # Factor 4: Distance score — inverse normalized to [0, 1]
        dist = target.distance_to(observer_position)
        distance_score = float(np.clip(1.0 - dist / max(max_distance, 1.0), 0.0, 1.0))

        # Factor 5: Urgency — time since first detection (normalised to 120s horizon)
        if target.detection_time is not None:
            age_s = current_time - target.detection_time
            urgency = float(np.clip(age_s / 120.0, 0.0, 1.0))
        else:
            urgency = 0.0

        # Factor 6: Freshness — inversely proportional to time since last seen
        if target.last_seen is not None:
            staleness_s = current_time - target.last_seen
            freshness = float(np.clip(1.0 - staleness_s / 60.0, 0.0, 1.0))
        else:
            freshness = 0.5  # unknown → neutral

        priority = (
            cfg.w_type * type_score
            + cfg.w_confidence * conf_score
            + cfg.w_movement * movement_bonus
            + cfg.w_distance * distance_score
            + cfg.w_urgency * urgency
            + cfg.w_freshness * freshness
        )

        target.priority = float(priority)
        return target.priority

    def score_all(
        self,
        targets: list[Target],
        observer_position: NDArray[np.float64],
        current_time: float,
        max_distance: float = 100.0,
    ) -> list[tuple[Target, float]]:
        """
        Score all targets and return sorted list (highest priority first).

        Each entry is (target, score).
        """
        scored = [
            (t, self.score(t, observer_position, current_time, max_distance))
            for t in targets
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _type_score(self, target: Target) -> float:
        """Return the base score for a target's type and movement state."""
        cfg = self._cfg
        if target.target_type == TargetType.DYNAMIC:
            if target.is_moving:
                # Distinguish moving human (TIME_VARYING originally from dynamic pool)
                # Use score_moving_human as top priority, vehicle as second
                return cfg.score_moving_human
            return cfg.score_moving_vehicle
        elif target.target_type == TargetType.TIME_VARYING:
            # Time-varying: score based on current movement state
            if target.is_moving:
                return cfg.score_moving_vehicle
            return cfg.score_static_human
        else:  # STATIC
            return cfg.score_static_human if target.is_human else cfg.score_static_object
