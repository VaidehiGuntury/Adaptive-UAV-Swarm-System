"""
DetectionSystem — per-UAV target sensing and shared detection database.

Responsibilities
----------------
- Scan each UAV's sensing disk against the active target population
- Generate DetectionEvent records with confidence values
- Deduplicate detections within a configurable time window
- Merge multi-UAV observations of the same target (confidence fusion)
- Notify TargetManager of new detections

Integration
-----------
SimulationEngine calls detection_system.scan_all(agents, target_manager, time_s)
each tick during BOTH the exploration and searching phases.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.config.search_config import DetectionConfig
from src.search.target import DetectionEvent, Target, TargetStatus, TargetType
from src.search.target_manager import TargetManager


class DetectionDatabase:
    """
    Shared detection event store with deduplication and confidence merge.

    Thread-safety: not required (single-threaded simulation).
    """

    def __init__(self, dedup_window_s: float = 2.0) -> None:
        self._dedup_window = dedup_window_s
        # target_id → list of DetectionEvent (most-recent first)
        self._events: dict[int, list[DetectionEvent]] = defaultdict(list)

    def add_event(self, event: DetectionEvent) -> bool:
        """
        Register a detection event if it is not a duplicate.

        Returns True if the event was accepted (not a duplicate).
        Duplicates: same target_id + uav_id within dedup_window_s.
        """
        recent = self._events[event.target_id]
        for existing in recent:
            if (
                existing.uav_id == event.uav_id
                and abs(existing.timestamp - event.timestamp) <= self._dedup_window
            ):
                return False
        recent.append(event)
        return True

    def events_for_target(self, target_id: int) -> list[DetectionEvent]:
        """Return all detection events for a given target."""
        return list(self._events.get(target_id, []))

    def recent_events(
        self,
        target_id: int,
        current_time: float,
        window_s: float,
    ) -> list[DetectionEvent]:
        """Return detection events within ``window_s`` seconds of ``current_time``."""
        all_events = self._events.get(target_id, [])
        return [e for e in all_events if current_time - e.timestamp <= window_s]

    def merged_confidence(
        self,
        target_id: int,
        current_time: float,
        window_s: float,
        merge_alpha: float,
    ) -> float:
        """
        Fuse multi-UAV observations into a single confidence estimate.

        Uses exponential weighted merge: c_merged = alpha * max(c) + (1-alpha) * mean(c)
        """
        recent = self.recent_events(target_id, current_time, window_s)
        if not recent:
            return 0.0
        confidences = [e.confidence for e in recent]
        return float(
            merge_alpha * max(confidences)
            + (1.0 - merge_alpha) * float(np.mean(confidences))
        )

    def all_detected_target_ids(self) -> set[int]:
        """Return set of all target IDs that have at least one detection."""
        return set(self._events.keys())

    def clear(self) -> None:
        """Remove all stored events."""
        self._events.clear()


class DetectionSystem:
    """
    Per-UAV scanning and detection pipeline.

    Parameters
    ----------
    config : DetectionConfig
        Detection model parameters.
    uav_sensing_range : float
        UAV sensor radius [m] — used when config.detection_radius == 0.0.
    """

    def __init__(
        self,
        config: DetectionConfig,
        uav_sensing_range: float,
    ) -> None:
        self._config = config
        self._detection_radius = (
            config.detection_radius
            if config.detection_radius > 0.0
            else uav_sensing_range
        )
        self.database = DetectionDatabase(
            dedup_window_s=config.dedup_time_window_s,
        )

    @property
    def detection_radius(self) -> float:
        """Effective detection radius [m]."""
        return self._detection_radius

    def scan_agent(
        self,
        agent: UAV,
        target_manager: TargetManager,
        current_time: float,
    ) -> list[DetectionEvent]:
        """
        Scan one UAV's sensing disk and return new DetectionEvents.

        Only UNDISCOVERED and previously detected (but untracked) targets
        within detection_radius generate events. Events below the minimum
        confidence threshold are discarded.
        """
        new_events: list[DetectionEvent] = []

        for target in target_manager.iter_targets():
            if target.status == TargetStatus.COMPLETED:
                continue

            dist = target.distance_to(agent.position)
            if dist > self._detection_radius:
                continue

            confidence = self._compute_confidence(dist, target)
            if confidence < self._config.min_detection_confidence:
                continue

            event = DetectionEvent(
                target_id=target.target_id,
                uav_id=agent.agent_id,
                position=target.position.copy(),
                timestamp=current_time,
                confidence=confidence,
                target_type=target.target_type,
            )

            accepted = self.database.add_event(event)
            if accepted:
                new_events.append(event)

        return new_events

    def scan_all(
        self,
        agents: list[UAV],
        target_manager: TargetManager,
        current_time: float,
    ) -> list[DetectionEvent]:
        """
        Scan all agents and update TargetManager with new detections.

        Returns the flat list of all new DetectionEvents this tick.
        """
        all_new: list[DetectionEvent] = []

        for agent in agents:
            agent_events = self.scan_agent(agent, target_manager, current_time)
            all_new.extend(agent_events)

        # Notify TargetManager and merge multi-UAV confidence
        for target_id in {e.target_id for e in all_new}:
            merged_conf = self.database.merged_confidence(
                target_id=target_id,
                current_time=current_time,
                window_s=self._config.dedup_time_window_s,
                merge_alpha=self._config.confidence_merge_alpha,
            )
            # Find position from most recent event for this target
            latest = max(
                (e for e in all_new if e.target_id == target_id),
                key=lambda e: e.timestamp,
            )
            target_manager.register_detection(
                target_id=target_id,
                confidence=merged_conf,
                position=latest.position,
                current_time=current_time,
            )

        return all_new

    def update_tracking_observations(
        self,
        agents: list[UAV],
        target_manager: TargetManager,
        current_time: float,
    ) -> dict[int, list[int]]:
        """
        During search phase: scan and return visible target mapping.

        Returns dict mapping uav_id → list of visible target_ids.
        Used by SearchController to trigger tracking updates.
        """
        visibility: dict[int, list[int]] = {agent.agent_id: [] for agent in agents}

        for agent in agents:
            for target in target_manager.iter_targets():
                if target.status == TargetStatus.COMPLETED:
                    continue
                dist = target.distance_to(agent.position)
                if dist <= self._detection_radius:
                    confidence = self._compute_confidence(dist, target)
                    if confidence >= self._config.min_detection_confidence:
                        visibility[agent.agent_id].append(target.target_id)
                        # Update last_seen for visible targets
                        target.last_seen = current_time
                        target.last_seen_position = target.position.copy()
                        target.confidence = max(target.confidence, confidence)

        return visibility

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self, distance: float, target: Target) -> float:
        """
        Sensor model: confidence decays linearly with distance.

        c(d) = base_confidence * (1 - d / detection_radius)
        Dynamic targets have a small additional confidence penalty (harder to detect)
        because they may have moved between observation and reporting.
        """
        if distance >= self._detection_radius:
            return 0.0

        linear_conf = self._config.base_confidence * (
            1.0 - distance / self._detection_radius
        )

        if target.target_type == TargetType.DYNAMIC:
            linear_conf *= 0.9
        elif target.target_type == TargetType.TIME_VARYING:
            linear_conf *= 0.95

        return float(np.clip(linear_conf, 0.0, 1.0))
