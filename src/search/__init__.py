"""
Target Search & Tracking extension for the DEBS Adaptive UAV Swarm System.

This package implements the post-exploration search mission (Layer 3 extension):

  Target Discovery  →  Prioritization  →  Assignment
       ↓                                       ↓
  Target Tracking  ←  Search Behaviour  ←  Search Controller

Modules
-------
target          — Target dataclass, TargetType/TargetStatus enums, DetectionEvent
target_manager  — TargetManager: spawn, lifecycle, movement, assignment API
detection       — DetectionSystem: per-UAV scan, deduplication, shared DB
prioritization  — PriorityScorer: multi-factor configurable priority score
assignment      — SearchAssigner: IDE-compatible distance-minimizing assignment
behaviours      — Search behaviour strategies (Direct, Spiral, Expanding, Recovery)
tracker         — TargetTracker: tracking state, loss/reacquisition FSM
search_controller — Per-UAV search FSM wiring all sub-modules
mission_phase   — MissionPhase enum and MissionOrchestrator

Integration
-----------
- World.target_manager holds the shared TargetManager instance
- SimulationEngine.mission_orchestrator drives phase transitions
- UAV.search_state carries per-agent search FSM state
- All search modules use agent.set_target() as the sole actuation interface
- BSA and search phases are mutually exclusive (never run simultaneously)
"""

from src.search.target import (
    Target,
    TargetType,
    TargetStatus,
    DetectionEvent,
    TrackingRecord,
)
from src.search.target_manager import TargetManager
from src.search.detection import DetectionSystem, DetectionDatabase
from src.search.prioritization import PriorityScorer
from src.search.assignment import SearchAssigner
from src.search.behaviours import (
    SearchBehaviour,
    DirectNavBehaviour,
    SpiralSearchBehaviour,
    ExpandingSearchBehaviour,
    RecoverySearchBehaviour,
)
from src.search.tracker import TargetTracker
from src.search.search_controller import SearchController, SearchAgentState
from src.search.mission_phase import MissionPhase, MissionOrchestrator

__all__ = [
    "Target",
    "TargetType",
    "TargetStatus",
    "DetectionEvent",
    "TrackingRecord",
    "TargetManager",
    "DetectionSystem",
    "DetectionDatabase",
    "PriorityScorer",
    "SearchAssigner",
    "SearchBehaviour",
    "DirectNavBehaviour",
    "SpiralSearchBehaviour",
    "ExpandingSearchBehaviour",
    "RecoverySearchBehaviour",
    "TargetTracker",
    "SearchController",
    "SearchAgentState",
    "MissionPhase",
    "MissionOrchestrator",
]
