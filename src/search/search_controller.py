"""
SearchController — per-UAV finite state machine for search & tracking.

Each UAV instance carries a SearchAgentState that this controller updates
each tick. The controller selects the appropriate behaviour strategy,
computes the next waypoint, and calls agent.set_target() — the same
actuation interface used by BSA during exploration.

State machine
-------------
IDLE → NAVIGATING_TO_SEARCH → SEARCHING → TRACKING → COMPLETED

Transitions
-----------
IDLE → NAVIGATING_TO_SEARCH  : target assigned
NAVIGATING_TO_SEARCH → SEARCHING : arrived near search area
SEARCHING → TRACKING         : target enters sensing range (via DetectionSystem)
TRACKING → SEARCHING         : target lost (TargetTracker event "lost")
TRACKING → COMPLETED         : tracking duration met (TargetTracker event "completed")
SEARCHING → COMPLETED        : max recovery attempts exceeded
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.config.search_config import SearchBehaviourConfig
from src.environment.world import World
from src.search.behaviours import (
    DirectNavBehaviour,
    ExpandingSearchBehaviour,
    RecoverySearchBehaviour,
    SearchBehaviour,
    SpiralSearchBehaviour,
)
from src.search.target import Target, TargetStatus
from src.search.target_manager import TargetManager
from src.search.tracker import TargetTracker


class AgentSearchPhase(Enum):
    """Per-agent search FSM state."""

    IDLE = auto()
    NAVIGATING_TO_SEARCH = auto()
    SEARCHING = auto()
    TRACKING = auto()
    COMPLETED = auto()


@dataclass
class SearchAgentState:
    """
    Search-specific state carried by each UAV.

    Attached to UAV.search_state by MissionOrchestrator at phase transition.

    assigned_target_id  : the target currently being actively pursued
                          (steered toward). None if idle.
    assigned_target_ids : all targets currently assigned to this UAV,
                          including assigned_target_id itself, in the
                          order they were assigned. A UAV can hold up to
                          config.assignment.max_targets_per_uav of these
                          at once, but only ever actively pursues one —
                          see SearchController.assign_target() /
                          _advance_to_next_target().
    idle_waypoint_index  : index into this agent's fixed sector lawnmower
                          waypoint list (SearchController._sector_waypoints),
                          advanced each time a new IDLE waypoint is issued;
                          wraps around (see SearchController._next_idle_waypoint())
                          once the sector has been fully swept.
    """

    agent_id: int
    phase: AgentSearchPhase = AgentSearchPhase.IDLE
    assigned_target_id: int | None = None
    assigned_target_ids: list[int] = field(default_factory=list)
    current_behaviour: SearchBehaviour = SearchBehaviour.DIRECT_NAV
    time_since_replan: float = 0.0
    waypoint: NDArray[np.float64] | None = None
    recovery_attempts: int = 0
    idle_waypoint_index: int = 0

    # Behaviour instances (created lazily)
    _direct_nav: DirectNavBehaviour | None = field(default=None, repr=False)
    _spiral: SpiralSearchBehaviour | None = field(default=None, repr=False)
    _expanding: ExpandingSearchBehaviour | None = field(default=None, repr=False)
    _recovery: RecoverySearchBehaviour | None = field(default=None, repr=False)


def _grid_factorization(n: int) -> tuple[int, int]:
    """
    Return (rows, cols) with rows*cols == n, as close to square as possible
    (rows <= cols). Falls back to a 1xN strip for prime n — an inherent
    consequence of requiring an exact factorization, not a special case.
    """
    n = max(1, n)
    for i in range(math.isqrt(n), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n  # unreachable (i=1 always divides n), kept for clarity


def _compute_sector_bounds(
    num_sectors: int,
    world_width: float,
    world_height: float,
) -> list[tuple[float, float, float, float]]:
    """
    Partition the world into ``num_sectors`` equal rectangular sectors on a
    grid as close to square as possible. Returns one (xmin, xmax, ymin, ymax)
    tuple per sector, in row-major order (sector index = row*cols + col).
    """
    rows, cols = _grid_factorization(num_sectors)
    sector_w = world_width / cols
    sector_h = world_height / rows
    bounds: list[tuple[float, float, float, float]] = []
    for s in range(num_sectors):
        row, col = divmod(s, cols)
        xmin = col * sector_w
        ymin = row * sector_h
        bounds.append((xmin, xmin + sector_w, ymin, ymin + sector_h))
    return bounds


def _generate_lawnmower_waypoints(
    bounds: tuple[float, float, float, float],
    row_spacing: float,
) -> list[NDArray[np.float64]]:
    """
    Boustrophedon (back-and-forth) waypoint list covering a rectangular
    sector. Only row endpoints are needed — straight-line travel between
    consecutive waypoints sweeps the full row width, so no intermediate
    points are required within a row. Always returns at least one waypoint,
    even for a sector smaller than one row_spacing.
    """
    xmin, xmax, ymin, ymax = bounds
    half = row_spacing / 2.0
    x_left = min(xmin + half, (xmin + xmax) / 2.0)
    x_right = max(xmax - half, (xmin + xmax) / 2.0)

    row_ys: list[float] = []
    y = ymin + half
    y_limit = ymax - half
    if y_limit >= ymin + half:
        while y <= y_limit:
            row_ys.append(y)
            y += row_spacing
    if not row_ys:
        row_ys = [(ymin + ymax) / 2.0]

    waypoints: list[NDArray[np.float64]] = []
    left_to_right = True
    for row_y in row_ys:
        first, second = (x_left, x_right) if left_to_right else (x_right, x_left)
        waypoints.append(np.array([first, row_y], dtype=np.float64))
        waypoints.append(np.array([second, row_y], dtype=np.float64))
        left_to_right = not left_to_right
    return waypoints


class SearchController:
    """
    Orchestrates per-UAV search behaviour selection and waypoint generation.

    Parameters
    ----------
    config : SearchBehaviourConfig
    tracker : TargetTracker
        Shared tracker — receives tracking events from DetectionSystem.
    rng : np.random.Generator
        Shared RNG for behaviour strategies.
    world_width, world_height : float
        World bounds, used to partition IDLE sector sweeps. Default to a
        single 100x100m sector so ``SearchController(config, tracker)``
        (no world/fleet info) remains valid — used by existing tests that
        never exercise the IDLE branch.
    num_uavs : int
        Fleet size — the world is divided into this many sectors, one per
        UAV (``agent_id % num_uavs``). Defaults to 1 (whole world, one
        sector) when unspecified.
    """

    def __init__(
        self,
        config: SearchBehaviourConfig,
        tracker: TargetTracker,
        rng: np.random.Generator | None = None,
        world_width: float = 100.0,
        world_height: float = 100.0,
        num_uavs: int = 1,
    ) -> None:
        self._cfg = config
        self._tracker = tracker
        self._rng = rng or np.random.default_rng()

        # IDLE sector sweep: fixed per-agent sector (agent_id % num_sectors),
        # precomputed once — no coordination needed between concurrently-IDLE
        # UAVs (each computes its own path purely from its own agent_id).
        self._num_sectors = max(1, num_uavs)
        self._sector_bounds = _compute_sector_bounds(
            self._num_sectors, world_width, world_height
        )
        self._sector_waypoints: list[list[NDArray[np.float64]]] = [
            _generate_lawnmower_waypoints(b, config.idle_sweep_row_spacing)
            for b in self._sector_bounds
        ]

    def sector_bounds_for_agent(self, agent_id: int) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the fixed IDLE sweep sector owned by agent_id."""
        return self._sector_bounds[agent_id % self._num_sectors]

    def initialize_agent(self, agent: UAV) -> None:
        """Create and attach a SearchAgentState to an agent."""
        agent.search_state = SearchAgentState(agent_id=agent.agent_id)

    def assign_target(
        self,
        agent: UAV,
        target: Target,
        world: World,
    ) -> None:
        """
        Add a search target to an agent's assignment queue.

        A UAV may hold multiple simultaneous assignments (up to
        config.assignment.max_targets_per_uav), but only ever actively
        pursues (steers toward) one at a time. If the agent is not
        currently TRACKING another assigned target, this target becomes
        the active one and navigation begins immediately. If the agent
        IS actively tracking another target, this one is queued —
        without interrupting the active track — and becomes active only
        once the current one is dropped (completed or permanently lost;
        see _advance_to_next_target()).
        """
        state = self._get_state(agent)
        if target.target_id not in state.assigned_target_ids:
            state.assigned_target_ids.append(target.target_id)

        if state.phase == AgentSearchPhase.TRACKING:
            return  # already actively tracking something — don't interrupt it

        state.assigned_target_id = target.target_id
        state.phase = AgentSearchPhase.NAVIGATING_TO_SEARCH
        state.time_since_replan = self._cfg.replan_interval_s  # force immediate replan
        state._direct_nav = DirectNavBehaviour(target)

        # Set initial waypoint toward target
        wp = self._compute_waypoint(state, agent, target, world)
        if wp is not None:
            agent.set_target(wp)

    def update(
        self,
        agent: UAV,
        target_manager: TargetManager,
        tracker_events: dict[int, str],
        visibility_map: dict[int, list[int]],
        world: World,
        dt: float,
    ) -> None:
        """
        Advance the per-agent search FSM by one timestep.

        Parameters
        ----------
        tracker_events : dict[int, str]
            Events from TargetTracker.update() this tick.
        visibility_map : dict[int, list[int]]
            uav_id → visible target_ids from DetectionSystem.
        """
        state = self._get_state(agent)

        if state.phase == AgentSearchPhase.COMPLETED:
            return

        target = self._get_assigned_target(state, target_manager)

        # IDLE: no target assigned yet → sweep this agent's fixed sector
        if state.phase == AgentSearchPhase.IDLE or (
            target is None and state.phase != AgentSearchPhase.COMPLETED
        ):
            state.time_since_replan += dt
            if state.time_since_replan >= self._cfg.replan_interval_s:
                state.time_since_replan = 0.0
                # Check if agent has arrived near its current waypoint
                arrived = (
                    state.waypoint is None
                    or float(np.linalg.norm(agent.position - state.waypoint))
                    < self._cfg.arrival_threshold_m * 2.0
                )
                if arrived or state.waypoint is None:
                    wp = self._next_idle_waypoint(agent, state, world)
                    agent.set_target(wp)
                    state.waypoint = wp
            return

        if target is None:
            state.phase = AgentSearchPhase.IDLE
            return

        # Handle tracker events for this agent's target
        tid = state.assigned_target_id
        if tid is not None:
            event = tracker_events.get(tid)
            if event == "completed":
                # "completed" covers both true completion and permanently-
                # lost-after-max-recovery-attempts (tracker.py emits the
                # same event for both) — either way this target is done;
                # hand off to the next queued assignment, if any.
                self._advance_to_next_target(state, agent, target_manager, world, tid)
                return
            elif event == "lost":
                self._transition_to_searching(state, agent, target, world, lost=True)
            elif event == "reacquired":
                state.phase = AgentSearchPhase.TRACKING
                state.current_behaviour = SearchBehaviour.DIRECT_NAV

        # Check current visibility
        visible_targets = visibility_map.get(agent.agent_id, [])
        target_visible = tid in visible_targets if tid else False

        # FSM transitions based on visibility
        if state.phase == AgentSearchPhase.NAVIGATING_TO_SEARCH:
            if target_visible:
                state.phase = AgentSearchPhase.TRACKING
                state.current_behaviour = SearchBehaviour.DIRECT_NAV
            else:
                # Check if arrived near last known position
                if target.last_seen_position is not None:
                    dist = float(np.linalg.norm(
                        agent.position - target.last_seen_position
                    ))
                    if dist < self._cfg.arrival_threshold_m * 3.0:
                        self._transition_to_searching(state, agent, target, world)

        elif state.phase == AgentSearchPhase.SEARCHING:
            if target_visible:
                state.phase = AgentSearchPhase.TRACKING
                state.current_behaviour = SearchBehaviour.DIRECT_NAV

        elif state.phase == AgentSearchPhase.TRACKING:
            if not target_visible:
                # Tracker will handle the LOST transition via loss_timeout
                # Keep navigating toward last_seen while awaiting timeout
                pass

        # Replan waypoint at replan_interval
        state.time_since_replan += dt
        if state.time_since_replan >= self._cfg.replan_interval_s:
            state.time_since_replan = 0.0
            wp = self._compute_waypoint(state, agent, target, world)
            if wp is not None:
                agent.set_target(wp)
                state.waypoint = wp

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_idle_waypoint(
        self,
        agent: UAV,
        state: SearchAgentState,
        world: World,
    ) -> NDArray[np.float64]:
        """
        Advance to the next waypoint in this agent's fixed lawnmower sector
        sweep (systematic coverage — see _generate_lawnmower_waypoints()),
        wrapping around once the sector has been fully swept once. Every
        waypoint is re-clipped/resolved against current world state at use
        time (the precomputed list holds raw sector-local coordinates only —
        dynamic obstacles may have since moved onto a given point, so
        resolve_collisions must be re-evaluated fresh, not baked in).
        """
        waypoints = self._sector_waypoints[agent.agent_id % self._num_sectors]
        wp = waypoints[state.idle_waypoint_index % len(waypoints)]
        state.idle_waypoint_index += 1
        wp = world.clip_position(wp)
        wp = world.resolve_collisions(wp)
        return wp

    def _get_state(self, agent: UAV) -> SearchAgentState:
        """Get or create SearchAgentState for an agent."""
        if not hasattr(agent, "search_state") or agent.search_state is None:
            agent.search_state = SearchAgentState(agent_id=agent.agent_id)
        return agent.search_state  # type: ignore[return-value]

    def _get_assigned_target(
        self,
        state: SearchAgentState,
        target_manager: TargetManager,
    ) -> Target | None:
        if state.assigned_target_id is None:
            return None
        return target_manager.get_target(state.assigned_target_id)

    def _advance_to_next_target(
        self,
        state: SearchAgentState,
        agent: UAV,
        target_manager: TargetManager,
        world: World,
        finished_target_id: int,
    ) -> None:
        """
        Drop a finished (completed or permanently-lost) target from the
        queue and, if another assigned target is waiting, make it active
        and begin navigation immediately. Otherwise mark COMPLETED.
        """
        if finished_target_id in state.assigned_target_ids:
            state.assigned_target_ids.remove(finished_target_id)

        if not state.assigned_target_ids:
            state.phase = AgentSearchPhase.COMPLETED
            state.assigned_target_id = None
            return

        next_id = state.assigned_target_ids[0]
        next_target = target_manager.get_target(next_id)
        state.assigned_target_id = next_id
        state.phase = AgentSearchPhase.NAVIGATING_TO_SEARCH
        state.time_since_replan = self._cfg.replan_interval_s
        if next_target is not None:
            state._direct_nav = DirectNavBehaviour(next_target)
            wp = self._compute_waypoint(state, agent, next_target, world)
            if wp is not None:
                agent.set_target(wp)

    def _transition_to_searching(
        self,
        state: SearchAgentState,
        agent: UAV,
        target: Target,
        world: World,
        lost: bool = False,
    ) -> None:
        """Transition agent to SEARCHING and select appropriate behaviour."""
        state.phase = AgentSearchPhase.SEARCHING

        center = (
            target.last_seen_position.copy()
            if target.last_seen_position is not None
            else target.position.copy()
        )

        if lost:
            state.recovery_attempts += 1
            if state.recovery_attempts >= self._cfg.max_recovery_attempts:
                # Switch to expanding search
                state.current_behaviour = SearchBehaviour.EXPANDING_SEARCH
                state._expanding = ExpandingSearchBehaviour(center, self._cfg, self._rng)
            else:
                state.current_behaviour = SearchBehaviour.RECOVERY
                state._recovery = RecoverySearchBehaviour(center, self._cfg, self._rng)
        else:
            state.current_behaviour = SearchBehaviour.SPIRAL_SEARCH
            state._spiral = SpiralSearchBehaviour(center, self._cfg, self._rng)

    def _compute_waypoint(
        self,
        state: SearchAgentState,
        agent: UAV,
        target: Target,
        world: World,
    ) -> NDArray[np.float64] | None:
        """Select behaviour and return next waypoint."""
        behaviour = state.current_behaviour

        if behaviour == SearchBehaviour.DIRECT_NAV:
            if state._direct_nav is None:
                state._direct_nav = DirectNavBehaviour(target)
            return state._direct_nav.next_waypoint(agent.position, world)

        elif behaviour == SearchBehaviour.SPIRAL_SEARCH:
            if state._spiral is None:
                center = (
                    target.last_seen_position.copy()
                    if target.last_seen_position is not None
                    else target.position.copy()
                )
                state._spiral = SpiralSearchBehaviour(center, self._cfg, self._rng)
            return state._spiral.next_waypoint(agent.position, world)

        elif behaviour == SearchBehaviour.EXPANDING_SEARCH:
            if state._expanding is None:
                center = (
                    target.last_seen_position.copy()
                    if target.last_seen_position is not None
                    else agent.position.copy()
                )
                state._expanding = ExpandingSearchBehaviour(center, self._cfg, self._rng)
            return state._expanding.next_waypoint(agent.position, world)

        elif behaviour == SearchBehaviour.RECOVERY:
            if state._recovery is None:
                center = (
                    target.last_seen_position.copy()
                    if target.last_seen_position is not None
                    else agent.position.copy()
                )
                state._recovery = RecoverySearchBehaviour(center, self._cfg, self._rng)
            return state._recovery.next_waypoint(agent.position, world)

        return None
