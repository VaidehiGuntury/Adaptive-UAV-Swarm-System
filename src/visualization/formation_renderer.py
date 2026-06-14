"""
Formation overlay rendering for Pygame (Paper 2 infrastructure).

Draws slot markers, edges, leader highlight, reference-frame axes, and
formation error vectors (agent → desired slot). No simulation or control logic.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.algorithms.formation.formation_types import DEFAULT_SLOT_TOLERANCE
from src.algorithms.formation.group_snapshot import (
    FormationGroupSnapshot,
    build_formation_group_snapshots,
)
from src.environment.formation_spec import FormationSpec
from src.visualization.coordinate_transform import CoordinateTransform
from src.visualization.render_palette import (
    COLOR_FORMATION_EDGE,
    COLOR_FORMATION_ERROR,
    COLOR_FORMATION_FRAME_LEFT,
    COLOR_FORMATION_FRAME_FORWARD,
    COLOR_FORMATION_LEADER,
    COLOR_FORMATION_LEADER_FILL,
    COLOR_FORMATION_SLOT_EMPTY,
    COLOR_FORMATION_SLOT_OCCUPIED,
    FORMATION_ERROR_MIN_DRAW_NORM,
)

Vector2 = NDArray[np.float64]


class AgentPoseView(Protocol):
    agent_id: int
    position: Vector2


def draw_world_formations(
    surface: Any,
    pg: Any,
    formation_specs: list[FormationSpec],
    formation_states: list[Any],
    agents: list[AgentPoseView],
    transform: CoordinateTransform,
    agent_radius_px: int,
    *,
    timestep: int = 0,
    time_s: float = 0.0,
    group_snapshots: tuple[FormationGroupSnapshot, ...] | None = None,
) -> None:
    """Resolve specs and snapshots from world state, then render overlays."""
    specs = list(formation_specs)
    if not specs and formation_states:
        from src.algorithms.formation.formation_state import FormationState

        specs = [
            FormationSpec.from_state(group_id=idx, state=fs)
            for idx, fs in enumerate(formation_states)
            if isinstance(fs, FormationState)
        ]

    positions = {agent.agent_id: agent.position for agent in agents}
    headings = {
        agent.agent_id: float(getattr(agent, "heading", 0.0))
        for agent in agents
        if hasattr(agent, "heading")
    }

    snapshots = group_snapshots
    if snapshots is None and formation_states:
        snapshots = build_formation_group_snapshots(
            formation_states,
            positions,
            timestep=timestep,
            time_s=time_s,
            headings=headings,
        )

    draw_formation_overlay(
        surface=surface,
        pg=pg,
        specs=specs,
        snapshots=snapshots or (),
        positions=positions,
        transform=transform,
        agent_radius_px=agent_radius_px,
    )


def draw_formation_overlay(
    surface: Any,
    pg: Any,
    specs: list[FormationSpec],
    snapshots: tuple[FormationGroupSnapshot, ...],
    positions: dict[int, Vector2],
    transform: CoordinateTransform,
    agent_radius_px: int,
) -> None:
    """Render formation slots, edges, leader frame, and error vectors."""
    if not specs and not snapshots:
        return

    snapshot_by_leader = {snap.leader.agent_id: snap for snap in snapshots}

    for spec in specs:
        leader_pos = positions.get(spec.leader_id)
        if leader_pos is None:
            continue

        snapshot = snapshot_by_leader.get(spec.leader_id)
        if snapshot is not None:
            _draw_reference_frame(surface, pg, snapshot, transform)
            _draw_leader_agent(surface, pg, snapshot.leader, transform, agent_radius_px)

        _draw_edges(surface, pg, spec, leader_pos, transform)
        _draw_slots(surface, pg, spec, leader_pos, transform, agent_radius_px, snapshot)

        if snapshot is not None:
            _draw_error_vectors(surface, pg, snapshot, positions, transform)
        else:
            _draw_legacy_occupancy(
                surface,
                pg,
                spec,
                leader_pos,
                positions,
                transform,
                DEFAULT_SLOT_TOLERANCE,
            )


def _draw_reference_frame(
    surface: Any,
    pg: Any,
    snapshot: FormationGroupSnapshot,
    transform: CoordinateTransform,
) -> None:
    """Draw leader-local +x (forward) and +y (left) axes."""
    origin = snapshot.leader.position
    length = snapshot.axis_length
    forward_end = origin + snapshot.leader.forward_direction * length
    left_end = origin + snapshot.leader.left_direction * length

    p0 = transform.world_to_screen(origin)
    pf = transform.world_to_screen(forward_end)
    pl = transform.world_to_screen(left_end)
    pg.draw.line(surface, COLOR_FORMATION_FRAME_FORWARD, p0, pf, width=2)
    pg.draw.line(surface, COLOR_FORMATION_FRAME_LEFT, p0, pl, width=2)


def _draw_leader_agent(
    surface: Any,
    pg: Any,
    leader: Any,
    transform: CoordinateTransform,
    agent_radius_px: int,
) -> None:
    cx, cy = transform.world_to_screen(leader.position)
    highlight_radius = agent_radius_px + 5
    pg.draw.circle(surface, COLOR_FORMATION_LEADER, (cx, cy), highlight_radius, width=2)
    pg.draw.circle(surface, COLOR_FORMATION_LEADER_FILL, (cx, cy), agent_radius_px - 1)


def _draw_edges(
    surface: Any,
    pg: Any,
    spec: FormationSpec,
    leader_pos: Vector2,
    transform: CoordinateTransform,
) -> None:
    for start, end in spec.edge_world_segments(leader_pos):
        p0 = transform.world_to_screen(start)
        p1 = transform.world_to_screen(end)
        pg.draw.line(surface, COLOR_FORMATION_EDGE, p0, p1, width=1)


def _draw_slots(
    surface: Any,
    pg: Any,
    spec: FormationSpec,
    leader_pos: Vector2,
    transform: CoordinateTransform,
    agent_radius_px: int,
    snapshot: FormationGroupSnapshot | None,
) -> None:
    radius = max(4, agent_radius_px - 2)
    occupied_slots = {f.slot_index for f in snapshot.followers if f.is_occupied} if snapshot else set()

    for slot_idx in sorted(spec.slot_offsets):
        world = spec.slot_world_position(slot_idx, leader_pos)
        if world is None:
            continue
        cx, cy = transform.world_to_screen(world)
        if slot_idx == 0:
            continue
        color = (
            COLOR_FORMATION_SLOT_OCCUPIED
            if slot_idx in occupied_slots
            else COLOR_FORMATION_SLOT_EMPTY
        )
        if slot_idx in occupied_slots:
            pg.draw.circle(surface, color, (cx, cy), radius)
        else:
            pg.draw.circle(surface, color, (cx, cy), radius, width=1)


def _draw_error_vectors(
    surface: Any,
    pg: Any,
    snapshot: FormationGroupSnapshot,
    positions: dict[int, Vector2],
    transform: CoordinateTransform,
) -> None:
    """Draw correction-direction arrows from agent position toward desired slot."""
    for follower in snapshot.followers:
        if follower.error_magnitude < FORMATION_ERROR_MIN_DRAW_NORM:
            continue
        agent_pos = positions.get(follower.agent_id)
        if agent_pos is None:
            continue
        p0 = transform.world_to_screen(agent_pos)
        p1 = transform.world_to_screen(follower.world_slot_position)
        pg.draw.line(surface, COLOR_FORMATION_ERROR, p0, p1, width=2)
        _draw_arrowhead(surface, pg, p0, p1, COLOR_FORMATION_ERROR)


def _draw_legacy_occupancy(
    surface: Any,
    pg: Any,
    spec: FormationSpec,
    leader_pos: Vector2,
    positions: dict[int, Vector2],
    transform: CoordinateTransform,
    slot_tolerance: float,
) -> None:
    """Fallback occupancy tether when no group snapshot is available."""
    from src.algorithms.formation.formation_types import is_slot_occupied_by_distance

    for agent_id in spec.member_ids:
        if agent_id == spec.leader_id:
            continue
        agent_pos = positions.get(agent_id)
        desired = spec.desired_world_position(leader_pos, agent_id)
        if agent_pos is None or desired is None:
            continue

        ax, ay = transform.world_to_screen(agent_pos)
        dx, dy = transform.world_to_screen(desired)
        dist = float(np.linalg.norm(agent_pos - desired))
        color = (
            COLOR_FORMATION_SLOT_OCCUPIED
            if is_slot_occupied_by_distance(dist, slot_tolerance)
            else COLOR_FORMATION_SLOT_EMPTY
        )
        pg.draw.line(surface, color, (ax, ay), (dx, dy), width=1)
        pg.draw.circle(surface, color, (dx, dy), 3)


def _draw_arrowhead(
    surface: Any,
    pg: Any,
    tail: tuple[int, int],
    tip: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 5.0
    left = (int(tip[0] - ux * size + px * size * 0.5), int(tip[1] - uy * size + py * size * 0.5))
    right = (int(tip[0] - ux * size - px * size * 0.5), int(tip[1] - uy * size - py * size * 0.5))
    pg.draw.polygon(surface, color, [tip, left, right])
