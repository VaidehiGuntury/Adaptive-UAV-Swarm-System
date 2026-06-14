"""
Formation geometry definitions and leader-local coordinate transforms (Paper 2).

All slot offsets live in the leader reference frame. World positions use
``world_pos = leader_pos + R(theta) @ local_offset``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

Vector2 = NDArray[np.float64]

# Default slot occupancy tolerance for renderer and future controllers (metres).
DEFAULT_SLOT_TOLERANCE: Final[float] = 0.5

# Default inter-agent spacing for formation templates (metres).
DEFAULT_FORMATION_SPACING: Final[float] = 2.0


def is_slot_occupied_by_distance(distance: float, slot_tolerance: float) -> bool:
    """
    Return True when ``distance`` is within the occupancy threshold.

    Occupancy is **inclusive** at the boundary:
    ``distance == slot_tolerance`` counts as occupied.
    """
    return float(distance) <= float(slot_tolerance)


def rotation_matrix(theta: float) -> NDArray[np.float64]:
    """2-D rotation matrix R(theta) mapping leader-local offsets to world axes."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_offset(offset: Vector2, theta: float) -> Vector2:
    """Rotate a leader-local offset by ``theta`` radians."""
    return rotation_matrix(theta) @ np.asarray(offset, dtype=np.float64)


def transform_to_world(
    local: Vector2,
    leader_pos: Vector2,
    theta: float,
) -> Vector2:
    """Map leader-local offset to world coordinates."""
    return np.asarray(leader_pos, dtype=np.float64) + rotate_offset(local, theta)


def transform_to_local(
    world: Vector2,
    leader_pos: Vector2,
    theta: float,
) -> Vector2:
    """Map world coordinates into the leader-local frame."""
    delta = np.asarray(world, dtype=np.float64) - np.asarray(leader_pos, dtype=np.float64)
    return rotation_matrix(-theta) @ delta


SlotGeneratorFn = Callable[[int, float], tuple[Vector2, ...]]
EdgeGeneratorFn = Callable[[int], tuple[tuple[int, int], ...]]


class FormationType(str, Enum):
    """Named formation templates (Paper 2 desired shapes M_i)."""

    LINE = "line"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    WEDGE = "wedge"


def _vec(x: float, y: float) -> Vector2:
    return np.array([x, y], dtype=np.float64)


def _generate_line(n: int, spacing: float) -> tuple[Vector2, ...]:
    """Leader at slot 0; followers extend along -y."""
    d = spacing
    return tuple(_vec(0.0, -i * d) for i in range(n))


def _edges_line(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, i + 1) for i in range(n - 1))


def _generate_triangle(n: int, spacing: float) -> tuple[Vector2, ...]:
    """Leader at apex (slot 0); base expands symmetrically behind."""
    d = spacing
    if n <= 0:
        return ()
    slots: list[Vector2] = [_vec(0.0, 0.0)]
    if n == 1:
        return tuple(slots)
    if n >= 2:
        slots.append(_vec(-d, -d))
    if n >= 3:
        slots.append(_vec(d, -d))
    row = 2
    while len(slots) < n:
        offset = (row - 1) * d
        for sign in (-1.0, 1.0):
            if len(slots) >= n:
                break
            slots.append(_vec(sign * offset, -row * d))
        row += 1
    return tuple(slots[:n])


def _edges_triangle(n: int) -> tuple[tuple[int, int], ...]:
    if n <= 1:
        return ()
    edges: list[tuple[int, int]] = [(0, 1), (0, 2)] if n >= 3 else [(0, 1)]
    if n >= 3:
        edges.append((1, 2))
    for i in range(3, n):
        parent = 1 if i % 2 == 1 else 2
        edges.append((parent, i))
    return tuple(edges)


def _generate_diamond(n: int, spacing: float) -> tuple[Vector2, ...]:
    """Leader at centre; cardinal points then outer ring."""
    d = spacing
    if n <= 0:
        return ()
    cardinal = (
        _vec(0.0, 0.0),
        _vec(0.0, d),
        _vec(0.0, -d),
        _vec(d, 0.0),
        _vec(-d, 0.0),
    )
    if n <= len(cardinal):
        return cardinal[:n]
    slots = list(cardinal)
    ring = 2
    while len(slots) < n:
        for dx, dy in ((ring, 0), (0, ring), (-ring, 0), (0, -ring)):
            if len(slots) >= n:
                break
            slots.append(_vec(dx * d, dy * d))
        ring += 1
    return tuple(slots[:n])


def _edges_diamond(n: int) -> tuple[tuple[int, int], ...]:
    if n <= 1:
        return ()
    edges: list[tuple[int, int]] = []
    if n >= 2:
        edges.append((0, 1))
    if n >= 3:
        edges.append((0, 2))
    if n >= 4:
        edges.append((0, 3))
    if n >= 5:
        edges.append((0, 4))
    if n >= 3:
        edges.append((1, 3))
    if n >= 4:
        edges.append((2, 4))
    if n >= 5:
        edges.append((3, 4))
    for i in range(5, n):
        edges.append((0, i))
    return tuple(edges)


def _generate_wedge(n: int, spacing: float) -> tuple[Vector2, ...]:
    """Leader at V tip; followers alternate left/right behind."""
    d = spacing
    if n <= 0:
        return ()
    slots: list[Vector2] = [_vec(0.0, 0.0)]
    rank = 1
    while len(slots) < n:
        depth = rank * d
        for sign in (-1.0, 1.0):
            if len(slots) >= n:
                break
            slots.append(_vec(sign * rank * d, -depth))
        rank += 1
    return tuple(slots)


def _edges_wedge(n: int) -> tuple[tuple[int, int], ...]:
    if n <= 1:
        return ()
    edges: list[tuple[int, int]] = []
    for i in range(1, n):
        edges.append((0, i))
    for i in range(2, n):
        parent = i - 1 if i % 2 == 0 else i - 2
        if parent >= 1:
            edges.append((parent, i))
    return tuple(edges)


@dataclass(frozen=True)
class FormationDefinition:
    """
    Extensible formation template catalog entry.

    ``generator_fn`` produces ``n`` leader-local slot offsets; ``edge_fn``
    returns slot-index pairs for rendering and future graph-based control.
    """

    formation_type: FormationType
    base_slots: tuple[Vector2, ...]
    generator_fn: SlotGeneratorFn
    edge_fn: EdgeGeneratorFn

    def slot_offsets(self, n: int, spacing: float = DEFAULT_FORMATION_SPACING) -> tuple[Vector2, ...]:
        """Return ``n`` leader-local slot offsets (slot 0 is leader)."""
        if n <= 0:
            return ()
        return self.generator_fn(n, spacing)

    def edges(self, n: int) -> tuple[tuple[int, int], ...]:
        """Return formation edge pairs for ``n`` active slots."""
        if n <= 1:
            return ()
        return self.edge_fn(n)


_FORMATION_DEFINITIONS: dict[FormationType, FormationDefinition] = {
    FormationType.LINE: FormationDefinition(
        formation_type=FormationType.LINE,
        base_slots=_generate_line(4, DEFAULT_FORMATION_SPACING),
        generator_fn=_generate_line,
        edge_fn=_edges_line,
    ),
    FormationType.TRIANGLE: FormationDefinition(
        formation_type=FormationType.TRIANGLE,
        base_slots=_generate_triangle(4, DEFAULT_FORMATION_SPACING),
        generator_fn=_generate_triangle,
        edge_fn=_edges_triangle,
    ),
    FormationType.DIAMOND: FormationDefinition(
        formation_type=FormationType.DIAMOND,
        base_slots=_generate_diamond(5, DEFAULT_FORMATION_SPACING),
        generator_fn=_generate_diamond,
        edge_fn=_edges_diamond,
    ),
    FormationType.WEDGE: FormationDefinition(
        formation_type=FormationType.WEDGE,
        base_slots=_generate_wedge(5, DEFAULT_FORMATION_SPACING),
        generator_fn=_generate_wedge,
        edge_fn=_edges_wedge,
    ),
}

DEFAULT_FORMATION_TYPE: Final[FormationType] = FormationType.WEDGE


def get_formation_definition(formation_type: FormationType) -> FormationDefinition:
    """Lookup a registered formation template."""
    return _FORMATION_DEFINITIONS[formation_type]


def register_formation_definition(definition: FormationDefinition) -> None:
    """Register or replace a formation template (for adaptive / procedural types)."""
    _FORMATION_DEFINITIONS[definition.formation_type] = definition
