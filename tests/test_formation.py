"""Unit tests for Paper 2 formation infrastructure."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.algorithms.formation.formation_state import FormationState, compute_formation_centroid
from src.algorithms.formation.formation_types import (
    DEFAULT_FORMATION_SPACING,
    DEFAULT_FORMATION_TYPE,
    DEFAULT_SLOT_TOLERANCE,
    FormationType,
    get_formation_definition,
    transform_to_local,
    transform_to_world,
)
from src.algorithms.formation.slot_assignment import assign_formation_slots
from src.environment.formation_spec import FormationSpec


class TestCoordinateTransforms(unittest.TestCase):
    def test_identity_round_trip(self) -> None:
        leader = np.array([10.0, 5.0])
        local = np.array([2.0, -3.0])
        world = transform_to_world(local, leader, 0.0)
        recovered = transform_to_local(world, leader, 0.0)
        np.testing.assert_allclose(recovered, local, atol=1e-12)

    def test_rotated_round_trip(self) -> None:
        leader = np.array([0.0, 0.0])
        local = np.array([1.0, 2.0])
        for theta in (0.3, np.pi / 4, np.pi / 2, -1.1):
            world = transform_to_world(local, leader, theta)
            recovered = transform_to_local(world, leader, theta)
            np.testing.assert_allclose(recovered, local, atol=1e-12)

    def test_rotated_world_position(self) -> None:
        leader = np.array([5.0, 5.0])
        local = np.array([1.0, 0.0])
        theta = np.pi / 2
        world = transform_to_world(local, leader, theta)
        np.testing.assert_allclose(world, np.array([5.0, 6.0]), atol=1e-12)


class TestFormationDefinitions(unittest.TestCase):
    def test_leader_at_slot_zero(self) -> None:
        for ftype in FormationType:
            definition = get_formation_definition(ftype)
            offsets = definition.slot_offsets(4, DEFAULT_FORMATION_SPACING)
            np.testing.assert_allclose(offsets[0], np.zeros(2), atol=1e-12)

    def test_wedge_is_default_type(self) -> None:
        self.assertEqual(DEFAULT_FORMATION_TYPE, FormationType.WEDGE)

    def test_scales_with_n(self) -> None:
        definition = get_formation_definition(FormationType.WEDGE)
        small = definition.slot_offsets(3, 2.0)
        large = definition.slot_offsets(6, 2.0)
        self.assertEqual(len(small), 3)
        self.assertEqual(len(large), 6)

    def test_edges_within_slot_range(self) -> None:
        definition = get_formation_definition(FormationType.DIAMOND)
        n = 5
        edges = definition.edges(n)
        for a, b in edges:
            self.assertLess(a, n)
            self.assertLess(b, n)


class TestSlotAssignment(unittest.TestCase):
    def test_leader_gets_slot_zero(self) -> None:
        state = assign_formation_slots([3, 1, 2], leader_id=2)
        self.assertEqual(state.slot_assignments[2], 0)
        np.testing.assert_allclose(state.desired_offsets[2], np.zeros(2))

    def test_followers_sorted_by_id(self) -> None:
        state = assign_formation_slots([5, 1, 3], leader_id=3)
        self.assertEqual(state.slot_assignments[1], 1)
        self.assertEqual(state.slot_assignments[5], 2)

    def test_slot_offsets_match_assignments(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0, formation_type=FormationType.LINE)
        for agent_id, slot_idx in state.slot_assignments.items():
            np.testing.assert_allclose(
                state.desired_offsets[agent_id],
                state.slot_offsets[slot_idx],
            )

    def test_default_formation_is_wedge(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0)
        self.assertEqual(state.formation_type, FormationType.WEDGE)

    def test_leader_must_be_in_agent_ids(self) -> None:
        with self.assertRaises(ValueError):
            assign_formation_slots([1, 2], leader_id=9)

    def test_heading_stored_on_state(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, leader_heading=0.7)
        self.assertAlmostEqual(state.leader_heading, 0.7)


class TestFormationState(unittest.TestCase):
    def test_centroid_computed_from_positions(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0)
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([2.0, 0.0]),
            2: np.array([0.0, 2.0]),
        }
        centroid = state.centroid(positions)
        np.testing.assert_allclose(centroid, np.array([2.0 / 3, 2.0 / 3]))

    def test_compute_formation_centroid_helper(self) -> None:
        members = (1, 2)
        positions = {1: np.array([1.0, 1.0]), 2: np.array([3.0, 5.0])}
        c = compute_formation_centroid(members, positions)
        np.testing.assert_allclose(c, np.array([2.0, 3.0]))

    def test_no_stored_centroid_field(self) -> None:
        state = FormationState(leader_id=0, formation_type=FormationType.WEDGE)
        self.assertFalse(hasattr(state, "formation_centroid"))

    def test_occupancy_uses_tolerance(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, spacing=2.0)
        state.slot_tolerance = DEFAULT_SLOT_TOLERANCE
        leader_pos = np.array([10.0, 10.0])
        desired = state.desired_world_position(1, leader_pos)
        assert desired is not None
        near = desired + np.array([0.4, 0.0])
        far = desired + np.array([1.0, 0.0])
        self.assertTrue(state.is_slot_occupied(1, near, leader_pos))
        self.assertFalse(state.is_slot_occupied(1, far, leader_pos))

    def test_desired_world_with_heading(self) -> None:
        state_zero = assign_formation_slots([0, 1], leader_id=0, spacing=1.0, leader_heading=0.0)
        state_rot = assign_formation_slots([0, 1], leader_id=0, spacing=1.0, leader_heading=np.pi / 2)
        leader_pos = np.array([0.0, 0.0])
        world_zero = state_zero.desired_world_position(1, leader_pos)
        world_rot = state_rot.desired_world_position(1, leader_pos)
        assert world_zero is not None and world_rot is not None
        np.testing.assert_allclose(world_zero, np.array([-1.0, -1.0]), atol=1e-12)
        np.testing.assert_allclose(world_rot, np.array([1.0, -1.0]), atol=1e-12)


class TestFormationSpec(unittest.TestCase):
    def test_frozen(self) -> None:
        spec = FormationSpec.from_state(0, assign_formation_slots([0, 1], leader_id=0))
        with self.assertRaises(Exception):
            spec.leader_id = 99  # type: ignore[misc]

    def test_from_state_round_trip(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0, formation_type=FormationType.WEDGE)
        spec = FormationSpec.from_state(group_id=7, state=state)
        self.assertEqual(spec.group_id, 7)
        self.assertEqual(spec.formation_type, FormationType.WEDGE)
        self.assertEqual(spec.slot_assignments, state.slot_assignments)
        self.assertGreater(len(spec.edges), 0)

    def test_desired_world_matches_transform(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, leader_heading=0.5)
        spec = FormationSpec.from_state(0, state)
        leader_pos = np.array([4.0, -2.0])
        from_state = state.desired_world_position(1, leader_pos)
        from_spec = spec.desired_world_position(leader_pos, 1)
        np.testing.assert_allclose(from_state, from_spec)


class TestFormationRenderer(unittest.TestCase):
    def test_headless_formation_overlay(self) -> None:
        try:
            import pygame
        except ImportError:
            self.skipTest("pygame is not installed")

        from dataclasses import replace
        from pathlib import Path

        from src.algorithms.aggregation.self_aggregation import SelfAggregationController
        from src.agents.uav import spawn_uavs
        from src.config.loader import load_config
        from src.environment.world import World
        from src.simulation.simulation_engine import SimulationEngine
        from src.visualization.pygame_renderer import PygameRenderer

        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        config = replace(load_config(config_path), duration=0.5, num_uavs=4)
        world = World.from_config(config.environment, config.uav)
        center = np.array([config.spawn_center_x, config.spawn_center_y])
        agents = spawn_uavs(
            count=config.num_uavs,
            center=center,
            spread_radius=config.uav.initial_spread_radius,
            mission_radius=config.aggregation.mission_region_radius,
            max_speed=config.uav.max_speed,
            max_angular_velocity=config.uav.max_angular_velocity,
            seed=0,
        )
        state = assign_formation_slots([a.agent_id for a in agents], leader_id=agents[0].agent_id)
        world.formation_states = [state]
        world.formation_specs = [FormationSpec.from_state(0, state)]

        aggregation = SelfAggregationController(
            config=config.aggregation,
            uav_config=config.uav,
            rng=np.random.default_rng(0),
        )
        engine = SimulationEngine(world, agents, aggregation, config)
        renderer = PygameRenderer(world, engine, screen_width=400, screen_height=400)
        self.assertTrue(renderer.toggles.show_formations)

        pygame.init()
        try:
            surface = pygame.Surface((400, 400))
            engine.step()
            renderer.draw_frame(engine.get_state(), surface=surface)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
