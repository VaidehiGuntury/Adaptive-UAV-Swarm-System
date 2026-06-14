"""Unit tests for Paper 2 leader-follower architecture."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.algorithms.formation import (
    assign_formation_slots,
    build_formation_group_snapshot,
    is_slot_occupied_by_distance,
    leader_agent_from_pose,
)
from src.algorithms.formation.follower_state import compute_follower_slot_state
from src.algorithms.formation.formation_types import transform_to_world
from src.environment.formation_spec import FormationSpec


class TestLeaderAgent(unittest.TestCase):
    def test_forward_and_left_directions(self) -> None:
        leader = leader_agent_from_pose(0, np.array([1.0, 2.0]), heading=np.pi / 2)
        np.testing.assert_allclose(leader.forward_direction, np.array([0.0, 1.0]), atol=1e-12)
        np.testing.assert_allclose(leader.left_direction, np.array([-1.0, 0.0]), atol=1e-12)

    def test_transform_round_trip(self) -> None:
        leader = leader_agent_from_pose(0, np.array([3.0, -1.0]), heading=0.4)
        local = np.array([1.5, -0.5])
        recovered = leader.to_local(leader.to_world(local))
        np.testing.assert_allclose(recovered, local, atol=1e-12)

    def test_formation_target_defaults_none(self) -> None:
        leader = leader_agent_from_pose(0, np.zeros(2), heading=0.0)
        self.assertIsNone(leader.formation_target)


class TestFollowerSlotState(unittest.TestCase):
    def test_world_slot_uses_heading(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, spacing=1.0, leader_heading=np.pi / 2)
        leader = leader_agent_from_pose(0, np.zeros(2), heading=np.pi / 2)
        follower = compute_follower_slot_state(
            leader,
            state,
            agent_id=1,
            agent_position=np.array([1.0, -1.0]),
        )
        assert follower is not None
        expected = transform_to_world(state.desired_offsets[1], leader.position, leader.heading)
        np.testing.assert_allclose(follower.world_slot_position, expected, atol=1e-12)

    def test_error_vector_is_agent_minus_slot(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, spacing=2.0)
        leader = leader_agent_from_pose(0, np.array([5.0, 5.0]), heading=0.0)
        agent_pos = np.array([6.0, 2.0])
        follower = compute_follower_slot_state(leader, state, 1, agent_pos)
        assert follower is not None
        np.testing.assert_allclose(
            follower.formation_error,
            agent_pos - follower.world_slot_position,
            atol=1e-12,
        )


class TestOccupancyBoundary(unittest.TestCase):
    def test_inclusive_at_tolerance(self) -> None:
        tolerance = 0.5
        self.assertTrue(is_slot_occupied_by_distance(tolerance, tolerance))
        self.assertTrue(is_slot_occupied_by_distance(tolerance - 1e-12, tolerance))
        self.assertFalse(is_slot_occupied_by_distance(tolerance + 1e-9, tolerance))

    def test_follower_occupied_at_exact_tolerance(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, spacing=2.0)
        state.slot_tolerance = 0.5
        leader = leader_agent_from_pose(0, np.zeros(2), heading=0.0)
        desired = transform_to_world(state.desired_offsets[1], leader.position, leader.heading)
        offset = np.array([0.5, 0.0])
        agent_pos = desired + offset
        self.assertAlmostEqual(float(np.linalg.norm(offset)), 0.5)

        follower = compute_follower_slot_state(leader, state, 1, agent_pos)
        assert follower is not None
        self.assertTrue(follower.is_occupied)
        self.assertTrue(
            state.is_slot_occupied(1, agent_pos, leader.position),
        )

    def test_follower_vacant_just_outside_tolerance(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, spacing=2.0)
        state.slot_tolerance = 0.5
        leader = leader_agent_from_pose(0, np.zeros(2), heading=0.0)
        desired = transform_to_world(state.desired_offsets[1], leader.position, leader.heading)
        agent_pos = desired + np.array([0.5 + 1e-6, 0.0])

        follower = compute_follower_slot_state(leader, state, 1, agent_pos)
        assert follower is not None
        self.assertFalse(follower.is_occupied)


class TestFormationMetrics(unittest.TestCase):
    def test_mean_error_excludes_leader(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0, spacing=2.0)
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 0.0]),
            2: np.array([0.0, 10.0]),
        }
        snapshot = build_formation_group_snapshot(state, positions, timestep=3, time_s=1.5)
        assert snapshot is not None
        self.assertEqual(len(snapshot.followers), 2)
        self.assertGreater(snapshot.metrics.mean_formation_error, 0.0)

    def test_occupancy_counts_include_leader_slot(self) -> None:
        state = assign_formation_slots([0, 1, 2], leader_id=0, spacing=2.0)
        state.slot_tolerance = 0.5
        positions = {0: np.zeros(2), 1: np.zeros(2), 2: np.zeros(2)}
        snapshot = build_formation_group_snapshot(state, positions, timestep=1, time_s=0.5)
        assert snapshot is not None
        self.assertEqual(snapshot.metrics.occupied_slot_count, 1)
        self.assertEqual(snapshot.metrics.vacant_slot_count, 2)


class TestFormationGroupSnapshot(unittest.TestCase):
    def test_timing_fields(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0)
        positions = {0: np.zeros(2), 1: np.array([1.0, 1.0])}
        snapshot = build_formation_group_snapshot(
            state,
            positions,
            timestep=42,
            time_s=2.1,
        )
        assert snapshot is not None
        self.assertEqual(snapshot.timestep, 42)
        self.assertAlmostEqual(snapshot.time_s, 2.1)

    def test_leader_formation_target_not_from_bsa(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0)
        positions = {0: np.zeros(2), 1: np.array([1.0, 0.0])}
        snapshot = build_formation_group_snapshot(state, positions)
        assert snapshot is not None
        self.assertIsNone(snapshot.leader.formation_target)

    def test_heading_from_agent_map(self) -> None:
        state = assign_formation_slots([0, 1], leader_id=0, leader_heading=0.0)
        positions = {0: np.zeros(2), 1: np.array([0.0, -2.0])}
        snapshot = build_formation_group_snapshot(
            state,
            positions,
            headings={0: np.pi / 2},
        )
        assert snapshot is not None
        self.assertAlmostEqual(snapshot.leader.heading, np.pi / 2)


class TestLeaderFollowerRenderer(unittest.TestCase):
    def test_headless_snapshot_overlay(self) -> None:
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
        formation = assign_formation_slots([a.agent_id for a in agents], leader_id=agents[0].agent_id)
        world.formation_states = [formation]
        world.formation_specs = [FormationSpec.from_state(0, formation)]

        aggregation = SelfAggregationController(
            config=config.aggregation,
            uav_config=config.uav,
            rng=np.random.default_rng(0),
        )
        engine = SimulationEngine(world, agents, aggregation, config)
        renderer = PygameRenderer(world, engine, screen_width=400, screen_height=400)

        pygame.init()
        try:
            surface = pygame.Surface((400, 400))
            engine.step()
            renderer.draw_frame(engine.get_state(), surface=surface)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
