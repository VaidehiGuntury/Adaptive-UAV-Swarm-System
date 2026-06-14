"""Unit tests for Paper 2 consensus formation tracking."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.algorithms.formation.consensus_controller import (
    ConsensusConfig,
    compute_consensus_correction,
    follower_neighbor_ids,
    scaled_consensus_command,
)
from src.algorithms.formation.consensus_metrics import compute_consensus_metrics
from src.algorithms.formation.formation_controller import (
    FormationAcquisitionConfig,
    FormationAcquisitionController,
)
from src.algorithms.formation.slot_assignment import assign_formation_slots
from src.algorithms.formation.velocity_control import clamp_velocity, proportional_slot_command
from src.agents.uav import UAV, spawn_uavs
from src.config.loader import load_config
from src.environment.communication import CommunicationGraph
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine


class TestVelocityControl(unittest.TestCase):
    def test_proportional_direction(self) -> None:
        error = np.array([2.0, 0.0])
        cmd = proportional_slot_command(error, proportional_gain=0.5, dead_zone=0.1)
        np.testing.assert_allclose(cmd, np.array([1.0, 0.0]))

    def test_dead_zone(self) -> None:
        error = np.array([0.1, 0.0])
        cmd = proportional_slot_command(error, proportional_gain=1.0, dead_zone=0.15)
        np.testing.assert_allclose(cmd, np.zeros(2))

    def test_speed_clamping(self) -> None:
        cmd = clamp_velocity(np.array([3.0, 4.0]), max_speed=1.0)
        self.assertAlmostEqual(float(np.linalg.norm(cmd)), 1.0)


class TestConsensusCorrection(unittest.TestCase):
    def test_consensus_vector_sum(self) -> None:
        positions = {
            0: np.array([0.0, 0.0]),
            1: np.array([2.0, 0.0]),
            2: np.array([0.0, 2.0]),
        }
        raw = compute_consensus_correction(0, positions, (1, 2))
        np.testing.assert_allclose(raw, np.array([2.0, 2.0]))

    def test_isolated_follower_zero_consensus(self) -> None:
        positions = {1: np.array([0.0, 0.0])}
        raw = compute_consensus_correction(1, positions, ())
        np.testing.assert_allclose(raw, np.zeros(2))

    def test_composition_with_gain(self) -> None:
        raw = np.array([2.0, 0.0])
        scaled = scaled_consensus_command(raw, 0.15)
        np.testing.assert_allclose(scaled, np.array([0.3, 0.0]))


class TestCommunicationGraphHelpers(unittest.TestCase):
    def test_follower_neighbors_exclude_leader(self) -> None:
        graph = CommunicationGraph.from_adjacency_matrix(
            np.array(
                [
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            agent_ids=[0, 1, 2],
        )
        followers = frozenset({1, 2})
        self.assertEqual(follower_neighbor_ids(graph, 1, followers), (2,))

    def test_disconnected_subgraph(self) -> None:
        graph = CommunicationGraph()
        graph.build_range_graph(
            [
                UAV(agent_id=1, position=np.array([0.0, 0.0])),
                UAV(agent_id=2, position=np.array([50.0, 0.0])),
            ],
            range_m=5.0,
        )
        self.assertFalse(graph.is_follower_subgraph_connected(frozenset({1, 2})))


class TestFormationActivation(unittest.TestCase):
    def test_activation_threshold(self) -> None:
        config = FormationAcquisitionConfig(activation_mean_pairwise=10.0)
        controller = FormationAcquisitionController(config=config)
        agents = [
            UAV(agent_id=0, position=np.array([0.0, 0.0])),
            UAV(agent_id=1, position=np.array([20.0, 0.0])),
        ]
        world = World(width=100.0, height=100.0, obstacles=__import__(
            "src.environment.obstacles", fromlist=["ObstacleField"]
        ).ObstacleField([]))
        controller.update_activation(agents, world, mean_pairwise_distance=15.0, time_s=1.0)
        self.assertFalse(controller.is_active)

        controller.update_activation(agents, world, mean_pairwise_distance=8.0, time_s=2.0)
        self.assertTrue(controller.is_active)
        self.assertEqual(controller.leader_id, 0)


class TestSimulationIntegration(unittest.TestCase):
    def setUp(self) -> None:
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        self.config = replace(load_config(config_path), duration=2.0, num_uavs=4)
        self.world = World.from_config(self.config.environment, self.config.uav)
        center = np.array([self.config.spawn_center_x, self.config.spawn_center_y])
        self.agents = spawn_uavs(
            count=self.config.num_uavs,
            center=center,
            spread_radius=3.0,
            mission_radius=self.config.aggregation.mission_region_radius,
            max_speed=self.config.uav.max_speed,
            max_angular_velocity=self.config.uav.max_angular_velocity,
            seed=0,
        )
        self.aggregation = SelfAggregationController(
            config=self.config.aggregation,
            uav_config=self.config.uav,
            rng=np.random.default_rng(0),
        )
        self.controller = FormationAcquisitionController(
            config=FormationAcquisitionConfig(activation_mean_pairwise=50.0),
            consensus_config=ConsensusConfig(enabled=True),
        )
        self.engine = SimulationEngine(
            self.world,
            self.agents,
            self.aggregation,
            self.config,
            formation_controller=self.controller,
        )

    def test_headless_smoke_with_consensus(self) -> None:
        for _ in range(10):
            self.engine.step()
        self.assertTrue(self.controller.is_active)
        metrics = self.engine.metrics_history[-1]
        self.assertTrue(metrics.formation_active)
        self.assertTrue(metrics.consensus_active)

    def test_heading_aware_slot_tracking(self) -> None:
        leader = self.agents[0]
        leader.heading = np.pi / 2
        state = assign_formation_slots([a.agent_id for a in self.agents], leader_id=0, spacing=1.0)
        self.controller.formation_state = state
        self.controller.is_active = True
        self.controller.apply_follower_control(
            self.agents,
            self.world,
            dt=0.1,
            communication_range=self.config.uav.sensing_range,
            time_s=1.0,
        )
        follower = self.agents[1]
        self.assertTrue(np.all(np.isfinite(follower.velocity)))


class TestConsensusMetrics(unittest.TestCase):
    def test_metrics_with_isolated_agent(self) -> None:
        follower_ids = frozenset({1, 2})
        graph = CommunicationGraph()
        graph.add_agent(1)
        graph.add_agent(2)
        positions = {1: np.zeros(2), 2: np.array([5.0, 0.0])}
        metrics = compute_consensus_metrics(
            follower_ids,
            graph,
            positions,
            scaled_corrections={1: np.zeros(2), 2: np.zeros(2)},
            neighbor_map={1: (), 2: ()},
        )
        self.assertEqual(metrics.isolated_follower_count, 2)
        self.assertFalse(metrics.graph_connectivity_status)


if __name__ == "__main__":
    unittest.main()
