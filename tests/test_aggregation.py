"""Unit tests for Paper 1 self-aggregation module."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.fitness_functions import (
    ViewpointCandidate,
    aggregation_cost_j_c,
    compute_lambda_beta,
    evaluate_viewpoint_cost,
    trail_penalty_j_l,
    turning_cost_j_v,
    utility_u_a,
)
from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.agents.uav import UAV, spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine


class TestAgentInitialization(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        self.config = load_config(config_path)
        center = np.array([50.0, 50.0], dtype=np.float64)
        self.agents = spawn_uavs(
            count=10,
            center=center,
            spread_radius=self.config.uav.initial_spread_radius,
            mission_radius=self.config.aggregation.mission_region_radius,
            max_speed=self.config.uav.max_speed,
            max_angular_velocity=self.config.uav.max_angular_velocity,
            seed=0,
        )

    def test_spawn_count(self) -> None:
        self.assertEqual(len(self.agents), 10)

    def test_unique_ids(self) -> None:
        ids = {agent.agent_id for agent in self.agents}
        self.assertEqual(len(ids), 10)

    def test_assigned_region_and_target(self) -> None:
        for agent in self.agents:
            self.assertIsNotNone(agent.assigned_region)
            self.assertIsNotNone(agent.assigned_target)
            np.testing.assert_array_equal(
                agent.assigned_region.center,
                agent.assigned_target,
            )

    def test_set_target_updates_state(self) -> None:
        agent = self.agents[0]
        target = np.array([10.0, 20.0])
        agent.set_target(target)
        np.testing.assert_array_equal(agent.assigned_target, target)

    def test_compute_distance(self) -> None:
        self.agents[0].position = np.array([0.0, 0.0])
        self.agents[1].position = np.array([3.0, 4.0])
        self.assertAlmostEqual(self.agents[0].compute_distance(self.agents[1]), 5.0)


class TestFitnessFunctions(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        self.config = load_config(config_path)
        self.agg = self.config.aggregation

    def test_lambda_beta_finite(self) -> None:
        beta = compute_lambda_beta(self.agg)
        self.assertTrue(np.isfinite(beta))

    def test_utility_zero_beyond_d0(self) -> None:
        p_a = np.array([0.0, 0.0])
        p_b = np.array([self.agg.d_0 + 1.0, 0.0])
        self.assertEqual(utility_u_a(p_a, p_b, self.agg), 0.0)

    def test_trail_penalty(self) -> None:
        candidate = ViewpointCandidate(
            viewpoint=np.zeros(2),
            yaw=0.0,
            cluster_id=0,
            is_trail=True,
        )
        self.assertEqual(trail_penalty_j_l(candidate, self.agg), self.agg.trail_penalty)

    def test_aggregation_prefers_own_target(self) -> None:
        agent = UAV(agent_id=0, position=np.array([0.0, 0.0]))
        agent.set_region(np.array([5.0, 0.0]), radius=4.5)
        other = UAV(agent_id=1, position=np.array([20.0, 0.0]))
        other.set_region(np.array([20.0, 0.0]), radius=4.5)

        near_own = ViewpointCandidate(
            viewpoint=np.array([5.5, 0.0]),
            yaw=0.0,
            cluster_id=1,
            is_trail=False,
        )
        near_other = ViewpointCandidate(
            viewpoint=np.array([19.5, 0.0]),
            yaw=0.0,
            cluster_id=2,
            is_trail=False,
        )
        cost_own = evaluate_viewpoint_cost(near_own, agent, [agent, other], self.agg)
        cost_other = evaluate_viewpoint_cost(near_other, agent, [agent, other], self.agg)
        self.assertGreater(cost_own, cost_other)


class TestAggregationBehaviour(unittest.TestCase):
    def test_simulation_advances_agents(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        config = replace(load_config(config_path), duration=2.0, num_uavs=3)

        world = World.from_config(config.environment, config.uav)
        center = np.array([config.spawn_center_x, config.spawn_center_y])
        agents = spawn_uavs(
            count=config.num_uavs,
            center=center,
            spread_radius=config.uav.initial_spread_radius,
            mission_radius=config.aggregation.mission_region_radius,
            max_speed=config.uav.max_speed,
            max_angular_velocity=config.uav.max_angular_velocity,
            seed=1,
        )
        aggregation = SelfAggregationController(
            config=config.aggregation,
            uav_config=config.uav,
            rng=np.random.default_rng(1),
        )
        engine = SimulationEngine(world, agents, aggregation, config)

        initial_positions = [agent.position.copy() for agent in agents]
        for _ in range(5):
            engine.step()

        moved = any(
            float(np.linalg.norm(agent.position - initial_positions[i])) > 1e-3
            for i, agent in enumerate(agents)
        )
        self.assertTrue(moved)
        self.assertGreater(engine.metrics_history[-1].explored_fraction, 0.0)

    def test_bsa_assigns_target_on_replan(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        config = load_config(config_path)
        world = World.from_config(config.environment, config.uav)
        agent = UAV(agent_id=0, position=np.array([50.0, 50.0]))
        agent.set_region(np.array([50.0, 50.0]), config.aggregation.mission_region_radius)
        agg_config = replace(config.aggregation, replan_interval=0.0)
        controller = SelfAggregationController(
            config=agg_config,
            uav_config=config.uav,
            rng=np.random.default_rng(0),
        )
        controller.update(agent, [agent], world, dt=0.1)
        self.assertIsNotNone(agent.assigned_target)


if __name__ == "__main__":
    unittest.main()
