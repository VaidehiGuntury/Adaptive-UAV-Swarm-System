"""Regression test: SimulationEngine's incremental revisit_ratio tracking
must stay identical to a from-scratch rescan of the same trajectory data.

This proves the O(agents*N) incremental fix in SimulationEngine (running
counters updated in step()) produces the same value as the original
O(agents*N^2) approach of calling revisit_ratio() on the full agent_histories
every tick — checked at multiple points during a run, not just at the end.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.agents.uav import spawn_uavs
from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.config.loader import load_config
from src.environment.world import World
from src.evaluation.exploration_metrics import revisit_ratio
from src.simulation.simulation_engine import SimulationEngine


class TestRevisitRatioIncrementalEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.yaml"
        config = replace(load_config(config_path), duration=2.0, num_uavs=4)

        world = World.from_config(config.environment, config.uav)
        center = np.array([config.spawn_center_x, config.spawn_center_y])
        agents = spawn_uavs(
            count=config.num_uavs,
            center=center,
            spread_radius=config.uav.initial_spread_radius,
            mission_radius=config.aggregation.mission_region_radius,
            max_speed=config.uav.max_speed,
            max_angular_velocity=config.uav.max_angular_velocity,
            seed=2,
        )
        aggregation = SelfAggregationController(
            config=config.aggregation,
            uav_config=config.uav,
            rng=np.random.default_rng(2),
        )
        self.engine = SimulationEngine(world, agents, aggregation, config)

    def test_incremental_matches_full_rescan_at_multiple_checkpoints(self) -> None:
        checkpoints: list[tuple[float, float]] = []  # (incremental, from_scratch)

        for step_num in range(1, 13):
            self.engine.step()
            if step_num in (5, 12):
                incremental = self.engine.metrics_history[-1].revisit_ratio
                # Snapshot agent_histories as they stand right now — a fresh
                # copy so later appends (from later steps) can't leak in.
                snapshot = {
                    agent_id: list(trail)
                    for agent_id, trail in self.engine.agent_histories.items()
                }
                from_scratch = revisit_ratio(snapshot, self.engine.world.map)
                checkpoints.append((incremental, from_scratch))

        self.assertEqual(len(checkpoints), 2)
        for incremental, from_scratch in checkpoints:
            self.assertAlmostEqual(incremental, from_scratch, places=12)

        # Sanity: with 4 agents confined to a coarse grid (resolution =
        # sensing_range/3 = 1.5m) moving at up to 0.15m/step, some revisits
        # are expected — otherwise the equality above would be trivially 0.0
        # vs 0.0 and wouldn't actually exercise the revisit-counting logic.
        self.assertGreater(checkpoints[-1][1], 0.0)


if __name__ == "__main__":
    unittest.main()
