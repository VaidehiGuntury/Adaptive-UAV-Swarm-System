"""Unit tests for belief map grid module."""

from __future__ import annotations

import unittest

import numpy as np

from src.environment.belief_map import BeliefMap


class TestBeliefMap(unittest.TestCase):
    def test_set_and_get_belief(self) -> None:
        belief = BeliefMap(width=40.0, height=40.0, resolution=1.0)
        belief.set_belief(5, 10, 0.75)
        self.assertAlmostEqual(belief.get_belief(5, 10), 0.75)

    def test_normalized_array_sums_to_one(self) -> None:
        belief = BeliefMap(width=10.0, height=10.0, resolution=1.0)
        belief.set_belief(0, 0, 2.0)
        belief.set_belief(1, 0, 2.0)
        normalized = belief.normalized_array()
        self.assertAlmostEqual(float(np.sum(normalized)), 1.0)
        self.assertAlmostEqual(float(normalized[0, 0]), 0.5)

    def test_set_uniform(self) -> None:
        belief = BeliefMap(width=4.0, height=4.0, resolution=1.0)
        belief.set_uniform()
        normalized = belief.normalized_array()
        self.assertAlmostEqual(float(np.sum(normalized)), 1.0)

    def test_world_to_grid_clamps(self) -> None:
        belief = BeliefMap(width=10.0, height=10.0, resolution=1.0)
        i, j = belief.world_to_grid(np.array([999.0, -5.0]))
        self.assertGreaterEqual(i, 0)
        self.assertGreaterEqual(j, 0)
        self.assertLess(i, belief.grid_shape[1])
        self.assertLess(j, belief.grid_shape[0])


if __name__ == "__main__":
    unittest.main()
