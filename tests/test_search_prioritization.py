"""Unit tests for STATIC target priority scoring (human vs. object)."""

from __future__ import annotations

import unittest

import numpy as np

from src.config.search_config import PriorityConfig, SearchConfig, TargetSpawnConfig
from src.search.prioritization import PriorityScorer
from src.search.target import Target, TargetType
from src.search.target_manager import TargetManager


class TestStaticTargetPriorityScoring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = PriorityConfig()
        self.scorer = PriorityScorer(self.cfg)

    def _make_static_target(self, is_human: bool) -> Target:
        return Target(
            target_id=0,
            target_type=TargetType.STATIC,
            position=np.array([10.0, 10.0], dtype=np.float64),
            is_human=is_human,
        )

    def test_static_human_type_score_uses_human_tier(self) -> None:
        target = self._make_static_target(is_human=True)
        self.assertEqual(self.scorer._type_score(target), self.cfg.score_static_human)

    def test_static_object_type_score_uses_object_tier(self) -> None:
        target = self._make_static_target(is_human=False)
        self.assertEqual(self.scorer._type_score(target), self.cfg.score_static_object)

    def test_static_human_scores_higher_than_static_object_overall(self) -> None:
        human = self._make_static_target(is_human=True)
        obj = self._make_static_target(is_human=False)
        observer = np.array([10.0, 10.0], dtype=np.float64)
        human_score = self.scorer.score(human, observer, current_time=0.0)
        obj_score = self.scorer.score(obj, observer, current_time=0.0)
        self.assertGreater(human_score, obj_score)


class TestStaticHumanSpawnIsAdditive(unittest.TestCase):
    def test_count_static_human_adds_to_count_static(self) -> None:
        spawn_cfg = TargetSpawnConfig(
            count_static=3,
            count_static_human=2,
            count_dynamic=0,
            count_time_varying=0,
        )
        manager = TargetManager(
            config=SearchConfig(targets=spawn_cfg),
            world_width=100.0,
            world_height=100.0,
        )
        spawned = manager.spawn_targets(current_time=0.0)

        self.assertEqual(len(spawned), 5)
        self.assertEqual(sum(1 for t in spawned if t.is_human), 2)
        self.assertEqual(sum(1 for t in spawned if not t.is_human), 3)
        self.assertTrue(all(t.target_type == TargetType.STATIC for t in spawned))


if __name__ == "__main__":
    unittest.main()
