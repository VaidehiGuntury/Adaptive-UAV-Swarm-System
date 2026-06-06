"""Unit tests for communication graph module."""

from __future__ import annotations

import unittest

import numpy as np

from src.agents.uav import UAV
from src.environment.communication import CommunicationGraph


class TestCommunicationGraph(unittest.TestCase):
    def test_range_graph_connects_nearby_agents(self) -> None:
        agents = [
            UAV(agent_id=0, position=np.array([0.0, 0.0])),
            UAV(agent_id=1, position=np.array([5.0, 0.0])),
            UAV(agent_id=2, position=np.array([50.0, 0.0])),
        ]
        graph = CommunicationGraph()
        graph.build_range_graph(agents, range_m=10.0)

        self.assertIn(1, graph.neighbors(0))
        self.assertIn(0, graph.neighbors(1))
        self.assertNotIn(2, graph.neighbors(0))

    def test_range_graph_boundary_excludes_distant_pair(self) -> None:
        agents = [
            UAV(agent_id=0, position=np.array([0.0, 0.0])),
            UAV(agent_id=1, position=np.array([10.0, 0.0])),
        ]
        graph = CommunicationGraph()
        graph.build_range_graph(agents, range_m=10.0)
        self.assertIn(1, graph.neighbors(0))

        graph.build_range_graph(agents, range_m=9.9)
        self.assertNotIn(1, graph.neighbors(0))

    def test_adjacency_matrix_directed_edges(self) -> None:
        w = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        graph = CommunicationGraph.from_adjacency_matrix(w, agent_ids=[0, 1])
        self.assertEqual(graph.neighbors(0), [1])
        self.assertEqual(graph.neighbors(1), [])

    def test_edges_returns_weights(self) -> None:
        agents = [
            UAV(agent_id=0, position=np.array([0.0, 0.0])),
            UAV(agent_id=1, position=np.array([3.0, 4.0])),
        ]
        graph = CommunicationGraph()
        graph.build_range_graph(agents, range_m=10.0, undirected=False)
        edges = graph.edges()
        self.assertEqual(len(edges), 1)
        self.assertAlmostEqual(edges[0][2], 5.0)


if __name__ == "__main__":
    unittest.main()
