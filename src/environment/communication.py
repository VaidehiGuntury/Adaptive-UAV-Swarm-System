"""
Communication topology for decentralized swarms.

Paper 1: range-limited neighbor discovery for pairwise IDE (future).
Paper 2: directed weighted graph G = (Q, E, W) with leader-follower tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from src.agents.base_agent import BaseAgent


@dataclass
class CommunicationGraph:
    """
    Directed communication graph backed by NetworkX.

    Supports range-based undirected discovery (Paper 1) and fixed
    adjacency matrices (Paper 2).
    """

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    @property
    def graph(self) -> nx.DiGraph:
        """Underlying NetworkX graph."""
        return self._graph

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._graph.clear()

    def add_agent(self, agent_id: int) -> None:
        """Register an agent as a graph node."""
        self._graph.add_node(agent_id)

    def build_range_graph(
        self,
        agents: list[BaseAgent],
        range_m: float,
        undirected: bool = True,
    ) -> None:
        """
        Build edges between agents within ``range_m`` (Paper 1 comm model).

        When ``undirected`` is True, adds both (i, j) and (j, i) with
        weight equal to Euclidean distance.
        """
        self.clear()
        for agent in agents:
            self.add_agent(agent.agent_id)

        for i, agent_i in enumerate(agents):
            for agent_j in agents[i + 1 :]:
                dist = agent_i.compute_distance(agent_j)
                if dist <= range_m:
                    self._graph.add_edge(agent_i.agent_id, agent_j.agent_id, weight=dist)
                    if undirected:
                        self._graph.add_edge(agent_j.agent_id, agent_i.agent_id, weight=dist)

    @classmethod
    def from_adjacency_matrix(
        cls,
        weight_matrix: NDArray[np.float64],
        agent_ids: list[int],
    ) -> CommunicationGraph:
        """
        Construct a directed graph from an adjacency matrix (Paper 2).

        Non-zero entry W[i, j] creates a directed edge agent_ids[i] → agent_ids[j].
        """
        if weight_matrix.shape[0] != len(agent_ids) or weight_matrix.shape[1] != len(agent_ids):
            raise ValueError("Adjacency matrix dimensions must match agent_ids length.")

        comm = cls()
        for agent_id in agent_ids:
            comm.add_agent(agent_id)

        n = len(agent_ids)
        for row in range(n):
            for col in range(n):
                weight = float(weight_matrix[row, col])
                if weight != 0.0:
                    comm._graph.add_edge(agent_ids[row], agent_ids[col], weight=weight)
        return comm

    def neighbors(self, agent_id: int) -> list[int]:
        """Return successor node IDs (out-neighbors) for ``agent_id``."""
        if agent_id not in self._graph:
            return []
        return list(self._graph.successors(agent_id))

    def is_follower_subgraph_connected(self, follower_ids: frozenset[int]) -> bool:
        """True when follower nodes form one weakly connected component."""
        if len(follower_ids) <= 1:
            return True
        subgraph = self._graph.subgraph(follower_ids)
        if subgraph.number_of_nodes() != len(follower_ids):
            return False
        return nx.is_weakly_connected(subgraph)

    def edges(self) -> list[tuple[int, int, float]]:
        """Return all directed edges as (source, target, weight) tuples."""
        result: list[tuple[int, int, float]] = []
        for source, target, data in self._graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            result.append((int(source), int(target), weight))
        return result
