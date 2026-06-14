"""
Communication graph refresh for formation control.
"""

from __future__ import annotations

from src.agents.base_agent import BaseAgent
from src.environment.communication import CommunicationGraph
from src.environment.world import World


def refresh_formation_comm_graph(
    world: World,
    agents: list[BaseAgent],
    range_m: float,
) -> CommunicationGraph:
    """Rebuild range-limited communication graph and store on ``world``."""
    if world.communication_graph is None:
        world.communication_graph = CommunicationGraph()
    graph = world.communication_graph
    graph.build_range_graph(agents, range_m=range_m, undirected=True)
    return graph
