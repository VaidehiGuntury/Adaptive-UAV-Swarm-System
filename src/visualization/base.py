"""
Shared renderer interface for Matplotlib and Pygame backends.
"""

from __future__ import annotations

from typing import Protocol

from src.simulation.simulation_engine import SimulationEngine, SimulationState


class RendererProtocol(Protocol):
    """Contract for simulation visualization backends."""

    world: object
    engine: SimulationEngine

    def record_step(self) -> None:
        """Capture current simulation state for playback."""
        ...

    def run_and_record(self) -> list[SimulationState]:
        """Run simulation and record every timestep."""
        ...
