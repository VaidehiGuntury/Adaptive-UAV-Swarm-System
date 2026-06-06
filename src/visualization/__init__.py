"""Simulation visualization backends."""

from src.visualization.base import RendererProtocol
from src.visualization.renderer import SimulationRenderer

__all__ = [
    "PygameRenderer",
    "RendererProtocol",
    "SimulationRenderer",
]


def __getattr__(name: str) -> object:
    """Lazy-load PygameRenderer so Matplotlib workflows work without pygame installed."""
    if name == "PygameRenderer":
        from src.visualization.pygame_renderer import PygameRenderer

        return PygameRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
