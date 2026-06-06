"""
Matplotlib animation renderer (independent from simulation logic).

Visualizes UAV positions, trajectories, obstacles, and aggregation targets.
"""

from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine, SimulationState


class SimulationRenderer:
    """FuncAnimation-based renderer consuming simulation snapshots."""

    def __init__(
        self,
        world: World,
        engine: SimulationEngine,
        interval_ms: int = 50,
    ) -> None:
        self.world = world
        self.engine = engine
        self.interval_ms = interval_ms
        self._history: list[SimulationState] = []
        self._fig: plt.Figure | None = None
        self._animation: FuncAnimation | None = None

    def record_step(self) -> None:
        """Capture current simulation state for playback."""
        self._history.append(self.engine.get_state())

    def run_and_record(self) -> list[SimulationState]:
        """Run simulation and record every timestep."""
        self._history.clear()
        self.record_step()
        while self.engine.time_s < self.engine.config.duration:
            self.engine.step()
            self.record_step()
        return list(self._history)

    def animate(self, show: bool = True) -> FuncAnimation:
        """
        Build FuncAnimation from recorded history.

        If no history exists, runs the simulation first.
        """
        if not self._history:
            self.run_and_record()

        self._fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.world.width)
        ax.set_ylim(0, self.world.height)
        ax.set_aspect("equal")
        ax.set_title("Paper 1 — Bio-inspired Self-Aggregation")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        for obstacle in self.world.obstacles.obstacles:
            patch = Circle(
                obstacle.center,
                obstacle.radius,
                color="gray",
                alpha=0.6,
            )
            ax.add_patch(patch)

        scatter = ax.scatter([], [], c="tab:blue", s=40, zorder=3)
        target_scatter = ax.scatter([], [], c="tab:orange", marker="x", s=30, zorder=3)
        trajectory_lines = [
            ax.plot([], [], linewidth=1, alpha=0.5)[0] for _ in self.engine.agents
        ]
        id_texts = [ax.text(0, 0, "", fontsize=8) for _ in self.engine.agents]
        time_text = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
        metrics_text = ax.text(
            0.02,
            0.92,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        def _update(frame: int) -> list:
            state = self._history[frame]
            positions = np.array([agent.position for agent in state.agents])
            scatter.set_offsets(positions)

            targets = [
                agent.assigned_target
                for agent in state.agents
                if agent.assigned_target is not None
            ]
            if targets:
                target_scatter.set_offsets(np.array(targets))
            else:
                target_scatter.set_offsets(np.empty((0, 2)))

            for idx, agent in enumerate(state.agents):
                history = self.engine.agent_histories[agent.agent_id]
                end = min(frame + 1, len(history))
                trail = np.array(history[:end])
                trajectory_lines[idx].set_data(trail[:, 0], trail[:, 1])
                id_texts[idx].set_position(
                    (agent.position[0] + 0.4, agent.position[1] + 0.4)
                )
                id_texts[idx].set_text(str(agent.agent_id))

            time_text.set_text(f"t = {state.time_s:.1f} s  |  step = {state.timestep}")
            metrics_text.set_text(
                f"explored = {state.metrics.explored_fraction * 100:.1f}%  "
                f"mean speed = {state.metrics.mean_speed:.2f} m/s"
            )
            artists: list = [
                scatter,
                target_scatter,
                time_text,
                metrics_text,
                *trajectory_lines,
                *id_texts,
            ]
            return artists

        self._animation = FuncAnimation(
            self._fig,
            _update,
            frames=len(self._history),
            interval=self.interval_ms,
            blit=False,
        )

        if show:
            plt.show()
        return self._animation

    @staticmethod
    def render_snapshot(
        ax: plt.Axes,
        world: World,
        agents: list[UAV],
        trajectories: dict[int, NDArray[np.float64]] | None = None,
        timestep: int = 0,
        time_s: float = 0.0,
    ) -> None:
        """Render a single static frame (useful for tests and notebooks)."""
        ax.clear()
        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
        ax.set_aspect("equal")

        for obstacle in world.obstacles.obstacles:
            ax.add_patch(Circle(obstacle.center, obstacle.radius, color="gray", alpha=0.6))

        positions = np.array([agent.position for agent in agents])
        ax.scatter(positions[:, 0], positions[:, 1], c="tab:blue", s=40)

        if trajectories:
            for agent_id, trail in trajectories.items():
                ax.plot(trail[:, 0], trail[:, 1], linewidth=1, alpha=0.5)

        for agent in agents:
            ax.text(
                agent.position[0] + 0.4,
                agent.position[1] + 0.4,
                str(agent.agent_id),
                fontsize=8,
            )

        ax.set_title(f"Swarm simulation — t={time_s:.1f}s, step={timestep}")
