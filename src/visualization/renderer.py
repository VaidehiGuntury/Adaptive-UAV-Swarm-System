"""
Matplotlib animation renderer (independent from simulation logic).

Visualizes UAV positions, trajectories, obstacles, and aggregation targets.
Dynamic obstacles are rendered with speed-coded colours when
``world.obstacle_manager`` is not None (SDS §37).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow, Circle
from numpy.typing import NDArray

from src.agents.uav import UAV
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine, SimulationState

# ---------------------------------------------------------------------------
# Speed-classification thresholds and colours (SDS §37)
# Relative to UAV max_speed (1.5 m/s by default):
#   slow   : speed < 0.8 × v_uav  → green
#   equal  : 0.8 ≤ speed ≤ 1.2   → orange
#   fast   : speed > 1.2 × v_uav  → red
# ---------------------------------------------------------------------------
_SPEED_SLOW_RATIO = 0.8
_SPEED_FAST_RATIO = 1.2
_COLOR_DYN_SLOW = "#50c878"     # green
_COLOR_DYN_EQUAL = "#ff9933"    # orange
_COLOR_DYN_FAST = "#e84040"     # red
_COLOR_DYN_PRED = "#a060d0"     # purple — prediction line
_COLOR_DYN_SAFETY = "#807840"   # muted yellow — safety ring
_VELOCITY_SCALE = 2.0           # world-metres per (m/s) for arrow length
_PREDICTION_HORIZON_S = 1.5     # seconds ahead for prediction overlay


@dataclass(frozen=True)
class _DynObstacleFrame:
    """
    Lightweight snapshot of one dynamic obstacle at a single simulation frame.

    Stored per-frame in ``SimulationRenderer._dyn_history`` so that playback
    shows the historically correct obstacle positions rather than the live
    end-of-run positions held by ``ObstacleManager``.

    All values are plain Python scalars or tuples so the dataclass is fully
    hashable and requires no NumPy state.
    """

    obstacle_id: str
    pos_x: float
    pos_y: float
    vel_x: float
    vel_y: float
    radius: float
    safety_margin: float


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
        # Parallel list of per-frame obstacle snapshots.
        # Entry i corresponds to _history[i].
        # Empty list when obstacle_manager is None (static world).
        self._dyn_history: list[list[_DynObstacleFrame]] = []
        self._fig: plt.Figure | None = None
        self._animation: FuncAnimation | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _capture_obstacle_snapshot(self) -> list[_DynObstacleFrame]:
        """
        Read the current state of all active dynamic obstacles.

        Returns an empty list when ``world.obstacle_manager`` is None so
        the call is always safe regardless of whether the dynamic environment
        is enabled.

        Returns
        -------
        list[_DynObstacleFrame]
            One entry per active obstacle, in insertion order.
        """
        manager = self.world.obstacle_manager
        if manager is None:
            return []
        frames: list[_DynObstacleFrame] = []
        for obs in manager:
            if obs.active:
                frames.append(
                    _DynObstacleFrame(
                        obstacle_id=obs.obstacle_id,
                        pos_x=float(obs.position[0]),
                        pos_y=float(obs.position[1]),
                        vel_x=float(obs.velocity[0]),
                        vel_y=float(obs.velocity[1]),
                        radius=obs.radius,
                        safety_margin=manager.safety_margin,
                    )
                )
        return frames

    def record_step(self) -> None:
        """Capture current simulation state and obstacle snapshot for playback."""
        self._history.append(self.engine.get_state())
        self._dyn_history.append(self._capture_obstacle_snapshot())

    def run_and_record(self) -> list[SimulationState]:
        """Run simulation and record every timestep."""
        self._history.clear()
        self._dyn_history.clear()
        self.record_step()
        while self.engine.time_s < self.engine.config.duration:
            self.engine.step()
            self.record_step()
        return list(self._history)

    def animate(self, show: bool = True) -> FuncAnimation:
        """
        Build FuncAnimation from recorded history.

        If no history exists, runs the simulation first.
        Dynamic obstacles are drawn from the per-frame snapshot stored in
        ``_dyn_history`` so each frame shows historically correct positions.
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

        # Static obstacles — drawn once, never updated
        for obstacle in self.world.obstacles.obstacles:
            patch = Circle(
                obstacle.center,
                obstacle.radius,
                color="gray",
                alpha=0.6,
            )
            ax.add_patch(patch)

        # Dynamic obstacle artists — one Circle + one Arrow per obstacle.
        # These are created eagerly using the first frame snapshot so their
        # count is stable across all frames (obstacles don't spawn/despawn
        # mid-mission in the current implementation).
        _dyn_circles: list[Circle] = []
        _dyn_arrows: list[Arrow] = []
        _dyn_safety_circles: list[Circle] = []
        _dyn_pred_lines: list = []  # Line2D objects

        first_snap = self._dyn_history[0] if self._dyn_history else []
        uav_max_speed: float = self.engine.config.uav.max_speed

        for snap in first_snap:
            color = _classify_dyn_color(
                float(np.hypot(snap.vel_x, snap.vel_y)),
                uav_max_speed,
            )
            # Body circle
            circ = Circle(
                (snap.pos_x, snap.pos_y),
                snap.radius,
                color=color,
                alpha=0.75,
                zorder=2,
            )
            ax.add_patch(circ)
            _dyn_circles.append(circ)

            # Safety-margin ring (dashed, always rendered — kept thin for clarity)
            safety_circ = Circle(
                (snap.pos_x, snap.pos_y),
                snap.radius + snap.safety_margin,
                fill=False,
                edgecolor=_COLOR_DYN_SAFETY,
                linewidth=0.8,
                linestyle="--",
                alpha=0.5,
                zorder=2,
            )
            ax.add_patch(safety_circ)
            _dyn_safety_circles.append(safety_circ)

            # Velocity arrow
            arrow = Arrow(
                snap.pos_x,
                snap.pos_y,
                snap.vel_x * _VELOCITY_SCALE,
                snap.vel_y * _VELOCITY_SCALE,
                width=snap.radius * 0.6,
                color=color,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(arrow)
            _dyn_arrows.append(arrow)

            # Prediction line
            pred_line, = ax.plot(
                [snap.pos_x, snap.pos_x + snap.vel_x * _PREDICTION_HORIZON_S],
                [snap.pos_y, snap.pos_y + snap.vel_y * _PREDICTION_HORIZON_S],
                color=_COLOR_DYN_PRED,
                linewidth=0.9,
                linestyle=":",
                alpha=0.6,
                zorder=2,
            )
            _dyn_pred_lines.append(pred_line)

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

            # Update dynamic obstacles from the per-frame snapshot
            snap_list = self._dyn_history[frame] if frame < len(self._dyn_history) else []
            for i, snap in enumerate(snap_list):
                speed = float(np.hypot(snap.vel_x, snap.vel_y))
                color = _classify_dyn_color(speed, uav_max_speed)

                # Update body circle
                _dyn_circles[i].center = (snap.pos_x, snap.pos_y)
                _dyn_circles[i].set_color(color)

                # Update safety ring
                _dyn_safety_circles[i].center = (snap.pos_x, snap.pos_y)

                # Replace Arrow (Matplotlib Arrow patches cannot be updated in-place)
                _dyn_arrows[i].remove()
                new_arrow = Arrow(
                    snap.pos_x,
                    snap.pos_y,
                    snap.vel_x * _VELOCITY_SCALE,
                    snap.vel_y * _VELOCITY_SCALE,
                    width=snap.radius * 0.6,
                    color=color,
                    alpha=0.85,
                    zorder=3,
                )
                ax.add_patch(new_arrow)
                _dyn_arrows[i] = new_arrow

                # Update prediction line
                px = snap.pos_x + snap.vel_x * _PREDICTION_HORIZON_S
                py = snap.pos_y + snap.vel_y * _PREDICTION_HORIZON_S
                _dyn_pred_lines[i].set_data(
                    [snap.pos_x, px],
                    [snap.pos_y, py],
                )

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
                *_dyn_circles,
                *_dyn_safety_circles,
                *_dyn_arrows,
                *_dyn_pred_lines,
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
        """
        Render a single static frame (useful for tests and notebooks).

        Dynamic obstacles are drawn from ``world.obstacle_manager`` when it
        is present, using the live state at the time of the call.
        """
        ax.clear()
        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
        ax.set_aspect("equal")

        # Static obstacles
        for obstacle in world.obstacles.obstacles:
            ax.add_patch(Circle(obstacle.center, obstacle.radius, color="gray", alpha=0.6))

        # Dynamic obstacles (if enabled)
        if world.obstacle_manager is not None:
            for obs in world.obstacle_manager:
                if not obs.active:
                    continue
                speed = float(np.linalg.norm(obs.velocity))
                # Use a nominal UAV max speed of 1.5 m/s as the classification
                # reference — render_snapshot is called without engine context.
                color = _classify_dyn_color(speed, uav_max_speed=1.5)
                ax.add_patch(
                    Circle(
                        obs.position,
                        obs.radius,
                        color=color,
                        alpha=0.75,
                        zorder=2,
                    )
                )
                # Safety ring
                ax.add_patch(
                    Circle(
                        obs.position,
                        obs.radius + world.obstacle_manager.safety_margin,
                        fill=False,
                        edgecolor=_COLOR_DYN_SAFETY,
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.5,
                        zorder=2,
                    )
                )
                # Velocity arrow
                if speed > 1e-6:
                    ax.annotate(
                        "",
                        xy=(
                            obs.position[0] + obs.velocity[0] * _VELOCITY_SCALE,
                            obs.position[1] + obs.velocity[1] * _VELOCITY_SCALE,
                        ),
                        xytext=(obs.position[0], obs.position[1]),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
                        zorder=3,
                    )
                # Prediction line
                pred = obs.predict_position(_PREDICTION_HORIZON_S)
                ax.plot(
                    [obs.position[0], pred[0]],
                    [obs.position[1], pred[1]],
                    color=_COLOR_DYN_PRED,
                    linewidth=0.9,
                    linestyle=":",
                    alpha=0.6,
                    zorder=2,
                )

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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _classify_dyn_color(speed: float, uav_max_speed: float) -> str:
    """
    Return the Matplotlib colour string for a dynamic obstacle based on its speed.

    Classification (SDS §37):
    - slow  : speed < 0.8 × uav_max_speed → green
    - equal : 0.8 ≤ speed ≤ 1.2 × uav_max_speed → orange
    - fast  : speed > 1.2 × uav_max_speed → red

    Parameters
    ----------
    speed:
        Current obstacle speed [m/s].
    uav_max_speed:
        UAV maximum speed used as the classification reference [m/s].

    Returns
    -------
    str
        Matplotlib colour string.
    """
    if speed < _SPEED_SLOW_RATIO * uav_max_speed:
        return _COLOR_DYN_SLOW
    if speed > _SPEED_FAST_RATIO * uav_max_speed:
        return _COLOR_DYN_FAST
    return _COLOR_DYN_EQUAL
