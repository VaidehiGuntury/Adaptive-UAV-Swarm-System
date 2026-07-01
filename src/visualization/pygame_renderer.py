"""
Pygame real-time renderer for Paper 1 BSA research demonstration.

Layers: exploration grid, frontiers, obstacles, sensor rings, trails,
assigned targets, velocity vectors, agents, and a research dashboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agents.base_agent import AgentRole
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine, SimulationState
from src.visualization.coordinate_transform import CoordinateTransform
from src.visualization.layer_toggles import LayerToggles
from src.visualization.render_palette import (
    COLOR_AGENT_ID,
    COLOR_BACKGROUND,
    COLOR_DASHBOARD_BG,
    COLOR_DASHBOARD_BORDER,
    COLOR_DASHBOARD_MUTED,
    COLOR_DASHBOARD_TEXT,
    COLOR_DASHBOARD_TITLE,
    COLOR_FRONTIER_CELL,
    COLOR_FRONTIER_CENTROID,
    COLOR_GRID_EXPLORED,
    COLOR_GRID_UNEXPLORED,
    COLOR_MASTER,
    COLOR_OBSTACLE,
    COLOR_SENSOR_RING,
    COLOR_TARGET_LINE,
    COLOR_TARGET_MARKER,
    COLOR_TRAIL_NEW,
    COLOR_TRAIL_OLD,
    COLOR_UGV,
    COLOR_UAV,
    COLOR_VELOCITY,
    DASHBOARD_WIDTH_PX,
    TRAIL_MAX_LENGTH,
    VELOCITY_ARROW_SCALE,
    # Dynamic environment palette (SDS §37)
    COLOR_DYN_EQUAL,
    COLOR_DYN_FAST,
    COLOR_DYN_PREDICTION,
    COLOR_DYN_SAFETY,
    COLOR_DYN_SLOW,
    COLOR_DYN_VELOCITY,
    DYN_PREDICTION_HORIZON_S,
    DYN_SPEED_FAST_RATIO,
    DYN_SPEED_SLOW_RATIO,
)

_AGENT_COLORS: dict[AgentRole, tuple[int, int, int]] = {
    AgentRole.UAV: COLOR_UAV,
    AgentRole.UGV: COLOR_UGV,
    AgentRole.MASTER: COLOR_MASTER,
}


def _require_pygame() -> Any:
    """Import pygame lazily so Matplotlib-only workflows work without it installed."""
    try:
        import pygame as pg
    except ImportError as exc:
        raise ImportError(
            "Pygame is required for PygameRenderer. Install with: pip install pygame-ce"
        ) from exc
    return pg


def _lerp_color(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linear RGB interpolation for trail fading."""
    t = float(np.clip(t, 0.0, 1.0))
    return (
        int(start[0] + (end[0] - start[0]) * t),
        int(start[1] + (end[1] - start[1]) * t),
        int(start[2] + (end[2] - start[2]) * t),
    )


def _draw_dashed_line(
    surface: Any,
    pg: Any,
    color: tuple[int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    dash_length: int = 6,
    gap_length: int = 4,
    width: int = 1,
) -> None:
    """
    Draw an approximated dashed line between *start* and *end* in screen space.

    Used for the dynamic obstacle predicted trajectory overlay.  Draws
    alternating filled and empty segments of *dash_length* / *gap_length*
    pixels respectively.

    Parameters
    ----------
    surface:
        Pygame drawing surface.
    pg:
        Pygame module reference.
    color:
        RGB colour tuple.
    start, end:
        Integer pixel coordinates.
    dash_length:
        Length of each drawn segment [px].
    gap_length:
        Length of each gap between segments [px].
    width:
        Line width [px].
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    total = float(np.hypot(dx, dy))
    if total < 1.0:
        return
    ux, uy = dx / total, dy / total
    step = dash_length + gap_length
    pos = 0.0
    while pos < total:
        dash_end = min(pos + dash_length, total)
        x0 = int(start[0] + ux * pos)
        y0 = int(start[1] + uy * pos)
        x1 = int(start[0] + ux * dash_end)
        y1 = int(start[1] + uy * dash_end)
        pg.draw.line(surface, color, (x0, y0), (x1, y1), width)
        pos += step


class PygameRenderer:
    """
    Real-time Pygame renderer consuming ``SimulationEngine`` snapshots.

    Toggle keys: G grid, F frontiers, T trails, V velocity, Y targets, S sensor.
    """

    def __init__(
        self,
        world: World,
        engine: SimulationEngine,
        screen_width: int = 800,
        screen_height: int = 800,
        fps: int = 20,
        agent_radius_px: int = 6,
    ) -> None:
        self.world = world
        self.engine = engine
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps
        self.agent_radius_px = agent_radius_px
        self.toggles = LayerToggles()
        self._history: list[SimulationState] = []
        self._transform = CoordinateTransform(
            world_width=world.width,
            world_height=world.height,
            screen_width=screen_width - DASHBOARD_WIDTH_PX,
            screen_height=screen_height,
        )
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None
        self._font: Any | None = None
        self._dashboard_font: Any | None = None
        self._title_font: Any | None = None

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

    def _ensure_pygame(self) -> Any:
        """Initialize Pygame subsystems on first use."""
        pg = self._pygame if self._pygame is not None else _require_pygame()
        if self._screen is None:
            pg.init()
            pg.display.init()
            self._screen = pg.display.set_mode((self.screen_width, self.screen_height))
            pg.display.set_caption("DEBS — BSA Viewpoint Selection (Paper 1)")
            self._clock = pg.time.Clock()
            self._font = pg.font.SysFont("consolas", 13)
            self._dashboard_font = pg.font.SysFont("consolas", 14)
            self._title_font = pg.font.SysFont("consolas", 15, bold=True)
            self._pygame = pg
        return pg

    def _init_fonts(self, pg: Any) -> None:
        if self._font is None:
            self._font = pg.font.SysFont("consolas", 13)
        if self._dashboard_font is None:
            self._dashboard_font = pg.font.SysFont("consolas", 14)
        if self._title_font is None:
            self._title_font = pg.font.SysFont("consolas", 15, bold=True)

    def draw_frame(
        self,
        state: SimulationState,
        surface: Any | None = None,
        frame_index: int | None = None,
    ) -> None:
        """Render one simulation frame with all enabled layers."""
        if surface is not None:
            pg = self._pygame if self._pygame is not None else _require_pygame()
            if self._pygame is None:
                pg.init()
                self._pygame = pg
            self._init_fonts(pg)
            target = surface
        else:
            pg = self._ensure_pygame()
            target = self._screen
            if target is None:
                raise RuntimeError("Pygame surface not initialized. Call run_live() first.")

        target.fill(COLOR_BACKGROUND)
        if self.toggles.show_grid:
            self._draw_exploration_grid(target, pg)
        if self.toggles.show_frontiers:
            self._draw_frontiers(target, pg)
        self._draw_obstacles(target, pg)
        self._draw_dynamic_obstacles(target, pg)
        if self.toggles.show_sensor_radius:
            self._draw_sensor_rings(target, state, pg)
        if self.toggles.show_trails:
            self._draw_trails(target, state, pg, frame_index)
        if self.toggles.show_targets:
            self._draw_assigned_targets(target, state, pg)
        if self.toggles.show_velocity:
            self._draw_velocity_vectors(target, state, pg)
        self._draw_agents(target, state, pg)
        self._draw_dashboard(target, state, pg)

    def _draw_exploration_grid(self, surface: Any, pg: Any) -> None:
        explored = self.world.map.explored_mask()
        obstacles = self.world.map.obstacle_mask()
        resolution = self.world.map.resolution
        rows, cols = explored.shape
        for row in range(rows):
            for col in range(cols):
                if obstacles[row, col]:
                    continue
                color = COLOR_GRID_EXPLORED if explored[row, col] else COLOR_GRID_UNEXPLORED
                rect = self._transform.cell_screen_rect(col, row, resolution)
                pg.draw.rect(surface, color, rect)

    def _draw_frontiers(self, surface: Any, pg: Any) -> None:
        frontier = self.world.map.frontier_mask()
        resolution = self.world.map.resolution
        rows, cols = frontier.shape
        for row in range(rows):
            for col in range(cols):
                if not frontier[row, col]:
                    continue
                rect = self._transform.cell_screen_rect(col, row, resolution)
                pg.draw.rect(surface, COLOR_FRONTIER_CELL, rect)

        for cluster in self.world.map.extract_frontier_clusters():
            cx, cy = self._transform.world_to_screen(cluster.centroid)
            pg.draw.circle(surface, COLOR_FRONTIER_CENTROID, (cx, cy), 4)
            pg.draw.circle(surface, COLOR_FRONTIER_CENTROID, (cx, cy), 6, width=1)

    def _draw_obstacles(self, surface: Any, pg: Any) -> None:
        for obstacle in self.world.obstacles.obstacles:
            cx, cy = self._transform.world_to_screen(obstacle.center)
            radius_px = self._transform.world_radius_to_pixels(obstacle.radius)
            pg.draw.circle(surface, COLOR_OBSTACLE, (cx, cy), radius_px)

    def _draw_dynamic_obstacles(self, surface: Any, pg: Any) -> None:
        """
        Render all active dynamic obstacles with speed-coded colours (SDS §37).

        Colour classification is relative to the UAV's configured max speed:

        - Slow  (speed < 0.8 × v_uav) → Green
        - Equal (0.8 ≤ speed ≤ 1.2 × v_uav) → Orange
        - Fast  (speed > 1.2 × v_uav) → Red

        Each obstacle is drawn as a filled circle (body) plus an outlined
        circle (boundary ring) for depth.  A velocity arrow extends from
        the obstacle centre using the same arrowhead helper as UAV vectors.

        Optional overlays (controlled by ``LayerToggles``):

        - ``show_dynamic_predictions`` (D key): a line from current position
          to predicted position at t + ``DYN_PREDICTION_HORIZON_S`` seconds.
        - ``show_dynamic_safety`` (R key): a thin circle at the obstacle
          boundary radius + the manager's safety_margin, visualising the
          near-miss zone.

        Performance
        -----------
        Guarded on ``world.obstacle_manager is not None`` so the loop is
        entirely skipped in static mode with zero overhead.
        """
        manager = self.world.obstacle_manager
        if manager is None:
            return

        uav_max_speed: float = self.engine.config.uav.max_speed
        slow_threshold = DYN_SPEED_SLOW_RATIO * uav_max_speed
        fast_threshold = DYN_SPEED_FAST_RATIO * uav_max_speed

        for obstacle in manager:
            if not obstacle.active:
                continue

            speed = float(np.linalg.norm(obstacle.velocity))
            if speed < slow_threshold:
                color = COLOR_DYN_SLOW
            elif speed > fast_threshold:
                color = COLOR_DYN_FAST
            else:
                color = COLOR_DYN_EQUAL

            cx, cy = self._transform.world_to_screen(obstacle.position)
            radius_px = self._transform.world_radius_to_pixels(obstacle.radius)

            # Filled body
            pg.draw.circle(surface, color, (cx, cy), radius_px)
            # Boundary ring (slightly brighter)
            pg.draw.circle(surface, color, (cx, cy), radius_px, width=2)

            # Velocity vector
            if speed > 1e-6:
                end_pos = obstacle.position + obstacle.velocity * VELOCITY_ARROW_SCALE
                p1 = self._transform.world_to_screen(end_pos)
                pg.draw.line(surface, COLOR_DYN_VELOCITY, (cx, cy), p1, width=2)
                self._draw_arrowhead(surface, pg, (cx, cy), p1, COLOR_DYN_VELOCITY)

            # Optional: predicted trajectory
            if self.toggles.show_dynamic_predictions:
                predicted_pos = obstacle.predict_position(DYN_PREDICTION_HORIZON_S)
                px_pred, py_pred = self._transform.world_to_screen(predicted_pos)
                # Draw as a dashed line (approximated with short segments)
                _draw_dashed_line(
                    surface, pg,
                    COLOR_DYN_PREDICTION,
                    (cx, cy), (px_pred, py_pred),
                    dash_length=6, gap_length=4,
                )
                pg.draw.circle(surface, COLOR_DYN_PREDICTION, (px_pred, py_pred), 3, width=1)

            # Optional: safety radius ring
            if self.toggles.show_dynamic_safety:
                safety_radius_px = self._transform.world_radius_to_pixels(
                    obstacle.radius + manager.safety_margin
                )
                pg.draw.circle(surface, COLOR_DYN_SAFETY, (cx, cy), safety_radius_px, width=1)

    def _draw_sensor_rings(self, surface: Any, state: SimulationState, pg: Any) -> None:
        sensing_range = self.engine.config.uav.sensing_range
        radius_px = self._transform.world_radius_to_pixels(sensing_range)
        for agent in state.agents:
            cx, cy = self._transform.world_to_screen(agent.position)
            pg.draw.circle(surface, COLOR_SENSOR_RING, (cx, cy), radius_px, width=1)

    def _draw_trails(
        self,
        surface: Any,
        state: SimulationState,
        pg: Any,
        frame_index: int | None,
    ) -> None:
        for agent in state.agents:
            history = self.engine.agent_histories.get(agent.agent_id, [])
            if frame_index is not None:
                end = min(frame_index + 1, len(history))
                history = history[:end]
            if len(history) < 2:
                continue
            trail = history[-TRAIL_MAX_LENGTH:]
            for idx in range(1, len(trail)):
                t = idx / max(1, len(trail) - 1)
                color = _lerp_color(COLOR_TRAIL_OLD, COLOR_TRAIL_NEW, t)
                p0 = self._transform.world_to_screen(trail[idx - 1])
                p1 = self._transform.world_to_screen(trail[idx])
                pg.draw.line(surface, color, p0, p1, width=1)

    def _draw_assigned_targets(self, surface: Any, state: SimulationState, pg: Any) -> None:
        for agent in state.agents:
            target = getattr(agent, "assigned_target", None)
            if target is None:
                continue
            ax, ay = self._transform.world_to_screen(agent.position)
            tx, ty = self._transform.world_to_screen(target)
            pg.draw.line(surface, COLOR_TARGET_LINE, (ax, ay), (tx, ty), width=1)
            size = 4
            pg.draw.line(surface, COLOR_TARGET_MARKER, (tx - size, ty), (tx + size, ty), width=2)
            pg.draw.line(surface, COLOR_TARGET_MARKER, (tx, ty - size), (tx, ty + size), width=2)

    def _draw_velocity_vectors(self, surface: Any, state: SimulationState, pg: Any) -> None:
        for agent in state.agents:
            speed = float(np.linalg.norm(agent.velocity))
            if speed < 1e-6:
                continue
            start = agent.position
            end = start + agent.velocity * VELOCITY_ARROW_SCALE
            p0 = self._transform.world_to_screen(start)
            p1 = self._transform.world_to_screen(end)
            pg.draw.line(surface, COLOR_VELOCITY, p0, p1, width=2)
            self._draw_arrowhead(surface, pg, p0, p1, COLOR_VELOCITY)

    def _draw_arrowhead(
        self,
        surface: Any,
        pg: Any,
        tail: tuple[int, int],
        tip: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        dx = tip[0] - tail[0]
        dy = tip[1] - tail[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux
        size = 5.0
        left = (int(tip[0] - ux * size + px * size * 0.5), int(tip[1] - uy * size + py * size * 0.5))
        right = (int(tip[0] - ux * size - px * size * 0.5), int(tip[1] - uy * size - py * size * 0.5))
        pg.draw.polygon(surface, color, [tip, left, right])

    def _draw_agents(self, surface: Any, state: SimulationState, pg: Any) -> None:
        font = self._font
        for agent in state.agents:
            color = _AGENT_COLORS.get(agent.role, COLOR_UAV)
            cx, cy = self._transform.world_to_screen(agent.position)
            pg.draw.circle(surface, color, (cx, cy), self.agent_radius_px)
            if font is not None:
                label = font.render(str(agent.agent_id), True, COLOR_AGENT_ID)
                surface.blit(label, (cx + self.agent_radius_px + 2, cy - self.agent_radius_px))

    def _draw_dashboard(self, surface: Any, state: SimulationState, pg: Any) -> None:
        title_font = self._title_font
        body_font = self._dashboard_font
        if title_font is None or body_font is None:
            return

        metrics = state.metrics
        panel_x = self.screen_width - DASHBOARD_WIDTH_PX
        panel_rect = pg.Rect(panel_x, 0, DASHBOARD_WIDTH_PX, self.screen_height)
        pg.draw.rect(surface, COLOR_DASHBOARD_BG, panel_rect)
        pg.draw.line(
            surface,
            COLOR_DASHBOARD_BORDER,
            (panel_x, 0),
            (panel_x, self.screen_height),
            width=1,
        )

        x = panel_x + 12
        y = 14
        title = title_font.render("Paper 1 — BSA Demo", True, COLOR_DASHBOARD_TITLE)
        surface.blit(title, (x, y))
        y += title_font.get_linesize() + 6

        subtitle = body_font.render("DEBS Stage 2", True, COLOR_DASHBOARD_MUTED)
        surface.blit(subtitle, (x, y))
        y += body_font.get_linesize() + 12

        lines = [
            ("Mission time", f"{state.time_s:.1f} s"),
            ("Step", str(state.timestep)),
            ("Fleet size", str(len(state.agents))),
            ("Coverage", f"{metrics.explored_fraction * 100:.1f} %"),
            ("Mean speed", f"{metrics.mean_speed:.2f} m/s"),
            ("Mean pair dist", f"{metrics.mean_pairwise_distance:.1f} m"),
            ("Target sep", f"{metrics.mean_target_separation:.1f} m"),
            ("Frontier reuse", f"{metrics.frontier_reuse_frequency:.3f}"),
            ("Reassigns/step", str(metrics.target_reassignment_count)),
            ("Revisit ratio", f"{metrics.revisit_ratio:.3f}"),
            ("Active frontiers", str(metrics.active_frontier_count)),
        ]
        for label, value in lines:
            label_surf = body_font.render(label, True, COLOR_DASHBOARD_MUTED)
            value_surf = body_font.render(value, True, COLOR_DASHBOARD_TEXT)
            surface.blit(label_surf, (x, y))
            surface.blit(value_surf, (x + 110, y))
            y += body_font.get_linesize() + 4

        y += 8
        hint_font = pg.font.SysFont("consolas", 11)
        hints = [
            "G grid  F frontiers",
            "T trails  V velocity",
            "Y targets  S sensor",
            "D dyn-predict  R safety",
        ]
        for hint in hints:
            surface.blit(hint_font.render(hint, True, COLOR_DASHBOARD_MUTED), (x, y))
            y += hint_font.get_linesize() + 2

    def _handle_events(self, pg: Any) -> bool:
        """Process pygame events. Returns False when the app should quit."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    return False
                self.toggles.handle_key(event.key, pg)
        return True

    def run_live(self, show: bool = True) -> None:
        """Run simulation with real-time Pygame display."""
        if not show:
            return

        pg = self._ensure_pygame()
        running = True
        self.engine.timestep = 0
        self.engine.time_s = 0.0
        self.engine.metrics_history.clear()
        self.engine.agent_histories = {
            agent.agent_id: [agent.position.copy()] for agent in self.engine.agents
        }

        while running and self.engine.time_s < self.engine.config.duration:
            running = self._handle_events(pg)
            if not running:
                break
            self.engine.step()
            state = self.engine.get_state()
            if self._screen is not None:
                self.draw_frame(state, self._screen)
                pg.display.flip()
            if self._clock is not None:
                self._clock.tick(self.fps)
        pg.quit()

    def playback(self, show: bool = True) -> None:
        """Playback recorded history in the Pygame window."""
        if not self._history:
            self.run_and_record()
        if not show:
            return

        pg = self._ensure_pygame()
        frame_index = 0
        running = True
        while running and frame_index < len(self._history):
            running = self._handle_events(pg)
            if not running:
                break
            state = self._history[frame_index]
            if self._screen is not None:
                self.draw_frame(state, self._screen, frame_index=frame_index)
                pg.display.flip()
            if self._clock is not None:
                self._clock.tick(self.fps)
            frame_index += 1
        pg.quit()
