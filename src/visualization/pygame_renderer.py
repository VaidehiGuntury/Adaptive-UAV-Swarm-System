"""
Pygame real-time renderer (primary visualization backend).

Week 1.5 skeleton: obstacles, UAV sprites, agent IDs, and basic HUD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agents.base_agent import AgentRole
from src.agents.uav import UAV
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine, SimulationState
from src.visualization.coordinate_transform import CoordinateTransform

if TYPE_CHECKING:
    import pygame

# Colour palette (R, G, B)
_COLOR_BACKGROUND = (18, 22, 28)
_COLOR_OBSTACLE = (90, 90, 95)
_COLOR_UAV = (66, 133, 244)
_COLOR_UGV = (76, 175, 80)
_COLOR_MASTER = (255, 193, 7)
_COLOR_HUD_BG = (30, 34, 42)
_COLOR_HUD_TEXT = (220, 225, 230)
_COLOR_ID = (200, 210, 220)

_AGENT_COLORS: dict[AgentRole, tuple[int, int, int]] = {
    AgentRole.UAV: _COLOR_UAV,
    AgentRole.UGV: _COLOR_UGV,
    AgentRole.MASTER: _COLOR_MASTER,
}


def _require_pygame() -> Any:
    """Import pygame lazily so Matplotlib-only workflows work without it installed."""
    try:
        import pygame as pg
    except ImportError as exc:
        raise ImportError(
            "Pygame is required for PygameRenderer. Install with: pip install pygame"
        ) from exc
    return pg


class PygameRenderer:
    """
    Real-time Pygame renderer consuming ``SimulationEngine`` snapshots.

    Renders obstacles, agent markers, agent IDs, and a metrics HUD.
    Formation overlays, comm links, and belief heatmaps are deferred.
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
        self._history: list[SimulationState] = []
        self._transform = CoordinateTransform(
            world_width=world.width,
            world_height=world.height,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self._pygame: Any | None = None
        self._screen: Any | None = None
        self._clock: Any | None = None
        self._font: Any | None = None
        self._hud_font: Any | None = None

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
            pg.display.set_caption("Swarm Simulation — Paper 1 BSA")
            self._clock = pg.time.Clock()
            self._init_fonts(pg)
            self._pygame = pg
        return pg

    def _init_fonts(self, pg: Any) -> None:
        """Create font objects once pygame is available."""
        if self._font is None:
            self._font = pg.font.SysFont("consolas", 14)
        if self._hud_font is None:
            self._hud_font = pg.font.SysFont("consolas", 16)

    def draw_frame(self, state: SimulationState, surface: Any | None = None) -> None:
        """
        Render a single simulation frame.

        Parameters
        ----------
        state:
            Snapshot to draw.
        surface:
            Target surface; defaults to the display window.
        """
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

        target.fill(_COLOR_BACKGROUND)
        self._draw_obstacles(target, pg)
        self._draw_agents(target, state, pg)
        self._draw_hud(target, state, pg)

    def _draw_obstacles(self, surface: Any, pg: Any) -> None:
        for obstacle in self.world.obstacles.obstacles:
            cx, cy = self._transform.world_to_screen(obstacle.center)
            radius_px = self._transform.world_radius_to_pixels(obstacle.radius)
            pg.draw.circle(surface, _COLOR_OBSTACLE, (cx, cy), radius_px)

    def _draw_agents(self, surface: Any, state: SimulationState, pg: Any) -> None:
        font = self._font
        for agent in state.agents:
            color = _AGENT_COLORS.get(agent.role, _COLOR_UAV)
            cx, cy = self._transform.world_to_screen(agent.position)
            pg.draw.circle(surface, color, (cx, cy), self.agent_radius_px)

            if font is not None:
                label = font.render(str(agent.agent_id), True, _COLOR_ID)
                surface.blit(label, (cx + self.agent_radius_px + 2, cy - self.agent_radius_px))

    def _draw_hud(self, surface: Any, state: SimulationState, pg: Any) -> None:
        hud_font = self._hud_font
        if hud_font is None:
            return

        metrics = state.metrics
        lines = [
            f"t = {state.time_s:.1f} s  |  step = {state.timestep}",
            f"explored = {metrics.explored_fraction * 100:.1f}%",
            f"mean speed = {metrics.mean_speed:.2f} m/s",
            f"agents = {len(state.agents)}",
        ]
        padding = 8
        line_height = hud_font.get_linesize()
        box_height = padding * 2 + line_height * len(lines)
        box_width = 320
        hud_rect = pg.Rect(10, 10, box_width, box_height)
        pg.draw.rect(surface, _COLOR_HUD_BG, hud_rect, border_radius=4)

        y = 10 + padding
        for line in lines:
            text = hud_font.render(line, True, _COLOR_HUD_TEXT)
            surface.blit(text, (10 + padding, y))
            y += line_height

    def run_live(self, show: bool = True) -> None:
        """
        Run simulation with real-time Pygame display.

        Steps the engine each frame until duration is reached or the user quits.
        """
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
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    running = False

            self.engine.step()
            state = self.engine.get_state()
            if self._screen is not None:
                self.draw_frame(state, self._screen)
                pg.display.flip()

            if self._clock is not None:
                self._clock.tick(self.fps)

        pg.quit()

    def playback(self, show: bool = True) -> None:
        """
        Playback recorded history in the Pygame window.

        If no history exists, runs ``run_and_record()`` first.
        """
        if not self._history:
            self.run_and_record()

        if not show:
            return

        pg = self._ensure_pygame()
        frame_index = 0
        running = True

        while running and frame_index < len(self._history):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    running = False

            state = self._history[frame_index]
            if self._screen is not None:
                self.draw_frame(state, self._screen)
                pg.display.flip()

            if self._clock is not None:
                self._clock.tick(self.fps)
            frame_index += 1

        pg.quit()
