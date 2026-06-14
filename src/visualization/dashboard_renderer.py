"""Research dashboard panel for Pygame renderer."""

from __future__ import annotations

from typing import Any

from src.simulation.simulation_engine import SimulationState
from src.visualization.render_palette import (
    COLOR_DASHBOARD_BG,
    COLOR_DASHBOARD_BORDER,
    COLOR_DASHBOARD_MUTED,
    COLOR_DASHBOARD_TEXT,
    COLOR_DASHBOARD_TITLE,
    COLOR_DASHBOARD_WARN,
    DASHBOARD_WIDTH_PX,
)


def draw_research_dashboard(
    surface: Any,
    pg: Any,
    state: SimulationState,
    screen_width: int,
    screen_height: int,
    title_font: Any,
    body_font: Any,
) -> None:
    """Draw the right-side metrics panel."""
    metrics = state.metrics
    panel_x = screen_width - DASHBOARD_WIDTH_PX
    panel_rect = pg.Rect(panel_x, 0, DASHBOARD_WIDTH_PX, screen_height)
    pg.draw.rect(surface, COLOR_DASHBOARD_BG, panel_rect)
    pg.draw.line(
        surface,
        COLOR_DASHBOARD_BORDER,
        (panel_x, 0),
        (panel_x, screen_height),
        width=1,
    )

    x = panel_x + 12
    y = 14
    title = title_font.render("Swarm Research Demo", True, COLOR_DASHBOARD_TITLE)
    surface.blit(title, (x, y))
    y += title_font.get_linesize() + 6

    if metrics.formation_active:
        mode = "FORMATION + CONSENSUS" if metrics.consensus_active else "FORMATION ACQUISITION"
        subtitle = body_font.render(mode, True, COLOR_DASHBOARD_TEXT)
    else:
        subtitle = body_font.render("BSA AGGREGATION", True, COLOR_DASHBOARD_MUTED)
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

    if metrics.formation_active:
        lines.extend(
            [
                ("RMS form err", f"{metrics.rms_formation_error:.2f} m"),
                ("Mean slot err", f"{metrics.mean_follower_slot_error:.2f} m"),
                ("Occupied slots", f"{metrics.occupied_slot_percentage * 100:.0f} %"),
                ("Avg neighbors", f"{metrics.average_neighbor_count:.1f}"),
                ("Consensus |u|", f"{metrics.consensus_residual_magnitude:.2f}"),
                ("Spacing var", f"{metrics.local_spacing_variance:.3f}"),
                (
                    "Graph connected",
                    "yes" if metrics.graph_connectivity_status else "NO",
                ),
            ]
        )
        if metrics.formation_convergence_time_s is not None:
            lines.append(("Converged at", f"{metrics.formation_convergence_time_s:.1f} s"))
        if metrics.isolated_follower_count > 0:
            lines.append(("Isolated", str(metrics.isolated_follower_count)))

    for label, value in lines:
        value_color = COLOR_DASHBOARD_TEXT
        if label == "Graph connected" and value == "NO":
            value_color = COLOR_DASHBOARD_WARN
        if label == "Isolated":
            value_color = COLOR_DASHBOARD_WARN
        label_surf = body_font.render(label, True, COLOR_DASHBOARD_MUTED)
        value_surf = body_font.render(value, True, value_color)
        surface.blit(label_surf, (x, y))
        surface.blit(value_surf, (x + 110, y))
        y += body_font.get_linesize() + 4

    y += 8
    hint_font = pg.font.SysFont("consolas", 11)
    hints = [
        "G grid  F frontiers",
        "T trails  V velocity",
        "Y targets  S sensor",
        "M formations  C comm",
    ]
    for hint in hints:
        surface.blit(hint_font.render(hint, True, COLOR_DASHBOARD_MUTED), (x, y))
        y += hint_font.get_linesize() + 2
