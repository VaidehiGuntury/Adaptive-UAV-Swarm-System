"""
Entry point for Paper 1 self-aggregation swarm simulation.

Usage (from project root):
    python -m src.main
    python -m src.main --config configs/simulation.yaml --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.algorithms.aggregation.self_aggregation import SelfAggregationController
from src.algorithms.formation.formation_controller import (
    FormationAcquisitionConfig,
    FormationAcquisitionController,
)
from src.agents.uav import spawn_uavs
from src.config.loader import load_config
from src.environment.world import World
from src.simulation.simulation_engine import SimulationEngine
from src.visualization.renderer import SimulationRenderer


def build_simulation(config_path: Path) -> tuple[SimulationEngine, SimulationRenderer]:
    """Construct world, agents, aggregation, engine, and renderer from config."""
    config = load_config(config_path)

    world = World.from_config(config.environment, config.uav)
    spawn_center = np.array(
        [config.spawn_center_x, config.spawn_center_y],
        dtype=np.float64,
    )
    agents = spawn_uavs(
        count=config.num_uavs,
        center=spawn_center,
        spread_radius=config.uav.initial_spread_radius,
        mission_radius=config.aggregation.mission_region_radius,
        max_speed=config.uav.max_speed,
        max_angular_velocity=config.uav.max_angular_velocity,
        seed=config.environment.obstacle_seed,
        spawn_mode=config.uav.spawn_mode,  # type: ignore[arg-type]
        spawn_angular_noise=config.uav.spawn_angular_noise,
    )

    aggregation = SelfAggregationController(
        config=config.aggregation,
        uav_config=config.uav,
        rng=np.random.default_rng(config.environment.obstacle_seed),
    )

    formation_controller = FormationAcquisitionController(
        config=FormationAcquisitionConfig(),
    )

    engine = SimulationEngine(
        world=world,
        agents=agents,
        aggregation=aggregation,
        config=config,
        formation_controller=formation_controller,
    )
    renderer = SimulationRenderer(
        world=world,
        engine=engine,
        interval_ms=config.animation_interval_ms,
    )
    return engine, renderer


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper 1 BSA swarm simulation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/simulation.yaml"),
        help="Path to simulation YAML configuration",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run simulation without displaying the matplotlib window",
    )
    args = parser.parse_args()

    _, renderer = build_simulation(args.config)
    renderer.run_and_record()

    if not args.no_show:
        renderer.animate(show=True)
    else:
        final = renderer.engine.metrics_history[-1]
        print(
            f"Simulation complete: {final.timestep} steps ({final.time_s:.1f}s), "
            f"explored {final.explored_fraction * 100:.1f}%, "
            f"frontier_reuse {final.frontier_reuse_frequency:.3f}, "
            f"mean speed {final.mean_speed:.2f} m/s"
        )


if __name__ == "__main__":
    main()
