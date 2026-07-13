"""
Task D2 — paper-style dynamic-obstacle scenario validation
(docs/dynamic_environment_design.md, section "36 standardized
experiment scenarios", Scenarios A-E).

Runs each of the 5 standardized scenarios (static / slow / equal_speed
/ fast / mixed) at 10 and 20 UAVs, 120s, single seed, using the full
builder (src.main.build_simulation) so dynamic obstacles are wired
into collision resolution (fix: commit 2bde77a). Search is disabled —
this validates the exploration+obstacle-avoidance layer only, per the
design doc's original scope.

Generates missing scenario configs under experiments/scratch/ (not
committed) by cloning configs/simulation.yaml with num_uavs,
dynamic_environment.enabled/scenario, and search.enabled overridden.

Usage:
    python -m experiments.scripts.run_obstacle_scenarios
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from src.evaluation.dynamic_environment_metrics import (
    InteractionRecord,
    average_obstacle_speed,
    collision_count,
    coverage_degradation,
    mission_completion_time,
    near_miss_count,
)
from src.main import build_simulation

BASE_CONFIG = Path("configs/simulation.yaml")
SCRATCH_DIR = Path("experiments/scratch")
RESULTS_DIR = Path("experiments/results")
DURATION = 120.0
SEED = 42
FLEET_SIZES = [10, 20]
SCENARIOS = ["static", "slow", "equal_speed", "fast", "mixed"]


def generate_scenario_config(scenario: str, num_uavs: int) -> Path:
    """Clone configs/simulation.yaml with scenario/fleet-size overrides."""
    text = BASE_CONFIG.read_text(encoding="utf-8")

    text = re.sub(r"^  num_uavs: \d+", f"  num_uavs: {num_uavs}", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^search:\n  enabled: true", "search:\n  enabled: false", text, count=1, flags=re.MULTILINE)

    if scenario == "static":
        text = re.sub(
            r"^dynamic_environment:\n  enabled: true",
            "dynamic_environment:\n  enabled: false",
            text, count=1, flags=re.MULTILINE,
        )
    else:
        text = re.sub(r"^  scenario: \w+", f"  scenario: {scenario}", text, count=1, flags=re.MULTILINE)

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH_DIR / f"obstacle_{scenario}_n{num_uavs}.yaml"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def run_scenario(config_path: Path, label: str) -> dict:
    engine, _renderer = build_simulation(config_path)

    interactions: list[InteractionRecord] = []
    timeseries: list[tuple[float, float]] = [(0.0, 0.0)]

    while engine.time_s < DURATION:
        m = engine.step()
        timeseries.append((m.time_s, m.explored_fraction))
        if engine.world.obstacle_manager is not None:
            for agent in engine.agents:
                result = engine.world.obstacle_manager.check_collision(agent.position)
                if result.obstacle_id is not None:
                    interactions.append(
                        InteractionRecord(
                            timestep=engine.timestep,
                            time_s=engine.time_s,
                            agent_id=agent.agent_id,
                            result=result,
                        )
                    )

    final_coverage = timeseries[-1][1]
    n_collisions = collision_count(interactions)
    n_near_misses = near_miss_count(interactions)
    completion_time = mission_completion_time(timeseries)
    avg_obs_speed = (
        average_obstacle_speed(engine.world.obstacle_manager)
        if engine.world.obstacle_manager is not None
        else 0.0
    )

    return {
        "label": label,
        "num_uavs": len(engine.agents),
        "final_coverage": final_coverage,
        "collision_count": n_collisions,
        "near_miss_count": n_near_misses,
        "mission_completion_time": completion_time if completion_time is not None else "",
        "avg_obstacle_speed": avg_obs_speed,
    }


def main() -> None:
    results = []
    for num_uavs in FLEET_SIZES:
        for scenario in SCENARIOS:
            label = f"{scenario}_n{num_uavs}"
            config_path = generate_scenario_config(scenario, num_uavs)
            print(f"[{label}] running ({config_path})...")
            result = run_scenario(config_path, label)
            results.append(result)
            print(
                f"[{label}] coverage={result['final_coverage']*100:.1f}% "
                f"collisions={result['collision_count']} "
                f"near_misses={result['near_miss_count']} "
                f"avg_obstacle_speed={result['avg_obstacle_speed']:.2f}"
            )

    # Coverage degradation vs static baseline, per fleet size.
    baseline_coverage = {
        r["num_uavs"]: r["final_coverage"] for r in results if r["label"].startswith("static")
    }
    for r in results:
        r["coverage_degradation_vs_static"] = coverage_degradation(
            baseline_coverage[r["num_uavs"]], r["final_coverage"]
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "obstacle_scenario_validation.csv"
    fields = [
        "label", "num_uavs", "final_coverage", "collision_count", "near_miss_count",
        "mission_completion_time", "avg_obstacle_speed", "coverage_degradation_vs_static",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== Task D2 obstacle scenario validation ===")
    for r in results:
        print(
            f"{r['label']:<16} coverage={r['final_coverage']*100:5.1f}%  "
            f"collisions={r['collision_count']:4d}  near_misses={r['near_miss_count']:4d}  "
            f"completion_t={r['mission_completion_time']}  "
            f"degradation={r['coverage_degradation_vs_static']*100:+.1f}pp"
        )
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
