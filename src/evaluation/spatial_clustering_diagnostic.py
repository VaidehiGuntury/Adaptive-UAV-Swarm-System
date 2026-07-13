"""
Task 1 diagnostic: target-to-target spatial clustering, before vs after the
hysteresis-commitment fix. Not part of the production evaluation pipeline —
one-off script for the exploration-phase-e regression investigation.

Usage:
    python -m src.evaluation.spatial_clustering_diagnostic \
        --config experiments/scratch/simulation_n30.yaml --duration 120 \
        --output experiments/results/after_hysteresis_n30_d120_targets.csv

The config passed in must already have the desired num_uavs set — this
script does not override fleet size after spawn.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.evaluation.run_experiment import build_engine

CHECKPOINT_INTERVAL_S = 10.0


def in_transit(engine, agent) -> bool:
    """assigned, not yet arrived (dist > d_c), not yet timed out.

    The pre-hysteresis-fix controller has no timeout concept (it replans
    unconditionally every replan_interval), so timeout only applies when
    the controller exposes _target_timeout.
    """
    if agent.assigned_target is None:
        return False
    dist = float(np.linalg.norm(agent.position - agent.assigned_target))
    if dist <= engine.config.aggregation.d_c:
        return False
    timeout_map = getattr(engine.aggregation, "_target_timeout", None)
    if timeout_map is not None:
        deadline = timeout_map.get(agent.agent_id, float("inf"))
        if engine.time_s >= deadline:
            return False
    return True


def target_pairwise_stats(agents) -> tuple[float, float]:
    """Mean and min pairwise distance between agents' CURRENT TARGETS."""
    targets = [a.assigned_target for a in agents if a.assigned_target is not None]
    if len(targets) < 2:
        return float("nan"), float("nan")
    dists = []
    for i in range(len(targets)):
        for j in range(i + 1, len(targets)):
            dists.append(float(np.linalg.norm(targets[i] - targets[j])))
    return float(np.mean(dists)), float(np.min(dists))


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial-clustering diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine = build_engine(args.config)

    rows = []
    next_checkpoint = CHECKPOINT_INTERVAL_S
    # record t=0 state before any stepping
    mean_d, min_d = target_pairwise_stats(engine.agents)
    transit_count = sum(1 for a in engine.agents if in_transit(engine, a))
    rows.append(
        {
            "time_s": 0.0,
            "mean_target_pairwise_dist": mean_d,
            "min_target_pairwise_dist": min_d,
            "uavs_in_transit": transit_count,
            "num_uavs": len(engine.agents),
        }
    )

    while engine.time_s < args.duration:
        engine.step()
        if engine.time_s + 1e-9 >= next_checkpoint:
            mean_d, min_d = target_pairwise_stats(engine.agents)
            transit_count = sum(1 for a in engine.agents if in_transit(engine, a))
            rows.append(
                {
                    "time_s": round(engine.time_s, 1),
                    "mean_target_pairwise_dist": mean_d,
                    "min_target_pairwise_dist": min_d,
                    "uavs_in_transit": transit_count,
                    "num_uavs": len(engine.agents),
                }
            )
            next_checkpoint += CHECKPOINT_INTERVAL_S

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} checkpoints to {args.output}")


if __name__ == "__main__":
    main()
