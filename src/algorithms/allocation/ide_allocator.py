# -*- coding: utf-8 -*-
"""IDE allocation algorithm implementation.

Implements the decentralized Iterative Distribution Estimation (IDE) allocator
as described in DEBS Paper 1 §4.  The implementation follows the equations
and algorithms enumerated in the user request:

* Eq. 1 – ``pairwise_objective``
* Eq. 2 – ``_adaptive_F``
* Eq. 3 – ``_adaptive_CR``
* Eq. 4 – ``_mutate`` (current‑to‑best/1)
* Eq. 5 – ``_crossover`` (binomial)
* LHS initialization (Section 4.2)
* Algorithm 1 – ``_run_ide``
* Algorithm 2 – ``allocate``

The population dimensionality is fixed to ``D = 2`` (2‑D position vectors).
All tunable parameters are supplied via :class:`IDEConfig` loaded from the
configuration system.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

# NOTE: ``IDEConfig`` will be defined in ``src.config.loader``.  Import here
# to keep the module self‑contained; the loader modification will ensure the
# symbol is available at runtime.
from src.config.loader import IDEConfig

logger = logging.getLogger(__name__)


def _pairwise_objective(p_i: np.ndarray, p_j: np.ndarray, d_star: float) -> float:
    """Pairwise objective function (DEBS Eq. 1).

    Parameters
    ----------
    p_i, p_j:
        2‑D position vectors of the two UAVs.
    d_star:
        Target separation distance.

    Returns
    -------
    float
        The objective value, which is non‑negative and equal to ``0`` when the
        Euclidean distance equals ``d_star``.

    Raises
    ------
    ValueError
        If the two positions coincide (singularity of the formula).
    """
    dist = np.linalg.norm(p_i - p_j)
    if dist == 0.0:
        raise ValueError("pairwise_objective: UAV positions must not be identical (singular case).")
    term1 = d_star * dist
    term2 = (d_star ** 4) / (2.0 * (dist ** 2))
    term3 = 1.5 * (d_star ** 2)
    return term1 + term2 - term3


def _adaptive_F(fe: int, config: IDEConfig) -> float:
    """Adaptive scaling factor (DEBS Eq. 2).

    ``fe`` is the current number of function evaluations.
    """
    return 2.0 * config.alpha * (fe / config.fe_max)


def _adaptive_CR(fe: int, config: IDEConfig) -> float:
    """Adaptive crossover rate (DEBS Eq. 3).

    ``fe`` is the current number of function evaluations.
    """
    return 2.0 * (1.0 - config.alpha) * (1.0 - fe / config.fe_max)


def _mutate(
    x_i: np.ndarray,
    x_best: np.ndarray,
    x_r1: np.ndarray,
    x_r2: np.ndarray,
    F: float,
) -> np.ndarray:
    """Mutation operator (DEBS Eq. 4 – current‑to‑best/1).

    Returns a new candidate vector.
    """
    return x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)


def _crossover(
    x_i: np.ndarray,
    x_new: np.ndarray,
    CR: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial crossover (DEBS Eq. 5).

    Guarantees that at least one dimension is taken from ``x_new``.
    """
    dim = x_i.shape[0]
    # j_rand guarantees at least one component from x_new
    j_rand = rng.integers(0, dim)
    mask = rng.random(dim) < CR
    mask[j_rand] = True
    trial = np.where(mask, x_new, x_i)
    return trial


def _lhs_init(
    centre: np.ndarray,
    config: IDEConfig,
    world_bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Local LHS around centre within Qs neighborhood.
    
    Paper §4: LHS samples locally in the vicinity of 
    the UAV's current mission region centre p̃*.
    Qs (bounds_padding reused as step size) defines
    the sampling neighborhood radius.
    """
    pop = np.empty((config.population_size, 2), dtype=float)
    min_x, max_x, min_y, max_y = world_bounds

    # Local neighborhood: ±Qs around current centre
    # Qs stored as bounds_padding in config (0.5m)
    Qs = config.bounds_padding
    lower = np.array([
        max(min_x, centre[0] - Qs),
        max(min_y, centre[1] - Qs)
    ])
    upper = np.array([
        min(max_x, centre[0] + Qs),
        min(max_y, centre[1] + Qs)
    ])

    # Degenerate case: bounds collapsed (at map edge)
    for d in range(2):
        if upper[d] <= lower[d]:
            upper[d] = lower[d] + 1e-6

    intervals = np.linspace(0, 1, config.population_size + 1)
    for d in range(2):
        points = (intervals[:-1]
                  + rng.random(config.population_size)
                  * (1.0 / config.population_size))
        rng.shuffle(points)
        pop[:, d] = lower[d] + points * (upper[d] - lower[d])

    pop[0] = centre.copy()
    return pop


class IDEAllocator:
    """Decentralized Iterative Distribution Estimation allocator.

    Implements Algorithm 2 from DEBS §4.  The allocator works on a per‑UAV
    basis; each UAV runs a differential‑evolution optimisation that seeks a
    new centre ``p̃_i*`` while treating its partner's centre ``p̃_j*`` as a
    constant.
    """

    def __init__(self, config: IDEConfig, world_bounds: Tuple[float, float, float, float]):
        """Create the allocator.

        Parameters
        ----------
        config:
            Configuration block (see :class:`IDEConfig`).
        world_bounds:
            Tuple ``(min_x, max_x, min_y, max_y)`` describing the
            simulation arena.
        """
        self.config = config
        self.world_bounds = world_bounds
        # Random generator seeded for reproducibility
        seed = config.seed if config.seed is not None else None
        self._rng = np.random.default_rng(seed)
        # Interaction timestamps – keys are sorted (min_id, max_id)
        self._last_interaction: Dict[Tuple[int, int], float] = {}

    # ---------------------------------------------------------------------
    # Helper methods (direct mapping to DEBS equations)
    # ---------------------------------------------------------------------
    def _pairwise_objective(self, p_i: np.ndarray, p_j: np.ndarray) -> float:
        """Wrapper around the module‑level ``_pairwise_objective``.
        """
        return _pairwise_objective(p_i, p_j, self.config.d_star)

    def _adaptive_F(self, fe: int) -> float:
        return _adaptive_F(fe, self.config)

    def _adaptive_CR(self, fe: int) -> float:
        return _adaptive_CR(fe, self.config)

    # ---------------------------------------------------------------------
    # DE operators
    # ---------------------------------------------------------------------
    def _mutate(self, x_i: np.ndarray, x_best: np.ndarray, x_r1: np.ndarray, x_r2: np.ndarray, F: float) -> np.ndarray:
        return _mutate(x_i, x_best, x_r1, x_r2, F)

    def _crossover(self, x_i: np.ndarray, x_new: np.ndarray, CR: float) -> np.ndarray:
        return _crossover(x_i, x_new, CR, self._rng)

    # ---------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------
    def _lhs_init(self, centre: np.ndarray) -> np.ndarray:
        return _lhs_init(centre, self.config, self.world_bounds, self._rng)

    # ---------------------------------------------------------------------
    # Core IDE optimisation (Algorithm 1)
    # ---------------------------------------------------------------------
    def _run_ide(
        self,
        init_pop: np.ndarray,
        objective: callable,
    ) -> Tuple[np.ndarray, float, int]:
        """Run the DE optimisation (Algorithm 1 of DEBS §4).

        Parameters
        ----------
        init_pop:
            Initial population of shape ``(N, 2)``.
        objective:
            Callable ``f(candidate)`` returning a non‑negative float.

        Returns
        -------
        best_individual, best_score, fe_used
            ``fe_used`` is the actual number of objective evaluations consumed,
            which may be less than ``fe_max`` in future early-stopping variants.
        """
        population = init_pop.copy()
        # Evaluate initial fitness
        fitness = np.array([objective(ind) for ind in population])
        best_idx = int(np.argmin(fitness))
        best = population[best_idx].copy()
        best_score = float(fitness[best_idx])

        fe = 0
        # Guard against fe_max == 0: return immediately without any evaluations.
        if self.config.fe_max == 0:
            return best, best_score, fe

        while fe < self.config.fe_max:
            for i in range(self.config.population_size):
                # Select distinct indices for best, r1, r2
                idxs = list(range(self.config.population_size))
                idxs.remove(i)
                # Ensure best is not i
                if best_idx == i:
                    # pick a random other as temporary "best"
                    tmp_best_idx = int(self._rng.choice(idxs))
                else:
                    tmp_best_idx = best_idx
                # Choose r1, r2 distinct from each other and i and best
                remaining = [idx for idx in idxs if idx != tmp_best_idx]
                if len(remaining) < 2:
                    # Population too small – skip mutation for this individual
                    continue
                r1, r2 = self._rng.choice(remaining, size=2, replace=False)
                F = self._adaptive_F(fe)
                CR = self._adaptive_CR(fe)
                x_i = population[i]
                x_best = population[tmp_best_idx]
                x_r1 = population[r1]
                x_r2 = population[r2]
                x_new = self._mutate(x_i, x_best, x_r1, x_r2, F)
                # Clip to world bounds
                min_x, max_x, min_y, max_y = self.world_bounds
                x_new = np.clip(x_new, [min_x, min_y], [max_x, max_y])
                x_trial = self._crossover(x_i, x_new, CR)
                # Evaluate trial
                trial_f = objective(x_trial)
                fe += 1
                if trial_f < fitness[i]:
                    population[i] = x_trial
                    fitness[i] = trial_f
                    if trial_f < best_score:
                        best = x_trial.copy()
                        best_score = float(trial_f)
                        best_idx = i
                if fe >= self.config.fe_max:
                    break
            # End for i
        # End while
        return best, best_score, fe

    # ---------------------------------------------------------------------
    # Partner selection (Algorithm 2, step 3)
    # ---------------------------------------------------------------------
    def _pick_partner(self, uav_i_id: int, uav_positions: Dict[int, np.ndarray], current_time: float) -> int | None:
        """Select a partner for UAV ``i``.

        Returns the partner's UAV id or ``None`` when no eligible partner exists.
        """
        # Gather eligible UAV ids (exclude self)
        eligible = [uid for uid in uav_positions if uid != uav_i_id]
        if not eligible:
            return None
        if self.config.communication_range > 0.0:
            pos_i = uav_positions[uav_i_id]
            eligible = [uid for uid in eligible if np.linalg.norm(pos_i - uav_positions[uid]) <= self.config.communication_range]
            if not eligible:
                return None
        # Choose the partner with the longest elapsed interaction time
        max_elapsed = -1.0
        partner_id = None
        for uid in eligible:
            key = tuple(sorted((uav_i_id, uid)))
            last = self._last_interaction.get(key, -math.inf)
            elapsed = current_time - last if last != -math.inf else math.inf
            if elapsed > max_elapsed:
                max_elapsed = elapsed
                partner_id = uid
        return partner_id

    # ---------------------------------------------------------------------
    # Public allocation interface (Algorithm 2)
    # ---------------------------------------------------------------------
    def allocate(
        self,
        uavs: Iterable[object],
        current_time: float,
    ) -> Dict[int, np.ndarray]:
        """Run the IDE allocation for the given UAV collection.

        Parameters
        ----------
        uavs:
            An iterable of UAV objects. Each object must expose:
            ``agent_id`` (int) and ``assigned_region.center`` (2‑D ``np.ndarray``).
        current_time:
            Simulation time in seconds.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from UAV ``agent_id`` to the newly allocated centre.
        """
        # Build convenient look‑ups
        uav_dict: Dict[int, object] = {uav.agent_id: uav for uav in uavs}
        if len(uav_dict) <= 1:
            return {}

        # Positions snapshot: all UAVs use the same p̃* values from the START
        # of this allocation cycle. This is intentional — in discrete-time
        # decentralized simulation, UAVs do not see mid-cycle updates from
        # neighbors. Each UAV acts on the state at the beginning of the step.
        # This matches DEBS Algorithm 2 at this simulation fidelity.
        positions: Dict[int, np.ndarray] = {
            uid: np.asarray(uav.assigned_region.center, dtype=float)
            for uid, uav in uav_dict.items()
        }

        new_allocations: Dict[int, np.ndarray] = {}
        # Track FE usage per interaction for diagnostic logging.
        fe_log: Dict[int, int] = {}

        # Iterate in deterministic order for reproducibility
        for uid in sorted(uav_dict.keys()):
            partner_id = self._pick_partner(uid, positions, current_time)
            if partner_id is None:
                continue
            # Enforce the minimum inter‑interaction time ``t_att``
            key = tuple(sorted((uid, partner_id)))
            last = self._last_interaction.get(key, -math.inf)
            if current_time - last < self.config.t_att:
                continue
            # Objective lambda for this pair (partner centre fixed)
            partner_center = positions[partner_id]
            def obj(p: np.ndarray, pc=partner_center):  # default arg binds current partner centre
                return self._pairwise_objective(p, pc)
            # Initialise population – perturb slightly if co-located to avoid singularity
            centre_i = positions[uid]
            if np.allclose(centre_i, partner_center):
                centre_i = centre_i + self._rng.uniform(-1e-6, 1e-6, size=2)
            init_pop = self._lhs_init(centre_i)
            best_p, _, fe_used = self._run_ide(init_pop, obj)
            new_allocations[uid] = best_p
            fe_log[uid] = fe_used
            # Record interaction timestamp
            self._last_interaction[key] = current_time

        return new_allocations