import math
import numpy as np
import pytest
from typing import Tuple

from src.algorithms.allocation.ide_allocator import (
    _pairwise_objective,
    _adaptive_F,
    _adaptive_CR,
    _mutate,
    _crossover,
    _lhs_init,
    IDEAllocator
)
from src.config.loader import IDEConfig

class DummyUAV:
    def __init__(self, agent_id: int, center: np.ndarray):
        self.agent_id = agent_id
        class DummyRegion:
            def __init__(self, center):
                self.center = center
        self.assigned_region = DummyRegion(center)


@pytest.fixture
def base_config() -> IDEConfig:
    return IDEConfig(
        population_size=10,
        fe_max=100,
        alpha=0.5,
        d_star=10.0,
        bounds_padding=0.5,
        communication_range=50.0,
        t_att=1.0,
        seed=42
    )


@pytest.fixture
def world_bounds() -> Tuple[float, float, float, float]:
    return (0.0, 100.0, 0.0, 100.0)


# T1: Test _pairwise_objective normal case
def test_pairwise_objective_normal():
    p_i = np.array([0.0, 0.0])
    p_j = np.array([10.0, 0.0])
    d_star = 10.0
    val = _pairwise_objective(p_i, p_j, d_star)
    
    # distance = 10.0
    # expected = 10 * 10 + (10^4) / (2 * 10^2) - 1.5 * 10^2
    # expected = 100 + 10000 / 200 - 150 = 100 + 50 - 150 = 0.0
    assert math.isclose(val, 0.0, abs_tol=1e-6)

    p_k = np.array([5.0, 0.0])
    val2 = _pairwise_objective(p_i, p_k, d_star)
    assert val2 > 0.0  # Should be positive since it's not optimal distance


# T2: Test _pairwise_objective singular case
def test_pairwise_objective_singular():
    p_i = np.array([5.0, 5.0])
    with pytest.raises(ValueError, match="singular case"):
        _pairwise_objective(p_i, p_i, 10.0)


# T3: Test _adaptive_F logic
def test_adaptive_f(base_config):
    # F = 2 * alpha * (fe / fe_max)
    # alpha = 0.5, fe_max = 100
    assert math.isclose(_adaptive_F(0, base_config), 0.0)
    assert math.isclose(_adaptive_F(50, base_config), 0.5)
    assert math.isclose(_adaptive_F(100, base_config), 1.0)


# T4: Test _adaptive_CR logic
def test_adaptive_cr(base_config):
    # CR = 2 * (1 - alpha) * (1 - fe / fe_max)
    # alpha = 0.5, fe_max = 100
    assert math.isclose(_adaptive_CR(0, base_config), 1.0)
    assert math.isclose(_adaptive_CR(50, base_config), 0.5)
    assert math.isclose(_adaptive_CR(100, base_config), 0.0)


# T5: Test _mutate operation (current-to-best/1)
def test_mutate():
    x_i = np.array([1.0, 1.0])
    x_best = np.array([2.0, 2.0])
    x_r1 = np.array([5.0, 0.0])
    x_r2 = np.array([3.0, 0.0])
    F = 0.5
    
    # Expected: x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    # 1.0 + 0.5 * (1.0) + 0.5 * (2.0) = 2.5
    # 1.0 + 0.5 * (1.0) + 0.5 * (0.0) = 1.5
    mutated = _mutate(x_i, x_best, x_r1, x_r2, F)
    np.testing.assert_array_almost_equal(mutated, np.array([2.5, 1.5]))


# T6: Test _crossover (binomial)
def test_crossover():
    rng = np.random.default_rng(42)
    x_i = np.array([0.0, 0.0, 0.0])
    x_new = np.array([1.0, 1.0, 1.0])
    
    # With CR = 0, at least one component (j_rand) should still come from x_new
    crossed_0 = _crossover(x_i, x_new, CR=0.0, rng=rng)
    assert np.sum(crossed_0) >= 1.0
    
    # With CR = 1, all components should come from x_new
    crossed_1 = _crossover(x_i, x_new, CR=1.0, rng=rng)
    np.testing.assert_array_almost_equal(crossed_1, x_new)


# T7: Test _lhs_init initialization
def test_lhs_init(base_config, world_bounds):
    rng = np.random.default_rng(42)
    centre = np.array([50.0, 50.0])
    pop = _lhs_init(centre, base_config, world_bounds, rng)
    
    # Population size should match config
    assert pop.shape == (base_config.population_size, 2)
    
    # First individual must be the exact centre
    np.testing.assert_array_almost_equal(pop[0], centre)
    
    # All individuals should be within Qs bounds
    Qs = base_config.bounds_padding
    assert np.all(pop >= centre - Qs)
    assert np.all(pop <= centre + Qs)


# T8: Test _run_ide (Algorithm 1)
def test_run_ide(base_config, world_bounds):
    allocator = IDEAllocator(base_config, world_bounds)
    
    # Dummy objective: distance from target (target is [60, 60])
    target = np.array([60.0, 60.0])
    def objective(p):
        return float(np.linalg.norm(p - target))
    
    # Start population at [50, 50]
    init_pop = allocator._lhs_init(np.array([50.0, 50.0]))
    
    best, best_score, fe = allocator._run_ide(init_pop, objective)
    
    assert fe > 0
    assert best_score >= 0.0
    # The best score should be reasonably small or better than the start
    initial_score = objective(np.array([50.0, 50.0]))
    assert best_score <= initial_score


# T9: Test _pick_partner
def test_pick_partner(base_config, world_bounds):
    allocator = IDEAllocator(base_config, world_bounds)
    
    # Setup positions for 3 UAVs
    positions = {
        1: np.array([10.0, 10.0]),
        2: np.array([10.0, 20.0]), # Distance 10 (within range 50)
        3: np.array([100.0, 100.0]) # Distance > 50 (outside range)
    }
    
    # Test partner selection for UAV 1
    # UAV 3 is out of range, UAV 2 is in range.
    partner = allocator._pick_partner(1, positions, current_time=0.0)
    assert partner == 2
    
    # Record an interaction with UAV 2
    allocator._last_interaction[(1, 2)] = 10.0
    
    # Add UAV 4 in range
    positions[4] = np.array([10.0, 15.0])
    
    # Now UAV 4 has not been interacted with (elapsed = inf), it should be picked over UAV 2
    partner2 = allocator._pick_partner(1, positions, current_time=15.0)
    assert partner2 == 4


# T10: Test allocate (Algorithm 2)
def test_allocate(base_config, world_bounds):
    allocator = IDEAllocator(base_config, world_bounds)
    
    uavs = [
        DummyUAV(1, np.array([10.0, 10.0])),
        DummyUAV(2, np.array([10.0 + base_config.d_star - 2, 10.0])), # Needs to spread out
    ]
    
    new_allocations = allocator.allocate(uavs, current_time=2.0)
    
    # Expect only one UAV to move because they pair with each other,
    # and the first one (UAV 1) updates the _last_interaction timestamp,
    # locking out UAV 2 due to t_att.
    assert len(new_allocations) == 1
    assert 1 in new_allocations
    
    # Verify that the new centers are within world bounds
    for uid, center in new_allocations.items():
        assert center[0] >= world_bounds[0] and center[0] <= world_bounds[1]
        assert center[1] >= world_bounds[2] and center[1] <= world_bounds[3]
