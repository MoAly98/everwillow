"""Tests for limit solvers (LimitSolver contract and built-in solvers).

Expected-value policy: every number asserted here is hand-derived in the test
docstring from closed forms (no expected value is produced by running reference
code). The workhorse objective is exp(-poi) crossing level=0.05, whose root is
    exp(-poi) = 1/20  =>  poi = ln 20 = ln 2 + ln 10 = 2.9957322735539909.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from everwillow._src.inference.hypotest.limit_solvers import (
    BisectionLimitSolver,
    GridScanLimitSolver,
    LimitSolver,
    RootFindingLimitSolver,
    StochasticLimitSolver,
)

LEVEL = 0.05
LN20 = 2.9957322735539909  # ln 20, root of exp(-x) = 0.05
LN10 = 2.302585092994046  # ln 10


def exp_objective(poi, key=None):
    """exp(-poi): crosses 0.05 at ln 20."""
    return jnp.exp(-poi)


def pytree_objective(poi, key=None):
    """Leaves cross 0.05 at ln 20 and ln 20 / 2 respectively."""
    return {"a": jnp.exp(-poi), "b": jnp.exp(-2.0 * poi)}


def noisy_objective(poi, key=None):
    """exp(-poi) plus key-dependent noise (for reproducibility contracts)."""
    noise = 0.0 if key is None else 0.05 * jax.random.normal(key)
    return jnp.exp(-poi) + noise


# =============================================================================
# Contract
# =============================================================================


class TestLimitSolverContract:
    """The LimitSolver ABC and the stochastic-safety marker hierarchy."""

    def test_abc_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LimitSolver()

    def test_stochastic_marker_membership(self):
        """Noise-validity is class membership: GridScan and Bisection declare it,
        RootFind (adaptive, assumes a deterministic criterion) does not."""
        assert isinstance(GridScanLimitSolver(scan=jnp.linspace(0.0, 1.0, 3)), StochasticLimitSolver)
        assert isinstance(BisectionLimitSolver(bounds=(0.0, 1.0)), StochasticLimitSolver)
        assert not isinstance(RootFindingLimitSolver(bounds=(0.0, 1.0)), StochasticLimitSolver)


# =============================================================================
# RootFind
# =============================================================================


class TestRootFind:
    """Adaptive root finding on deterministic objectives."""

    def test_exponential_root(self):
        """exp(-x) = 0.05 at x = ln 20; default rtol=1e-4 => rel 1e-3 is ample."""
        solver = RootFindingLimitSolver(bounds=(0.1, 8.0))

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(LN20, rel=1e-3)

    def test_tight_solver_override(self):
        """A user-supplied optimistix root finder controls the precision."""
        solver = RootFindingLimitSolver(bounds=(0.1, 8.0), solver=optx.Bisection(rtol=1e-6, atol=1e-8))

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(LN20, rel=1e-5)

    def test_pytree_objective_per_leaf_roots(self):
        """exp(-x) and exp(-2x) cross 0.05 at ln 20 and (ln 20)/2 = 1.4978661."""
        solver = RootFindingLimitSolver(bounds=(0.1, 8.0))

        limits = solver.solve(pytree_objective, LEVEL)

        assert float(limits["a"]) == pytest.approx(LN20, rel=1e-3)
        assert float(limits["b"]) == pytest.approx(LN20 / 2.0, rel=1e-3)


# =============================================================================
# GridScan
# =============================================================================


class TestGridScan:
    """Fixed grid + per-leaf linear interpolation."""

    def test_fine_grid_root(self):
        """Grid spacing h=0.01. Linear-interpolation bias for a convex f is
        bounded by max|f''| h^2 / 8 / |f'| = h^2/8 = 1.25e-5 for exp(-x)
        (f''/f' = -1), so abs=1e-4 is a derived, not generous, tolerance."""
        solver = GridScanLimitSolver(scan=jnp.linspace(0.1, 6.0, 591))

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(LN20, abs=1e-4)

    def test_coarse_grid_returns_chord_crossing(self):
        """The systematic interpolation bias is asserted, not hidden.

        Grid [0, 2, 4, 6]: the crossing is bracketed by (2, 4). Linear
        interpolation returns the chord crossing
            x* = 2 + 2 (e^-2 - 0.05) / (e^-2 - e^-4)
               = 2 + 2 * 0.0853352832 / 0.1170196443 = 3.4585
        which sits 0.46 above the true root ln 20 = 2.9957 (chord above a
        convex decreasing curve pushes the crossing right).
        """
        solver = GridScanLimitSolver(scan=jnp.linspace(0.0, 6.0, 4))

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(3.4585, abs=5e-3)

    def test_pytree_objective_single_pass(self):
        """Per-leaf interpolation from one grid evaluation."""
        solver = GridScanLimitSolver(scan=jnp.linspace(0.1, 6.0, 591))

        limits = solver.solve(pytree_objective, LEVEL)

        assert float(limits["a"]) == pytest.approx(LN20, abs=1e-4)
        assert float(limits["b"]) == pytest.approx(LN20 / 2.0, abs=1e-4)

    def test_pytree_objective_costs_no_extra_passes(self):
        """A multi-leaf objective is evaluated exactly as often as a scalar
        one — per-leaf limits come from ONE grid pass (call-count glue; this
        is what makes Brazil bands cheap on a grid). Holds whether the grid
        is vmapped (one traced call per grid point) or looped."""
        solver = GridScanLimitSolver(scan=jnp.linspace(0.1, 6.0, 60))
        scalar_calls = []
        pytree_calls = []

        def counting_scalar(poi, key=None):
            scalar_calls.append(poi)
            return jnp.exp(-poi)

        def counting_pytree(poi, key=None):
            pytree_calls.append(poi)
            return {"a": jnp.exp(-poi), "b": jnp.exp(-2.0 * poi)}

        solver.solve(counting_scalar, LEVEL)
        solver.solve(counting_pytree, LEVEL)

        assert len(pytree_calls) == len(scalar_calls)

    def test_keyed_reproducibility(self):
        """Same key -> same limit; different keys -> different limits
        (key-sensitive objective; values themselves are not asserted)."""
        solver = GridScanLimitSolver(scan=jnp.linspace(0.1, 6.0, 60))

        limit_a1 = solver.solve(noisy_objective, LEVEL, key=jax.random.key(1))
        limit_a2 = solver.solve(noisy_objective, LEVEL, key=jax.random.key(1))
        limit_b = solver.solve(noisy_objective, LEVEL, key=jax.random.key(2))

        assert float(limit_a1) == float(limit_a2)
        assert float(limit_a1) != float(limit_b)


# =============================================================================
# Bisection
# =============================================================================


class TestBisection:
    """Bracket-halving with an early-exit tolerance on the objective value."""

    def test_exponential_root_tol_zero(self):
        """tol=0 disables the early exit: max_iterations=100 halvings collapse
        the bracket to (8-0.1)/2^100, so the result is exact to solver noise."""
        solver = BisectionLimitSolver(bounds=(0.1, 8.0), tol=0.0)

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(LN20, rel=1e-3)

    def test_coarse_tol_stops_at_first_in_tolerance_midpoint(self):
        """Hand-traced early exit on round numbers (the documented contract:
        evaluate midpoint -> update bracket -> return bracket midpoint).

        bounds (0, 8), tol=0.5: iteration 1 evaluates mid=4,
        f(4)=e^-4=0.0183, |0.0183-0.05|=0.032 < 0.5 -> converged;
        f < level puts the crossing left: bracket becomes (0, 4),
        and the returned value is its midpoint 2.0 — far from ln 20,
        which is exactly the early-exit artifact this documents.
        """
        solver = BisectionLimitSolver(bounds=(0.0, 8.0), tol=0.5)

        limit = solver.solve(exp_objective, LEVEL)

        assert float(limit) == pytest.approx(2.0, abs=1e-6)

    def test_pytree_objective_per_leaf_roots(self):
        solver = BisectionLimitSolver(bounds=(0.1, 8.0), tol=0.0)

        limits = solver.solve(pytree_objective, LEVEL)

        assert float(limits["a"]) == pytest.approx(LN20, rel=1e-3)
        assert float(limits["b"]) == pytest.approx(LN20 / 2.0, rel=1e-3)

    def test_keyed_reproducibility(self):
        """Same key -> same limit; different keys -> different limits."""
        solver = BisectionLimitSolver(bounds=(0.1, 8.0), tol=0.0)

        limit_a1 = solver.solve(noisy_objective, LEVEL, key=jax.random.key(1))
        limit_a2 = solver.solve(noisy_objective, LEVEL, key=jax.random.key(1))
        limit_b = solver.solve(noisy_objective, LEVEL, key=jax.random.key(2))

        assert float(limit_a1) == float(limit_a2)
        assert float(limit_a1) != float(limit_b)
