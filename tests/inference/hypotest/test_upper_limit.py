"""Tests for upper limit finding functions.

Tests upper_limit, upper_limit_scan, upper_limit_toys, and expected_upper_limit
with concrete expected values.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from everwillow.inference.hypotest import (
    ExpectedBands,
    HypoTestResult,
    expected_upper_limit,
    upper_limit,
    upper_limit_scan,
    upper_limit_toys,
)
from everwillow.inference.hypotest import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)

# =============================================================================
# upper_limit Tests
# =============================================================================


class TestUpperLimit:
    """Tests for upper_limit function with exact expected values."""

    def test_linear_objective(self):
        """Test: f(poi) = 1 - 0.5*poi = 0.05 → poi = 1.9"""

        def objective(poi):
            return 1.0 - 0.5 * poi

        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.05)
        assert float(limit) == pytest.approx(1.9, rel=1e-4)

    def test_linear_objective_level_0p1(self):
        """Test: f(poi) = 1 - 0.5*poi = 0.10 → poi = 1.8"""

        def objective(poi):
            return 1.0 - 0.5 * poi

        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.10)
        assert float(limit) == pytest.approx(1.8, rel=1e-4)

    def test_quadratic_objective(self):
        """Test: f(poi) = 1 - 0.25*poi² = 0.05 → poi = sqrt(3.8) = 1.9493..."""

        def objective(poi):
            return 1.0 - 0.25 * poi**2

        expected = math.sqrt(0.95 / 0.25)  # 1.9493588...
        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.05)
        assert float(limit) == pytest.approx(expected, rel=1e-4)

    def test_exponential_objective(self):
        """Test: f(poi) = exp(-poi) = 0.05 → poi = -log(0.05) = 2.9957..."""

        def objective(poi):
            return jnp.exp(-poi)

        expected = -math.log(0.05)  # 2.9957322...
        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.05)
        assert float(limit) == pytest.approx(expected, rel=1e-4)

    def test_custom_solver(self):
        """Test with custom solver gives same result."""

        def objective(poi):
            return 1.0 - 0.5 * poi

        limit = upper_limit(
            objective,
            bounds=(0.0, 5.0),
            level=0.05,
            solver=optx.Bisection(rtol=1e-6, atol=1e-8),
        )
        assert float(limit) == pytest.approx(1.9, rel=1e-5)

    def test_tight_tolerance(self):
        """Test with tight tolerance for high precision."""

        def objective(poi):
            return jnp.exp(-poi)

        expected = -math.log(0.05)
        limit = upper_limit(
            objective,
            bounds=(0.0, 5.0),
            level=0.05,
            rtol=1e-8,
            atol=1e-10,
        )
        assert float(limit) == pytest.approx(expected, rel=1e-6)

    def test_jit_compatibility(self):
        """Test that upper_limit is JIT-compatible."""

        @jax.jit
        def find_limit(level):
            return upper_limit(
                lambda poi: 1.0 - 0.5 * poi,
                bounds=(0.0, 5.0),
                level=level,
            )

        assert float(find_limit(0.05)) == pytest.approx(1.9, rel=1e-4)
        assert float(find_limit(0.10)) == pytest.approx(1.8, rel=1e-4)


# =============================================================================
# upper_limit_scan Tests
# =============================================================================


class TestUpperLimitScan:
    """Tests for upper_limit_scan function (grid search)."""

    def test_linear_objective_coarse(self):
        """Test with coarse grid (50 points)."""

        def objective(poi):
            return 1.0 - 0.5 * poi

        scan = jnp.linspace(0.0, 5.0, 50)
        limit = upper_limit_scan(objective, scan, level=0.05)
        # Coarse grid: ~2% accuracy
        assert float(limit) == pytest.approx(1.9, rel=0.02)

    def test_linear_objective_fine(self):
        """Test with fine grid (500 points)."""

        def objective(poi):
            return 1.0 - 0.5 * poi

        scan = jnp.linspace(0.0, 5.0, 500)
        limit = upper_limit_scan(objective, scan, level=0.05)
        # Fine grid: ~0.5% accuracy
        assert float(limit) == pytest.approx(1.9, rel=0.005)

    def test_exponential_objective(self):
        """Test: exp(-poi) = 0.05 → poi = 2.9957..."""

        def objective(poi):
            return jnp.exp(-poi)

        expected = -math.log(0.05)
        scan = jnp.linspace(0.0, 5.0, 200)
        limit = upper_limit_scan(objective, scan, level=0.05)
        assert float(limit) == pytest.approx(expected, rel=0.01)

    def test_accuracy_improves_with_grid_density(self):
        """Verify finer grid gives better accuracy for nonlinear objective."""

        def objective(poi):
            # Use nonlinear function where interpolation error matters
            return jnp.exp(-poi)

        expected = -math.log(0.05)  # 2.9957...

        coarse = upper_limit_scan(objective, jnp.linspace(0, 5, 10), level=0.05)
        fine = upper_limit_scan(objective, jnp.linspace(0, 5, 200), level=0.05)

        assert abs(float(fine) - expected) < abs(float(coarse) - expected)

    def test_jit_compatibility(self):
        """Test that upper_limit_scan is JIT-compatible."""

        @jax.jit
        def find_limit(scan):
            return upper_limit_scan(lambda poi: 1.0 - 0.5 * poi, scan, level=0.05)

        scan = jnp.linspace(0.0, 5.0, 100)
        assert float(find_limit(scan)) == pytest.approx(1.9, rel=0.02)


# =============================================================================
# upper_limit_toys Tests
# =============================================================================


class TestUpperLimitToys:
    """Tests for upper_limit_toys function (stochastic)."""

    def test_deterministic_objective(self):
        """Test with deterministic objective (ignores key).

        f(poi) = 1 - 0.5*poi = 0.05 → poi = 1.9
        """

        def objective(poi, key):
            return 1.0 - 0.5 * poi

        limit = upper_limit_toys(
            objective,
            bounds=(0.0, 5.0),
            key=jax.random.key(42),
            level=0.05,
            tol=0.01,
        )
        assert float(limit) == pytest.approx(1.9, rel=0.02)

    def test_exponential_deterministic(self):
        """Test: exp(-poi) = 0.05 → poi = 2.9957..."""

        def objective(poi, key):
            return jnp.exp(-poi)

        expected = -math.log(0.05)
        limit = upper_limit_toys(
            objective,
            bounds=(0.0, 5.0),
            key=jax.random.key(42),
            level=0.05,
            tol=0.001,  # Tighter tolerance
            max_iterations=50,  # More iterations
        )
        assert float(limit) == pytest.approx(expected, rel=0.02)

    def test_reproducibility(self):
        """Same key gives same result."""

        def objective(poi, key):
            noise = jax.random.normal(key) * 0.01
            return 1.0 - 0.5 * poi + noise

        limit1 = upper_limit_toys(
            objective, bounds=(0.0, 5.0), key=jax.random.key(123), level=0.05
        )
        limit2 = upper_limit_toys(
            objective, bounds=(0.0, 5.0), key=jax.random.key(123), level=0.05
        )

        assert float(limit1) == float(limit2)

    def test_with_significant_noise(self):
        """Test convergence with significant stochastic noise."""

        def objective(poi, key):
            # Larger noise to test robustness
            noise = jax.random.normal(key) * 0.1
            return 1.0 - 0.5 * poi + noise

        limit = upper_limit_toys(
            objective,
            bounds=(0.0, 5.0),
            key=jax.random.key(42),
            level=0.05,
            tol=0.1,  # Loose tolerance for noisy objective
            max_iterations=50,
        )

        # Should converge to something reasonable within bounds
        assert 0.0 < float(limit) < 5.0
        # And be in the ballpark of 1.9
        assert float(limit) == pytest.approx(1.9, rel=0.5)


# =============================================================================
# expected_upper_limit Tests
# =============================================================================


class TestExpectedUpperLimit:
    """Tests for expected_upper_limit with concrete expected values."""

    def test_with_known_cls_function(self):
        """Test expected_upper_limit with a mock calc_fn.

        Mock CLs(poi) = exp(-poi), so:
        - observed limit: exp(-poi) = 0.05 → poi = 2.996
        - All expected bands use the same formula, so all limits = 2.996
        """
        expected_limit = -math.log(0.05)  # 2.9957...

        def mock_calc_fn(poi):
            """Mock calculator returning CLs = exp(-poi)."""
            cls_val = jnp.exp(-poi)
            # Create mock expected bands - all return same CLs for simplicity
            bands = ExpectedBands(
                minus_2sigma=(jnp.array(0.5), cls_val * jnp.array(0.5)),
                minus_1sigma=(jnp.array(0.5), cls_val * jnp.array(0.5)),
                median=(jnp.array(0.5), cls_val * jnp.array(0.5)),
                plus_1sigma=(jnp.array(0.5), cls_val * jnp.array(0.5)),
                plus_2sigma=(jnp.array(0.5), cls_val * jnp.array(0.5)),
            )
            return HypoTestResult(
                q_obs=jnp.array(0.0),
                pnull=jnp.array(0.5),
                palt=cls_val * jnp.array(0.5),
                cl_s=cls_val,
                expected_bands=bands,
                test_stat_result=TSResult(q=jnp.array(0.0), extras={}),
            )

        result = expected_upper_limit(mock_calc_fn, bounds=(0.0, 5.0), level=0.05)

        assert float(result.observed) == pytest.approx(expected_limit, rel=1e-3)
        assert float(result.expected) == pytest.approx(expected_limit, rel=1e-3)
        assert float(result.minus_2sigma) == pytest.approx(expected_limit, rel=1e-3)
        assert float(result.plus_2sigma) == pytest.approx(expected_limit, rel=1e-3)

    def test_with_varying_bands(self):
        """Test expected_upper_limit with different CLs for each band.

        Mock where:
        - observed CLs = exp(-poi)     → limit = 2.996
        - median CLs = exp(-0.8*poi)   → limit = 3.745
        - -1σ CLs = exp(-0.6*poi)      → limit = 4.993
        - +1σ CLs = exp(-1.0*poi)      → limit = 2.996
        """
        expected_observed = -math.log(0.05)  # 2.996
        expected_median = -math.log(0.05) / 0.8  # 3.745
        expected_minus1 = -math.log(0.05) / 0.6  # 4.993
        expected_plus1 = -math.log(0.05) / 1.0  # 2.996

        def mock_calc_fn(poi):
            """Mock with different sensitivities per band."""
            cls_obs = jnp.exp(-poi)
            bands = ExpectedBands(
                minus_2sigma=(jnp.array(0.5), jnp.exp(-0.5 * poi) * 0.5),
                minus_1sigma=(jnp.array(0.5), jnp.exp(-0.6 * poi) * 0.5),
                median=(jnp.array(0.5), jnp.exp(-0.8 * poi) * 0.5),
                plus_1sigma=(jnp.array(0.5), jnp.exp(-1.0 * poi) * 0.5),
                plus_2sigma=(jnp.array(0.5), jnp.exp(-1.2 * poi) * 0.5),
            )
            return HypoTestResult(
                q_obs=jnp.array(0.0),
                pnull=jnp.array(0.5),
                palt=cls_obs * 0.5,
                cl_s=cls_obs,
                expected_bands=bands,
                test_stat_result=TSResult(q=jnp.array(0.0), extras={}),
            )

        result = expected_upper_limit(mock_calc_fn, bounds=(0.0, 10.0), level=0.05)

        assert float(result.observed) == pytest.approx(expected_observed, rel=1e-3)
        assert float(result.expected) == pytest.approx(expected_median, rel=1e-3)
        assert float(result.minus_1sigma) == pytest.approx(expected_minus1, rel=1e-3)
        assert float(result.plus_1sigma) == pytest.approx(expected_plus1, rel=1e-3)
