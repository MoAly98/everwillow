"""Tests for upper limit finding functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from everwillow.inference.hypotest import (
    BandValues,
    ExpectedBands,
    HypoTestResult,
    QMuAsymptotic,
    cl_s,
    expected_upper_limit,
    significance,
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

    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (0.05, 1.9),  # f(poi) = 1 - 0.5*poi = 0.05 → poi = 1.9
            (0.10, 1.8),  # f(poi) = 1 - 0.5*poi = 0.10 → poi = 1.8
        ],
    )
    def test_linear_objective(self, level, expected):
        """Test: f(poi) = 1 - 0.5*poi = level → poi = 2*(1 - level)."""

        def objective(poi):
            return 1.0 - 0.5 * poi

        limit = upper_limit(objective, bounds=(0.0, 5.0), level=level)
        assert float(limit) == pytest.approx(expected, rel=1e-4)

    def test_quadratic_objective(self):
        """Test: f(poi) = 1 - 0.25*poi² = 0.05 → poi = sqrt(0.95/0.25) = 1.94936."""

        def objective(poi):
            return 1.0 - 0.25 * poi**2

        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.05)
        assert float(limit) == pytest.approx(1.94936, rel=1e-4)

    def test_exponential_objective(self):
        """Test: f(poi) = exp(-poi) = 0.05 → poi = -ln(0.05) = 2.99573."""

        def objective(poi):
            return jnp.exp(-poi)

        limit = upper_limit(objective, bounds=(0.0, 5.0), level=0.05)
        assert float(limit) == pytest.approx(2.99573, rel=1e-4)

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

        limit = upper_limit(
            objective,
            bounds=(0.0, 5.0),
            level=0.05,
            rtol=1e-8,
            atol=1e-10,
        )
        assert float(limit) == pytest.approx(2.99573, rel=1e-6)

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

    @pytest.mark.parametrize(
        "bounds",
        [(3.0, 5.0), (0.0, 1.0)],
        ids=["root-below", "root-above"],
    )
    def test_root_outside_bounds(self, bounds):
        """Raises when root (1.9) is outside the search range."""
        with pytest.raises(RuntimeError, match="root is not contained"):
            upper_limit(lambda poi: 1.0 - 0.5 * poi, bounds=bounds, level=0.05)


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
        # Coarse grid: ~2% accuracy (random)
        assert float(limit) == pytest.approx(1.9, rel=0.02)

    def test_linear_objective_fine(self):
        """Test with fine grid (500 points)."""

        def objective(poi):
            return 1.0 - 0.5 * poi

        scan = jnp.linspace(0.0, 5.0, 500)
        limit = upper_limit_scan(objective, scan, level=0.05)
        # Fine grid: ~0.5% accuracy (random)
        assert float(limit) == pytest.approx(1.9, rel=0.005)

    def test_exponential_objective(self):
        """Test: exp(-poi) = 0.05 → poi = -ln(0.05) = 2.99573."""

        def objective(poi):
            return jnp.exp(-poi)

        scan = jnp.linspace(0.0, 5.0, 200)
        limit = upper_limit_scan(objective, scan, level=0.05)
        assert float(limit) == pytest.approx(2.99573, rel=0.005)

    def test_accuracy_improves_with_grid_density(self):
        """Verify finer grid gives better accuracy for nonlinear objective."""

        def objective(poi):
            # Use nonlinear function where interpolation error matters
            return jnp.exp(-poi)

        coarse = upper_limit_scan(objective, jnp.linspace(0, 5, 10), level=0.05)
        fine = upper_limit_scan(objective, jnp.linspace(0, 5, 200), level=0.05)

        assert abs(float(fine) - 2.99573) < abs(float(coarse) - 2.99573)

    def test_jit_compatibility(self):
        """Test that upper_limit_scan is JIT-compatible."""

        @jax.jit
        def find_limit(scan):
            return upper_limit_scan(lambda poi: 1.0 - 0.5 * poi, scan, level=0.05)

        scan = jnp.linspace(0.0, 5.0, 100)
        assert float(find_limit(scan)) == pytest.approx(1.9, rel=0.02)

    @pytest.mark.parametrize(
        "scan_range",
        [(3.0, 5.0), (0.0, 1.0)],
        ids=["root-below", "root-above"],
    )
    def test_root_outside_scan_range(self, scan_range):
        """Raises when root (1.9) is outside the scan range."""
        scan = jnp.linspace(*scan_range, 50)
        with pytest.raises(RuntimeError, match="root not found within scan range"):
            upper_limit_scan(lambda poi: 1.0 - 0.5 * poi, scan, level=0.05)


# =============================================================================
# upper_limit_toys Tests
# =============================================================================


class TestUpperLimitToys:
    """Tests for upper_limit_toys function (stochastic bisection)."""

    def test_noisy_objective(self):
        """Converges to correct root despite per-iteration noise.

        f(poi) = 1 - 0.5*poi + noise = 0.05 → true root at poi = 1.9.
        Each iteration gets a fresh key via fold_in, producing independent noise.
        """

        def objective(poi, key):
            noise = jax.random.normal(key) * 0.1
            return 1.0 - 0.5 * poi + noise

        limit = upper_limit_toys(
            objective,
            bounds=(0.0, 5.0),
            key=jax.random.key(42),
            level=0.05,
            tol=0.1,
            max_iterations=50,
        )

        assert float(limit) == pytest.approx(1.9, rel=0.2)

    @pytest.mark.parametrize(
        "bounds",
        [(3.0, 5.0), (0.0, 1.0)],
        ids=["root-below", "root-above"],
    )
    def test_root_outside_bounds(self, bounds):
        """Raises when root (1.9) is outside the search range."""

        def objective(poi, key):
            return 1.0 - 0.5 * poi

        with pytest.raises(RuntimeError, match="root not found within bounds"):
            upper_limit_toys(
                objective,
                bounds=bounds,
                key=jax.random.key(42),
                level=0.05,
                tol=0.01,
            )


# =============================================================================
# expected_upper_limit Tests
# =============================================================================


def _make_mock_bands(pnulls, palts):
    """Build ExpectedBands from parallel lists of pnull/palt values."""
    band_names = [
        "minus_2sigma",
        "minus_1sigma",
        "median",
        "plus_1sigma",
        "plus_2sigma",
    ]
    return ExpectedBands(
        null_pvalue=BandValues(**dict(zip(band_names, pnulls, strict=False))),
        alt_pvalue=BandValues(**dict(zip(band_names, palts, strict=False))),
        cl_s=BandValues(
            **{
                n: cl_s(pn, pa)
                for n, pn, pa in zip(band_names, pnulls, palts, strict=False)
            }
        ),
        null_sig=BandValues(
            **{n: significance(pn) for n, pn in zip(band_names, pnulls, strict=False)}
        ),
        alt_sig=BandValues(
            **{n: significance(pa) for n, pa in zip(band_names, palts, strict=False)}
        ),
    )


class TestExpectedUpperLimit:
    """Tests for expected_upper_limit with concrete expected values."""

    def test_with_varying_bands(self):
        """Test expected_upper_limit with different CLs for each band.

        CLs = pnull/palt, with palt=0.5 constant.
        Mock where:
        - observed CLs = exp(-poi)     → limit = 2.996
        - median CLs = exp(-0.8*poi)   → limit = 3.745
        - -1σ CLs = exp(-0.6*poi)      → limit = 4.993
        - +1σ CLs = exp(-1.0*poi)      → limit = 2.996
        """
        expected_observed = 2.99573  # -ln(0.05)
        expected_median = 3.74466  # -ln(0.05) / 0.8
        expected_minus1 = 4.99289  # -ln(0.05) / 0.6
        expected_plus1 = 2.99573  # -ln(0.05) / 1.0

        def mock_calc_fn(poi):
            """Mock with different sensitivities per band."""
            palt = jnp.array(0.5)
            pnull_obs = jnp.exp(-poi) * palt
            rates = [0.5, 0.6, 0.8, 1.0, 1.2]
            pnulls = [jnp.exp(-r * poi) * palt for r in rates]
            palts = [palt] * 5
            bands = _make_mock_bands(pnulls, palts)
            return HypoTestResult(
                q_obs=jnp.array(0.0),
                pnull=pnull_obs,
                palt=palt,
                cl_s=jnp.exp(-poi),
                expected_bands=bands,
                test_stat_result=TSResult(value=jnp.array(0.0), test=jnp.array(0.0)),
            )

        result = expected_upper_limit(mock_calc_fn, bounds=(0.0, 10.0), level=0.05)

        assert float(result.observed) == pytest.approx(expected_observed, rel=1e-3)
        assert float(result.expected.median) == pytest.approx(expected_median, rel=1e-3)
        assert float(result.expected.minus_1sigma) == pytest.approx(
            expected_minus1, rel=1e-3
        )
        assert float(result.expected.plus_1sigma) == pytest.approx(
            expected_plus1, rel=1e-3
        )


# =============================================================================
# expected_upper_limit with real asymptotic distributions
# =============================================================================


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Φ⁻¹)."""
    return float(jax.scipy.stats.norm.ppf(jnp.array(p)))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class TestExpectedUpperLimitAsymptotic:
    """Integration test: expected_upper_limit with QMuAsymptotic.

    Uses a distribution-level calc_fn with constant σ=1 so the
    analytic formula μ_up(N) = σ·(Φ⁻¹(1 - α·Φ(N)) + N) applies.
    """

    SIGMA = 1.0
    ALPHA = 0.05

    @staticmethod
    def _analytic_upper_limit(n_sigma: float, sigma: float, alpha: float) -> float:
        """μ_up(N) = σ·(Φ⁻¹(1 - α·Φ(N)) + N)."""
        phi_n = _normal_cdf(n_sigma)
        return sigma * (_normal_ppf(1.0 - alpha * phi_n) + n_sigma)

    def _make_calc_fn(self):
        """Build a calc_fn wrapping QMuAsymptotic with constant σ=1."""
        dist = QMuAsymptotic()

        def calc_fn(poi):
            # q_asimov = (poi/σ)² with σ=1 → q_asimov = poi²
            q_asimov = poi**2
            # Observed: use q_obs = q_asimov (Asimov observation)
            result = TSResult(
                value=q_asimov,
                test=poi,
                q_asimov=q_asimov,
            )
            pnull = dist.null_pval(result)
            palt = dist.alt_pval(result)
            bands = dist.expected_pvalues(result)

            return HypoTestResult(
                q_obs=result.value,
                pnull=pnull,
                palt=palt,
                cl_s=cl_s(pnull, palt),
                expected_bands=bands,
                test_stat_result=result,
            )

        return calc_fn

    @pytest.mark.parametrize(
        ("band_name", "n_sigma"),
        [
            ("minus_2sigma", -2.0),
            ("minus_1sigma", -1.0),
            ("median", 0.0),
            ("plus_1sigma", 1.0),
            ("plus_2sigma", 2.0),
        ],
    )
    def test_expected_band(self, band_name: str, n_sigma: float):
        """Each expected band matches the analytic formula.

        Hardcoded expected values (σ=1, α=0.05):
            -2σ: 1.052, -1σ: 1.412, median: 1.960, +1σ: 2.727, +2σ: 3.656
        """
        expected_values = {
            "minus_2sigma": 1.052,
            "minus_1sigma": 1.412,
            "median": 1.960,
            "plus_1sigma": 2.727,
            "plus_2sigma": 3.656,
        }
        expected = expected_values[band_name]

        # Cross-check with analytic formula
        analytic = self._analytic_upper_limit(n_sigma, self.SIGMA, self.ALPHA)
        assert analytic == pytest.approx(expected, abs=0.001)

        calc_fn = self._make_calc_fn()
        result = expected_upper_limit(calc_fn, bounds=(0.01, 8.0), level=self.ALPHA)
        actual = float(result.expected[band_name])

        assert actual == pytest.approx(expected, rel=1e-2)
