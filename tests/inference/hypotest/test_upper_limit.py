"""Tests for upper limit finding functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

import everwillow.statelib as sl
from everwillow.hypotest.calculators import HypoTestCalculator
from everwillow.hypotest.distributions import Distribution, QMuAsymptotic
from everwillow.hypotest.results import BandValues, ExpectedBands
from everwillow.hypotest.results import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)
from everwillow.hypotest.test_statistics import TestStatistic
from everwillow.hypotest.upper_limit import (
    expected_upper_limit,
    upper_limit,
    upper_limit_scan,
    upper_limit_toys,
)
from everwillow.hypotest.utils import cl_s, significance

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
# expected_upper_limit Helpers
# =============================================================================


def _dummy_nll(params, observation):
    """No-op NLL for tests that bypass fitting."""
    return 0.0


_DUMMY_PARAMS = sl.State.from_pytree({"mu": 0.0})
_DUMMY_OBS = {}


class _IdentityTestStat(TestStatistic):
    """Returns poi_test as the test stat value (no fitting)."""

    def _compute(self, nll_fn, params, observation, poi_key, poi_test, **kwargs):
        return jnp.asarray(poi_test), {}


class _ConstantSigmaTestStat(TestStatistic):
    """Returns q = poi², q_asimov = poi² (emulates σ=1 model).

    Overrides __call__ to set q_asimov (base class leaves it None).
    """

    def compute(self, nll_fn, params, observation, poi_key, poi_test, **kwargs):
        q = jnp.asarray(poi_test) ** 2
        return TSResult(value=q, test=jnp.asarray(poi_test), q_asimov=q)

    def _compute(self, nll_fn, params, observation, poi_key, poi_test, **kwargs):
        return jnp.asarray(poi_test) ** 2, {}


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
        cl_s=BandValues(**{n: cl_s(pn, pa) for n, pn, pa in zip(band_names, pnulls, palts, strict=False)}),
        null_sig=BandValues(**{n: significance(pn) for n, pn in zip(band_names, pnulls, strict=False)}),
        alt_sig=BandValues(**{n: significance(pa) for n, pa in zip(band_names, palts, strict=False)}),
    )


class _VaryingBandDist(Distribution):
    """Distribution with different exponential decay rates per expected band.

    Observed: CLs = exp(-poi) (pnull = exp(-poi)*0.5, palt = 0.5)
    Bands: CLs = exp(-rate*poi) with rates [0.5, 0.6, 0.8, 1.0, 1.2]
    """

    def cdf(self, q, mu, mu_prime, sigma):
        raise NotImplementedError

    def null_pval(self, result):
        return jnp.exp(-result.test) * 0.5

    def alt_pval(self, result):
        return jnp.array(0.5)

    def pvalue_bands(self, result):
        poi = result.test
        palt = jnp.array(0.5)
        rates = [0.5, 0.6, 0.8, 1.0, 1.2]
        pnulls = [jnp.exp(-r * poi) * palt for r in rates]
        palts = [palt] * 5
        return _make_mock_bands(pnulls, palts)


# =============================================================================
# expected_upper_limit Tests
# =============================================================================


class TestExpectedUpperLimit:
    """Tests for expected_upper_limit with concrete expected values."""

    def test_with_varying_bands(self):
        """Test expected_upper_limit with different CLs for each band.

        CLs = pnull/palt, with palt=0.5 constant.
        Distribution varies sensitivity per band:
        - -2σ CLs = exp(-0.5*poi)      → limit = -ln(0.05)/0.5 = 5.991
        - -1σ CLs = exp(-0.6*poi)      → limit = -ln(0.05)/0.6 = 4.993
        - median CLs = exp(-0.8*poi)   → limit = -ln(0.05)/0.8 = 3.745
        - +1σ CLs = exp(-1.0*poi)      → limit = -ln(0.05)/1.0 = 2.996
        - +2σ CLs = exp(-1.2*poi)      → limit = -ln(0.05)/1.2 = 2.496
        """
        expected_minus2 = 5.99146  # -ln(0.05) / 0.5
        expected_minus1 = 4.99289  # -ln(0.05) / 0.6
        expected_median = 3.74466  # -ln(0.05) / 0.8
        expected_plus1 = 2.99573  # -ln(0.05) / 1.0
        expected_plus2 = 2.49644  # -ln(0.05) / 1.2

        calc = HypoTestCalculator(
            nll_fn=_dummy_nll,
            params=_DUMMY_PARAMS,
            observation=_DUMMY_OBS,
            poi_key="mu",
            test_statistic=_IdentityTestStat(),
            distribution=_VaryingBandDist(),
        )

        def band_cls_objective(poi):
            result = calc.test(poi)
            bands = calc.pvalue_bands(result)
            assert bands is not None
            return bands.cl_s

        result = expected_upper_limit(band_cls_objective, bounds=(0.0, 10.0), level=0.05)

        assert float(result.minus_2sigma) == pytest.approx(expected_minus2, rel=1e-3)
        assert float(result.minus_1sigma) == pytest.approx(expected_minus1, rel=1e-3)
        assert float(result.median) == pytest.approx(expected_median, rel=1e-3)
        assert float(result.plus_1sigma) == pytest.approx(expected_plus1, rel=1e-3)
        assert float(result.plus_2sigma) == pytest.approx(expected_plus2, rel=1e-3)


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

    Uses _ConstantSigmaTestStat (q = poi², q_asimov = poi²) to emulate
    a σ=1 model so the analytic formula μ_up(N) = σ·(Φ⁻¹(1 - α·Φ(N)) + N)
    applies.
    """

    SIGMA = 1.0
    ALPHA = 0.05

    @staticmethod
    def _analytic_upper_limit(n_sigma: float, sigma: float, alpha: float) -> float:
        """μ_up(N) = σ·(Φ⁻¹(1 - α·Φ(N)) + N)."""
        phi_n = _normal_cdf(n_sigma)
        return sigma * (_normal_ppf(1.0 - alpha * phi_n) + n_sigma)

    @pytest.fixture
    def calc(self) -> HypoTestCalculator:
        """Calculator with constant-σ test stat and QMuAsymptotic distribution."""
        return HypoTestCalculator(
            nll_fn=_dummy_nll,
            params=_DUMMY_PARAMS,
            observation=_DUMMY_OBS,
            poi_key="mu",
            test_statistic=_ConstantSigmaTestStat(),
            distribution=QMuAsymptotic(),
        )

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
    def test_expected_band(self, calc: HypoTestCalculator, band_name: str, n_sigma: float):
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

        def band_cls_objective(poi):
            result = calc.test(poi)
            bands = calc.pvalue_bands(result)
            assert bands is not None
            return bands.cl_s

        result = expected_upper_limit(band_cls_objective, bounds=(0.01, 8.0), level=self.ALPHA)
        actual = float(result[band_name])

        assert actual == pytest.approx(expected, rel=1e-2)

    def test_zero_lower_bound_handled(self, calc: HypoTestCalculator):
        """bounds=(0.0, ...) must not produce NaN from poi=0 singularity.

        Asymptotic formulas have σ = μ/√q_A, which is 0/0 at poi=0.
        pvalue_bands must handle this gracefully.
        Expected median limit = 1.960 (same as with bounds=(0.01, 8.0)).
        """

        def band_cls_objective(poi):
            result = calc.test(poi)
            bands = calc.pvalue_bands(result)
            assert bands is not None
            return bands.cl_s

        result = expected_upper_limit(band_cls_objective, bounds=(0.0, 8.0), level=self.ALPHA)

        for _, val in result:
            assert jnp.isfinite(val)
        assert float(result.median) == pytest.approx(1.960, rel=1e-2)
