"""Tests for hypotest function (asymptotic hypothesis testing)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

import everwillow.statelib as sl
from everwillow.inference.calculators import cls
from everwillow.inference.hypotest import HypoTestResult, hypotest

jax.config.update("jax_enable_x64", True)


def make_counting_nll(signal: float, bkg: float, bkg_unc: float, observed: float):
    """Create NLL matching pyhf's uncorrelated_background model.

    Uses Poisson auxiliary constraint (matching pyhf).
    """
    aux_data = (bkg / bkg_unc) ** 2

    def nll(params):
        mu = params["mu"]
        gamma = params["gamma"]
        expected = mu * signal + bkg * gamma
        main_nll = (
            expected
            - observed * jnp.log(jnp.maximum(expected, 1e-10))
            + gammaln(observed + 1)
        )
        aux_expected = gamma * aux_data
        constraint_nll = (
            aux_expected
            - aux_data * jnp.log(jnp.maximum(aux_expected, 1e-10))
            + gammaln(aux_data + 1)
        )
        return main_nll + constraint_nll

    return nll


class TestHypoTestResult:
    """Tests for HypoTestResult dataclass."""

    def test_result_has_required_fields(self):
        """HypoTestResult should have p_alt, p_null, q_obs, expected_pvalues."""
        result = HypoTestResult(
            p_alt=0.084,
            p_null=0.5,
            q_obs=1.9,
            expected_pvalues=[
                (0.01, 0.02),
                (0.05, 0.16),
                (0.08, 0.5),
                (0.2, 0.84),
                (0.4, 0.98),
            ],
        )
        assert hasattr(result, "p_alt")
        assert hasattr(result, "p_null")
        assert hasattr(result, "q_obs")
        assert hasattr(result, "expected_pvalues")


class TestHypoTestBasic:
    """Basic tests for hypotest function."""

    def test_hypotest_returns_result(self):
        """hypotest should return HypoTestResult."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        assert isinstance(result, HypoTestResult)

    def test_hypotest_pvalues_in_valid_range(self):
        """P-values should be in [0, 1]."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        assert 0.0 <= float(result.p_alt) <= 1.0
        assert 0.0 <= float(result.p_null) <= 1.0

    def test_hypotest_q_obs_non_negative(self):
        """Test statistic q should be non-negative."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        assert float(result.q_obs) >= 0.0

    def test_cls_can_be_computed_from_pvalues(self):
        """User can compute CLs from returned p-values."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)
        cls_obs = cls(result.p_alt, result.p_null)

        assert float(cls_obs) >= 0.0
        assert jnp.isfinite(cls_obs)


class TestHypoTestExpectedBands:
    """Tests for expected p-value bands."""

    def test_expected_bands_length(self):
        """Expected bands should have 5 elements [-2s, -1s, median, +1s, +2s]."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        assert len(result.expected_pvalues) == 5

    def test_expected_bands_are_tuples(self):
        """Each expected band should be a (p_alt, p_null) tuple."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        for band in result.expected_pvalues:
            assert isinstance(band, tuple)
            assert len(band) == 2
            p_alt_exp, p_null_exp = band
            assert 0.0 <= p_alt_exp <= 1.0
            assert 0.0 <= p_null_exp <= 1.0

    def test_expected_cls_bands_ordered(self):
        """Expected CLs bands should be ordered: -2s < -1s < median < +1s < +2s."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        # Compute CLs for each band
        cls_bands = [cls(p_alt, p_null) for p_alt, p_null in result.expected_pvalues]
        minus2, minus1, median, plus1, plus2 = cls_bands

        assert (
            float(minus2) < float(minus1) < float(median) < float(plus1) < float(plus2)
        )

    def test_expected_median_equals_observed_when_data_equals_asimov(self):
        """When observed = Asimov expectation, observed p-values should equal median expected."""
        # In this model, observed=9 and bkg=9, so data equals Asimov
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        # Median is index 2
        p_alt_median, p_null_median = result.expected_pvalues[2]
        assert jnp.isclose(result.p_alt, p_alt_median, rtol=0.01)
        assert jnp.isclose(result.p_null, p_null_median, rtol=0.01)


class TestHypoTestValues:
    """Tests for specific numerical values (validated against pyhf)."""

    def test_hypotest_cls_matches_expected_order_of_magnitude(self):
        """CLs at mu=1 for this model should be around 0.17."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)
        cls_obs = cls(result.p_alt, result.p_null)

        # Should be around 0.168 (from pyhf)
        assert 0.1 < float(cls_obs) < 0.3

    def test_hypotest_p_null_approximately_half(self):
        """For Asimov-like data, p_null should be close to 0.5."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)

        # p_null should be 0.5 when data equals Asimov expectation
        assert jnp.isclose(result.p_null, 0.5, atol=0.01)


class TestHypoTestAtDifferentMu:
    """Tests for hypotest at different mu values."""

    def test_cls_decreases_with_mu_for_deficit(self):
        """For deficit data (obs < expected), CLs decreases as mu increases.

        In this model: observed=9, background=9, signal=6.
        At mu=0.5: expected = 12, small deficit, low q, high CLs.
        At mu=2.0: expected = 21, large deficit, high q, low CLs.
        """
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result_low = hypotest(nll_fn, params, poi_name="mu", poi_test=0.5)
        result_high = hypotest(nll_fn, params, poi_name="mu", poi_test=2.0)

        cls_low = cls(result_low.p_alt, result_low.p_null)
        cls_high = cls(result_high.p_alt, result_high.p_null)

        # Higher mu has larger deficit -> easier to exclude -> lower CLs
        assert float(cls_low) > float(cls_high)

    def test_cls_at_mu_zero(self):
        """At mu=0, CLs should be 1 (can't exclude no signal)."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        result = hypotest(nll_fn, params, poi_name="mu", poi_test=0.0)
        cls_obs = cls(result.p_alt, result.p_null)

        # At mu=0, q should be 0 (best fit), so CLs = 1
        assert jnp.isclose(cls_obs, 1.0, atol=0.01)
