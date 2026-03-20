"""Tests for hypothesis test calculators."""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import pytest

from everwillow.inference.hypotest import (
    AsymptoticCalculator,
    HypoTestCalculator,
    HypoTestResult,
    QMu,
    QMuAsymptotic,
    QTilde,
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
    TMuAsymptotic,
)
from everwillow.inference.hypotest import (
    TestStatResult as TSResult,
)

from ._counting_model import (
    create_observation,
    create_params,
    poisson_nll,
    predict_fn,
)

# =============================================================================
# AsymptoticCalculator Tests
# =============================================================================


class TestAsymptoticCalculator:
    """Tests for AsymptoticCalculator."""

    def test_basic_result_structure(self):
        """Test that calculator returns correct values at MLE.

        n_obs=15, poi_test=1.0 (at MLE): q_obs=0.
        Asimov at mu=0 (n=5), tested at mu=1: q_asimov = 9.0139.
        pnull = 1-Φ(0) = 0.5, palt = 1-Φ(0-3.002) = 0.99866.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)
        assert float(result.pnull) == pytest.approx(0.5, rel=1e-3)
        assert float(result.palt) == pytest.approx(0.99866, rel=1e-3)
        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_q_asimov_with_asimov_observation(self):
        """Test q_asimov with explicit Asimov observation.

        Asimov at mu=1 (n=15), testing at mu=1: q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc.test(poi_test=1.0, asimov_observation=asimov)

        assert result.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

    def test_q_asimov_with_predict_fn(self):
        """Test that predict_fn generates Asimov at mu_asimov.

        mu_asimov=0 by default.
        Asimov at mu=0: n_asimov = 5
        Testing at mu=1: q_asimov = 2*(15-5-5*ln(3)) = 9.0139
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_q_asimov_at_different_mu_test(self):
        """Test that Asimov is always at mu_asimov, regardless of mu_test.

        At mu_test=0: Asimov at mu=0 (n=5), testing at 0 → q_asimov=0
        At mu_test=2: Asimov at mu=0 (n=5), testing at 2 → q_asimov = 23.9056
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            predict_fn=predict_fn,
        )

        # Test at mu=0
        result_0 = calc.test(poi_test=0.0)
        assert result_0.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

        # Test at mu=2
        result_2 = calc.test(poi_test=2.0)
        assert result_2.test_stat_result.q_asimov == pytest.approx(23.9056, rel=1e-3)

    def test_pvalues_computed(self):
        """Test that pnull and palt have correct values.

        n_obs=10, mu_test=1: q_obs = 1.8907, q_asimov = 9.0139.
        Standard region (q < q_asimov):
        pnull = 1-Φ(√1.8907) = 1-Φ(1.375) = 0.08456
        palt = 1-Φ(1.375-3.002) = 1-Φ(-1.627) = 0.94816
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert float(result.pnull) == pytest.approx(0.08456, rel=1e-3)
        assert float(result.palt) == pytest.approx(0.94816, rel=1e-3)

    def test_without_predict_fn(self):
        """Without predict_fn, q_asimov is None and p-values are None.

        QTildeAsymptotic requires q_asimov for both null_pval and alt_pval.
        Without predict_fn, the Asimov dataset is not generated.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc.test(poi_test=1.0)

        assert result.test_stat_result.q_asimov is None
        assert result.pnull is None
        assert result.palt is None


# =============================================================================
# HypoTestCalculator Tests (generic base)
# =============================================================================


class TestHypoTestCalculator:
    """Tests for the generic HypoTestCalculator base."""

    def test_default_test_statistic(self):
        """Test that default test statistic is QTilde."""
        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key=("mu",),
        )
        assert isinstance(calc.test_statistic, QTilde)

    @mock.patch.object(
        QTilde,
        "__call__",
        return_value=TSResult(value=jnp.array(0.0), test=jnp.array(1.0)),
    )
    def test_kwargs_passthrough(self, mock_ts):
        """Verify kwargs are forwarded verbatim to the test statistic."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            calc.test(
                poi_test=1.0,
                predict_fn=predict_fn,
                mu_asimov=0.0,
            )

        mock_ts.assert_called_once_with(
            poisson_nll,
            params,
            observed,
            ("mu",),
            1.0,
            predict_fn=predict_fn,
            mu_asimov=0.0,
        )

    def test_result_without_asimov(self):
        """QMu + QMuAsymptotic gives correct pnull without Asimov data.

        n_obs=10, mu_test=1: q_mu = 1.8907.
        QMuAsymptotic.null_pval = 1 - Φ(√1.8907) = 1 - Φ(1.375) = 0.08456.
        alt_pval needs q_asimov → None (no predict_fn provided).
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=params,
            observation=observed,
            poi_key=("mu",),
            test_statistic=QMu(),
            distribution=QMuAsymptotic(),
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc.test(poi_test=1.0)

        assert result.q_obs == pytest.approx(1.8907, rel=1e-3)
        assert float(result.pnull) == pytest.approx(0.08456, rel=1e-3)
        assert result.palt is None


# =============================================================================
# Calculator cls() and expected() Tests
# =============================================================================


class TestCalculatorCls:
    """Tests for HypoTestCalculator.cls()."""

    def test_cls_asymptotic_counting_model(self):
        """calc.cls() returns CLs = pnull/palt for counting model.

        n_obs=10, mu_test=1: pnull=0.08456, palt=0.94816.
        CLs = 0.08456 / 0.94816 = 0.08919.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(10.0),
            poi_key=("mu",),
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)

        assert float(calc.cls(result)) == pytest.approx(0.08919, rel=1e-2)

    def test_cls_empirical_known_arrays(self):
        """calc.cls() with empirical distribution and explicit arrays.

        q_null = [0.5..5], q_alt = [1..10], q_obs=5:
        pnull = 1/10 = 0.1, palt = 6/10 = 0.6.
        CLs = 0.1/0.6 = 0.16667.
        """
        q_null = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        q_alt = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        dist = SimpleEmpiricalDistribution(q_null=q_null, q_alt=q_alt)

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key=("mu",),
            distribution=dist,
        )
        ts = TSResult(value=jnp.array(5.0), test=jnp.array(1.0))
        result = HypoTestResult(
            q_obs=ts.value,
            pnull=dist.null_pval(ts),
            palt=dist.alt_pval(ts),
            test_stat_result=ts,
        )

        assert float(calc.cls(result)) == pytest.approx(0.16667, rel=1e-4)

    def test_cls_none_when_palt_none(self):
        """cls() returns None when palt is None (no q_alt toys)."""
        dist = SimpleEmpiricalDistribution(q_null=jnp.array([1.0, 2.0, 3.0]))

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key=("mu",),
            distribution=dist,
        )
        ts = TSResult(value=jnp.array(1.0), test=jnp.array(1.0))
        with pytest.warns(UserWarning, match="cannot be performed without q_alt"):
            palt = dist.alt_pval(ts)
        result = HypoTestResult(
            q_obs=ts.value,
            pnull=dist.null_pval(ts),
            palt=palt,
            test_stat_result=ts,
        )

        assert calc.cls(result) is None


class TestCalculatorExpected:
    """Tests for HypoTestCalculator.expected()."""

    def test_expected_asymptotic_median_cls(self):
        """calc.expected() median CLs for counting model.

        n_obs=15, mu_test=1.0, Asimov at mu=0 (n=5).
        q_asimov = 9.0139, σ = μ/√q_A = 1/3.002 = 0.3331.
        Median expected q = max(0, μ/σ-0)² = (3.002)² = 9.012.
        pnull = 1-Φ(3.002) = 0.00134, palt = 1-Φ(0) = 0.5.
        CLs_median = 0.00134/0.5 = 0.00269.
        """
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key=("mu",),
            predict_fn=predict_fn,
        )
        result = calc.test(poi_test=1.0)
        bands = calc.expected(result)

        assert float(bands.cl_s.median) == pytest.approx(0.00269, rel=0.05)

    @pytest.mark.parametrize(
        ("band_name", "expected_cls"),
        [
            # q_alt quantile at Φ(-2)=0.0228 → q≈0.455.
            # pnull=(10-0.455)/10=0.9545, palt=(20-0.455)/20=0.9773.
            # CLs=0.9545/0.9773=0.9767
            ("minus_2sigma", 0.9767),
            # q_alt quantile at Φ(-1)=0.1587 → q≈3.173.
            # pnull=(10-3.173)/10=0.6827, palt=(20-3.173)/20=0.8413.
            # CLs=0.6827/0.8413=0.8114
            ("minus_1sigma", 0.8114),
            # q_alt quantile at Φ(0)=0.5 → q=10.0.
            # pnull=(10-10)/10=0.0, palt=(20-10)/20=0.5.
            # CLs=0.0/0.5=0.0
            ("median", 0.0),
            # q_alt quantile at Φ(1)=0.8413 → q≈16.827.
            # pnull=(10-16.827)/10=0.0 (clamped), palt=(20-16.827)/20=0.1587.
            # CLs=0.0/0.1587=0.0
            ("plus_1sigma", 0.0),
            # q_alt quantile at Φ(2)=0.9772 → q≈19.545.
            # pnull=(10-19.545)/10=0.0 (clamped), palt=(20-19.545)/20=0.0228.
            # CLs=0.0/0.0228=0.0
            ("plus_2sigma", 0.0),
        ],
    )
    def test_expected_empirical_cls_per_band(self, band_name, expected_cls):
        """calc.expected() with empirical uniform distributions.

        q_null = linspace(0, 10, 10001), q_alt = linspace(0, 20, 10001).
        CLs at each band from quantiles of q_alt.
        """
        q_null = jnp.linspace(0.0, 10.0, 10001)
        q_alt = jnp.linspace(0.0, 20.0, 10001)
        dist = SimpleEmpiricalDistribution(q_null=q_null, q_alt=q_alt)

        calc = HypoTestCalculator(
            nll_fn=poisson_nll,
            params=create_params(),
            observation=create_observation(1.0),
            poi_key=("mu",),
            distribution=dist,
        )
        ts = TSResult(value=jnp.array(5.0), test=jnp.array(1.0))
        result = HypoTestResult(
            q_obs=ts.value,
            pnull=dist.null_pval(ts),
            palt=dist.alt_pval(ts),
            test_stat_result=ts,
        )
        bands = calc.expected(result)

        assert float(bands.cl_s[band_name]) == pytest.approx(expected_cls, abs=0.02)

    def test_expected_raises_when_distribution_lacks_implementation(self):
        """TMuAsymptotic does not implement expected_pvalues → NotImplementedError."""
        calc = AsymptoticCalculator(
            nll_fn=poisson_nll,
            params=create_params(mu_init=1.0),
            observation=create_observation(15.0),
            poi_key=("mu",),
            predict_fn=predict_fn,
            distribution=TMuAsymptotic(),
        )
        result = calc.test(poi_test=1.0)

        with pytest.raises(NotImplementedError):
            calc.expected(result)
