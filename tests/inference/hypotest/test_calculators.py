"""Tests for hypothesis test calculators."""

from __future__ import annotations

from unittest import mock

import jax.numpy as jnp
import pytest

from everwillow.inference.hypotest import (
    AsymptoticCalculator,
    HypoTestCalculator,
    QMu,
    QMuAsymptotic,
    QTilde,
    QTildeAsymptotic,
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
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)
        assert float(result.pnull) == pytest.approx(0.5, rel=1e-3)
        assert float(result.palt) == pytest.approx(0.99866, rel=1e-3)
        assert result.expected_bands is not None
        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_q_obs_at_mle(self):
        """At MLE, q=0.

        n_obs=15 for mu=1: mu_hat=1=mu_test, q=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)

    def test_q_asimov_with_asimov_observation(self):
        """Test q_asimov with explicit Asimov observation.

        Asimov at mu=1 (n=15), testing at mu=1: q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)

        calc = HypoTestCalculator(
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            asimov_observation=asimov,
        )

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
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.test_stat_result.q_asimov == pytest.approx(9.0139, rel=1e-3)

    def test_q_asimov_at_different_mu_test(self):
        """Test that Asimov is always at mu_asimov, regardless of mu_test.

        At mu_test=0: Asimov at mu=0 (n=5), testing at 0 → q_asimov=0
        At mu_test=2: Asimov at mu=0 (n=5), testing at 2 → q_asimov = 23.9056
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = AsymptoticCalculator(
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )

        # Test at mu=0
        result_0 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=0.0,
            predict_fn=predict_fn,
        )
        assert result_0.test_stat_result.q_asimov == pytest.approx(0.0, abs=1e-4)

        # Test at mu=2
        result_2 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=2.0,
            predict_fn=predict_fn,
        )
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
            test_statistic=QTilde(),
            distribution=QTildeAsymptotic(),
        )
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

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
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc(
                poisson_nll,
                params,
                observed,
                ("mu",),
                poi_test=1.0,
            )

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
        calc = HypoTestCalculator()
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

        calc = HypoTestCalculator()
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            calc(
                poisson_nll,
                params,
                observed,
                ("mu",),
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

        calc = HypoTestCalculator(test_statistic=QMu(), distribution=QMuAsymptotic())
        with pytest.warns(UserWarning, match="cannot be performed without an Asimov"):
            result = calc(
                poisson_nll,
                params,
                observed,
                ("mu",),
                poi_test=1.0,
            )

        assert result.q_obs == pytest.approx(1.8907, rel=1e-3)
        assert float(result.pnull) == pytest.approx(0.08456, rel=1e-3)
        assert result.palt is None
