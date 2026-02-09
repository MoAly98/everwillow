"""Tests for HypoTestCalculator.

Tests the calculator orchestration with concrete expected values.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    ExpectedBands,
    HypoTestCalculator,
    QMu,
    QTilde,
    QTildeAsymptotic,
)

# =============================================================================
# Test fixtures: Simple Poisson counting experiment
# =============================================================================

S = 10.0  # signal yield
B = 5.0  # background yield


def poisson_nll(params, observation):
    """Poisson NLL for a simple counting experiment."""
    mu = params["mu"]
    n_expected = mu * S + B
    n_observed = observation["n"]
    return n_expected - n_observed * jnp.log(n_expected)


def create_params(mu_init: float = 1.0) -> sl.State:
    """Create initial parameter state."""
    return sl.State.from_pytree({"mu": mu_init})


def create_observation(n: float) -> dict[str, float]:
    """Create observation dict."""
    return {"n": n}


def predict_fn(params_state: sl.State) -> dict[str, float]:
    """Prediction function for Asimov data."""
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * S + B}


def expected_q(n_obs: float, mu_test: float) -> float:
    """Compute analytical q value."""
    n_exp_test = mu_test * S + B
    return 2.0 * (n_exp_test - n_obs - n_obs * math.log(n_exp_test / n_obs))


def normal_sf(x: float) -> float:
    """Standard normal survival function: 1 - CDF."""
    return 0.5 * (1 - math.erf(x / math.sqrt(2)))


# =============================================================================
# HypoTestCalculator Tests
# =============================================================================


class TestHypoTestCalculator:
    """Tests for HypoTestCalculator."""

    def test_basic_result_structure(self):
        """Test that calculator returns proper HypoTestResult."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
        )

        assert hasattr(result, "q_obs")
        assert hasattr(result, "pnull")
        assert hasattr(result, "palt")
        assert hasattr(result, "cl_s")
        assert hasattr(result, "expected_bands")
        assert hasattr(result, "test_stat_result")

    def test_cls_at_mle(self):
        """At MLE, q=0 and CLs should be 1.0.

        n_obs=15 for mu=1: q=0, palt=0.5, pnull=0.5, CLs=1.0
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
        )

        assert result.q_obs == pytest.approx(0.0, abs=1e-5)
        assert result.cl_s == pytest.approx(1.0, rel=1e-3)

    def test_cls_with_sensitivity(self):
        """Test CLs < 1 when testing above observed with sensitivity.

        n_obs=10, mu_test=2.0, asimov at null (mu=0):
        - mu_hat = 0.5 < 2.0
        - Asimov at null gives n=5, so q_asimov > 0 when testing mu=2
        - With positive q_asimov, pnull > palt, so CLs < 1
        """
        n_obs = 10.0
        mu_test = 2.0
        expected_q_value = expected_q(n_obs, mu_test)

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)
        # Asimov at null hypothesis (background only)
        asimov_null = create_observation(5.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=mu_test,
            distribution=QTildeAsymptotic(),
            asimov_observation=asimov_null,
        )

        assert result.q_obs == pytest.approx(expected_q_value, rel=1e-3)
        # With sensitivity (q_asimov > 0), CLs < 1
        assert result.test_stat_result.extras["q_asimov"] > 0
        assert float(result.cl_s) < 1.0

    def test_cls_with_asimov(self):
        """Test CLs with explicit Asimov observation."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
            asimov_observation=asimov,
        )

        # q_asimov should be 0
        assert result.test_stat_result.extras["q_asimov"] == pytest.approx(
            0.0, abs=1e-4
        )

    def test_predict_fn_generates_asimov(self):
        """Test that predict_fn generates correct Asimov data.

        When predict_fn is used, Asimov is generated at mu_test.
        For mu_test=1.0: n_asimov = 1*10 + 5 = 15
        Testing at mu=1 on Asimov(mu=1) gives q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
            predict_fn=predict_fn,
        )

        # Asimov at mu_test=1 gives MLE=1, so q_asimov=0
        assert result.test_stat_result.extras["q_asimov"] == pytest.approx(
            0.0, abs=1e-4
        )
        # And CLs=1 because palt=pnull when q_asimov=0
        assert result.cl_s == pytest.approx(1.0, rel=1e-3)

    def test_predict_fn_different_mu_test(self):
        """Test predict_fn at different mu_test values.

        At mu_test=0: n_asimov = 0*10 + 5 = 5
        Testing at mu=0 on Asimov(mu=0) gives q_asimov=0.

        At mu_test=2: n_asimov = 2*10 + 5 = 25
        Testing at mu=2 on Asimov(mu=2) gives q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())

        # Test at mu=0
        result_0 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=0.0,
            distribution=QTildeAsymptotic(),
            predict_fn=predict_fn,
        )
        assert result_0.test_stat_result.extras["q_asimov"] == pytest.approx(
            0.0, abs=1e-4
        )

        # Test at mu=2
        result_2 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=2.0,
            distribution=QTildeAsymptotic(),
            predict_fn=predict_fn,
        )
        assert result_2.test_stat_result.extras["q_asimov"] == pytest.approx(
            0.0, abs=1e-4
        )

    def test_expected_bands_present(self):
        """Test that expected bands are computed."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
            predict_fn=predict_fn,
        )

        bands = result.expected_bands
        assert hasattr(bands, "minus_2sigma")
        assert hasattr(bands, "minus_1sigma")
        assert hasattr(bands, "median")
        assert hasattr(bands, "plus_1sigma")
        assert hasattr(bands, "plus_2sigma")

    def test_cls_bands(self):
        """Test that expected CLs bands can be extracted."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
            predict_fn=predict_fn,
        )

        # At MLE (q_asimov=0), all CLs bands should be 1.0
        cls_bands = result.expected_bands.cls_bands()
        assert len(cls_bands) == 5
        for cls_val in cls_bands:
            assert float(cls_val) == pytest.approx(1.0, rel=1e-3)

    def test_custom_test_statistic(self):
        """Test calculator with QMu instead of QTilde."""
        params = create_params(mu_init=1.0)
        observed = create_observation(25.0)

        calc = HypoTestCalculator(test_statistic=QMu())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            distribution=QTildeAsymptotic(),
        )

        # QMu doesn't have boundary, so q > 0 for upward fluctuation
        assert float(result.q_obs) > 0.0

    def test_default_test_statistic(self):
        """Test that default test statistic is QTilde."""
        calc = HypoTestCalculator()
        assert isinstance(calc.test_statistic, QTilde)


class TestHypoTestCalculatorCLsValues:
    """Tests for specific CLs values with known inputs."""

    def test_strong_exclusion(self):
        """Test strong exclusion case (low CLs).

        Large downward fluctuation with non-zero q_asimov gives low CLs.
        n_obs=6, mu_test=2.0, asimov at background (mu=0) gives n=5:
        - q_obs is large (data far below expectation)
        - q_asimov computed at null gives sensitivity
        - CLs should be small
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(6.0)
        # Asimov at null hypothesis (background only, mu=0)
        asimov_null = create_observation(5.0)

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=2.0,
            distribution=QTildeAsymptotic(),
            asimov_observation=asimov_null,
        )

        # q_asimov > 0 when testing mu=2 on null Asimov
        assert result.test_stat_result.extras["q_asimov"] > 0
        # With sensitivity, CLs < 1
        assert float(result.cl_s) < 1.0

    def test_no_sensitivity(self):
        """Test no sensitivity case (CLs ≈ 1).

        Testing at mu=0 (no signal hypothesis).
        """
        params = create_params(mu_init=0.0)
        observed = create_observation(5.0)  # background only

        calc = HypoTestCalculator(test_statistic=QTilde())
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=0.0,
            distribution=QTildeAsymptotic(),
        )

        # At mu=0 with background-only observation, CLs should be near 1
        assert float(result.cl_s) == pytest.approx(1.0, rel=0.1)


class TestExpectedBandsClsBands:
    """Tests for ExpectedBands.cls_bands() with known values."""

    def test_cls_bands_with_known_pvalues(self):
        """Test cls_bands computes CLs = palt / pnull correctly.

        Each band tuple is (pnull, palt). CLs = palt / pnull.
        """
        bands = ExpectedBands(
            minus_2sigma=(jnp.array(0.5), jnp.array(0.1)),  # CLs = 0.2
            minus_1sigma=(jnp.array(0.4), jnp.array(0.2)),  # CLs = 0.5
            median=(jnp.array(0.3), jnp.array(0.15)),  # CLs = 0.5
            plus_1sigma=(jnp.array(0.2), jnp.array(0.1)),  # CLs = 0.5
            plus_2sigma=(jnp.array(0.1), jnp.array(0.05)),  # CLs = 0.5
        )

        cls_values = bands.cls_bands()

        assert float(cls_values[0]) == pytest.approx(0.2, rel=1e-6)
        assert float(cls_values[1]) == pytest.approx(0.5, rel=1e-6)
        assert float(cls_values[2]) == pytest.approx(0.5, rel=1e-6)
        assert float(cls_values[3]) == pytest.approx(0.5, rel=1e-6)
        assert float(cls_values[4]) == pytest.approx(0.5, rel=1e-6)
