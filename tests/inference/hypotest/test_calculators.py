"""Tests for hypothesis test calculators.

Tests the calculator orchestration with concrete expected values.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    AsymptoticCalculator,
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
# AsymptoticCalculator Tests
# =============================================================================


class TestAsymptoticCalculator:
    """Tests for AsymptoticCalculator."""

    def test_basic_result_structure(self):
        """Test that calculator returns proper HypoTestResult."""
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

        assert hasattr(result, "q_obs")
        assert hasattr(result, "pnull")
        assert hasattr(result, "palt")
        assert hasattr(result, "cl_s")
        assert hasattr(result, "expected_bands")
        assert hasattr(result, "test_stat_result")

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
        Testing at mu=1: q_asimov = 2*(15-5-5*ln(3)) ≈ 9.014
        """
        expected_q_asimov = expected_q(5.0, 1.0)  # ~9.014

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

        assert result.test_stat_result.q_asimov == pytest.approx(
            expected_q_asimov, rel=1e-3
        )

    def test_q_asimov_at_different_mu_test(self):
        """Test that Asimov is always at mu_asimov, regardless of mu_test.

        At mu_test=0: Asimov at mu=0 (n=5), testing at 0 → q_asimov=0
        At mu_test=2: Asimov at mu=0 (n=5), testing at 2 → q_asimov≈23.9
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
        expected_q_asimov_2 = expected_q(5.0, 2.0)
        result_2 = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=2.0,
            predict_fn=predict_fn,
        )
        assert result_2.test_stat_result.q_asimov == pytest.approx(
            expected_q_asimov_2, rel=1e-3
        )

    def test_pvalues_computed(self):
        """Test that pnull and palt are finite after distribution call."""
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

        assert jnp.isfinite(result.pnull)
        assert jnp.isfinite(result.palt)

    def test_expected_bands_none(self):
        """Test that expected bands are None (not yet implemented)."""
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

        assert result.expected_bands is None

    def test_custom_test_statistic(self):
        """Test calculator with QMu instead of QTilde."""
        params = create_params(mu_init=1.0)
        observed = create_observation(25.0)

        calc = AsymptoticCalculator(
            test_statistic=QMu(),
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

        # QMu doesn't have boundary, so q > 0 for upward fluctuation
        assert float(result.q_obs) > 0.0


# =============================================================================
# HypoTestCalculator Tests (generic base)
# =============================================================================


class TestHypoTestCalculator:
    """Tests for the generic HypoTestCalculator base."""

    def test_default_test_statistic(self):
        """Test that default test statistic is QTilde."""
        calc = HypoTestCalculator()
        assert isinstance(calc.test_statistic, QTilde)

    def test_kwargs_passthrough(self):
        """Test that kwargs are forwarded to test statistic."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(
            test_statistic=QTilde(), distribution=QTildeAsymptotic()
        )
        # predict_fn passed as kwarg, forwarded to CowanTestStatistic
        result = calc(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.test_stat_result.q_asimov is not None

    def test_without_predict_fn(self):
        """Test that calculator works without predict_fn (no Asimov).

        Without Asimov, q_asimov is None and piecewise distributions
        (QTildeAsymptotic) return None for p-values that need it.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        calc = HypoTestCalculator(
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
# ExpectedBands Tests
# =============================================================================


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
