"""Tests for hypothesis test statistics.

Tests each test statistic (QTilde, QMu, Q0, TMu) with concrete expected values.
Uses a simple Poisson counting experiment where we can compute expected q values.

Model: n_expected = mu * s + b, with s=10, b=5
Poisson NLL: nll = n_exp - n_obs * log(n_exp)

Analytical solutions:
- MLE: mu_hat = (n_obs - b) / s
- q = 2 * [n_exp_test - n_obs - n_obs * log(n_exp_test / n_obs)]
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.inference.hypotest import Q0, QMu, QTilde, TMu

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


def expected_mu_hat(n_obs: float) -> float:
    """Compute analytical MLE for mu."""
    return (n_obs - B) / S


def expected_q(n_obs: float, mu_test: float) -> float:
    """Compute analytical q value.

    q = 2 * [n_exp_test - n_obs - n_obs * log(n_exp_test / n_obs)]
    """
    n_exp_test = mu_test * S + B
    return 2.0 * (n_exp_test - n_obs - n_obs * math.log(n_exp_test / n_obs))


# =============================================================================
# QTilde Tests
# =============================================================================


class TestQTilde:
    """Tests for QTilde test statistic."""

    def test_at_mle(self):
        """Test q̃=0 when observation matches mu_test exactly.

        n_obs=15 is exactly expected for mu=1, so mu_hat=1=mu_test, q=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = QTilde()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.value == pytest.approx(0.0, abs=1e-5)
        assert result.extras["mu_hat"] == pytest.approx(1.0, rel=1e-3)

    def test_downward_fluctuation(self):
        """Test q̃ > 0 when mu_hat < mu_test.

        n_obs=10: mu_hat = (10-5)/10 = 0.5 < mu_test=1.0
        q = 2 * [15 - 10 - 10*log(15/10)] = 2 * [5 - 4.0546] = 1.8907
        """
        n_obs = 10.0
        mu_test = 1.0
        expected_q_value = expected_q(n_obs, mu_test)  # 1.8907...

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = QTilde()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(0.5, rel=1e-3)
        assert result.value == pytest.approx(expected_q_value, rel=1e-3)

    def test_upward_fluctuation_boundary(self):
        """Test q̃=0 boundary when mu_hat > mu_test.

        n_obs=25: mu_hat = (25-5)/10 = 2.0 > mu_test=1.0
        QTilde sets q=0 for upward fluctuations.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(25.0)

        result = QTilde()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.extras["mu_hat"] == pytest.approx(2.0, rel=1e-3)
        assert result.value == pytest.approx(0.0, abs=1e-5)

    def test_with_asimov_observation(self):
        """Test q_asimov=0 when Asimov matches mu_test."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)  # Expected for mu=1

        result = QTilde()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            asimov_observation=asimov,
        )

        assert result.q_asimov == pytest.approx(0.0, abs=1e-4)

    def test_with_predict_fn(self):
        """Test q_asimov with predict_fn at explicit mu_asimov=1.

        Asimov at mu=1 gives n=15. Testing at mu=1: q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        result = QTilde()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
            mu_asimov=1.0,
        )

        assert result.q_asimov == pytest.approx(0.0, abs=1e-4)

    def test_with_predict_fn_default_mu_asimov(self):
        """Test q_asimov with predict_fn at default mu_asimov=0.

        Asimov at mu=0 gives n=5. Testing at mu=1:
        q_asimov = 2*(15 - 5 - 5*ln(3)) ≈ 9.014
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        expected_q_asimov = expected_q(5.0, 1.0)  # ~9.014

        result = QTilde()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.q_asimov == pytest.approx(expected_q_asimov, rel=1e-3)


# =============================================================================
# QMu Tests
# =============================================================================


class TestQMu:
    """Tests for QMu test statistic (no boundary handling)."""

    def test_at_mle(self):
        """Test q_mu=0 when observation matches mu_test exactly."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = QMu()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.value == pytest.approx(0.0, abs=1e-5)

    def test_downward_fluctuation(self):
        """Test q_mu value for downward fluctuation.

        Same as QTilde when mu_hat < mu_test.
        """
        n_obs = 10.0
        mu_test = 1.0
        expected_q_value = expected_q(n_obs, mu_test)

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = QMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.value == pytest.approx(expected_q_value, rel=1e-3)

    def test_upward_fluctuation_no_boundary(self):
        """Test q_mu > 0 for upward fluctuation (no boundary unlike QTilde).

        n_obs=25: mu_hat = 2.0 > mu_test=1.0
        q = 2 * [15 - 25 - 25*log(15/25)] = 2 * [-10 + 12.78] = 5.56
        """
        n_obs = 25.0
        mu_test = 1.0
        expected_q_value = expected_q(n_obs, mu_test)  # 5.56...

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = QMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(2.0, rel=1e-3)
        assert result.value == pytest.approx(expected_q_value, rel=1e-3)

    def test_comparison_with_qtilde_upward(self):
        """QMu > 0 while QTilde = 0 for upward fluctuation."""
        params = create_params(mu_init=1.0)
        observed = create_observation(25.0)

        q_mu = QMu()(poisson_nll, params, observed, ("mu",), poi_test=1.0).value
        q_tilde = QTilde()(poisson_nll, params, observed, ("mu",), poi_test=1.0).value

        assert float(q_mu) > 5.0  # Should be ~5.56
        assert q_tilde == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# Q0 Tests
# =============================================================================


class TestQ0:
    """Tests for Q0 discovery test statistic."""

    def test_always_tests_at_zero(self):
        """Q0 ignores poi_test and always tests at mu=0."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        # Different poi_test values should give identical results
        result1 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=0.0)
        result2 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=1.0)
        result3 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=2.0)

        assert result1.value == pytest.approx(float(result2.value), rel=1e-5)
        assert result1.value == pytest.approx(float(result3.value), rel=1e-5)

    def test_discovery_significance(self):
        """Test Q0 value for signal-like observation.

        n_obs=15: mu_hat = 1.0 >= 0
        Testing at mu=0: n_exp_test = 5
        q0 = 2 * [5 - 15 - 15*log(5/15)] = 2 * [-10 + 16.48] = 12.96
        """
        n_obs = 15.0
        expected_q0 = expected_q(n_obs, mu_test=0.0)  # 12.96...

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = Q0()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.extras["mu_hat"] == pytest.approx(1.0, rel=1e-3)
        assert result.value == pytest.approx(expected_q0, rel=1e-3)

    def test_background_only_observation(self):
        """Test Q0=0 when observation matches background.

        n_obs=5 = b: mu_hat = 0, testing at mu=0 gives q0=0.
        """
        params = create_params(mu_init=0.0)
        observed = create_observation(5.0)

        result = Q0()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.extras["mu_hat"] == pytest.approx(0.0, abs=1e-3)
        assert result.value == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# TMu Tests
# =============================================================================


class TestTMu:
    """Tests for TMu signed test statistic."""

    def test_positive_sign_upward(self):
        """Test t_mu > 0 when mu_hat > mu_test.

        n_obs=25: mu_hat = 2.0 > mu_test = 1.0
        t_mu = +1 * q_mu = q_mu > 0
        """
        n_obs = 25.0
        mu_test = 1.0
        expected_t = expected_q(n_obs, mu_test)  # positive, ~5.56

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = TMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(2.0, rel=1e-3)
        assert result.value == pytest.approx(expected_t, rel=1e-3)

    def test_negative_sign_downward(self):
        """Test t_mu < 0 when mu_hat < mu_test.

        n_obs=10: mu_hat = 0.5 < mu_test = 1.0
        t_mu = -1 * q_mu < 0
        """
        n_obs = 10.0
        mu_test = 1.0
        expected_t = -expected_q(n_obs, mu_test)  # negative, ~-1.89

        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = TMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(0.5, rel=1e-3)
        assert result.value == pytest.approx(expected_t, rel=1e-3)

    def test_zero_at_mle(self):
        """Test t_mu=0 when mu_hat = mu_test."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = TMu()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.value == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# General Tests
# =============================================================================


class TestTestStatisticGeneral:
    """General tests that apply to all test statistics."""

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, Q0, TMu])
    def test_result_structure(self, TestStatClass):
        """Test that all test statistics return proper TestStatResult."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = TestStatClass()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert hasattr(result, "value")
        assert hasattr(result, "test")
        assert hasattr(result, "q_asimov")
        assert hasattr(result, "extras")
        assert "fit_free" in result.extras
        assert "fit_constrained" in result.extras
        assert "mu_hat" in result.extras

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, Q0, TMu])
    def test_jit_compatibility(self, TestStatClass):
        """Test that all test statistics are JIT-compatible."""
        params = create_params(mu_init=1.0)
        test_stat = TestStatClass()

        @jax.jit
        def compute_q(obs_n):
            obs = {"n": obs_n}
            return test_stat(poisson_nll, params, obs, ("mu",), poi_test=1.0).value

        q = compute_q(15.0)
        assert jnp.isfinite(q)

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, Q0, TMu])
    def test_q_asimov_none_without_asimov(self, TestStatClass):
        """Without predict_fn or asimov_observation, q_asimov is None."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = TestStatClass()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.q_asimov is None


# =============================================================================
# Tests with nuisance parameters
# =============================================================================


class TestWithNuisanceParameters:
    """Tests with models that have nuisance parameters (covers constrained_fit branch)."""

    def test_qtilde_with_nuisance(self):
        """Test QTilde with a nuisance parameter (background uncertainty).

        Model: n_expected = mu * s + b * theta
        where theta is a nuisance parameter for background normalization.

        n_obs = 12, s = 10, b = 5
        At mu_test = 1.0: n_expected = 10 + 5*theta
        Free fit finds mu_hat and theta_hat that minimize NLL.
        Constrained fit fixes mu=1 but optimizes theta.
        """

        def nll_with_nuisance(params, observation):
            mu = params["mu"]
            theta = params["theta"]
            n_expected = mu * S + B * theta
            n_observed = observation["n"]
            # Poisson + Gaussian constraint on theta
            poisson_term = n_expected - n_observed * jnp.log(n_expected)
            constraint = 0.5 * (theta - 1.0) ** 2 / 0.1**2  # 10% uncertainty
            return poisson_term + constraint

        params = sl.State.from_pytree({"mu": 1.0, "theta": 1.0})
        observed = create_observation(12.0)

        result = QTilde()(nll_with_nuisance, params, observed, ("mu",), poi_test=1.0)

        # With nuisance, the fit should optimize theta
        # mu_hat should be close to (12 - 5) / 10 = 0.7
        assert result.extras["mu_hat"] == pytest.approx(0.7, rel=0.1)
        # q should be small since we're testing near the best fit
        assert float(result.value) >= 0.0

    def test_qmu_with_nuisance(self):
        """Test QMu with nuisance parameter."""

        def nll_with_nuisance(params, observation):
            mu = params["mu"]
            theta = params["theta"]
            n_expected = mu * S + B * theta
            n_observed = observation["n"]
            poisson_term = n_expected - n_observed * jnp.log(n_expected)
            constraint = 0.5 * (theta - 1.0) ** 2 / 0.1**2
            return poisson_term + constraint

        params = sl.State.from_pytree({"mu": 1.0, "theta": 1.0})
        observed = create_observation(20.0)  # Higher count

        result = QMu()(nll_with_nuisance, params, observed, ("mu",), poi_test=1.0)

        # With n_obs=20, mu_hat > 1 (upward fluctuation)
        assert result.extras["mu_hat"] > 1.0
        # QMu doesn't have boundary, so q > 0
        assert float(result.value) > 0.0
