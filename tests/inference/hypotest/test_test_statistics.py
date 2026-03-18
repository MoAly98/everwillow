"""Tests for hypothesis test statistics."""

from __future__ import annotations

import jax
import pytest

from everwillow.inference.hypotest import Q0, QMu, QTilde, TMu
from everwillow.inference.hypotest.test_statistics import TestStatistic

from ._counting_model import (
    create_observation,
    create_params,
    poisson_nll,
    predict_fn,
)

# =============================================================================
# TestStatistic base class Tests
# =============================================================================


class TestTestStatisticBase:
    """Tests for TestStatistic.__call__ (base class, not CowanTestStatistic)."""

    def test_call_returns_correct_fields(self):
        """Base __call__ sets value, test=poi_test, q_asimov=None, extras."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        # Call the base TestStatistic.__call__ directly, bypassing
        # CowanTestStatistic's override (which adds Asimov handling).
        result = TestStatistic.__call__(
            QTilde(), poisson_nll, params, observed, ("mu",), poi_test=1.0
        )

        assert result.value == pytest.approx(0.0, abs=1e-4)
        assert result.test == pytest.approx(1.0)
        assert result.q_asimov is None
        assert result.extras["mu_hat"] == pytest.approx(1.0, abs=1e-3)


# =============================================================================
# QTilde Tests
# =============================================================================


class TestQTilde:
    """Tests for QTilde test statistic."""

    @pytest.mark.parametrize(
        ("n_obs", "mu_test", "expected_mu_hat", "expected_q"),
        [
            (3.0, 1.0, -0.2, 14.3434),
            (5.0, 1.0, 0.0, 9.0139),
            (10.0, 1.0, 0.5, 1.8907),
            (15.0, 1.0, 1.0, 0.0),
            (25.0, 1.0, 2.0, 0.0),
        ],
        ids=[
            "deep-downward",
            "at-background",
            "moderate-downward",
            "at-mle",
            "upward-boundary",
        ],
    )
    def test_values(self, n_obs, mu_test, expected_mu_hat, expected_q):
        """Test QTilde values across multiple scenarios.

        Boundary: q̃=0 when mu_hat > mu_test.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = QTilde()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, abs=1e-3)
        assert result.value == pytest.approx(expected_q, abs=1e-4)


# =============================================================================
# QMu Tests
# =============================================================================


class TestQMu:
    """Tests for QMu test statistic (no boundary handling)."""

    @pytest.mark.parametrize(
        ("n_obs", "mu_test", "expected_mu_hat", "expected_q"),
        [
            (3.0, 1.0, -0.2, 14.3434),
            (5.0, 1.0, 0.0, 9.0139),
            (10.0, 1.0, 0.5, 1.8907),
            (15.0, 1.0, 1.0, 0.0),
            (25.0, 1.0, 2.0, 5.5413),
        ],
        ids=[
            "deep-downward",
            "at-background",
            "moderate-downward",
            "at-mle",
            "upward-no-boundary",
        ],
    )
    def test_values(self, n_obs, mu_test, expected_mu_hat, expected_q):
        """Test QMu values across multiple scenarios.

        No boundary — always returns the raw profile likelihood ratio.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = QMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, abs=1e-3)
        assert result.value == pytest.approx(expected_q, abs=1e-4)


# =============================================================================
# TMu Tests
# =============================================================================


class TestTMu:
    """Tests for TMu signed test statistic."""

    @pytest.mark.parametrize(
        ("n_obs", "mu_test", "expected_mu_hat", "expected_t"),
        [
            (3.0, 1.0, -0.2, -14.3434),
            (5.0, 1.0, 0.0, -9.0139),
            (10.0, 1.0, 0.5, -1.8907),
            (15.0, 1.0, 1.0, 0.0),
            (25.0, 1.0, 2.0, 5.5413),
        ],
        ids=[
            "deep-downward-negative",
            "at-background-negative",
            "moderate-downward-negative",
            "at-mle-zero",
            "upward-positive",
        ],
    )
    def test_values(self, n_obs, mu_test, expected_mu_hat, expected_t):
        """Test TMu values across multiple scenarios.

        Signed: t = sign(mu_hat - mu_test) * q_mu.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = TMu()(poisson_nll, params, observed, ("mu",), poi_test=mu_test)

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, abs=1e-3)
        assert result.value == pytest.approx(expected_t, abs=1e-4)


# =============================================================================
# Q0 Tests
# =============================================================================


class TestQ0:
    """Tests for Q0 discovery test statistic."""

    @pytest.mark.parametrize(
        ("n_obs", "expected_mu_hat", "expected_q"),
        [
            (3.0, -0.2, 0.0),
            (5.0, 0.0, 0.0),
            (10.0, 0.5, 3.8629),
            (15.0, 1.0, 12.9584),
            (25.0, 2.0, 40.4719),
        ],
        ids=[
            "negative-muhat-boundary",
            "at-null-mle",
            "moderate-excess",
            "signal-like",
            "strong-excess",
        ],
    )
    def test_values(self, n_obs, expected_mu_hat, expected_q):
        """Test Q0 values across multiple scenarios.

        Boundary: q0=0 when mu_hat < 0 (no "discovery" of negative signal).
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = Q0()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, abs=1e-3)
        assert result.value == pytest.approx(expected_q, abs=1e-4)

    def test_ignores_poi_test(self):
        """Q0 ignores poi_test and always tests at mu=0."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result0 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=0.0)
        result1 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=1.0)
        result2 = Q0()(poisson_nll, params, observed, ("mu",), poi_test=2.0)

        assert result0.value == pytest.approx(result1.value, abs=1e-6)
        assert result1.value == pytest.approx(result2.value, abs=1e-6)
        assert result0.test == pytest.approx(0.0)
        assert result1.test == pytest.approx(0.0)
        assert result2.test == pytest.approx(0.0)


# =============================================================================
# Cross-cutting CowanTestStatistic Tests
# =============================================================================


class TestCowanTestStatisticGeneral:
    """Shared CowanTestStatistic behavior across all four stats."""

    @pytest.mark.parametrize(
        ("TestStatClass", "expected_test"),
        [
            (QTilde, 1.0),
            (QMu, 1.0),
            (TMu, 1.0),
            (Q0, 0.0),  # Q0 always tests at mu=0
        ],
    )
    def test_result_test_field(self, TestStatClass, expected_test):
        """result.test stores the tested POI value."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = TestStatClass()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.test == pytest.approx(expected_test)

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, Q0, TMu])
    def test_q_asimov_none_without_asimov(self, TestStatClass):
        """Without predict_fn or asimov_observation, q_asimov is None."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        result = TestStatClass()(poisson_nll, params, observed, ("mu",), poi_test=1.0)

        assert result.q_asimov is None

    @pytest.mark.parametrize(
        ("TestStatClass", "expected_q"),
        [
            (QTilde, 0.0),
            (QMu, 0.0),
            (Q0, 12.9584),
            (TMu, 0.0),
        ],
    )
    def test_jit_compatibility(self, TestStatClass, expected_q):
        """Test that all test statistics are JIT-compatible."""
        params = create_params(mu_init=1.0)
        test_stat = TestStatClass()

        @jax.jit
        def compute_q(obs_n):
            obs = {"n": obs_n}
            return test_stat(poisson_nll, params, obs, ("mu",), poi_test=1.0).value

        q = compute_q(15.0)
        assert q == pytest.approx(expected_q, abs=1e-4)

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, TMu])
    def test_asimov_observation(self, TestStatClass):
        """Test q_asimov=0 when Asimov matches mu_test."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)
        asimov = create_observation(15.0)  # Expected for mu=1

        result = TestStatClass()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            asimov_observation=asimov,
        )

        assert result.q_asimov == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize("TestStatClass", [QTilde, QMu, TMu])
    def test_predict_fn_explicit_mu_asimov(self, TestStatClass):
        """Test q_asimov=0 with predict_fn at mu_asimov=1.0.

        Asimov at mu=1 gives n=15. Testing at mu=1: q_asimov=0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        result = TestStatClass()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
            mu_asimov=1.0,
        )

        assert result.q_asimov == pytest.approx(0.0, abs=1e-4)

    @pytest.mark.parametrize(
        ("TestStatClass", "expected_q_asimov"),
        [
            # Asimov at mu=0 → n=5. Testing at mu=1:
            # q_raw = 2*(15 - 5 - 5*ln(3)) = 9.0139
            # QTilde/QMu: q_asimov = 9.0139 (mu_hat=0 < mu_test=1)
            # TMu: t_asimov = sign(0-1)*9.0139 = -9.0139
            (QTilde, 9.0139),
            (QMu, 9.0139),
            (TMu, -9.0139),
        ],
    )
    def test_predict_fn_default_mu_asimov(self, TestStatClass, expected_q_asimov):
        """Test q_asimov with predict_fn at default mu_asimov=0."""
        params = create_params(mu_init=1.0)
        observed = create_observation(10.0)

        result = TestStatClass()(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            predict_fn=predict_fn,
        )

        assert result.q_asimov == pytest.approx(expected_q_asimov, rel=1e-3)
