"""Tests for hypothesis testing utilities."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.hypotest.test_statistics import QMu, QTilde
from everwillow.hypotest.utils import (
    cl_s,
    make_asimov,
    sigma_from_asimov,
    significance,
)

from ._counting_model import (
    create_observation,
    create_params,
    nll_with_nuisance,
    poisson_nll,
    predict_fn,
)


class TestClS:
    """Tests for cl_s utility: CLs = pnull / max(palt, 1e-10)."""

    @pytest.mark.parametrize(
        ("pnull", "palt", "expected"),
        [
            (0.025, 0.5, 0.05),
            (0.1, 0.5, 0.2),
            (0.001, 0.002, 0.5),
        ],
        ids=["small-pnull", "moderate-pnull", "similar-values"],
    )
    def test_values(self, pnull, palt, expected):
        """cl_s returns pnull / palt."""
        result = cl_s(jnp.array(pnull), jnp.array(palt))
        assert float(result) == pytest.approx(expected, rel=1e-5)

    def test_palt_zero(self):
        """cl_s(0.05, 0.0) returns a large value (0.05 / 1e-10 = 5e8)."""
        result = cl_s(jnp.array(0.05), jnp.array(0.0))
        assert float(result) == pytest.approx(5e8, rel=1e-3)

    def test_both_zero(self):
        """cl_s(0.0, 0.0) returns 0.0 (0/1e-10 = 0)."""
        result = cl_s(jnp.array(0.0), jnp.array(0.0))
        assert float(result) == pytest.approx(0.0, abs=1e-5)


# =============================================================================
# significance Tests
# =============================================================================


class TestSignificance:
    """Tests for significance(): Z = Φ⁻¹(1 - p)."""

    def test_known_values(self):
        """Test standalone significance() with known p-value → Z mappings."""
        assert float(significance(jnp.array(0.5))) == pytest.approx(0.0, abs=1e-6)
        assert float(significance(jnp.array(0.02275))) == pytest.approx(2.0, abs=0.01)
        assert float(significance(jnp.array(0.15866))) == pytest.approx(1.0, abs=0.01)
        assert float(significance(jnp.array(0.00135))) == pytest.approx(3.0, abs=0.01)


# =============================================================================
# sigma_from_asimov Tests
# =============================================================================


class TestSigmaFromAsimov:
    """Tests for the sigma_from_asimov utility.

    σ = |μ - μ_asimov| / √q_asimov (with floor at q_asimov=1e-10).
    """

    def test_unit_sigma(self):
        """μ=2, q_asimov=4, μ_asimov=0 → σ = 2/√4 = 1.0."""
        sigma = sigma_from_asimov(jnp.array(2.0), jnp.array(4.0))
        assert float(sigma) == pytest.approx(1.0, rel=1e-5)

    def test_fractional_sigma(self):
        """μ=1, q_asimov=9, μ_asimov=0 → σ = 1/√9 = 0.33333."""
        sigma = sigma_from_asimov(jnp.array(1.0), jnp.array(9.0))
        assert float(sigma) == pytest.approx(0.33333, rel=1e-4)

    def test_division_by_zero_guard(self):
        """q_asimov=0 triggers the 1e-10 floor, giving σ = 1e5."""
        sigma = sigma_from_asimov(jnp.array(1.0), jnp.array(0.0))
        assert float(sigma) == pytest.approx(1e5, rel=1e-3)


# =============================================================================
# make_asimov Tests
# =============================================================================


class TestMakeAsimov:
    """Tests for the make_asimov utility."""

    def test_asimov_at_background_only(self):
        """Asimov at mu=0 gives background-only expectation: n=5."""
        params = create_params(mu_init=1.0)
        asimov = make_asimov(predict_fn, params, "mu", mu_asimov=0.0)
        assert asimov["n"] == pytest.approx(5.0, rel=1e-5)

    def test_asimov_at_signal_plus_background(self):
        """Asimov at mu=1 gives signal+background expectation: n=15."""
        params = create_params(mu_init=1.0)
        asimov = make_asimov(predict_fn, params, "mu", mu_asimov=1.0)
        assert asimov["n"] == pytest.approx(15.0, rel=1e-5)

    def test_asimov_at_arbitrary_mu(self):
        """Asimov at mu=2 gives n = 2*10 + 5 = 25."""
        params = create_params(mu_init=1.0)
        asimov = make_asimov(predict_fn, params, "mu", mu_asimov=2.0)
        assert asimov["n"] == pytest.approx(25.0, rel=1e-5)


# =============================================================================
# constrained_fit Tests
# =============================================================================


class TestConstrainedFit:
    """Tests for constrained_fit: both code paths.

    constrained_fit has two branches:
    - No free params after fixing POI → evaluates NLL directly
    - Free nuisance params remain → runs ew.fit optimizer
    """

    @pytest.mark.parametrize(
        ("TestStatClass", "n_obs", "expected_mu_hat", "expected_q"),
        [
            (QTilde, 10.0, 0.5, 1.8907),
            (QMu, 10.0, 0.5, 1.8907),
        ],
        ids=["qtilde", "qmu"],
    )
    def test_no_nuisance(self, TestStatClass, n_obs, expected_mu_hat, expected_q):
        """No free params after fixing POI → NLL evaluation shortcut."""
        params = create_params(mu_init=1.0)
        observed = create_observation(n_obs)

        result = TestStatClass().compute(
            poisson_nll, params, observed, "mu", poi_test=1.0
        )

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, abs=1e-3)
        assert result.value == pytest.approx(expected_q, abs=1e-4)

    @pytest.mark.parametrize(
        ("TestStatClass", "n_obs", "expected_mu_hat", "expected_q"),
        [
            # QTilde, n_obs=12: mu_hat ≈ (12-5)/10 = 0.7
            # With nuisance theta (10% bkg uncertainty), optimizer finds
            # mu_hat ≈ 0.700, q ≈ 0.635
            (QTilde, 12.0, 0.700, 0.635),
            # QMu, n_obs=12: mu_hat ≈ (12-5)/10 = 0.7
            # Same non-boundary region as QTilde
            (QMu, 12.0, 0.700, 0.635),
        ],
        ids=["qtilde", "qmu"],
    )
    def test_with_nuisance(self, TestStatClass, n_obs, expected_mu_hat, expected_q):
        """Free nuisance params after fixing POI → optimizer profiles theta.

        Model: n_expected = mu * s + b * theta
        where theta is a nuisance parameter for background normalization
        with a Gaussian constraint (10% uncertainty).
        """
        params = sl.State.from_pytree({"mu": 1.0, "theta": 1.0})
        observed = create_observation(n_obs)

        result = TestStatClass().compute(
            nll_with_nuisance, params, observed, "mu", poi_test=1.0
        )

        assert result.extras["mu_hat"] == pytest.approx(expected_mu_hat, rel=0.05)
        assert result.value == pytest.approx(expected_q, rel=0.05)
