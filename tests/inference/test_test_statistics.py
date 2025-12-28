"""Tests for test statistics (q_tilde) implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import everwillow as ew
import everwillow.statelib as sl
from everwillow.inference.test_statistics import q_tilde

jax.config.update("jax_enable_x64", True)


def simple_nll(params):
    """Simple quadratic NLL: minimum at mu=2, sigma=1."""
    return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2


def counting_nll(params, observed: float, background: float):
    """Simple counting experiment NLL: signal + background."""
    expected = params["mu"] * 6.0 + background  # signal=6 at mu=1
    return expected - observed * jnp.log(jnp.maximum(expected, 1e-10))


class TestQTildeBasic:
    """Basic tests for q_tilde test statistic."""

    def test_q_tilde_at_best_fit_is_zero(self):
        """q should be 0 when poi_test equals best-fit value."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        # First do a free fit to find best-fit mu
        free_result = ew.fit(simple_nll, params)
        best_fit_mu = float(free_result.params["mu"])

        # Now compute q at best-fit mu - should be 0
        q, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=best_fit_mu)

        assert jnp.isclose(q, 0.0, atol=1e-6)

    def test_q_tilde_increases_away_from_best_fit(self):
        """q should increase as poi_test moves away from best-fit."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        # Best fit is at mu=2
        q_at_2, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=2.0)
        q_at_3, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=3.0)
        q_at_4, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=4.0)

        # q should increase as we move away from best-fit
        assert float(q_at_2) < float(q_at_3) < float(q_at_4)

    def test_q_tilde_clips_at_zero_for_excess(self):
        """q should be 0 when best_fit > poi_test (observed excess)."""
        # For simple_nll, best-fit mu is 2.0
        # Testing at mu=1.0 < 2.0 should give q=0 (excess scenario)
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        q, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=1.0)

        # With q_tilde clamping, q should be 0 when best_fit > poi_test
        assert float(q) == 0.0

    def test_q_tilde_profiles_nuisances(self):
        """Non-POI parameters should be profiled in both fits."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        _q, constrained_fit, free_fit = q_tilde(
            simple_nll, params, poi_name="mu", poi_test=3.0
        )

        # In free fit, sigma should be at its optimal value (1.0)
        assert jnp.isclose(free_fit.params["sigma"], 1.0, atol=1e-4)

        # In constrained fit (mu fixed at 3.0), sigma should still be optimized
        assert jnp.isclose(constrained_fit.params["sigma"], 1.0, atol=1e-4)

        # mu should be fixed at 3.0 in constrained fit
        assert jnp.isclose(constrained_fit.params["mu"], 3.0, atol=1e-10)


class TestQTildeFormula:
    """Tests for q_tilde formula correctness."""

    def test_q_tilde_formula_2_times_delta_nll(self):
        """q = 2 * (NLL_constrained - NLL_free)."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        q_val, constrained_fit, free_fit = q_tilde(
            simple_nll, params, poi_name="mu", poi_test=3.0
        )

        expected_q = 2.0 * (float(constrained_fit.nll) - float(free_fit.nll))

        # Should match (but with clamping at 0 for excess)
        if float(free_fit.params["mu"]) <= 3.0:
            assert jnp.isclose(q_val, expected_q, atol=1e-6)
        else:
            assert float(q_val) == 0.0

    def test_q_tilde_non_negative(self):
        """q should always be non-negative."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        for poi_test in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            q, _, _ = q_tilde(simple_nll, params, poi_name="mu", poi_test=poi_test)
            assert float(q) >= 0.0


class TestQTildeWithCountingExperiment:
    """Tests with a counting experiment model (like pyhf example)."""

    def test_q_tilde_counting_experiment(self):
        """Test q_tilde with counting experiment NLL."""
        # Model: signal=6, background=9, observed=9
        # At mu=1: expected = 15, observed = 9 → deficit → should have positive q

        def nll(params):
            # Include background normalization as nuisance parameter
            signal = 6.0
            background = 9.0 * params["gamma"]  # gamma ~ 1.0
            expected = params["mu"] * signal + background
            # Poisson NLL + Gaussian constraint on gamma
            poisson = expected - 9.0 * jnp.log(jnp.maximum(expected, 1e-10))
            constraint = ((params["gamma"] - 1.0) / 0.33) ** 2  # ~33% uncertainty
            return poisson + constraint

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        q, _constrained_fit, free_fit = q_tilde(
            nll, params, poi_name="mu", poi_test=1.0
        )

        # Free fit should find mu < 1 (since observed < expected at mu=1)
        # So q at mu=1 should be > 0
        assert float(q) > 0.0
        assert float(free_fit.params["mu"]) < 1.0

    def test_q_tilde_counting_at_mu_zero(self):
        """Test q_tilde at mu=0 for counting experiment."""

        def nll(params):
            # Include background normalization as nuisance parameter
            signal = 6.0
            background = 9.0 * params["gamma"]
            expected = params["mu"] * signal + background
            poisson = expected - 9.0 * jnp.log(jnp.maximum(expected, 1e-10))
            constraint = ((params["gamma"] - 1.0) / 0.33) ** 2
            return poisson + constraint

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        q, _, _free_fit = q_tilde(nll, params, poi_name="mu", poi_test=0.0)

        # At mu=0: expected = background = 9*gamma, observed = 9
        # Best fit should have gamma ~ 1.0
        # q might be clamped to 0 if best_fit mu > 0
        assert float(q) >= 0.0


class TestQTildeReturns:
    """Tests for q_tilde return values."""

    def test_returns_fit_results(self):
        """q_tilde should return both fit results."""
        params: sl.State[float] = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})

        q, constrained_fit, free_fit = q_tilde(
            simple_nll, params, poi_name="mu", poi_test=3.0
        )

        # Check types
        assert isinstance(q, jax.Array)
        assert isinstance(constrained_fit, ew.FitResult)
        assert isinstance(free_fit, ew.FitResult)

        # Check fit results have expected structure
        assert "mu" in constrained_fit.params
        assert "sigma" in constrained_fit.params
        assert "mu" in free_fit.params
        assert "sigma" in free_fit.params
