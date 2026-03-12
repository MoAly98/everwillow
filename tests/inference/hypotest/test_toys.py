"""Tests for ToyGenerator.

Tests toy generation with both sample_fn and predict_fn (Poisson sampler).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import everwillow.statelib as sl
from everwillow.inference.hypotest import (
    QTilde,
    SimpleEmpiricalDistribution,
    ToyGenerator,
    ToyResult,
)
from everwillow.inference.hypotest import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)

# =============================================================================
# Test fixtures
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
    """Prediction function for expected counts."""
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * S + B}


def sample_fn(params_state: sl.State, key) -> dict[str, float]:
    """Sample function for toy generation."""
    expected = predict_fn(params_state)
    return {"n": jax.random.poisson(key, expected["n"])}


# =============================================================================
# ToyGenerator Tests
# =============================================================================


class TestToyGenerator:
    """Tests for ToyGenerator."""

    def test_generate_with_sample_fn(self):
        """Test toy generation with explicit sample_fn."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=100)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(42),
            sample_fn=sample_fn,
        )

        assert isinstance(toys, ToyResult)
        assert toys.q_alt.shape == (100,)
        assert toys.q_null.shape == (100,)

    def test_generate_with_predict_fn(self):
        """Test toy generation with predict_fn (default Poisson sampler)."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=100)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(42),
            predict_fn=predict_fn,
        )

        assert isinstance(toys, ToyResult)
        assert toys.q_alt.shape == (100,)
        assert toys.q_null.shape == (100,)

    def test_requires_sample_or_predict(self):
        """Test that either sample_fn or predict_fn is required."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=10)

        with pytest.raises(ValueError, match="Either sample_fn or predict_fn"):
            toy_gen.generate(
                poisson_nll,
                params,
                observed,
                ("mu",),
                poi_test=1.0,
                key=jax.random.key(42),
            )

    def test_q_alt_vs_q_null_distribution(self):
        """Test that q_alt and q_null have different distributions.

        Under alternative (mu=1), signal is present -> q_alt tends to be smaller.
        Under null (mu=0), no signal -> q_null tends to be larger for same poi_test.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=500)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(123),
            predict_fn=predict_fn,
        )

        # Mean of q_null should be larger than mean of q_alt
        mean_q_alt = float(jnp.mean(toys.q_alt))
        mean_q_null = float(jnp.mean(toys.q_null))

        assert mean_q_null > mean_q_alt

    def test_reproducibility(self):
        """Test that same key gives same results."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=50)

        toys1 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(999),
            predict_fn=predict_fn,
        )

        toys2 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(999),
            predict_fn=predict_fn,
        )

        assert jnp.allclose(toys1.q_alt, toys2.q_alt)
        assert jnp.allclose(toys1.q_null, toys2.q_null)

    def test_different_keys_different_results(self):
        """Test that different keys give different results."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=50)

        toys1 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(111),
            predict_fn=predict_fn,
        )

        toys2 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(222),
            predict_fn=predict_fn,
        )

        assert not jnp.allclose(toys1.q_alt, toys2.q_alt)
        assert not jnp.allclose(toys1.q_null, toys2.q_null)


class TestToyGeneratorPoissonSampler:
    """Tests for the default Poisson sampler."""

    def test_poisson_samples_finite(self):
        """Test that Poisson samples produce finite q values."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=100)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(42),
            predict_fn=predict_fn,
        )

        assert jnp.all(jnp.isfinite(toys.q_alt))
        assert jnp.all(jnp.isfinite(toys.q_null))

    def test_poisson_mean_matches_expectation(self):
        """Test that Poisson samples have correct mean.

        Under alternative with mu=1, expected count is 15.
        Mean of many Poisson samples should be close to 15.
        """
        params = create_params(mu_init=1.0)
        key = jax.random.key(42)
        keys = jax.random.split(key, 1000)

        sampler = ToyGenerator._make_poisson_sampler(predict_fn)
        samples = jax.vmap(lambda k: sampler(params, k)["n"])(keys)

        mean_sample = float(jnp.mean(samples))
        assert mean_sample == pytest.approx(15.0, rel=0.05)

    def test_poisson_variance_matches_expectation(self):
        """Test that Poisson samples have correct variance.

        For Poisson distribution, variance = mean = 15.
        """
        params = create_params(mu_init=1.0)
        key = jax.random.key(42)
        keys = jax.random.split(key, 2000)

        sampler = ToyGenerator._make_poisson_sampler(predict_fn)
        samples = jax.vmap(lambda k: sampler(params, k)["n"])(keys)

        var_sample = float(jnp.var(samples))
        assert var_sample == pytest.approx(15.0, rel=0.15)


class TestToyGeneratorIntegration:
    """Integration tests for ToyGenerator with calculator."""

    def test_empirical_pvalues_in_range(self):
        """Test that empirical p-values are in valid range [0, 1]."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=200)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(42),
            predict_fn=predict_fn,
        )
        dist = SimpleEmpiricalDistribution.from_toys(toys)

        # Test p-values at q=0
        result = TSResult(value=jnp.array(0.0), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        assert 0.0 <= float(pnull) <= 1.0
        assert 0.0 <= float(palt) <= 1.0

    def test_empirical_pvalues_at_q_zero(self):
        """At q=0, most toys should have q >= 0, so p-values should be high."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=200)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            ("mu",),
            poi_test=1.0,
            key=jax.random.key(42),
            predict_fn=predict_fn,
        )
        dist = SimpleEmpiricalDistribution.from_toys(toys)

        result = TSResult(value=jnp.array(0.0), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        # At q=0, all toys with q >= 0 contribute
        assert float(palt) > 0.8
        assert float(pnull) > 0.8
