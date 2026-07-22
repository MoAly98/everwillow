"""Tests for ToyGenerator."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import pytest

from everwillow.hypotest.distributions import SimpleEmpiricalDistribution
from everwillow.hypotest.results import (
    TestStatResult as TSResult,  # Alias avoids pytest collection
)
from everwillow.hypotest.results import ToyResult
from everwillow.hypotest.test_statistics import QTilde
from everwillow.hypotest.toys import ToyGenerator

from ._counting_model import (
    create_observation,
    create_params,
    poisson_nll,
    predict_fn,
    sample_fn,
)

# =============================================================================
# ToyGenerator Tests
# =============================================================================


class TestToyGenerator:
    """Tests for ToyGenerator."""

    @pytest.mark.parametrize(
        ("gen_method", "gen_fn"),
        [
            ("sample_fn", sample_fn),
            ("predict_fn", predict_fn),
        ],
    )
    def test_generate(self, gen_method, gen_fn):
        """Test toy generation with sample_fn and predict_fn."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(ntoys=100, **{gen_method: gen_fn})
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            test_statistic=QTilde(),
            poi_alt=0.0,
            key=jax.random.key(42),
        )

        assert isinstance(toys, ToyResult)
        assert toys.q_alt.shape == (100,)
        assert toys.q_null.shape == (100,)
        # can we somehow test values?

    def test_requires_sample_or_predict(self):
        """Test that either sample_fn or predict_fn is required."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(ntoys=10)

        with pytest.raises(ValueError, match="Either sample_fn or predict_fn"):
            toy_gen.generate(
                poisson_nll,
                params,
                observed,
                "mu",
                poi_null=1.0,
                test_statistic=QTilde(),
                key=jax.random.key(42),
            )

    def test_q_alt_vs_q_null_distribution(self):
        """Test that q_alt and q_null have different distributions.

        Under null (poi_null=1.0), data is consistent -> q_null tends to be small.
        Under alt (poi_alt=0.0), data inconsistent with poi_test -> q_alt large.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(predict_fn=predict_fn, ntoys=500)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(123),
        )

        # Mean of q_alt should be larger than mean of q_null
        mean_q_alt = float(jnp.mean(toys.q_alt))
        mean_q_null = float(jnp.mean(toys.q_null))

        assert mean_q_alt > mean_q_null

    def test_reproducibility(self):
        """Test that same key gives same results."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(predict_fn=predict_fn, ntoys=50)

        toys1 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(999),
        )

        toys2 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(999),
        )

        assert jnp.allclose(toys1.q_alt, toys2.q_alt)
        assert jnp.allclose(toys1.q_null, toys2.q_null)

    def test_different_keys_different_results(self):
        """Test that different keys give different results."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(predict_fn=predict_fn, ntoys=50)

        toys1 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(111),
        )

        toys2 = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(222),
        )

        assert not jnp.allclose(toys1.q_alt, toys2.q_alt)
        assert not jnp.allclose(toys1.q_null, toys2.q_null)

    @pytest.mark.parametrize(
        ("map_fn_id", "map_fn"),
        [
            ("lax.map", lambda fn: partial(jax.lax.map, fn)),
            (
                "python-loop",
                lambda fn: lambda keys: jnp.stack([fn(k) for k in keys]),
            ),
        ],
    )
    def test_custom_map_fn(self, map_fn_id, map_fn):
        """Custom map_fn produces same results as default jax.vmap."""
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)
        key = jax.random.key(42)

        gen_default = ToyGenerator(predict_fn=predict_fn, ntoys=20)
        gen_custom = ToyGenerator(predict_fn=predict_fn, ntoys=20, map_fn=map_fn)

        toys_default = gen_default.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=key,
        )
        toys_custom = gen_custom.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=key,
        )

        assert jnp.allclose(toys_default.q_null, toys_custom.q_null, atol=1e-4)
        assert jnp.allclose(toys_default.q_alt, toys_custom.q_alt, atol=1e-4)


class TestToyGeneratorPoissonSampler:
    """Tests for the default Poisson sampler."""

    def test_poisson_samples_values(self):
        """Test that Poisson toys produce expected q distributions.

        Under null (poi_null=1.0): testing at true mu, so
        QTilde gives q=0 for most toys (mu_hat ≈ mu_test).
        Median q_null should be near 0.0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(predict_fn=predict_fn, ntoys=100)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(42),
        )

        assert jnp.all(jnp.isfinite(toys.q_alt))
        assert jnp.all(jnp.isfinite(toys.q_null))
        # Under null (poi_null=1.0), testing at true mu: median q_null ≈ 0
        assert float(jnp.median(toys.q_null)) == pytest.approx(0.0, abs=0.5)

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

    def test_empirical_pvalues_at_q_zero(self):
        """At q=0, nearly all toys have q >= 0, so p-values ≈ 1.0.

        QTilde produces q >= 0 by construction (boundary + max(0,...)).
        At q_obs=0, the fraction of toys with q >= 0 is ~1.0.
        """
        params = create_params(mu_init=1.0)
        observed = create_observation(15.0)

        toy_gen = ToyGenerator(predict_fn=predict_fn, ntoys=200)
        toys = toy_gen.generate(
            poisson_nll,
            params,
            observed,
            "mu",
            poi_null=1.0,
            poi_alt=0.0,
            test_statistic=QTilde(),
            key=jax.random.key(42),
        )
        dist = SimpleEmpiricalDistribution.from_toys(toys)

        result = TSResult(value=jnp.array(0.0), test=jnp.array(1.0))
        pnull = dist.null_pval(result)
        palt = dist.alt_pval(result)

        # All q values are >= 0, so at q_obs=0 p-values should be ~1.0
        assert float(pnull) == pytest.approx(1.0, abs=0.1)
        assert float(palt) == pytest.approx(1.0, abs=0.1)
