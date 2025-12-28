"""Tests for hypotest_toys function (toy-based hypothesis testing)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

import everwillow.statelib as sl
from everwillow.inference.calculators import cls
from everwillow.inference.hypotest import hypotest
from everwillow.inference.hypotest_toys import HypoTestToysResult, hypotest_toys

jax.config.update("jax_enable_x64", True)


def make_counting_nll_factory(signal: float, bkg: float, bkg_unc: float):
    """Return NLL factory that takes observed dict as argument.

    This allows creating NLLs for different toy observations.
    The observed dict should have "main" and "aux" keys.
    """
    aux_data = (bkg / bkg_unc) ** 2

    def nll_fn(params, observed):
        mu = params["mu"]
        gamma = params["gamma"]

        # Extract observations (can be scalar or traced)
        main_obs = observed["main"]
        aux_obs = observed["aux"]

        # Main channel: Poisson NLL
        expected = mu * signal + bkg * gamma
        main_nll = (
            expected
            - main_obs * jnp.log(jnp.maximum(expected, 1e-10))
            + gammaln(main_obs + 1)
        )

        # Auxiliary channel: Poisson constraint
        aux_expected = gamma * aux_data
        constraint_nll = (
            aux_expected
            - aux_obs * jnp.log(jnp.maximum(aux_expected, 1e-10))
            + gammaln(aux_obs + 1)
        )
        return main_nll + constraint_nll

    return nll_fn, aux_data


def make_sample_fn(signal: float, bkg: float, aux_data: float):
    """Create a sample function that generates toy data.

    This function matches pyhf's Poisson sampling for both main
    and auxiliary channels.
    """

    def sample_fn(params, key):
        """Generate toy data given parameters and PRNG key.

        Args:
            params: Dictionary with "mu" and "gamma" parameters
            key: JAX PRNG key

        Returns:
            Dictionary with "main" and "aux" observed counts
        """
        mu = params["mu"]
        gamma = params["gamma"]

        key1, key2 = jax.random.split(key)

        # Main channel: Poisson(mu*signal + bkg*gamma)
        expected_main = mu * signal + bkg * gamma
        main_obs = jax.random.poisson(key1, expected_main)

        # Auxiliary channel: Poisson(gamma * aux_data)
        expected_aux = gamma * aux_data
        aux_obs = jax.random.poisson(key2, expected_aux)

        return {"main": main_obs, "aux": aux_obs}

    return sample_fn


class TestHypoTestToysResult:
    """Tests for HypoTestToysResult dataclass."""

    def test_result_has_required_fields(self):
        """HypoTestToysResult should have p_alt, p_null, q_obs, ntoys, q_alt, q_null."""
        result = HypoTestToysResult(
            p_alt=0.084,
            p_null=0.5,
            q_obs=1.9,
            ntoys=500,
            q_alt=jnp.ones(500),
            q_null=jnp.ones(500),
        )
        assert hasattr(result, "p_alt")
        assert hasattr(result, "p_null")
        assert hasattr(result, "q_obs")
        assert hasattr(result, "ntoys")
        assert hasattr(result, "q_alt")
        assert hasattr(result, "q_null")


class TestHypoTestToysBasic:
    """Basic tests for hypotest_toys function."""

    def test_hypotest_toys_returns_result(self):
        """hypotest_toys should return HypoTestToysResult."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)
        key = jax.random.key(42)

        result = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key,
            ntoys=10,  # Few toys for basic test
            max_steps=512,  # More steps for edge case toys
        )

        assert isinstance(result, HypoTestToysResult)

    def test_hypotest_toys_pvalues_in_valid_range(self):
        """P-values should be in [0, 1]."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)
        key = jax.random.key(42)

        result = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key,
            ntoys=20,
        )

        assert 0.0 <= float(result.p_alt) <= 1.0
        assert 0.0 <= float(result.p_null) <= 1.0

    def test_hypotest_toys_q_obs_non_negative(self):
        """Test statistic q should be non-negative."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)
        key = jax.random.key(42)

        result = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key,
            ntoys=20,
        )

        assert float(result.q_obs) >= 0.0


class TestHypoTestToysReproducibility:
    """Tests for reproducibility with PRNG keys."""

    def test_same_key_gives_same_result(self):
        """Same PRNG key should give identical results."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        key1 = jax.random.key(12345)
        key2 = jax.random.key(12345)

        result1 = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key1,
            ntoys=50,
        )

        result2 = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key2,
            ntoys=50,
        )

        assert result1.p_alt == result2.p_alt
        assert result1.p_null == result2.p_null

    def test_different_keys_give_different_results(self):
        """Different PRNG keys should (very likely) give different results."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        key1 = jax.random.key(11111)
        key2 = jax.random.key(99999)

        result1 = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key1,
            ntoys=20,
        )

        result2 = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key2,
            ntoys=20,
        )

        # With high probability, results differ (not guaranteed but very likely)
        # At least one of p_alt or p_null should differ
        results_differ = (result1.p_alt != result2.p_alt) or (
            result1.p_null != result2.p_null
        )
        assert results_differ


class TestHypoTestToysConvergence:
    """Tests for statistical properties of toy-based CLs."""

    def test_rough_agreement_with_asymptotic(self):
        """Toy CLs should roughly match asymptotic (within stat fluctuations).

        This test verifies that toy-based CLs is statistically consistent
        with the asymptotic calculation. We use a relaxed tolerance since
        toy estimates have statistical uncertainty.
        """
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        # Asymptotic result
        asymp_result = hypotest(nll_fn, params, poi_name="mu", poi_test=1.0)
        cls_asymp = cls(asymp_result.p_alt, asymp_result.p_null)

        # Toy result with many toys for better convergence
        key = jax.random.key(42)
        toy_result = hypotest_toys(
            nll_fn,
            params,
            poi_name="mu",
            poi_test=1.0,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key,
            ntoys=500,
        )
        cls_toys = cls(toy_result.p_alt, toy_result.p_null)

        # Relaxed tolerance: toys should be within ~0.1 of asymptotic
        # (Statistical uncertainty for 500 toys is ~1/sqrt(500) ~ 0.04)
        assert jnp.isclose(cls_toys, cls_asymp, atol=0.1)

    def test_more_toys_reduces_variance(self):
        """More toys should give more stable results."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        # Run multiple times with different keys at different ntoys
        def get_cls_variance(ntoys: int, n_trials: int = 5) -> float:
            """Get variance of CLs estimates across trials."""
            cls_values = []
            base_key = jax.random.key(0)
            for i in range(n_trials):
                key = jax.random.fold_in(base_key, i)
                result = hypotest_toys(
                    nll_fn,
                    params,
                    poi_name="mu",
                    poi_test=1.0,
                    nll_factory=nll_factory,
                    sample_fn=sample_fn,
                    key=key,
                    ntoys=ntoys,
                )
                cls_values.append(float(cls(result.p_alt, result.p_null)))
            return float(jnp.var(jnp.array(cls_values)))

        # More toys should have lower variance
        var_few = get_cls_variance(50)
        var_many = get_cls_variance(200)

        # We can't guarantee var_many < var_few for all seeds, but the
        # expected variance scales as 1/ntoys, so we check it's not worse
        # (This is a statistical test, so we use a relaxed check)
        assert var_many < var_few * 2  # Should be roughly 4x smaller, allow 2x
