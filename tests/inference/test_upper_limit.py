"""Tests for upper_limit function (root finding for exclusion limits)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

import everwillow.statelib as sl
from everwillow.inference.calculators import cls
from everwillow.inference.hypotest import hypotest
from everwillow.inference.upper_limit import upper_limit
from everwillow.inference.upper_limit_toys import upper_limit_toys

jax.config.update("jax_enable_x64", True)


def make_counting_nll(signal: float, bkg: float, bkg_unc: float, observed: float):
    """Create NLL matching pyhf's uncorrelated_background model."""
    aux_data = (bkg / bkg_unc) ** 2

    def nll(params):
        mu = params["mu"]
        gamma = params["gamma"]
        expected = mu * signal + bkg * gamma
        main_nll = (
            expected
            - observed * jnp.log(jnp.maximum(expected, 1e-10))
            + gammaln(observed + 1)
        )
        aux_expected = gamma * aux_data
        constraint_nll = (
            aux_expected
            - aux_data * jnp.log(jnp.maximum(aux_expected, 1e-10))
            + gammaln(aux_data + 1)
        )
        return main_nll + constraint_nll

    return nll


class TestUpperLimitBasic:
    """Basic tests for upper_limit function."""

    def test_upper_limit_returns_float(self):
        """upper_limit should return a float."""

        # Simple monotonic function: f(x) = x
        def objective(x):
            return x

        result = upper_limit(objective, bounds=(0.0, 10.0), level=5.0)
        assert isinstance(result, float)

    def test_upper_limit_finds_root(self):
        """upper_limit should find where objective = level."""

        # f(x) = x, find where f(x) = 3
        def objective(x):
            return x

        result = upper_limit(objective, bounds=(0.0, 10.0), level=3.0)
        assert jnp.isclose(result, 3.0, atol=1e-4)

    def test_upper_limit_quadratic(self):
        """upper_limit should work with non-linear functions."""

        # f(x) = x^2, find where f(x) = 4 -> x = 2
        def objective(x):
            return x**2

        result = upper_limit(objective, bounds=(0.0, 10.0), level=4.0)
        assert jnp.isclose(result, 2.0, atol=1e-4)

    def test_upper_limit_within_bounds(self):
        """Result should be within specified bounds."""

        def objective(x):
            return x

        bounds = (1.0, 5.0)
        result = upper_limit(objective, bounds=bounds, level=3.0)
        assert bounds[0] <= result <= bounds[1]


class TestUpperLimitWithHypotest:
    """Tests for upper_limit with hypotest-based objectives."""

    def test_upper_limit_cls_criterion(self):
        """Find upper limit using CLs criterion."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        def cls_criterion(poi):
            # Objective function must be JAX-traceable (no float() calls)
            result = hypotest(nll_fn, params, poi_name="mu", poi_test=poi)
            return cls(result.p_alt, result.p_null)

        # Find 95% CL upper limit (CLs = 0.05)
        limit = upper_limit(cls_criterion, bounds=(0.0, 5.0), level=0.05)

        # Verify CLs at the limit is close to 0.05
        result_at_limit = hypotest(nll_fn, params, poi_name="mu", poi_test=limit)
        cls_at_limit = cls(result_at_limit.p_alt, result_at_limit.p_null)
        assert jnp.isclose(cls_at_limit, 0.05, atol=0.001)

    def test_upper_limit_palt_criterion(self):
        """Find upper limit using p_alt only (frequentist)."""
        nll_fn = make_counting_nll(signal=6.0, bkg=9.0, bkg_unc=3.0, observed=9.0)
        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})

        def palt_criterion(poi):
            # Objective function must be JAX-traceable (no float() calls)
            result = hypotest(nll_fn, params, poi_name="mu", poi_test=poi)
            return result.p_alt

        # Find where p_alt = 0.05
        limit = upper_limit(palt_criterion, bounds=(0.0, 5.0), level=0.05)

        # Verify p_alt at the limit is close to 0.05
        result_at_limit = hypotest(nll_fn, params, poi_name="mu", poi_test=limit)
        assert jnp.isclose(result_at_limit.p_alt, 0.05, atol=0.001)


class TestUpperLimitMonotonicity:
    """Tests for monotonicity assumptions."""

    def test_upper_limit_decreasing_function(self):
        """upper_limit should work with decreasing functions."""

        # CLs typically decreases as mu increases (for deficit data)
        # f(x) = 1 - x/10, find where f(x) = 0.3 -> x = 7
        def objective(x):
            return 1.0 - x / 10.0

        result = upper_limit(objective, bounds=(0.0, 10.0), level=0.3)
        assert jnp.isclose(result, 7.0, atol=1e-4)

    def test_upper_limit_increasing_function(self):
        """upper_limit should work with increasing functions."""

        # f(x) = x/10, find where f(x) = 0.5 -> x = 5
        def objective(x):
            return x / 10.0

        result = upper_limit(objective, bounds=(0.0, 10.0), level=0.5)
        assert jnp.isclose(result, 5.0, atol=1e-4)


class TestUpperLimitEdgeCases:
    """Edge case tests for upper_limit."""

    def test_upper_limit_level_at_bound(self):
        """Handle case where solution is near bounds."""

        def objective(x):
            return x

        # Solution is at x=0.1, close to lower bound
        result = upper_limit(objective, bounds=(0.0, 10.0), level=0.1)
        assert jnp.isclose(result, 0.1, atol=1e-4)

    def test_upper_limit_narrow_bounds(self):
        """Work with narrow search bounds."""

        def objective(x):
            return x

        result = upper_limit(objective, bounds=(2.9, 3.1), level=3.0)
        assert jnp.isclose(result, 3.0, atol=1e-4)


# ============================================================
# Tests for upper_limit_toys
# ============================================================

from everwillow.inference.hypotest_toys import hypotest_toys


def make_counting_nll_factory(signal: float, bkg: float, bkg_unc: float):
    """Create NLL factory for toy-based testing."""
    aux_data = (bkg / bkg_unc) ** 2

    def nll_factory(params, observed):
        mu = params["mu"]
        gamma = params["gamma"]

        main_obs = observed["main"]
        aux_obs = observed["aux"]

        expected = mu * signal + bkg * gamma
        main_nll = (
            expected
            - main_obs * jnp.log(jnp.maximum(expected, 1e-10))
            + gammaln(main_obs + 1)
        )

        aux_expected = gamma * aux_data
        constraint_nll = (
            aux_expected
            - aux_obs * jnp.log(jnp.maximum(aux_expected, 1e-10))
            + gammaln(aux_obs + 1)
        )
        return main_nll + constraint_nll

    return nll_factory, aux_data


def make_sample_fn(signal: float, bkg: float, aux_data: float):
    """Create sampling function for toy generation."""

    def sample_fn(params, key):
        mu = params["mu"]
        gamma = params["gamma"]

        key1, key2 = jax.random.split(key)

        expected_main = mu * signal + bkg * gamma
        main_obs = jax.random.poisson(key1, expected_main)

        expected_aux = gamma * aux_data
        aux_obs = jax.random.poisson(key2, expected_aux)

        return {"main": main_obs, "aux": aux_obs}

    return sample_fn


class TestUpperLimitToys:
    """Tests for upper_limit_toys function."""

    def test_upper_limit_toys_returns_float(self):
        """upper_limit_toys should return a float."""
        # Simple objective function that takes (poi, key) -> value
        def objective(poi, key):
            # Simulate CLs-like behavior: decreases as poi increases
            return 1.0 / (1.0 + poi)

        key = jax.random.key(42)
        result = upper_limit_toys(objective, bounds=(0.0, 10.0), key=key, level=0.5)
        assert isinstance(result, float)

    def test_upper_limit_toys_finds_root(self):
        """upper_limit_toys should find where objective = level."""
        # f(x, key) = 1/(1+x), find where f(x) = 0.5 -> x = 1
        def objective(poi, key):
            return 1.0 / (1.0 + poi)

        key = jax.random.key(42)
        result = upper_limit_toys(
            objective, bounds=(0.0, 10.0), key=key, level=0.5, tolerance=0.01
        )
        assert jnp.isclose(result, 1.0, atol=0.05)

    def test_upper_limit_toys_within_bounds(self):
        """Result should be within specified bounds."""

        def objective(poi, key):
            return 1.0 / (1.0 + poi)

        key = jax.random.key(42)
        bounds = (0.5, 5.0)
        result = upper_limit_toys(objective, bounds=bounds, key=key, level=0.3)
        assert bounds[0] <= result <= bounds[1]

    def test_upper_limit_toys_with_hypotest(self):
        """Test with actual hypotest_toys objective."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        # Create objective that calls hypotest_toys
        def cls_objective(poi, key):
            result = hypotest_toys(
                nll_fn,
                params,
                poi_name="mu",
                poi_test=poi,
                nll_factory=nll_factory,
                sample_fn=sample_fn,
                key=key,
                ntoys=50,
            )
            return float(cls(result.p_alt, result.p_null))

        key = jax.random.key(42)
        result = upper_limit_toys(
            cls_objective,
            bounds=(0.0, 3.0),
            key=key,
            level=0.05,
            tolerance=0.03,
            max_iterations=8,
        )

        assert isinstance(result, float)
        assert 0.0 <= result <= 3.0

    def test_upper_limit_toys_rough_agreement_with_asymptotic(self):
        """Toy-based limit should roughly match asymptotic limit."""
        signal, bkg, bkg_unc = 6.0, 9.0, 3.0
        nll_factory, aux_data = make_counting_nll_factory(signal, bkg, bkg_unc)
        observed = {"main": 9.0, "aux": aux_data}

        def nll_fn(params):
            return nll_factory(params, observed)

        params: sl.State[float] = sl.State.from_pytree({"mu": 1.0, "gamma": 1.0})
        sample_fn = make_sample_fn(signal, bkg, aux_data)

        # Asymptotic limit
        def cls_criterion_asymp(poi):
            result = hypotest(nll_fn, params, poi_name="mu", poi_test=poi)
            return cls(result.p_alt, result.p_null)

        limit_asymp = upper_limit(cls_criterion_asymp, bounds=(0.0, 3.0), level=0.05)

        # Toy-based objective
        def cls_objective_toys(poi, key):
            result = hypotest_toys(
                nll_fn,
                params,
                poi_name="mu",
                poi_test=poi,
                nll_factory=nll_factory,
                sample_fn=sample_fn,
                key=key,
                ntoys=200,
            )
            return float(cls(result.p_alt, result.p_null))

        key = jax.random.key(42)
        limit_toys = upper_limit_toys(
            cls_objective_toys,
            bounds=(0.0, 3.0),
            key=key,
            level=0.05,
            tolerance=0.02,
            max_iterations=10,
        )

        # Should be within ~50% of each other (toys have large uncertainty)
        assert 0.5 * limit_asymp < limit_toys < 2.0 * limit_asymp
