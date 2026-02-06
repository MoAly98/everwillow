"""Toy generation for hypothesis testing.

This module provides ToyGenerator for Monte Carlo-based hypothesis testing.
It generates toys under both hypotheses and returns an EmpiricalDistribution.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._utils import constrained_fit
from everwillow.inference.hypotest.distributions import EmpiricalDistribution
from everwillow.inference.hypotest.test_statistics import TestStatistic

__all__ = ["ToyGenerator"]


class ToyGenerator(eqx.Module):
    """Generates toy experiments for hypothesis testing.

    Creates toys under both alternative and null hypotheses,
    computes test statistics for each, and returns an
    EmpiricalDistribution for p-value computation.

    Attributes:
        test_statistic: Test statistic to compute for each toy.
        ntoys: Number of toys per hypothesis. Defaults to 1000.

    Example:
        >>> toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=10000)
        >>> dist = toy_gen.generate(
        ...     nll_fn, params, ("mu",), 1.0,
        ...     sample_fn=my_sampler,
        ...     nll_factory=my_nll_factory,
        ...     key=jax.random.key(42),
        ... )
        >>> # Use with HypoTestCalculator
        >>> result = calc(nll_fn, params, ("mu",), 1.0, distribution=dist)
    """

    test_statistic: TestStatistic
    ntoys: int = 1000

    def generate(
        self,
        nll_fn: tp.Callable[[PyTree], float],
        params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        *,
        sample_fn: tp.Callable[[sl.State, PRNGKeyArray], tp.Any],
        nll_factory: tp.Callable[[sl.State, tp.Any], tp.Callable[[PyTree], float]],
        key: PRNGKeyArray,
        **fit_kwargs: tp.Any,
    ) -> EmpiricalDistribution:
        """Generate toys and return empirical distribution.

        Args:
            nll_fn: Negative log-likelihood function for observed data
                   (used to profile nuisance parameters).
            params: Initial parameter state.
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            sample_fn: Function to generate toy data. Called as
                sample_fn(params_state, key) -> toy_data.
            nll_factory: Function to create NLL from toy data. Called as
                nll_factory(params_state, toy_data) -> nll_fn.
            key: JAX PRNG key for reproducibility.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            EmpiricalDistribution with q_alt and q_null arrays.
        """
        # Split keys for alt and null toys
        keys = jax.random.split(key, self.ntoys * 2)
        keys_alt = keys[: self.ntoys]
        keys_null = keys[self.ntoys :]

        # Profile nuisance parameters by fitting with POI fixed
        # This ensures toys are generated at the best-fit point for each hypothesis

        # Alternative hypothesis: POI = poi_test (signal)
        fixed_alt: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        alt_result = constrained_fit(nll_fn, params, fixed_alt, **fit_kwargs)
        params_alt: sl.State[ArrayLike] = sl.State.from_pytree(alt_result.params)

        # Null hypothesis: POI = 0 (background-only)
        fixed_null: sl.State[float] = sl.State.from_pytree({poi_key: 0.0})
        null_result = constrained_fit(nll_fn, params, fixed_null, **fit_kwargs)
        params_null: sl.State[ArrayLike] = sl.State.from_pytree(null_result.params)

        # Generate toys under alternative hypothesis
        q_alt = self._run_toys(
            params_alt,
            params,
            poi_key,
            poi_test,
            sample_fn,
            nll_factory,
            keys_alt,
            fit_kwargs,
        )

        # Generate toys under null hypothesis
        q_null = self._run_toys(
            params_null,
            params,
            poi_key,
            poi_test,
            sample_fn,
            nll_factory,
            keys_null,
            fit_kwargs,
        )

        return EmpiricalDistribution(q_alt=q_alt, q_null=q_null)

    def _run_toys(
        self,
        sample_params: sl.State,
        fit_params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        sample_fn: tp.Callable[[sl.State, PRNGKeyArray], tp.Any],
        nll_factory: tp.Callable[[sl.State, tp.Any], tp.Callable[[PyTree], float]],
        keys: PRNGKeyArray,
        fit_kwargs: dict,
    ) -> Array:
        """Run toys and return test statistic values.

        Uses jax.vmap for parallel computation across toys.

        Args:
            sample_params: Parameters to use for sampling (State).
            fit_params: Parameters to use for fitting (State).
            poi_key: Canonical key for the POI.
            poi_test: Test value for POI.
            sample_fn: Sampling function.
            nll_factory: NLL factory function.
            keys: Array of PRNG keys, one per toy.
            fit_kwargs: Additional fit arguments.

        Returns:
            Array of test statistic values, shape (ntoys,).
        """

        def single_toy(key: PRNGKeyArray) -> Array:
            # Generate toy data
            toy_data = sample_fn(sample_params, key)
            # Create NLL for this toy
            toy_nll = nll_factory(sample_params, toy_data)
            # Compute test statistic
            result = self.test_statistic(
                toy_nll, fit_params, poi_key, poi_test, **fit_kwargs
            )
            return result.q

        # Run toys in parallel using vmap
        return jax.vmap(single_toy)(keys)
