"""Toy generation for hypothesis testing.

This module provides ToyGenerator for Monte Carlo-based hypothesis testing.
It generates toys under both hypotheses and returns a ToyResult.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

import everwillow.statelib as sl
from everwillow.inference.hypotest._utils import constrained_fit
from everwillow.inference.hypotest.results import ToyResult
from everwillow.inference.hypotest.test_statistics import TestStatistic

__all__ = ["ToyGenerator"]


class ToyGenerator(eqx.Module):
    """Generates toy experiments for hypothesis testing.

    Creates toys under both alternative and null hypotheses,
    computes test statistics for each, and returns a ToyResult.
    The raw arrays can then be fed into any EmpiricalDistribution
    subclass for p-value computation.

    Attributes:
        test_statistic: Test statistic to compute for each toy.
        ntoys: Number of toys per hypothesis. Defaults to 1000.

    Example:
        >>> toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=10000)
        >>> toys = toy_gen.generate(
        ...     nll_fn, params, observed, ("mu",), 1.0,
        ...     key=jax.random.key(42),
        ...     predict_fn=my_predict_fn,
        ... )
        >>> # Choose how to interpret the toys (open-world)
        >>> dist = SimpleEmpiricalDistribution.from_toys(toys)
        >>> # Use with HypoTestCalculator
        >>> calc = HypoTestCalculator(test_statistic=QTilde(), distribution=dist)
        >>> result = calc(nll_fn, params, observed, ("mu",), 1.0)
    """

    test_statistic: TestStatistic
    ntoys: int = 1000

    def generate(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        *,
        key: PRNGKeyArray,
        sample_fn: tp.Callable[[sl.State, PRNGKeyArray], PyTree] | None = None,
        predict_fn: tp.Callable[[sl.State], PyTree] | None = None,
        **fit_kwargs: tp.Any,
    ) -> ToyResult:
        """Generate toys and return raw test statistic arrays.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data (used to profile nuisance parameters).
            poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
            poi_test: Test value for the POI.
            key: JAX PRNG key for reproducibility.
            sample_fn: Function to generate toy data. Called as
                sample_fn(params_state, key) -> toy_observation. If None,
                a default Poisson sampler is created using predict_fn.
            predict_fn: Function returning expected observation given parameters.
                Used to create default Poisson sampler if sample_fn is None.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            ToyResult with q_alt and q_null arrays.

        Raises:
            ValueError: If neither sample_fn nor predict_fn is provided.
        """
        # Create default Poisson sampler if sample_fn not provided
        if sample_fn is None:
            if predict_fn is None:
                raise ValueError("Either sample_fn or predict_fn must be provided")
            sample_fn = self._make_poisson_sampler(predict_fn)
        # Split keys for alt and null toys
        keys = jax.random.split(key, self.ntoys * 2)
        keys_alt = keys[: self.ntoys]
        keys_null = keys[self.ntoys :]

        # Profile nuisance parameters by fitting with POI fixed
        # This ensures toys are generated at the best-fit point for each hypothesis

        # Alternative hypothesis: POI = poi_test (signal)
        fixed_alt: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        alt_result = constrained_fit(
            nll_fn, params, observation, fixed_alt, **fit_kwargs
        )
        params_alt: sl.State[ArrayLike] = sl.State.from_pytree(alt_result.params)

        # Null hypothesis: POI = 0 (background-only)
        fixed_null: sl.State[float] = sl.State.from_pytree({poi_key: 0.0})
        null_result = constrained_fit(
            nll_fn, params, observation, fixed_null, **fit_kwargs
        )
        params_null: sl.State[ArrayLike] = sl.State.from_pytree(null_result.params)

        # Generate toys under alternative hypothesis
        q_alt = self._run_toys(
            nll_fn,
            params_alt,
            params,
            poi_key,
            poi_test,
            sample_fn,
            keys_alt,
            fit_kwargs,
        )

        # Generate toys under null hypothesis
        q_null = self._run_toys(
            nll_fn,
            params_null,
            params,
            poi_key,
            poi_test,
            sample_fn,
            keys_null,
            fit_kwargs,
        )

        return ToyResult(q_alt=q_alt, q_null=q_null)

    def _run_toys(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        sample_params: sl.State,
        fit_params: sl.State,
        poi_key: sl.K,
        poi_test: float,
        sample_fn: tp.Callable[[sl.State, PRNGKeyArray], PyTree],
        keys: PRNGKeyArray,
        fit_kwargs: dict[str, tp.Any],
    ) -> Array:
        """Run toys and return test statistic values.

        Uses jax.vmap for parallel computation across toys.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            sample_params: Parameters to use for sampling (State).
            fit_params: Parameters to use for fitting (State).
            poi_key: Canonical key for the POI.
            poi_test: Test value for POI.
            sample_fn: Sampling function.
            keys: Array of PRNG keys, one per toy.
            fit_kwargs: Additional fit arguments.

        Returns:
            Array of test statistic values, shape (ntoys,).
        """

        def single_toy(key: PRNGKeyArray) -> Array:
            # Generate toy observation
            toy_observation = sample_fn(sample_params, key)
            # Compute test statistic using toy as observation
            result = self.test_statistic(
                nll_fn, fit_params, toy_observation, poi_key, poi_test, **fit_kwargs
            )
            return result.value

        # Run toys in parallel using vmap
        return jax.vmap(single_toy)(keys)

    @staticmethod
    def _make_poisson_sampler(
        predict_fn: tp.Callable[[sl.State], PyTree],
    ) -> tp.Callable[[sl.State, PRNGKeyArray], PyTree]:
        """Create a Poisson sampler from a prediction function.

        Args:
            predict_fn: Function returning expected observation given parameters.

        Returns:
            Sampling function that generates Poisson-distributed observations.
        """

        def sample_fn(params_state: sl.State, key: PRNGKeyArray) -> PyTree:
            expected = predict_fn(params_state)
            leaves, treedef = jax.tree_util.tree_flatten(expected)
            subkeys = jax.random.split(key, len(leaves))
            keys_tree = jax.tree_util.tree_unflatten(treedef, subkeys)
            return jax.tree.map(jax.random.poisson, keys_tree, expected)

        return sample_fn
