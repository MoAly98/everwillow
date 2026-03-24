"""Toy generation for hypothesis testing.

This module provides ToyGenerator for Monte Carlo-based hypothesis testing.
It generates toys under both hypotheses and returns a ToyResult.
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

import everwillow._src.statelib as sl
from everwillow._src.inference.hypotest.results import ToyResult
from everwillow._src.inference.hypotest.test_statistics import TestStatistic
from everwillow._src.inference.hypotest.utils import constrained_fit

__all__ = ["ToyGenerator"]


class ToyGenerator(eqx.Module):
    """Generates toy experiments for hypothesis testing.

    Creates toys under the tested hypothesis (poi_test) and optionally
    under an alternative hypothesis (poi_alt). Returns a ToyResult
    with raw test statistic arrays that can be fed into any
    EmpiricalDistribution subclass for p-value computation.

    Attributes:
        test_statistic: Test statistic to compute for each toy.
        ntoys: Number of toys per hypothesis. Defaults to 1000.
        map_fn: Function that maps a scalar function over an array of keys.
            Defaults to ``jax.vmap``. Replace with e.g.
            ``partial(jax.lax.map, batch_size=8)`` for batched sequential
            execution, or a Python loop for debugging.

    Example:
        >>> toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=10000)
        >>> toys = toy_gen.generate(
        ...     nll_fn, params, observed, "mu", 1.0,
        ...     poi_alt=0.0,
        ...     key=jax.random.key(42),
        ...     predict_fn=my_predict_fn,
        ... )
        >>> # Choose how to interpret the toys (open-world)
        >>> dist = SimpleEmpiricalDistribution.from_toys(toys)
        >>> # Use with HypoTestCalculator
        >>> calc = HypoTestCalculator(
        ...     nll_fn=nll_fn,
        ...     params=params,
        ...     observation=observed,
        ...     poi_key="mu",
        ...     test_statistic=QTilde(),
        ...     distribution=dist,
        ... )
        >>> result = calc.test(1.0)
    """

    test_statistic: TestStatistic
    ntoys: int = 1000
    map_fn: tp.Callable = eqx.field(default=jax.vmap, static=True)

    def generate(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_key: sl.K,
        poi_test: float,
        *,
        poi_alt: float | None = None,
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
            poi_key: Canonical key for the parameter of interest, e.g. "mu".
            poi_test: Test value for the POI (the tested hypothesis).
            poi_alt: Alternative hypothesis POI value. If provided, toys are
                generated under both hypotheses. If None, only null (tested)
                toys are generated and q_alt will be None in the result.
                For exclusion tests, typically 0.0. For discovery, typically 1.0.
            key: JAX PRNG key for reproducibility.
            sample_fn: Function to generate toy data. Called as
                sample_fn(params_state, key) -> toy_observation. If None,
                a default Poisson sampler is created using predict_fn.
            predict_fn: Function returning expected observation given parameters.
                Used to create default Poisson sampler if sample_fn is None.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            ToyResult with q_null (always) and q_alt (if poi_alt provided).

        Raises:
            ValueError: If neither sample_fn nor predict_fn is provided.
        """
        # Create default Poisson sampler if sample_fn not provided
        if sample_fn is None:
            if predict_fn is None:
                raise ValueError("Either sample_fn or predict_fn must be provided")
            sample_fn = self._make_poisson_sampler(predict_fn)

        # Null hypothesis (tested): POI = poi_test
        fixed_null: sl.State[float] = sl.State.from_pytree({poi_key: poi_test})
        null_result = constrained_fit(
            nll_fn, params, observation, fixed_null, **fit_kwargs
        )
        params_null: sl.State[ArrayLike] = null_result.params

        # Alternative hypothesis: POI = poi_alt (only if provided)
        q_alt = None
        if poi_alt is not None:
            keys = jax.random.split(key, self.ntoys * 2)
            keys_null = keys[: self.ntoys]
            keys_alt = keys[self.ntoys :]

            fixed_alt: sl.State[float] = sl.State.from_pytree({poi_key: poi_alt})
            alt_result = constrained_fit(
                nll_fn, params, observation, fixed_alt, **fit_kwargs
            )
            params_alt: sl.State[ArrayLike] = alt_result.params

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
        else:
            keys_null = jax.random.split(key, self.ntoys)

        # Generate toys under null (tested) hypothesis
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

        return ToyResult(q_null=q_null, q_alt=q_alt)

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

        Uses ``self.map_fn`` to map across toys.

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

        return self.map_fn(single_toy)(keys)

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
