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
from everwillow._src.inference.hypotest.test_statistics import PoiPoint, TestStatistic
from everwillow._src.inference.hypotest.utils import constrained_fit

__all__ = ["ToyGenerator"]


class ToyGenerator(eqx.Module):
    """Sampling engine generating toy experiments for hypothesis testing.

    Creates toys under the null hypothesis (poi_null) and optionally
    under an alternative hypothesis (poi_alt). Returns a ToyResult
    with raw test statistic arrays that can be fed into any
    EmpiricalDistribution subclass for p-value computation.

    Each toy dataset is drawn in one of two modes, chosen by which field
    is set (``sample_fn`` wins if both are):

    ``predict_fn`` — counting experiments (Poisson data)::

        expected_yields = predict_fn(params_state)
        toy_observation = poisson_sample(key, expected_yields)  # each yield fluctuated independently

    ``sample_fn`` — any other sampling scheme::

        toy_observation = sample_fn(params_state, key)  # user draws the full pseudo-dataset

    In both modes ``toy_observation`` has the same format as the observed
    data passed to the NLL.

    The generator holds only sampling configuration; the test statistic is
    supplied per ``generate()`` call, so when composed into a
    :class:`~everwillow._src.inference.hypotest.calculators.ToyCalculator`
    there is a single definition of it (on the calculator).

    Attributes:
        sample_fn: Draws one complete pseudo-dataset,
            ``sample_fn(params_state, key) -> toy_observation``. For
            example, a Gaussian measurement:
            ``lambda state, key: {"x": model(state) + sigma * jax.random.normal(key)}``.
        predict_fn: Model prediction returning the expected event yields,
            ``predict_fn(params_state) -> expected_observation``.
        ntoys: Number of toys per hypothesis. Defaults to 1000.
        map_fn: Function that maps a scalar function over an array of keys.
            Defaults to ``jax.vmap``. Replace with e.g.
            ``lambda fn: partial(jax.lax.map, fn, batch_size=8)`` to
            process toys in groups instead of all at once, or a Python
            loop for step-through debugging.

    Example:
        >>> toy_gen = ToyGenerator(predict_fn=my_predict_fn, ntoys=10000)
        >>> toys = toy_gen.generate(
        ...     nll_fn, params, observed, {"mu": 1.0},
        ...     test_statistic=QTilde(),
        ...     poi_alt={"mu": 0.0},
        ...     key=jax.random.key(42),
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

    sample_fn: tp.Callable[[sl.State, PRNGKeyArray], PyTree] | None = None
    predict_fn: tp.Callable[[sl.State], PyTree] | None = None
    ntoys: int = 1000
    map_fn: tp.Callable = eqx.field(default=jax.vmap, static=True)

    def generate(
        self,
        nll_fn: tp.Callable[[PyTree, PyTree], float],
        params: sl.State,
        observation: PyTree,
        poi_null: PoiPoint,
        *,
        test_statistic: TestStatistic,
        poi_alt: PoiPoint | None = None,
        key: PRNGKeyArray,
        **fit_kwargs: tp.Any,
    ) -> ToyResult:
        """Generate toys and return raw test statistic arrays.

        Args:
            nll_fn: Negative log-likelihood function taking (params, observation).
            params: Initial parameter state.
            observation: Observed data (used to profile nuisance parameters).
            poi_null: Null hypothesis POI point, a mapping from POI key to value
                (e.g. ``{"mu": 1.0}``). Toys generated under this hypothesis
                populate q_null, and the test statistic is evaluated at this
                point for each toy.
            test_statistic: Test statistic to compute for each toy.
            poi_alt: Alternative hypothesis POI point. If provided, toys are
                generated under both hypotheses. If None, only null toys are
                generated and q_alt will be None in the result. For exclusion
                tests, typically ``{"mu": 0.0}``; for discovery, ``{"mu": 1.0}``.
            key: JAX PRNG key for reproducibility.
            **fit_kwargs: Additional arguments passed to fit().

        Returns:
            ToyResult with q_null (always) and q_alt (if poi_alt provided).

        Raises:
            ValueError: If neither the sample_fn nor predict_fn field is set.
        """
        # Create default Poisson sampler if sample_fn not provided
        sample_fn = self.sample_fn
        if sample_fn is None:
            if self.predict_fn is None:
                msg = "Either sample_fn or predict_fn must be provided"
                raise ValueError(msg)
            sample_fn = self._make_poisson_sampler(self.predict_fn)

        # Null hypothesis: POI = poi_null
        null_result = constrained_fit(nll_fn, params, observation, poi_null, **fit_kwargs)
        params_null: sl.State[ArrayLike] = null_result.params

        # Alternative hypothesis: POI = poi_alt (only if provided)
        q_alt = None
        if poi_alt is not None:
            keys = jax.random.split(key, self.ntoys * 2)
            keys_null = keys[: self.ntoys]
            keys_alt = keys[self.ntoys :]

            alt_result = constrained_fit(nll_fn, params, observation, poi_alt, **fit_kwargs)
            params_alt: sl.State[ArrayLike] = alt_result.params

            q_alt = self._run_toys(
                nll_fn,
                params_alt,
                params,
                poi_null,
                test_statistic,
                sample_fn,
                keys_alt,
                fit_kwargs,
            )
        else:
            keys_null = jax.random.split(key, self.ntoys)

        # Generate toys under null hypothesis
        q_null = self._run_toys(
            nll_fn,
            params_null,
            params,
            poi_null,
            test_statistic,
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
        poi_null: PoiPoint,
        test_statistic: TestStatistic,
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
            poi_null: Null hypothesis POI point (mapping from POI key to value).
            test_statistic: Test statistic to compute for each toy.
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
            result = test_statistic.compute(nll_fn, fit_params, toy_observation, poi_null, **fit_kwargs)
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
