"""Toy-based hypothesis testing using Monte Carlo simulation.

This module provides hypothesis testing via toy Monte Carlo sampling,
as an alternative to asymptotic approximations. It uses JAX's vmap
for efficient parallel evaluation across all toys.

Mathematical Background
-----------------------
The p-value is estimated empirically as the fraction of toys with
test statistics at least as extreme as observed:

    p = (1/N) * sum(q_toy >= q_obs)

where N is the number of toys and q_toy is the test statistic for each toy.

For CLs calculation:
    - p_alt: Toys generated under alternative hypothesis (poi = poi_test)
    - p_null: Toys generated under null hypothesis (poi = 0)
    - CLs = p_alt / p_null

Statistical uncertainty on p-values scales as ~1/sqrt(N), so more toys
give more precise estimates.

References:
    - Cowan et al., "Asymptotic formulae for likelihood-based tests"
      Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
    - pyhf ToyCalculator
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

import everwillow.statelib as sl
from everwillow.inference import fitting
from everwillow.inference.test_statistics import q_tilde


@dataclass
class HypoTestToysResult:
    """Result of toy-based hypothesis test.

    Attributes:
        p_alt: P-value under alternative hypothesis (poi = poi_test).
        p_null: P-value under null hypothesis (poi = 0).
        q_obs: Observed test statistic.
        ntoys: Number of toys used.
        q_alt: Test statistics for all alternative hypothesis toys.
        q_null: Test statistics for all null hypothesis toys.

    To compute CLs, use the cls() function:
        >>> from everwillow.inference.calculators import cls
        >>> cls_obs = cls(result.p_alt, result.p_null)

    Note:
        Some toy optimizations may not fully converge (when throw=False is used
        internally). The q values for those toys are still included but may be
        approximate. With sufficient toys, this should not significantly affect
        the p-value estimates.
    """

    p_alt: float | Float[Array, ""]
    p_null: float | Float[Array, ""]
    q_obs: float | Float[Array, ""]
    ntoys: int
    q_alt: Float[Array, " "]  # Test statistics for each alternative toy
    q_null: Float[Array, " "]  # Test statistics for each null toy


def hypotest_toys(
    nll_fn: tp.Callable[[PyTree], float],
    params: sl.State,
    poi_name: str,
    poi_test: float,
    *,
    nll_factory: tp.Callable[[PyTree, tp.Any], float],
    sample_fn: tp.Callable[[dict, PRNGKeyArray], dict],
    key: PRNGKeyArray,
    ntoys: int = 500,
    null_value: float = 0.0,
    solver: tp.Any = None,
    max_steps: int = 256,
) -> HypoTestToysResult:
    """Toy-based hypothesis test using Monte Carlo sampling.

    Computes CLs using toy Monte Carlo instead of asymptotic approximations.
    Uses vmap for parallel evaluation across all toys.

    The workflow is:
        1. Compute q_obs on observed data
        2. Fit to get best-fit params under alternative (poi = poi_test)
        3. Fit to get best-fit params under null (poi = null_value)
        4. Generate toys under both hypotheses
        5. Compute test statistics for each toy
        6. Estimate p-values as fraction of toys with q >= q_obs

    Args:
        nll_fn: Negative log-likelihood function for observed data.
        params: Initial parameter state.
        poi_name: Name of parameter of interest (e.g., "mu", "c_tG").
        poi_test: Test value for the POI.
        nll_factory: Factory function that creates NLL for any observation.
                     Signature: (params_dict, observation) -> nll_value
        sample_fn: Function to generate toy observations.
                   Signature: (params_dict, prng_key) -> observation
                   The observation format must match what nll_factory expects.
        key: JAX PRNG key for reproducibility.
        ntoys: Number of toy experiments (default 500).
        null_value: Null hypothesis value for POI (default 0.0).
        solver: Optional solver for optimization.
        max_steps: Maximum optimization steps.

    Returns:
        HypoTestToysResult with empirical p-values and CLs.

    Note:
        Statistical uncertainty on p-values is ~1/sqrt(ntoys). Use more toys
        for more precise estimates. 500 toys gives ~4% relative uncertainty.

    Examples:
        >>> def nll_factory(params, obs):
        ...     return poisson_nll(params, obs["counts"])
        >>> def sample_fn(params, key):
        ...     return {"counts": jax.random.poisson(key, params["rate"])}
        >>> result = hypotest_toys(
        ...     nll_fn, params, "mu", 1.0,
        ...     nll_factory=nll_factory, sample_fn=sample_fn,
        ...     key=jax.random.key(42), ntoys=500
        ... )
    """
    # 1. Compute q on observed data
    q_obs, _constrained_fit, _free_fit = q_tilde(
        nll_fn, params, poi_name, poi_test, solver=solver, max_steps=max_steps
    )

    # 2. Get best-fit params under alternative hypothesis (poi = poi_test)
    fixed_alt: sl.State[tp.Any] = sl.State.from_pytree({poi_name: poi_test})
    alt_result = fitting.fit(
        nll_fn,
        params,
        fixed=fixed_alt,
        solver=solver,
        max_steps=max_steps,
    )
    params_alt = alt_result.params  # Already a pytree dict

    # 3. Get best-fit params under null hypothesis (poi = null_value)
    fixed_null: sl.State[tp.Any] = sl.State.from_pytree({poi_name: null_value})
    null_result = fitting.fit(
        nll_fn,
        params,
        fixed=fixed_null,
        solver=solver,
        max_steps=max_steps,
    )
    params_null = null_result.params  # Already a pytree dict

    # 4. Generate PRNG keys for all toys
    keys = jax.random.split(key, ntoys * 2)
    keys_alt = keys[:ntoys]
    keys_null = keys[ntoys:]

    # 5. Generate toys under both hypotheses (vectorized)
    toys_alt = jax.vmap(sample_fn, in_axes=(None, 0))(params_alt, keys_alt)
    toys_null = jax.vmap(sample_fn, in_axes=(None, 0))(params_null, keys_null)

    # 6. Compute test statistic for each toy using proper q_tilde
    # Use throw=False so optimization failures don't break vmap
    def compute_q_for_toy(toy_obs):
        """Compute q_tilde for a single toy observation."""

        def toy_nll(p):
            return nll_factory(p, toy_obs)

        q, _, _ = q_tilde(
            toy_nll,
            params,
            poi_name,
            poi_test,
            solver=solver,
            max_steps=max_steps,
            throw=False,  # Don't raise on non-convergence
        )
        return q

    # Vectorize q computation across all toys
    q_alt = jax.vmap(compute_q_for_toy)(toys_alt)
    q_null = jax.vmap(compute_q_for_toy)(toys_null)

    # 7. Compute p-values as fraction of toys with q >= q_obs
    p_alt = jnp.mean(q_alt >= q_obs)
    p_null = jnp.mean(q_null >= q_obs)

    return HypoTestToysResult(
        p_alt=p_alt,
        p_null=p_null,
        q_obs=q_obs,
        ntoys=ntoys,
        q_alt=q_alt,
        q_null=q_null,
    )
