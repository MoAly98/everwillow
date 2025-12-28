"""Upper limit finding via toy-based hypothesis testing.

This module provides a function for computing upper limits using
Monte Carlo toys instead of asymptotic approximations.

The bisection search is performed in Python (not JIT-compiled) because
each iteration requires generating new toys with a fresh PRNG key.

Example usage:
    >>> limit = upper_limit_toys(
    ...     nll_fn, params, poi_name="mu",
    ...     nll_factory=nll_factory, sample_fn=sample_fn,
    ...     key=jax.random.key(42), bounds=(0, 5), ntoys=500
    ... )
"""

from __future__ import annotations

import typing as tp

import jax
from jaxtyping import PRNGKeyArray, PyTree

import everwillow.statelib as sl
from everwillow.inference.calculators import cls
from everwillow.inference.hypotest_toys import hypotest_toys


def upper_limit_toys(
    nll_fn: tp.Callable[[PyTree], float],
    params: sl.State,
    poi_name: str,
    *,
    nll_factory: tp.Callable[[PyTree, tp.Any], float],
    sample_fn: tp.Callable[[dict, PRNGKeyArray], dict],
    key: PRNGKeyArray,
    bounds: tuple[float, float] = (0.0, 10.0),
    level: float = 0.05,
    ntoys: int = 500,
    null_value: float = 0.0,
    tolerance: float = 0.02,
    max_iterations: int = 15,
    solver: tp.Any = None,
    max_steps: int = 256,
) -> float:
    """Find upper limit on POI using toy-based CLs.

    Uses bisection search with hypotest_toys at each point.
    Each iteration uses a fresh subset of the PRNG key for reproducibility.

    Args:
        nll_fn: Negative log-likelihood function for observed data.
        params: Initial parameter state.
        poi_name: Name of parameter of interest (e.g., "mu", "c_tG").
        nll_factory: Factory function that creates NLL for any observation.
                     Signature: (params_dict, observation) -> nll_value
        sample_fn: Function to generate toy observations.
                   Signature: (params_dict, prng_key) -> observation
        key: JAX PRNG key for reproducibility.
        bounds: (lower, upper) search range for POI value.
        level: Target CLs level (default 0.05 for 95% CL).
        ntoys: Number of toys per evaluation (default 500).
        null_value: Null hypothesis value for POI (default 0.0).
        tolerance: Stop when |CLs - level| < tolerance (default 0.02).
        max_iterations: Maximum bisection iterations (default 15).
        solver: Optional solver for optimization.
        max_steps: Maximum optimization steps.

    Returns:
        POI value where CLs ≈ level (the upper limit).

    Note:
        The result has statistical uncertainty from the Monte Carlo sampling.
        With ntoys toys, the CLs uncertainty is ~1/sqrt(ntoys).
        Increase ntoys for more precise limits.

        Unlike the asymptotic upper_limit, this function is NOT JIT-compiled
        because each bisection iteration needs a fresh PRNG key.

    Examples:
        >>> # Find 95% CL upper limit on signal strength
        >>> limit = upper_limit_toys(
        ...     nll_fn, params, "mu",
        ...     nll_factory=nll_factory, sample_fn=sample_fn,
        ...     key=jax.random.key(42), bounds=(0, 5), ntoys=500
        ... )

        >>> # Find limit on Wilson coefficient with custom tolerance
        >>> limit = upper_limit_toys(
        ...     nll_fn, params, "c_tG",
        ...     nll_factory=nll_factory, sample_fn=sample_fn,
        ...     key=key, bounds=(-2, 2), tolerance=0.01, ntoys=1000
        ... )
    """
    lo, hi = bounds

    for iteration in range(max_iterations):
        mid = (lo + hi) / 2.0

        # Get fresh key for this iteration
        key_iter = jax.random.fold_in(key, iteration)

        # Compute CLs at midpoint using toys
        result = hypotest_toys(
            nll_fn,
            params,
            poi_name,
            mid,
            nll_factory=nll_factory,
            sample_fn=sample_fn,
            key=key_iter,
            ntoys=ntoys,
            null_value=null_value,
            solver=solver,
            max_steps=max_steps,
        )
        cls_mid = float(cls(result.p_alt, result.p_null))

        # Check convergence
        if abs(cls_mid - level) < tolerance:
            return mid

        # Bisection update
        # CLs typically decreases as POI increases (for exclusion)
        if cls_mid > level:
            lo = mid  # Need higher POI to get lower CLs
        else:
            hi = mid  # Need lower POI to get higher CLs

    # Return best estimate after max iterations
    return (lo + hi) / 2.0
