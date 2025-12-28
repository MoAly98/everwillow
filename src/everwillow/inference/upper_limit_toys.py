"""Upper limit finding via toy-based root search.

This module provides a generic root-finding function for computing upper limits
using Monte Carlo toys. It mirrors the API of upper_limit.py but accepts a
PRNG key for stochastic objective functions.

The user provides an objective function that maps (poi, key) -> value,
and upper_limit_toys finds where that value equals the target level.

This is criterion-agnostic: the objective can compute CLs, p_alt, or any
other quantity. The user composes the objective function to implement
their desired exclusion criterion.

Note on bisection implementation:
    Currently uses a Python loop instead of optimistix.Bisection because:
    1. Each iteration needs a fresh PRNG key for new toys
    2. The toy-based hypotest contains Python loops (not fully vmap'd)

    In principle, this could be converted to optimistix by:
    - Passing iteration count as aux state and using fold_in for keys
    - Making hypotest_toys fully vmap-compatible

    For now, the Python loop is simpler and works correctly.
"""

from __future__ import annotations

import typing as tp

import jax
from jaxtyping import PRNGKeyArray


def upper_limit_toys(
    objective_fn: tp.Callable[[float, PRNGKeyArray], float],
    bounds: tuple[float, float],
    key: PRNGKeyArray,
    level: float = 0.05,
    *,
    tolerance: float = 0.02,
    max_iterations: int = 15,
) -> float:
    """Find POI value where objective function equals target level using toys.

    This is a generic root finder for stochastic objectives. The user composes
    the objective function to implement their desired exclusion criterion
    (CLs, p_alt, or any other quantity).

    Uses bisection search with the objective function evaluated at each point.
    Each iteration uses a fresh PRNG key derived from the base key.

    Args:
        objective_fn: Function mapping (poi, key) to quantity of interest.
                      Should be monotonic (typically decreasing) as POI increases.
                      Signature: (poi_value, prng_key) -> float
        bounds: (lower, upper) search range for POI value.
        key: JAX PRNG key for reproducibility.
        level: Target value for the objective function (default 0.05).
        tolerance: Stop when |objective - level| < tolerance (default 0.02).
        max_iterations: Maximum bisection iterations (default 15).

    Returns:
        POI value where objective_fn(poi, key) ≈ level.

    Note:
        The result has statistical uncertainty from the Monte Carlo sampling.
        The tolerance should account for this - setting it too tight may
        cause the search to use all iterations without converging.

    Examples:
        >>> # CLs-based upper limit with toys
        >>> def cls_objective(poi, key):
        ...     result = hypotest_toys(
        ...         nll_fn, params, "mu", poi,
        ...         nll_factory=nll_factory, sample_fn=sample_fn,
        ...         key=key, ntoys=500
        ...     )
        ...     return float(cls(result.p_alt, result.p_null))
        >>> limit = upper_limit_toys(cls_objective, bounds=(0, 5), key=key)

        >>> # p_alt-based upper limit (non-CLs frequentist)
        >>> def palt_objective(poi, key):
        ...     result = hypotest_toys(
        ...         nll_fn, params, "mu", poi,
        ...         nll_factory=nll_factory, sample_fn=sample_fn,
        ...         key=key, ntoys=500
        ...     )
        ...     return float(result.p_alt)
        >>> limit = upper_limit_toys(palt_objective, bounds=(0, 5), key=key)
    """
    lo, hi = bounds

    for iteration in range(max_iterations):
        mid = (lo + hi) / 2.0

        # Get fresh key for this iteration
        key_iter = jax.random.fold_in(key, iteration)

        # Evaluate objective at midpoint
        obj_mid = objective_fn(mid, key_iter)

        # Check convergence
        if abs(obj_mid - level) < tolerance:
            return mid

        # Bisection update
        # Objective typically decreases as POI increases (for exclusion)
        if obj_mid > level:
            lo = mid  # Need higher POI to get lower objective
        else:
            hi = mid  # Need lower POI to get higher objective

    # Return best estimate after max iterations
    return (lo + hi) / 2.0
