"""Upper limit finding via root search.

This module provides a generic root-finding function for computing upper limits.
It uses optimistix.root_find with Bisection for pure JAX, JIT-compatible operation.

The user provides an objective function that maps POI values to some quantity,
and upper_limit finds where that quantity equals a target level.

This is criterion-agnostic: the objective can compute CLs, p_alt, or any
other quantity. The user composes the objective function to implement
their desired exclusion criterion.

Note:
    The objective function must be JAX-traceable. Avoid calling float() or other
    Python operations that break JAX tracing. Return JAX arrays directly instead.

Example usage:
    >>> # CLs-based upper limit
    >>> def cls_criterion(poi):
    ...     result = hypotest(nll_fn, params, "mu", poi)
    ...     return cls(result.p_alt, result.p_null)  # Return JAX array
    >>> limit = upper_limit(cls_criterion, bounds=(0, 5), level=0.05)

    >>> # p_alt-based upper limit (frequentist)
    >>> def palt_criterion(poi):
    ...     result = hypotest(nll_fn, params, "mu", poi)
    ...     return result.p_alt  # Return JAX array
    >>> limit = upper_limit(palt_criterion, bounds=(0, 5), level=0.05)
"""

from __future__ import annotations

import typing as tp

import optimistix as optx


def upper_limit(
    objective_fn: tp.Callable[[float], float],
    bounds: tuple[float, float],
    level: float = 0.05,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 100,
) -> float:
    """Find POI value where objective function equals target level.

    Uses bisection root-finding to find where objective_fn(poi) = level.
    Pure JAX implementation via optimistix, fully JIT-compatible.

    This is a generic root finder - the user composes the objective function
    to implement their desired exclusion criterion (CLs, p_alt, etc.).

    Args:
        objective_fn: Function mapping POI value to quantity of interest.
                      Must be JAX-traceable (no float() calls on traced values).
                      Should be monotonic within bounds for reliable convergence.
        bounds: (lower, upper) search range for POI value.
        level: Target value for the objective function (default 0.05).
        rtol: Relative tolerance for convergence.
        atol: Absolute tolerance for convergence.
        max_steps: Maximum bisection iterations.

    Returns:
        POI value where objective_fn(poi) = level.

    Raises:
        ValueError: If root is not bracketed by bounds.

    Note:
        The objective function is JIT-compiled by optimistix. Avoid calling
        float() or other Python operations that break JAX tracing. Return
        JAX arrays directly instead.

    Examples:
        >>> # Find where CLs = 0.05 (95% CL upper limit)
        >>> def cls_criterion(poi):
        ...     result = hypotest(nll_fn, params, "mu", poi)
        ...     return cls(result.p_alt, result.p_null)  # Return JAX array directly
        >>> limit = upper_limit(cls_criterion, bounds=(0, 5), level=0.05)

        >>> # Find where p_alt = 0.05 (frequentist limit)
        >>> def palt_criterion(poi):
        ...     result = hypotest(nll_fn, params, "mu", poi)
        ...     return result.p_alt  # Return JAX array directly
        >>> limit = upper_limit(palt_criterion, bounds=(0, 5), level=0.05)
    """

    def root_objective(poi, _args):
        """Objective for root finding: f(poi) - level = 0."""
        return objective_fn(poi) - level

    solver = optx.Bisection(rtol=rtol, atol=atol)

    # Initial guess at midpoint
    y0 = (bounds[0] + bounds[1]) / 2.0

    solution = optx.root_find(
        root_objective,
        solver,
        y0,
        args=None,
        options={"lower": bounds[0], "upper": bounds[1]},
        max_steps=max_steps,
        throw=False,
    )

    return float(solution.value)
