"""Upper limit finding via root search.

This module provides generic root-finding functions for computing upper limits.
The functions are criterion-agnostic: the user provides an objective function
that maps POI values to some quantity, and the root finder locates where
that quantity equals a target level.

Both implementations are pure JAX and fully JIT-compatible:
    - upper_limit: Uses optimistix.Bisection for deterministic objectives
    - upper_limit_toys: Uses jax.lax.while_loop for stochastic (toy-based) objectives

Example usage:
    >>> # CLs-based upper limit (asymptotic)
    >>> def cls_objective(poi):
    ...     return calc.cls(calc.test(poi))
    >>> limit = upper_limit(cls_objective, bounds=(0, 5), level=0.05)

    >>> # CLs-based upper limit (toy-based)
    >>> def cls_objective_toys(poi, key):
    ...     result = toy_calc.test(poi, key=key)
    ...     return toy_calc.cls(result)
    >>> limit = upper_limit_toys(cls_objective_toys, bounds=(0, 5), key=key)
"""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, PRNGKeyArray

from everwillow._src.inference.hypotest.calculators import HypoTestCalculator
from everwillow._src.inference.hypotest.results import (
    BandValues,
    ExpectedBands,
    ExpectedLimitResult,
)
from everwillow._src.inference.hypotest.utils import cl_s

__all__ = [
    "expected_upper_limit",
    "upper_limit",
    "upper_limit_scan",
    "upper_limit_toys",
]


def upper_limit(
    objective_fn: tp.Callable[[float], Array],
    bounds: tuple[float, float],
    level: float = 0.05,
    *,
    solver: optx.AbstractRootFinder | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    max_steps: int = 100,
) -> Array:
    """Find POI value where objective function equals target level.

    Uses root-finding to find where objective_fn(poi) = level.
    Pure JAX implementation via optimistix, fully JIT-compatible.

    This is a generic root finder - the user composes the objective function
    to implement their desired exclusion criterion (CLs, p_alt, etc.).

    Args:
        objective_fn: Function mapping POI value to quantity of interest.
                      Must be JAX-traceable (no float() calls on traced values).
                      Should be monotonic within bounds for reliable convergence.
        bounds: (lower, upper) search range for POI value.
        level: Target value for the objective function (default 0.05).
        solver: Root-finding solver to use. Defaults to optx.Bisection.
        rtol: Relative tolerance for convergence (used by default solver).
        atol: Absolute tolerance for convergence (used by default solver).
        max_steps: Maximum iterations.

    Returns:
        POI value where objective_fn(poi) = level.

    Note:
        The objective function is JIT-compiled by optimistix. Avoid calling
        float() or other Python operations that break JAX tracing.

    Examples:
        >>> # Find where CLs = 0.05 (95% CL upper limit)
        >>> def cls_objective(poi):
        ...     return calc.cls(calc.test(poi))
        >>> limit = upper_limit(cls_objective, bounds=(0, 5), level=0.05)

        >>> # With custom solver
        >>> solver = optx.Newton(rtol=1e-5, atol=1e-5)
        >>> limit = upper_limit(cls_objective, bounds=(0, 5), solver=solver)
    """

    def root_objective(poi, _args):
        """Objective for root finding: f(poi) - level = 0."""
        return objective_fn(poi) - level

    if solver is None:
        solver = optx.Bisection(rtol=rtol, atol=atol)  # type: ignore[call-arg]

    # Initial guess at midpoint
    y0 = jnp.array((bounds[0] + bounds[1]) / 2.0)

    solution: optx.Solution = optx.root_find(
        root_objective,
        solver,
        y0,
        args=None,
        options={"lower": bounds[0], "upper": bounds[1]},
        max_steps=max_steps,
        throw=False,
    )

    return solution.value


def upper_limit_toys(
    objective_fn: tp.Callable[[float, PRNGKeyArray], Array],
    bounds: tuple[float, float],
    key: PRNGKeyArray,
    level: float = 0.05,
    *,
    tol: float = 1e-2,
    max_iterations: int = 100,
) -> Array:
    """Find POI value where stochastic objective equals target level.

    Uses bisection search with fresh PRNG keys at each iteration.
    Pure JAX implementation via jax.lax.while_loop, fully JIT-compatible.

    Args:
        objective_fn: Function mapping (poi, key) to quantity of interest.
                      Should be monotonic (typically decreasing) as POI increases.
                      Must be JAX-traceable (no float() calls on traced values).
        bounds: (lower, upper) search range for POI value.
        key: JAX PRNG key for reproducibility.
        level: Target value for the objective function (default 0.05).
        tol: Stop when objective is within tol of level (default 1e-2).
        max_iterations: Maximum bisection iterations (default 100).

    Returns:
        POI value where objective_fn(poi, key) ≈ level.

    Note:
        The result has statistical uncertainty from Monte Carlo sampling.
        The tolerance should account for this.

    Examples:
        >>> # CLs-based upper limit with toys
        >>> def cls_objective(poi, key):
        ...     result = toy_calc(
        ...         nll_fn, params, ("mu",), poi,
        ...         sample_fn=sample_fn, nll_factory=nll_factory, key=key
        ...     )
        ...     return result.cl_s
        >>> limit = upper_limit_toys(cls_objective, bounds=(0, 5), key=key)
    """

    def cond_fn(state):
        iteration, _lo, _hi, converged = state
        return (iteration < max_iterations) & ~converged

    def body_fn(state):
        iteration, lo, hi, _ = state
        mid = (lo + hi) / 2.0

        # Fresh key for this iteration
        key_iter = jax.random.fold_in(key, iteration)

        # Evaluate objective at midpoint
        obj_mid = objective_fn(mid, key_iter)

        # Check convergence (objective close to target level)
        converged = jnp.abs(obj_mid - level) < tol

        # Bisection: objective typically decreases as POI increases
        new_lo = jnp.where(obj_mid > level, mid, lo)
        new_hi = jnp.where(obj_mid > level, hi, mid)

        return (iteration + 1, new_lo, new_hi, converged)

    # Initial state: (iteration, lo, hi, converged)
    lo0 = jnp.asarray(bounds[0])
    hi0 = jnp.asarray(bounds[1])
    init_state = (0, lo0, hi0, jnp.array(False))

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, lo, hi, _ = final_state

    result = (lo + hi) / 2.0
    at_lower = jnp.isclose(result, lo0, rtol=tol)
    at_upper = jnp.isclose(result, hi0, rtol=tol)
    result = eqx.error_if(
        result,
        at_lower | at_upper,
        "upper_limit_toys: root not found within bounds. "
        "The limit is at the search boundary, suggesting the bounds are too narrow.",
    )
    return result  # noqa: RET504


def upper_limit_scan(
    objective_fn: tp.Callable[[float], Array],
    scan: Array,
    level: float = 0.05,
) -> Array:
    """Find POI value where objective equals target level via grid scan.

    Evaluates the objective function on a grid of POI values, then
    interpolates to find where it crosses the target level.
    Fully JIT-compatible via jax.vmap and jnp.interp.

    This is useful when:
    - The objective function is expensive and you want to reuse evaluations
    - You need to visualize the objective curve
    - Root-finding fails due to non-monotonicity

    Args:
        objective_fn: Function mapping POI value to quantity of interest.
                      Must be JAX-traceable.
        scan: Array of POI values to evaluate. Should be monotonically
              increasing and span the expected limit location.
        level: Target value for the objective function (default 0.05).

    Returns:
        POI value where objective_fn(poi) = level, found by interpolation.

    Note:
        The accuracy depends on the density of scan points near the crossing.
        For CLs limits, the objective typically decreases as POI increases.

    Examples:
        >>> # Scan CLs on a grid
        >>> scan = jnp.linspace(0, 2, 50)
        >>> limit = upper_limit_scan(
        ...     lambda poi: calc.cls(calc.test(poi)),
        ...     scan,
        ...     level=0.05,
        ... )

        >>> # With finer grid near expected limit
        >>> scan = jnp.concatenate([
        ...     jnp.linspace(0, 0.5, 10),
        ...     jnp.linspace(0.5, 1.5, 30),
        ...     jnp.linspace(1.5, 3, 10),
        ... ])
        >>> limit = upper_limit_scan(cls_objective, scan, level=0.05)
    """
    # Evaluate objective on the scan grid
    values = jax.vmap(objective_fn)(scan)

    # Interpolate to find crossing point
    # For CLs: objective decreases as POI increases, so reverse for interp
    # jnp.interp expects xp to be increasing, so we reverse both arrays
    result = jnp.interp(level, values[::-1], scan[::-1])
    at_lower = jnp.isclose(result, scan[0])
    at_upper = jnp.isclose(result, scan[-1])
    result = eqx.error_if(
        result,
        at_lower | at_upper,
        "upper_limit_scan: root not found within scan range. "
        "The limit is at the scan boundary, suggesting the scan range is too narrow.",
    )
    return result  # noqa: RET504


def expected_upper_limit(
    calc: HypoTestCalculator,
    bounds: tuple[float, float],
    level: float = 0.05,
    objective_fn: tp.Callable[[Array, Array], Array] | None = None,
    **solver_kwargs: tp.Any,
) -> ExpectedLimitResult:
    """Compute observed and expected upper limits with Brazil bands.

    Uses the calculator's expected bands to compute limits at
    standard sigma fluctuations (-2σ, -1σ, median, +1σ, +2σ).

    Args:
        calc: Hypothesis test calculator (stores nll_fn, params, etc.).
        bounds: (lower, upper) search range for POI value.
        level: Target level (default 0.05 for 95% CL).
        objective_fn: Function mapping (pnull, palt) to scalar objective.
                      Defaults to CLs = pnull / palt.
        **solver_kwargs: Additional arguments passed to :func:`upper_limit`
            (e.g., ``rtol``, ``atol``, ``max_steps``).

    Returns:
        ExpectedLimitResult with observed and expected limits at all bands.

    Example:
        >>> calc = AsymptoticCalculator(
        ...     nll_fn=nll_fn, params=params, observation=observed,
        ...     poi_key=("mu",), predict_fn=my_predict_fn,
        ... )
        >>> result = expected_upper_limit(calc, bounds=(0, 5))
        >>> print(f"Observed: {result.observed:.3f}")
        >>> print(f"Expected: {result.expected.median:.3f}")
    """

    if objective_fn is None:
        objective_fn = cl_s

    # Observed limit
    def _observed_objective(poi: float) -> Array:
        r = calc.test(poi)
        return objective_fn(r.pnull, r.palt)

    observed = upper_limit(
        _observed_objective,
        bounds,
        level,
        **solver_kwargs,
    )

    # Expected limits at each sigma band
    def _get_bands(poi: Array) -> ExpectedBands:
        result = calc.test(poi)
        bands = calc.expected(result)
        if bands is None:
            msg = "Distribution does not support expected p-values"
            raise ValueError(msg)
        return bands

    def _band_objective(band: str) -> tp.Callable[[float], Array]:
        def fn(poi: float) -> Array:
            bands = _get_bands(poi)
            return objective_fn(bands.null_pvalue[band], bands.alt_pvalue[band])

        return fn

    band_limits = {}
    for band_name in BandValues._NAMES:
        band_limits[band_name] = upper_limit(
            _band_objective(band_name),
            bounds,
            level,
            **solver_kwargs,
        )
    return ExpectedLimitResult(
        observed=observed,
        expected=BandValues(**band_limits),
    )
