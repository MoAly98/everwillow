"""Test statistics for hypothesis testing.

Mathematical Background
-----------------------
The profile likelihood ratio test statistic is used to test a hypothesized
value of the parameter of interest (POI) against the best-fit value.

For a POI mu (signal strength) or c (Wilson coefficient), the test statistic is:

    q_mu = -2 * ln(L(mu, hat{theta}_mu) / L(hat{mu}, hat{theta}))
         = 2 * (NLL(mu, hat{theta}_mu) - NLL(hat{mu}, hat{theta}))

where:
    - hat{mu}, hat{theta}: Global best-fit (free fit)
    - hat{theta}_mu: Conditional best-fit with mu fixed (profiled nuisances)

The "tilde" variant (q_tilde) includes boundary handling:

    q_tilde_mu = q_mu   if hat{mu} <= mu  (deficit or match)
               = 0      if hat{mu} > mu   (excess, no exclusion power)

This ensures q_tilde >= 0 and provides one-sided exclusion limits.

Asymptotic Distribution
-----------------------
Under the asymptotic approximation (Wilks' theorem + Wald approximation):
    - sqrt(q_mu) ~ N(0, 1) under null hypothesis (mu = 0)
    - sqrt(q_mu) ~ N(sqrt(q_A), 1) under alternative (mu = mu')

where q_A is the "Asimov" test statistic computed on the Asimov dataset
(expected data under the null hypothesis with best-fit nuisances).

References:
    - Cowan et al., "Asymptotic formulae for likelihood-based tests"
      Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727
"""

from __future__ import annotations

import typing as tp

import jax.numpy as jnp
from jaxtyping import PyTree

import everwillow.statelib as sl
from everwillow.inference.fitting import FitResult, fit


def q_tilde(
    nll_fn: tp.Callable[[PyTree], float],
    params: sl.State,
    poi_name: str,
    poi_test: float,
    *,
    solver: tp.Any = None,
    max_steps: int = 256,
    **minimise_kwargs,
) -> tuple[jnp.ndarray, FitResult, FitResult]:
    """Profile likelihood ratio test statistic with boundary handling.

    Formula:
        q_tilde = 2 * (NLL(poi=test, nuisances=profiled) - NLL(poi=free, nuisances=free))

    with boundary condition:
        q_tilde = 0  if best_fit_poi > poi_test  (excess scenario)

    This implements the "tilde" variant that clips to zero when the observed
    data shows an excess over the tested hypothesis, providing one-sided
    exclusion limits.

    Profiling:
        All non-POI parameters (nuisance parameters) are profiled (minimized)
        in both the constrained and free fits. For EFT fits with multiple
        Wilson coefficients, fixing one coefficient profiles the others.

    Args:
        nll_fn: Negative log-likelihood function taking a parameter dict.
        params: Initial parameter state with all parameters.
        poi_name: Name of the parameter of interest (e.g., "mu", "c_tG").
        poi_test: Test value for the POI.
        solver: Optional solver for optimization. Defaults to BFGS.
        max_steps: Maximum optimization steps.
        **minimise_kwargs: Additional keyword arguments passed to optx.minimise.
            Common options:
            - throw (bool): If False, don't raise on non-convergence (default True).
              Use throw=False when using jax.vmap over multiple fits.

    Returns:
        Tuple of (q, constrained_fit, free_fit):
            - q: Test statistic value (non-negative float).
            - constrained_fit: FitResult with POI fixed at poi_test.
            - free_fit: FitResult with all parameters free.

    Note:
        When using throw=False, check FitResult.success to identify fits that
        did not converge. Non-converged fits may have inaccurate q values.

    Examples:
        >>> # Simple example with signal strength
        >>> q, fixed_fit, free_fit = q_tilde(nll_fn, params, "mu", poi_test=1.0)
        >>> print(f"q = {q:.4f}, best-fit mu = {free_fit.params['mu']:.4f}")

        >>> # EFT example with Wilson coefficient
        >>> q, fixed_fit, free_fit = q_tilde(nll_fn, params, "c_tG", poi_test=0.5)
    """
    # 1. Free fit: all parameters float → get best_fit POI and NLL
    free_fit = fit(
        nll_fn,
        params,
        solver=solver,
        max_steps=max_steps,
        **minimise_kwargs,
    )

    # 2. Constrained fit: POI fixed to poi_test, others profiled
    fixed_state: sl.State[tp.Any] = sl.State.from_pytree({poi_name: poi_test})
    constrained_fit = fit(
        nll_fn,
        params,
        fixed=fixed_state,
        solver=solver,
        max_steps=max_steps,
        **minimise_kwargs,
    )

    # 3. q = 2 * (NLL_constrained - NLL_free)
    delta_nll = constrained_fit.nll - free_fit.nll
    q_raw = 2.0 * delta_nll

    # 4. Boundary: q = 0 if best_fit > poi_test (observed excess)
    best_fit_poi = free_fit.params[poi_name]
    q = jnp.where(best_fit_poi <= poi_test, q_raw, 0.0)

    # Ensure non-negative (numerical safety)
    q = jnp.maximum(q, 0.0)

    return q, constrained_fit, free_fit
