"""Internal utilities for hypothesis testing."""

from __future__ import annotations

import typing as tp

import jax.numpy as jnp
from jaxtyping import Array, PyTree

import everwillow as ew
import everwillow.statelib as sl


def make_asimov(
    predict_fn: tp.Callable[[sl.State], PyTree],
    params: sl.State,
    poi_key: sl.K,
    mu_asimov: float,
) -> PyTree:
    """Generate an Asimov dataset at a given POI value.

    Sets the POI to ``mu_asimov`` in the parameter state and calls
    ``predict_fn`` to produce the expected observation.

    Args:
        predict_fn: Function mapping parameter state to expected observation.
        params: Parameter state (used as template).
        poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
        mu_asimov: POI value at which to generate the Asimov dataset.

    Returns:
        Expected observation (Asimov dataset).
    """
    asimov_params = sl.update(params, updates={poi_key: mu_asimov})
    return predict_fn(asimov_params)


def cl_s(palt: Array, pnull: Array) -> Array:
    """Compute CLs = palt / pnull.

    The CLs method protects against excluding signal
    hypotheses when there is no sensitivity.

    Args:
        palt: p-value under alternative hypothesis (signal+background).
        pnull: p-value under null hypothesis (background-only).

    Returns:
        CLs value. Protected against division by zero.
    """
    return palt / jnp.maximum(pnull, 1e-10)


def constrained_fit(
    nll_fn: tp.Callable[[PyTree, PyTree], float],
    params: sl.State,
    observation: PyTree,
    fixed: sl.State,
    **fit_kwargs: tp.Any,
) -> ew.FitResult:
    """Perform constrained fit, handling the case where all params are fixed.

    When the POI is the only parameter and it's being fixed, there are no
    free parameters to optimize. In this case, we simply evaluate the NLL
    at the fixed point rather than running the optimizer.

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Initial parameter state.
        observation: Observed data passed to nll_fn.
        fixed: State specifying which parameters to fix and their values.
        **fit_kwargs: Additional arguments passed to fit().

    Returns:
        FitResult with fitted parameters and NLL value.
    """
    # Check if fixing these params leaves any free parameters
    free_keys = set(params.mapping.keys()) - set(fixed.mapping.keys())

    if len(free_keys) == 0:
        # All parameters are fixed - just evaluate NLL
        updated_params = sl.update(params, updates=fixed)
        nll_value = jnp.asarray(nll_fn(updated_params.to_pytree(), observation))
        return ew.FitResult(
            params=updated_params.to_pytree(),
            nll=nll_value,
            success=jnp.asarray(True),
            solver_result=None,
        )

    return ew.fit(nll_fn, params, observation, fixed=fixed, **fit_kwargs)
