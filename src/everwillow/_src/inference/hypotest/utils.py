"""Utilities for hypothesis testing."""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

import everwillow._src.statelib as sl
from everwillow._src.inference.fitting import FitResult, fit

__all__ = [
    "cl_s",
    "constrained_fit",
    "make_asimov",
    "sigma_from_asimov",
    "significance",
]


def make_asimov(
    predict_fn: tp.Callable[[PyTree], PyTree],
    params: PyTree,
    poi_key: sl.K,
    mu_asimov: float,
) -> PyTree:
    """Generate an Asimov dataset at a given POI value.

    Sets the POI to ``mu_asimov`` in the parameter pytree and calls
    ``predict_fn`` to produce the expected observation.

    Args:
        predict_fn: Function mapping parameters to expected observation.
        params: Parameter pytree (used as template).
        poi_key: Canonical key for the parameter of interest, e.g. ("mu",).
        mu_asimov: POI value at which to generate the Asimov dataset.

    Returns:
        Expected observation (Asimov dataset).
    """
    params = sl.State.from_pytree(params)
    asimov_params = sl.update(params, updates={poi_key: mu_asimov})
    return predict_fn(asimov_params.to_pytree())


def sigma_from_asimov(mu: Array, q_asimov: Array, mu_asimov: float = 0.0) -> Array:
    r"""Extract :math:`\sigma` (uncertainty on :math:`\hat{\mu}`) from an Asimov test statistic.

    Uses the relation :math:`t_{\mu,A} \approx (\mu - \mu')^2/\sigma^2` to solve for :math:`\sigma`.

    Args:
        mu: POI value being tested.
        q_asimov: Test statistic evaluated on Asimov data.
        mu_asimov: POI value used to generate the Asimov dataset.
            Defaults to 0.0 (background-only, for exclusion tests).

    Returns:
        Estimated :math:`\sigma = |\mu - \mu_\text{asimov}| / \sqrt{q_\text{asimov}}`.
    """
    return jnp.abs(mu - mu_asimov) / jnp.sqrt(jnp.maximum(q_asimov, 1e-10))


def significance(p: Array) -> Array:
    r"""Convert p-value to significance: :math:`Z = \Phi^{-1}(1 - p)`.

    Args:
        p: p-value (scalar or array).

    Returns:
        Significance Z.
    """
    return -jax.scipy.stats.norm.ppf(p)


def cl_s(pnull: Array, palt: Array) -> Array:
    r"""Compute :math:`\text{CL}_s = p_\text{null} / p_\text{alt}`.

    :math:`\text{CL}_s = P(q \geq q_\text{obs} \mid \text{signal+background}) / P(q \geq q_\text{obs} \mid \text{background})`

    The CLs method protects against excluding signal
    hypotheses when there is no sensitivity: if palt is small
    (background also finds data unlikely), CLs stays large.

    Args:
        pnull: p-value under null hypothesis (:math:`\mu' = \mu`, signal+background).
        palt: p-value under alternative hypothesis (:math:`\mu' = 0`, background-only).

    Returns:
        CLs value. Protected against division by zero.
    """
    return pnull / jnp.maximum(palt, 1e-10)


def constrained_fit(
    nll_fn: tp.Callable[[PyTree, PyTree], float],
    params: PyTree,
    observation: PyTree,
    fixed: PyTree,
    **fit_kwargs: tp.Any,
) -> FitResult:
    """Perform constrained fit, handling the case where all params are fixed.

    When the POI is the only parameter and it's being fixed, there are no
    free parameters to optimize. In this case, we simply evaluate the NLL
    at the fixed point rather than running the optimizer.

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Initial parameter pytree.
        observation: Observed data passed to nll_fn.
        fixed: Pytree specifying which parameters to fix and their values.
        **fit_kwargs: Additional arguments passed to fit().

    Returns:
        FitResult with fitted parameters and NLL value.
    """
    params = sl.State.from_pytree(params)
    fixed = sl.State.from_pytree(fixed)

    # Check if fixing these params leaves any free parameters
    free_keys = set(params.mapping.keys()) - set(fixed.mapping.keys())

    if len(free_keys) == 0:
        # All parameters are fixed - just evaluate NLL
        updated_params = sl.update(params, updates=fixed)
        nll_value = jnp.asarray(nll_fn(updated_params.to_pytree(), observation))
        return FitResult(
            params=updated_params.to_pytree(),
            nll=nll_value,
            success=jnp.asarray(True),
            solver_result=None,
        )

    return fit(nll_fn, params, observation, fixed=fixed, **fit_kwargs)
