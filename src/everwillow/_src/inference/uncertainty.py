"""Parameter uncertainty estimation from Hessian-based covariance.

This module provides functions for computing parameter uncertainties using
the Fisher information matrix, which is obtained by inverting the Hessian
of the negative log-likelihood at the maximum likelihood estimate.

The key relationship is the Cramér-Rao bound: the covariance matrix of
unbiased estimators is bounded below by the inverse Fisher information.
At the MLE, under regularity conditions, this bound is saturated.
"""

from __future__ import annotations

import typing as tp
from types import EllipsisType

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

import everwillow._src.statelib as sl
from everwillow._src.statelib import V


def hessian_matrix(
    nll_fn: tp.Callable[[PyTree[V], PyTree], float],
    params: PyTree[V],
    observation: PyTree,
    *,
    fixed: PyTree[V | EllipsisType] | None = None,
) -> Float[Array, "n_free n_free"]:
    """Compute the Hessian matrix of the NLL at given parameters.

    The Hessian H_ij = ∂²NLL/∂θ_i∂θ_j is computed only for free (non-fixed)
    parameters using JAX automatic differentiation.

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Full parameter pytree at which to evaluate the Hessian.
        observation: Observed data passed to nll_fn.
        fixed: Parameters to treat as constants (excluded from Hessian).

    Returns:
        2D JAX array of shape (n_free, n_free).
    """
    params = sl.State.from_pytree(params)
    fixed = sl.State.from_pytree({}) if fixed is None else sl.State.from_pytree(fixed)

    # Split into fixed and free
    fixed_state, free_state = sl.partition(
        params, predicate=lambda key, _: key in fixed
    )

    # Get flat array of free values
    free_keys = tuple(free_state.notnone.keys())
    flat_values = jnp.array([free_state[k] for k in free_keys])

    def _flat_nll(flat_free: Float[Array, ...]) -> Float[Array, ""]:
        """Compute the negative log-likelihood for the flat parameter vector.
        Necessary for jax.hessian.

        Args:
            flat_free (Float[Array, "..."]): Flattened free parameter values.

        Returns:
            Float[Array, ""]: Negative log-likelihood value.
        """
        free_mapping = {k: flat_free[i] for i, k in enumerate(free_keys)}
        new_free = sl.update(free_state, updates=free_mapping)
        combined = sl.combine_partitions(fixed_state, new_free)
        return nll_fn(combined.to_pytree(), observation)

    return jax.hessian(_flat_nll)(flat_values)


def covariance_matrix(
    nll_fn: tp.Callable[[PyTree[V], PyTree], float],
    params: PyTree[V],
    observation: PyTree,
    *,
    fixed: PyTree[V | EllipsisType] | None = None,
) -> Float[Array, "nparams nparams"]:
    """Compute the covariance matrix (inverse Hessian) at given parameters.

    The Fisher information matrix is obtained by inverting the Hessian:
        Cov(θ) = H⁻¹ where H_ij = ∂²NLL/∂θ_i∂θ_j

    This is the Laplace approximation to the posterior covariance.

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Full parameter pytree (typically fitted values).
        observation: Observed data passed to nll_fn.
        fixed: Parameters to exclude from covariance computation.

    Returns:
        2D JAX array of shape (n_free, n_free).
    """
    hess = hessian_matrix(nll_fn, params, observation, fixed=fixed)
    # Invert Hessian to get Fisher information matrix (covariance)
    return jnp.linalg.inv(hess)


def correlation_matrix(
    nll_fn: tp.Callable[[PyTree[V], PyTree], float],
    params: PyTree[V],
    observation: PyTree,
    *,
    fixed: PyTree[V | EllipsisType] | None = None,
) -> jax.Array:
    """Compute the correlation matrix (normalized covariance).

    The covariance matrix is normalized so that diagonal entries are 1.0:
        ρ_ij = Cov_ij / √(Cov_ii · Cov_jj)

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Full parameter pytree at which to evaluate.
        observation: Observed data passed to nll_fn.
        fixed: Parameters to exclude from correlation computation.

    Returns:
        2D JAX array with diagonal = 1.0, off-diagonal in [-1, 1].
    """
    cov = covariance_matrix(nll_fn, params, observation, fixed=fixed)
    # Normalize: ρ_ij = Cov_ij / (σ_i · σ_j)
    d = jnp.sqrt(jnp.diag(cov))
    corr = cov / jnp.outer(d, d)
    # Ensure diagonal is exactly 1.0 (numerical stability)
    return jnp.where(jnp.eye(corr.shape[0], dtype=corr.dtype), 1.0, corr)


def uncertainties(
    nll_fn: tp.Callable[[PyTree[V], PyTree], float],
    params: PyTree[V],
    observation: PyTree,
    *,
    fixed: PyTree[V | EllipsisType] | None = None,
) -> PyTree[V]:
    """Extract parameter uncertainties as sqrt(diag(covariance)).

    The uncertainties are the square roots of the diagonal of the
    covariance matrix (inverse Fisher information), following the
    Cramér-Rao bound: σ_i = √(Cov_ii) = √((H⁻¹)_ii)

    Args:
        nll_fn: Negative log-likelihood function taking (params, observation).
        params: Full parameter pytree (typically fitted values).
        observation: Observed data passed to nll_fn.
        fixed: Parameters to exclude from uncertainty computation.

    Returns:
        Pytree containing uncertainty values for free parameters only.
    """
    params = sl.State.from_pytree(params)
    fixed = sl.State.from_pytree({}) if fixed is None else sl.State.from_pytree(fixed)

    # Get covariance matrix for free parameters
    cov = covariance_matrix(nll_fn, params, observation, fixed=fixed)
    # Cramér-Rao: uncertainties are sqrt of diagonal elements
    stderrs = jnp.sqrt(jnp.diag(cov))

    # Get free_state with same structure/ordering as used for Hessian
    fixed_state, free_state = sl.partition(
        params, predicate=lambda key, _: key in fixed
    )

    # Unflatten stderrs back into the same pytree structure as free_state
    _, treedef = jax.tree_util.tree_flatten(free_state)
    free_uncertainty = jax.tree_util.tree_unflatten(treedef, stderrs)

    fixed_uncertainty = jax.tree.map(lambda _: None, fixed_state)

    return sl.combine_partitions(fixed_uncertainty, free_uncertainty).to_pytree()
