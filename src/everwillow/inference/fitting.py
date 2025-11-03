"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import typing as tp
from types import EllipsisType

import equinox as eqx
import jax
import optimistix as optx
from jaxtyping import PyTree

import everwillow.parameters as ewp
import everwillow.statelib as sl
from everwillow.statelib import K, V

Args: tp.TypeAlias = tuple[
    sl.PartitionedMapping[V],  # fixed_state
    sl.TreeDefMeta,  # treedefmeta
    sl.State[ewp.TransformBase],  # bounds
]


def _reconstruct_full_state(
    free_state: sl.PartitionedMapping[V],
    *,
    args: Args,
) -> sl.State[V]:
    """Reconstruct full parameter pytree from free state and Args."""
    (fixed_state, treedefmeta, bounds) = args

    # Combine partitions back together (still in unbounded space)
    combined_mapping = sl.combine_partitions(fixed_state, free_state)

    # Caution: using treedefmeta to preserve the original key order
    full_state_transformed = sl.State(combined_mapping, treedefmeta=treedefmeta)

    # Transform back to bounded space for NLL evaluation
    return ewp.wrap(full_state_transformed, bounds.mapping)


class FitResult(eqx.Module, tp.Generic[V]):
    """Result of a fit operation."""

    params: PyTree[V]  #: Fitted parameter state.
    nll: jax.Array  #: Negative log-likelihood at the optimum.
    success: jax.Array  #: Whether the optimisation converged.
    solver_result: PyTree  #: Raw solver result.


def fit(
    nll_fn: tp.Callable[[PyTree[V]], float],
    params: sl.State[V],
    *,
    fixed: sl.State[V | EllipsisType] | None = None,
    bounds: sl.State[ewp.TransformBase] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    **minimise_kwargs,
) -> FitResult[V]:
    """Perform an unconditional maximum-likelihood fit.

    The negative log-likelihood (NLL) provided via ``nll_fn`` is minimised with
    respect to all parameters except those explicitly marked as fixed. Internally
    the parameter pytree is converted into a :class:`~everwillow.statelib.state.State`
    so that subsets of the state can be frozen using
    :func:`everwillow.statelib.state.partition`.

    Parameter bounds are supported through automatic transformation to unbounded space.

    Args:
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument, followed by any positional or keyword
            arguments supplied via ``args``/``kwargs``.
        params: Initial parameter values organised as a state (e.g. mapping or
            nested containers).
        fixed: Optional state of canonicalized keys to fixed values for
            identifying parameters that should remain unchanged during the fit.
        bounds: Optional state of :class:`~everwillow.parameters.transforms.TransformBase`
            instances. When provided, parameters are unwrapped via the transform's
            ``unwrap`` method prior to optimisation and wrapped back afterwards.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        ``**minimise_kwargs``: Additional keyword arguments forwarded to
            :func:`optimistix.minimise`.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Examples:
        >>> import everwillow as ew
        >>> import everwillow.statelib as sl

        >>> # Basic usage
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2 + (params["sigma"] - 1)**2
        >>> initial_params = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        >>> result = ew.fit(my_nll, initial_params)
        >>> result.params["mu"]  # Should be close to 2.0

        >>> # Fix 'sigma' while fitting mu
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> fixed = sl.State.from_pytree({"sigma": ...})
        >>> result = ew.fit(
        ...     my_nll,
        ...     initial_params,
        ...     fixed=fixed,
        ... )
        >>> result.params["sigma"]  # Remains fixed
        0.5

        >>> # With parameter bounds
        >>> from everwillow.parameters.transforms import MinuitTransform
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> bounds = sl.State.from_pytree({"mu": MinuitTransform(lower=0.0, upper=5.0)})
        >>> result = ew.fit(
        ...     my_nll,
        ...     initial_params,
        ...     bounds=bounds,
        ... )
        >>> 0.0 <= result.params["mu"] <= 5.0  # Respects bounds
        True
    """
    # ensure args are properly typed
    if not isinstance(params, sl.State):
        raise TypeError("params must be a State")

    # normalize fixed and bounds inputs
    if fixed is None:
        fixed = sl.State.from_pytree({})
    if not isinstance(fixed, sl.State):
        raise TypeError("fixed must be a State or None")

    if bounds is None:
        bounds = sl.State.from_pytree({})
    if not isinstance(bounds, sl.State):
        raise TypeError("bounds must be a State or None")

    # Set fixed values
    updated_params = sl.update(params, updates=fixed)

    # Apply bounds transformations and get inverse transform map `wrap` for later
    param_state_transformed = ewp.unwrap(updated_params, transform_mapping=bounds)

    # Partition state into fixed and free components
    def predicate(key: K, value: V) -> bool:
        del value  # unused
        return key in fixed

    fixed_state, free_state = sl.partition(
        param_state_transformed,
        predicate=predicate,
    )

    # Prepare args for reconstructing full state
    args: Args = (fixed_state, params.treedefmeta, bounds)

    # Wrap nll to only take free parameters (as flat array)
    def wrapped_nll(new_state, args):
        full_state = _reconstruct_full_state(new_state, args=args)
        # Call user's nll_fn with params pytree
        return nll_fn(full_state.to_pytree())

    # Set up solver
    if solver is None:
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    # Minimize
    solution = optx.minimise(
        wrapped_nll,
        solver,
        y0=free_state,
        args=args,
        **minimise_kwargs,
    )

    # Reconstruct full fitted state
    fitted_state = _reconstruct_full_state(solution.value, args=args)

    # Return result
    return FitResult(
        params=fitted_state.to_pytree(),
        nll=solution.state.f_info.f,
        success=jax.numpy.asarray(solution.result == optx.RESULTS.successful),
        solver_result=solution,
    )
