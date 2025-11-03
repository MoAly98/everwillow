"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import typing as tp

import equinox as eqx
import jax
import optimistix as optx
from jaxtyping import PyTree

import everwillow.statelib as sl
from everwillow.parameters import AbstractParameterTransformation, unwrap, wrap
from everwillow.statelib import K, V


class FitResult(eqx.Module, tp.Generic[V]):
    """Result of a fit operation."""

    params: PyTree[V]  #: Fitted parameter pytree.
    nll: jax.Array  #: Negative log-likelihood at the optimum.
    success: jax.Array  #: Whether the optimisation converged.
    solver_result: PyTree  #: Raw solver result.


def fit(
    nll_fn: tp.Callable[[PyTree], float],
    params: PyTree[V],
    *,
    fixed: tp.Mapping[K, V] | None = None,
    bounds: tp.Mapping[K, AbstractParameterTransformation] | None = None,
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
        params: Initial parameter values organised as a pytree (e.g. mapping or
            nested containers).
        fixed: Optional mapping of canonicalized keys to fixed values for
            identifying parameters that should remain unchanged during the fit.
        bounds: Optional mapping from parameter names (``str``) or canonical key
            tuples to :class:`~everwillow.parameters.transforms.AbstractParameterTransformation`
            instances. When provided, parameters are unwrapped via the transform's
            ``unwrap`` method prior to optimisation and wrapped back afterwards.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        ``**minimise_kwargs``: Additional keyword arguments forwarded to
            :func:`optimistix.minimise`.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Examples:
        >>> # Simple case: nll_fn takes only params
        >>> import everwillow as ew
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2 + (params["sigma"] - 1)**2
        >>> result = ew.fit(my_nll, {"mu": 0.0, "sigma": 0.5})
        >>> result.params["mu"]  # Should be close to 2.0

        >>> # With additional arguments (partial)
        >>> from functools import partial
        >>>
        >>> def my_nll(params, data, templates, *, config):
        ...     return compute_loss(params, data, templates, config)
        >>> result = ew.fit(partial(my_nll, data, templates, config=cfg), initial_params)

        >>> # Fix background while fitting mu and sigma
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> result = ew.fit(
        ...     my_nll,
        ...     {"mu": 0.0, "sigma": 0.5, "background": 50.0},
        ...     fixed={"background": ...},
        ... )
        >>> result.params["background"]  # Remains fixed
        50.0

        >>> # With parameter bounds
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> from everwillow.parameters.transforms import MinuitTransform
        >>> result = ew.fit(
        ...     my_nll,
        ...     {"mu": 0.5, "sigma": 0.1},
        ...     bounds={"mu": MinuitTransform(lower=0.0, upper=5.0)},
        ... )
        >>> 0.0 <= result.params["mu"] <= 5.0  # Respects bounds
        True
    """
    # Convert to State for manipulation
    param_state: sl.State[V] = sl.State.from_pytree(params, sep="/")

    if fixed is None:
        fixed = {}

    # set fixed values
    param_state = sl.update(param_state, fixed)

    if bounds is None:
        bounds = {}

    # Apply bounds transformations and get inverse transform map `wrap` for later
    param_state_transformed = unwrap(param_state, bounds)

    # Partition state into fixed and free components
    fixed_state, free_state = sl.partition(
        param_state_transformed.mapping,
        predicate=lambda key, _value: key in fixed,
    )

    # Wrap nll to only take free parameters (as flat array)
    def wrapped_nll(new_state, args):
        (fixed_state,) = args

        # Combine partitions back together (still in unbounded space)
        combined_mapping = sl.combine_partitions(fixed_state, new_state)

        # Caution: using param_state.treedefmeta to preserve the original key order
        full_state_t = sl.State(combined_mapping, treedefmeta=param_state.treedefmeta)

        # Transform back to bounded space for NLL evaluation
        full_state = wrap(full_state_t, bounds)

        # Convert to pytree for user's nll_fn
        full_pytree = full_state.to_pytree()

        # Call user's nll_fn with params + additional args/kwargs
        return nll_fn(full_pytree)

    # Set up solver
    if solver is None:
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    # Minimize
    solution: tp.Any = optx.minimise(
        wrapped_nll,
        solver,
        y0=free_state,
        args=(fixed_state,),
        **minimise_kwargs,
    )

    # Combine with fixed partition if one existed (still in unbounded space)
    combined_solution = sl.combine_partitions(fixed_state, solution.value)
    fitted_full_state_t = sl.update(param_state_transformed, combined_solution)

    # Transform back to bounded space for final result
    fitted_full_state = wrap(fitted_full_state_t, bounds)

    # Convert back to original pytree structure
    fitted_params = fitted_full_state.to_pytree()

    # Return result
    return FitResult(
        params=fitted_params,
        nll=solution.state.f_info.f,
        success=jax.numpy.asarray(solution.result == optx.RESULTS.successful),
        solver_result=solution,
    )
