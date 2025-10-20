"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
import optimistix as optx

import everwillow.parameters.bounds as bounds_module
import everwillow.statelib as sl
from everwillow.inference.utils import _prepare_fixed_param_state, _resolve_keys


@dataclass(frozen=True)
class FitResult:
    """Result of a fit operation."""

    params: tp.Any  #: Fitted parameter pytree (same structure as input).
    nll: float  #: Negative log-likelihood at the optimum.
    success: bool  #: Whether the optimisation converged.
    solver_result: tp.Any = None  #: Raw solver result (optional).


def _minimize_fast(wrapped_nll, solver, y0, **kwargs):
    """Fast minimization without side effects (can be JIT compiled by user)."""
    return optx.minimise(wrapped_nll, solver, y0=y0, **kwargs)


def _minimize_interactive(wrapped_nll, solver, y0, **kwargs):
    """Interactive minimization with callbacks and checkpointing (cannot be JIT compiled)."""
    # TODO: Implement interactive minimization features
    # For now, just delegate to fast minimization
    return optx.minimise(wrapped_nll, solver, y0=y0, **kwargs)


def _fit(
    nll_fn: tp.Callable[..., float],
    params: tp.Any,
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], bounds_module.TransformSpec] | None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None,
    solver: optx.AbstractMinimiser | None,
    args: tuple,
    kwargs: dict | None,
    interactive: bool,
    **solver_kwargs,
) -> FitResult:
    """Internal fitting implementation shared by fit() and ifit().

    Handles parameter state management, bounds transformations, fixed parameter
    partitioning, and delegates to mode-specific minimization.

    Args:
        nll_fn: Callable returning the scalar NLL.
        params: Initial parameter values as a pytree.
        fixed: Optional parameters to hold fixed (names or key tuples).
        bounds: Optional parameter bounds specification.
        fixed_predicate: Optional callable to identify additional fixed parameters.
        solver: Optimizer instance (defaults to BFGS if None).
        args: Positional arguments forwarded to ``nll_fn``.
        kwargs: Keyword arguments forwarded to ``nll_fn``.
        interactive: If True, use interactive minimization with side effects.
        **solver_kwargs: Additional arguments forwarded to minimizer.

    Returns:
        FitResult containing fitted parameters and diagnostics.
    """
    # Convert to FlatState for manipulation
    param_state = sl.FlatState.from_pytree(params)

    resolved_bounds: dict[str | tuple[tp.Any, ...], bounds_module.TransformSpec] = (
        {} if bounds is None else dict(bounds)
    )

    (
        unbounded_param_state,
        _unwrap_transforms,
        wrap_transforms,
    ) = bounds_module.apply_bounds_transform(param_state, resolved_bounds)

    # Identify fixed parameters
    fixed_keys = _resolve_keys(unbounded_param_state, fixed) if fixed else set()
    if fixed_predicate is not None:
        fixed_keys |= {
            key for key, value in unbounded_param_state.raw_mapping.items()
            if fixed_predicate(key, value)
        }

    if fixed_keys:
        fixed_state, free_state = sl.partition_state(
            unbounded_param_state, keys=fixed_keys
        )
    else:
        fixed_state = None
        free_state = unbounded_param_state

    # Get keys for free parameters in consistent order
    free_keys = list(free_state.raw_mapping.keys())

    # Handle kwargs default
    kwargs = kwargs or {}

    # Wrap nll to only take free parameters (as flat array)
    def wrapped_nll(free_values, _args):
        # Update free state with new values (in unbounded space)
        updates = dict(zip(free_keys, free_values, strict=True))
        updated_free = sl.update_state(free_state, updates)

        # Combine partitions back together (still in unbounded space)
        full_unbounded_state = (
            updated_free
            if fixed_state is None
            else sl.combine_partitions(fixed_state, updated_free)
        )

        # Transform back to bounded space for NLL evaluation
        if wrap_transforms:
            full_bounded_state = sl.apply_transformations(
                full_unbounded_state, wrap_transforms
            )
        else:
            full_bounded_state = full_unbounded_state

        # Convert to pytree for user's nll_fn
        full_pytree = full_bounded_state.to_pytree()

        # Call user's nll_fn with params + additional args/kwargs
        return nll_fn(full_pytree, *args, **kwargs)

    # Set up solver
    if solver is None:
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    # Initial values for free parameters (in same order as free_keys)
    y0 = jnp.array([free_state[k] for k in free_keys])

    # Delegate to mode-specific minimization
    if interactive:
        solution = _minimize_interactive(
            wrapped_nll, solver, y0,
            **solver_kwargs
        )
    else:
        solution = _minimize_fast(wrapped_nll, solver, y0, **solver_kwargs)

    # Reconstruct fitted parameters (in unbounded space)
    fitted_updates = dict(zip(free_keys, solution.value, strict=True))
    fitted_free = sl.update_state(free_state, fitted_updates)

    # Combine with fixed partition if one existed (still in unbounded space)
    fitted_full_unbounded_state = (
        fitted_free
        if fixed_state is None
        else sl.combine_partitions(fixed_state, fitted_free)
    )

    # Transform back to bounded space for final result
    if wrap_transforms:
        fitted_full_state = sl.apply_transformations(
            fitted_full_unbounded_state, wrap_transforms
        )
    else:
        fitted_full_state = fitted_full_unbounded_state

    # Convert back to original pytree structure
    fitted_pytree = fitted_full_state.to_pytree()

    # Get the final NLL value by evaluating wrapped_nll at the solution
    # (wrapped_nll already handles transformation internally)
    final_nll = wrapped_nll(solution.value, None)

    return FitResult(
        params=fitted_pytree,
        nll=float(final_nll),
        success=True,  # TODO: Check convergence
        solver_result=solution,
    )


def fit(
    nll_fn: tp.Callable[..., float],
    params: tp.Any,
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None = None,
    *,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], bounds_module.TransformSpec]
    | None = None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """Perform an unconditional maximum-likelihood fit.

    The negative log-likelihood (NLL) provided via ``nll_fn`` is minimised with
    respect to all parameters except those explicitly marked as fixed. Internally
    the parameter pytree is converted into a :class:`~everwillow.statelib.state.FlatState`
    so that subsets of the state can be frozen using
    :func:`everwillow.statelib.state.partition_state`.

    Parameter bounds are supported through automatic transformation to unbounded space.

    Args:
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument, followed by any positional or keyword
            arguments supplied via ``args``/``kwargs``.
        params: Initial parameter values organised as a pytree (e.g. mapping or
            nested containers).
        fixed: Optional sequence of leaf names (``str``) or canonical key tuples
            identifying parameters that should remain unchanged during the fit.
        bounds: Optional mapping from parameter names (``str``) or canonical key
            tuples to :class:`~everwillow.parameters.transforms.AbstractParameterTransformation`
            instances. When provided, parameters are unwrapped via the transform's
            ``unwrap`` method prior to optimisation and wrapped back afterwards.
        fixed_predicate: Optional callable invoked for each ``(key, value)`` pair
            in the flattened state. Returning ``True`` marks the parameter as
            fixed. This is evaluated in addition to ``fixed`` when provided.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        args: Positional arguments forwarded to ``nll_fn`` after the parameter
            pytree.
        kwargs: Keyword arguments forwarded to ``nll_fn``.
        ``**solver_kwargs``: Additional keyword arguments forwarded to
            :func:`optimistix.minimise`.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Examples:
        >>> # Simple case: nll_fn takes only params
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2 + (params["sigma"] - 1)**2
        >>> result = fit(my_nll, {"mu": 0.0, "sigma": 0.5})
        >>> result.params["mu"]  # Should be close to 2.0

        >>> # With additional arguments
        >>> def my_nll(params, data, templates, *, config):
        ...     return compute_loss(params, data, templates, config)
        >>> result = fit(my_nll, initial_params, args=(data, templates), kwargs={"config": cfg})

        >>> # Fix background while fitting mu and sigma
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> result = fit(
        ...     my_nll,
        ...     {"mu": 0.0, "sigma": 0.5, "background": 50.0},
        ...     fixed=["background"],
        ... )
        >>> result.params["background"]  # Remains fixed
        50.0

        >>> # With parameter bounds
        >>> def my_nll(params):
        ...     return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2
        >>> from everwillow.parameters.transforms import MinuitTransform
        >>> result = fit(
        ...     my_nll,
        ...     {"mu": 0.5, "sigma": 0.1},
        ...     bounds={"mu": MinuitTransform(lower=0.0, upper=5.0)},
        ... )
        >>> 0.0 <= result.params["mu"] <= 5.0  # Respects bounds
        True
    """
    return _fit(
        nll_fn,
        params,
        fixed,
        bounds,
        fixed_predicate,
        solver,
        args,
        kwargs,
        interactive=False,
        **solver_kwargs,
    )


def ifit(
    nll_fn: tp.Callable[..., float],
    params: tp.Any,
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None = None,
    *,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], bounds_module.TransformSpec]
    | None = None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """Perform an interactive maximum-likelihood fit with monitoring.

    Similar to :func:`fit` but supports callbacks, checkpointing, and progress
    monitoring. This function **cannot be JIT compiled** due to side effects (I/O).

    The negative log-likelihood (NLL) provided via ``nll_fn`` is minimised with
    respect to all parameters except those explicitly marked as fixed. Parameter
    state management and bounds handling are identical to :func:`fit`.

    Args:
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument, followed by any positional or keyword
            arguments supplied via ``args``/``kwargs``.
        params: Initial parameter values organised as a pytree (e.g. mapping or
            nested containers).
        fixed: Optional sequence of leaf names (``str``) or canonical key tuples
            identifying parameters that should remain unchanged during the fit.
        bounds: Optional mapping from parameter names (``str``) or canonical key
            tuples to :class:`~everwillow.parameters.transforms.AbstractParameterTransformation`
            instances. When provided, parameters are unwrapped via the transform's
            ``unwrap`` method prior to optimisation and wrapped back afterwards.
        fixed_predicate: Optional callable invoked for each ``(key, value)`` pair
            in the flattened state. Returning ``True`` marks the parameter as
            fixed. This is evaluated in addition to ``fixed`` when provided.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        args: Positional arguments forwarded to ``nll_fn`` after the parameter
            pytree.
        kwargs: Keyword arguments forwarded to ``nll_fn``.
        ``**solver_kwargs``: Additional keyword arguments forwarded to minimizer.
            Interactive-specific options like ``callback``, ``checkpoint_dir``, etc.
            can be passed here.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Note:
        This function is designed for long-running fits where you need progress
        updates or want to save intermediate results. For fast fits that can be
        JIT compiled by the user, use :func:`fit` instead.

    See Also:
        :func:`fit`: Fitting function that can be JIT compiled.

    Examples:
        >>> # Interactive fit (same interface as fit)
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2 + (params["sigma"] - 1)**2
        >>> result = ifit(my_nll, {"mu": 0.0, "sigma": 0.5})
        >>> result.params["mu"]  # Should be close to 2.0

        >>> # With callback for monitoring (future feature)
        >>> def progress_callback(step, params, nll):
        ...     print(f"Step {step}: NLL = {nll:.4f}")
        >>> result = ifit(my_nll, initial_params, callback=progress_callback)
    """
    return _fit(
        nll_fn,
        params,
        fixed,
        bounds,
        fixed_predicate,
        solver,
        args,
        kwargs,
        interactive=True,
        **solver_kwargs,
    )


def fixed_param_fit(
    param_values: dict[str, float],
    nll_fn: tp.Callable[..., float],
    params: tp.Any,
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None = None,
    *,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], bounds_module.TransformSpec]
    | None = None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """
    Perform a profile-likelihood style fit with selected parameters frozen.

    ``param_values`` supplies concrete values for one or more leaves in the
    parameter pytree. These leaves remain fixed while the remaining parameters
    are optimised. Additional parameters can be frozen via ``fixed`` or
    ``fixed_predicate``. Internally this routine relies on
    :func:`everwillow.statelib.state.partition_state` to separate the fixed and
    free portions of the state.

    Args:
        param_values: Mapping from parameter names to the values that should be
            injected prior to optimisation.
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument, followed by any positional or keyword
            arguments supplied via ``args``/``kwargs``.
        params: Initial parameter values organised as a pytree.
        fixed: Additional parameters to hold fixed, expressed as leaf names or
            canonical key tuples.
        bounds: Optional parameter bounds specification (same format as :func:`fit`).
        fixed_predicate: Optional callable evaluated for each leaf after
            ``param_values`` have been applied. Returning ``True`` marks the leaf
            as fixed.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        args: Positional arguments forwarded to ``nll_fn`` after the parameter
            pytree.
        kwargs: Keyword arguments forwarded to ``nll_fn``.
        ``**solver_kwargs``: Additional keyword arguments forwarded to minimizer.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Examples:
        >>> # Simple case
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2
        >>> # Fix mu=1.5 and fit everything else
        >>> result = fixed_param_fit(
        ...     {"mu": 1.5},
        ...     my_nll,
        ...     {"mu": 0.0, "data": 100},
        ...     fixed=["data"],
        ... )
        >>> result.params["mu"]  # Will be 1.5 (fixed)

        >>> # With additional arguments
        >>> def my_nll(params, data, *, verbose):
        ...     return compute_loss(params, data, verbose=verbose)
        >>> result = fixed_param_fit({"mu": 1.5}, my_nll, initial_params,
        ...                          args=(data,), kwargs={"verbose": True})
    """
    # Prepare parameter state with injected values and identify all fixed parameters
    updated_pytree, fixed_sequence = _prepare_fixed_param_state(
        params, param_values, fixed, fixed_predicate
    )

    return _fit(
        nll_fn,
        updated_pytree,
        fixed_sequence,
        bounds,
        fixed_predicate,
        solver,
        args,
        kwargs,
        interactive=False,
        **solver_kwargs,
    )


def ifixed_param_fit(
    param_values: dict[str, float],
    nll_fn: tp.Callable[..., float],
    params: tp.Any,
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None = None,
    *,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], bounds_module.TransformSpec]
    | None = None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """
    Interactive profile-likelihood fit with monitoring (cannot be JIT compiled).

    Similar to :func:`fixed_param_fit` but supports callbacks, checkpointing,
    and progress monitoring. This function **cannot be JIT compiled** due to
    side effects (I/O).

    ``param_values`` supplies concrete values for one or more leaves in the
    parameter pytree. These leaves remain fixed while the remaining parameters
    are optimised. Additional parameters can be frozen via ``fixed`` or
    ``fixed_predicate``.

    Args:
        param_values: Mapping from parameter names to the values that should be
            injected prior to optimisation.
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument, followed by any positional or keyword
            arguments supplied via ``args``/``kwargs``.
        params: Initial parameter values organised as a pytree.
        fixed: Additional parameters to hold fixed, expressed as leaf names or
            canonical key tuples.
        bounds: Optional parameter bounds specification (same format as :func:`fit`).
        fixed_predicate: Optional callable evaluated for each leaf after
            ``param_values`` have been applied. Returning ``True`` marks the leaf
            as fixed.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        args: Positional arguments forwarded to ``nll_fn`` after the parameter
            pytree.
        kwargs: Keyword arguments forwarded to ``nll_fn``.
        ``**solver_kwargs``: Additional keyword arguments forwarded to minimizer.
            Interactive-specific options like ``callback``, ``checkpoint_dir``, etc.
            can be passed here.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Note:
        This function is designed for long-running profile scans where you need
        progress updates. For fast fits that can be JIT compiled, use
        :func:`fixed_param_fit` instead.

    See Also:
        :func:`fixed_param_fit`: Fast version that can be JIT compiled.
        :func:`ifit`: Interactive unconditional fit.

    Examples:
        >>> # Interactive fixed parameter fit
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2
        >>> result = ifixed_param_fit(
        ...     {"mu": 1.5},
        ...     my_nll,
        ...     {"mu": 0.0, "data": 100},
        ...     fixed=["data"],
        ... )
        >>> result.params["mu"]  # Will be 1.5 (fixed)
    """
    # Prepare parameter state with injected values and identify all fixed parameters
    updated_pytree, fixed_sequence = _prepare_fixed_param_state(
        params, param_values, fixed, fixed_predicate
    )

    return _fit(
        nll_fn,
        updated_pytree,
        fixed_sequence,
        bounds,
        fixed_predicate,
        solver,
        args,
        kwargs,
        interactive=True,
        **solver_kwargs,
    )
