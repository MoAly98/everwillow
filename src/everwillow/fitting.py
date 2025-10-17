"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
import optimistix as optx

import everwillow.statelib as sl

@dataclass(frozen=True)
class FitResult:
    """
    Result of a fit operation.

    Attributes:
        params: Fitted parameter pytree (same structure as input)
        nll: Negative log-likelihood at minimum
        success: Whether optimization converged
        solver_result: Raw solver result (optional)
    """

    params: tp.Any  # pytree
    nll: float
    success: bool
    solver_result: tp.Any = None


def _canonical_fixed_keys(
    state: sl.FlatState[tp.Any],
    fixed: list[str] | None,
) -> set[sl.KeyPath]:
    """Resolve user-supplied parameter names to canonical FlatState keys."""

    if not fixed:
        return set()

    requested = set(fixed)
    resolved: dict[str, list[sl.KeyPath]] = {}
    for key in state.raw_mapping.keys():
        if not key:
            continue
        name = key[-1]
        if name in requested:
            resolved.setdefault(name, []).append(key)

    missing = requested - resolved.keys()
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Fixed parameter(s) not found in parameter state: {missing_list}")

    canonical_keys: set[sl.KeyPath] = set()
    for name, keys in resolved.items():
        canonical_keys.update(keys)

    return canonical_keys


def fit(
    nll_fn: tp.Callable[..., float],
    params: tp.Any,  # pytree
    fixed: list[str] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """
    Unconditional maximum likelihood fit.

    Minimizes negative log-likelihood with respect to all parameters
    except those specified as fixed.

    Args:
        nll_fn: Negative log-likelihood function. First argument must be parameter pytree,
                followed by any additional positional/keyword arguments.
        params: Initial parameter values (pytree, e.g. dict, nested dict, etc.)
        fixed: List of parameter names to hold fixed during fit
        solver: Optimistix solver (default: BFGS)
        args: Additional positional arguments to pass to nll_fn after params
        kwargs: Additional keyword arguments to pass to nll_fn
        **solver_kwargs: Additional kwargs for solver

    Returns:
        FitResult with fitted parameters and diagnostics

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
    """
    # Convert to FlatState for manipulation
    param_state = sl.FlatState.from_pytree(params)

    fixed_keys = _canonical_fixed_keys(param_state, fixed)

    if fixed_keys:
        fixed_state, free_state = sl.partition_state(param_state, keys=fixed_keys)
    else:
        fixed_state = None
        free_state = param_state

    # Get keys for free parameters in consistent order
    free_keys = list(free_state.raw_mapping.keys())

    # Handle kwargs default
    kwargs = kwargs or {}

    # Wrap nll to only take free parameters (as flat array)
    def wrapped_nll(free_values, _args):
        # Update free state with new values
        updates = dict(zip(free_keys, free_values, strict=True))
        updated_free = sl.update_state(free_state, updates)

        # Combine partitions back together
        full_state = (
            updated_free
            if fixed_state is None
            else sl.combine_partitions(fixed_state, updated_free)
        )

        # Convert to pytree for user's nll_fn
        full_pytree = full_state.to_pytree()

        # Call user's nll_fn with params + additional args/kwargs
        return nll_fn(full_pytree, *args, **kwargs)

    # Set up solver
    if solver is None:
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    # Initial values for free parameters (in same order as free_keys)
    y0 = jnp.array([free_state[k] for k in free_keys])

    # Minimize
    solution = optx.minimise(wrapped_nll, solver, y0=y0, **solver_kwargs)

    # Reconstruct fitted parameters
    fitted_updates = dict(zip(free_keys, solution.value, strict=True))
    fitted_free = sl.update_state(free_state, fitted_updates)

    # Combine with fixed partition if one existed
    fitted_full_state = (
        fitted_free
        if fixed_state is None
        else sl.combine_partitions(fixed_state, fitted_free)
    )

    # Convert back to original pytree structure
    fitted_pytree = fitted_full_state.to_pytree()

    # Get the final NLL value by evaluating wrapped_nll at the solution
    final_nll = wrapped_nll(solution.value, None)

    return FitResult(
        params=fitted_pytree,
        nll=float(final_nll),
        success=True,  # TODO: Check convergence
        solver_result=solution,
    )


def fixed_param_fit(
    param_values: dict[str, float],
    nll_fn: tp.Callable[..., float],
    params: tp.Any,  # pytree
    fixed: list[str] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    **solver_kwargs,
) -> FitResult:
    """
    Maximum likelihood fit with specific parameters fixed to given values.

    This is the building block for profile likelihood scans. It performs
    a fit with one or more parameters held fixed at specified values.

    Args:
        param_values: Dict of {param_name: value} to fix
        nll_fn: Negative log-likelihood function. First argument must be parameter pytree,
                followed by any additional positional/keyword arguments.
        params: Initial parameter values (pytree)
        fixed: Additional parameters to hold fixed (beyond param_values)
        solver: Optimistix solver (default: BFGS)
        args: Additional positional arguments to pass to nll_fn after params
        kwargs: Additional keyword arguments to pass to nll_fn
        **solver_kwargs: Additional kwargs for solver

    Returns:
        FitResult with fitted parameters

    Examples:
        >>> # Simple case
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2
        >>> # Fix mu=1.5 and fit everything else
        >>> result = fixed_param_fit({"mu": 1.5}, my_nll, {"mu": 0.0, "data": 100}, fixed=["data"])
        >>> result.params["mu"]  # Will be 1.5 (fixed)

        >>> # With additional arguments
        >>> def my_nll(params, data, *, verbose):
        ...     return compute_loss(params, data, verbose=verbose)
        >>> result = fixed_param_fit({"mu": 1.5}, my_nll, initial_params,
        ...                          args=(data,), kwargs={"verbose": True})
    """
    # Convert to FlatState for manipulation
    param_state = sl.FlatState.from_pytree(params)

    # Resolve canonical keys and apply the fixed values
    fixed_keys = _canonical_fixed_keys(param_state, list(param_values.keys()))

    updates = {}
    for key in fixed_keys:
        name = key[-1]
        # param_values contains the desired fixed value for each trailing name
        updates[key] = param_values[name]

    updated_state = sl.update_state(param_state, updates)

    # Combine fixed lists
    fixed = fixed or []
    fixed_combined = fixed + list(param_values.keys())

    # Convert back to pytree and call regular fit
    updated_pytree = updated_state.to_pytree()

    return fit(
        nll_fn,
        updated_pytree,
        fixed=fixed_combined,
        solver=solver,
        args=args,
        kwargs=kwargs,
        **solver_kwargs,
    )
