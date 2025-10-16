"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import jax.numpy as jnp
import optimistix as optx

from everwillow.state import ParamState, partition_state, update_state


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
    # Convert to ParamState for manipulation
    param_state = ParamState.from_pytree(params)

    # Determine fixed keys
    if fixed is None:
        fixed_keys = set()
    else:
        fixed_keys = set()
        for name in fixed:
            for key in param_state.keys():
                if key[-1] == name or (len(key) == 1 and key[0] == name):
                    fixed_keys.add(key)

    # Partition into free and fixed
    free_state, fixed_state, original_treedef = partition_state(param_state, fixed_keys)

    # Store the original key order for reconstruction
    original_keys = list(param_state.keys())
    free_keys = list(free_state.keys())

    # Handle kwargs default
    if kwargs is None:
        kwargs = {}

    # Wrap nll to only take free parameters (as flat array)
    def wrapped_nll(free_values, _args):
        # Reconstruct free_state from flat array
        free_mapping = dict(zip(free_keys, free_values, strict=True))
        free_state_updated = ParamState._new(free_mapping)

        # Merge back in original order
        full_mapping = {}
        for key in original_keys:
            if key in free_state_updated._mapping:
                full_mapping[key] = free_state_updated._mapping[key]
            else:
                full_mapping[key] = fixed_state._mapping[key]

        # Create full state with original treedef
        full_state = ParamState._new(full_mapping, treedef=original_treedef)

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

    # Reconstruct fitted parameters in original order
    fitted_free_mapping = dict(zip(free_keys, solution.value, strict=True))

    full_mapping = {}
    for key in original_keys:
        if key in fitted_free_mapping:
            full_mapping[key] = fitted_free_mapping[key]
        else:
            full_mapping[key] = fixed_state[key]

    fitted_full_state = ParamState._new(full_mapping, treedef=original_treedef)

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
    # Convert to ParamState
    param_state = ParamState.from_pytree(params)

    # Update state with fixed values
    updates = {}
    for name, value in param_values.items():
        # Find matching key
        for key in param_state.keys():
            if key[-1] == name or (len(key) == 1 and key[0] == name):
                updates[key] = value
                break

    updated_state = update_state(param_state, updates)

    # Combine fixed lists
    if fixed is None:
        fixed = []
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
