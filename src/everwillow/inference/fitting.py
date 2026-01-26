"""Core fitting functionality for statistical inference."""

from __future__ import annotations

import contextlib
import typing as tp
from types import EllipsisType

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import PyTree

import everwillow.parameters as ewp
import everwillow.statelib as sl
from everwillow.statelib import K, V

Args: tp.TypeAlias = tuple[
    sl.State[V],  # fixed_state
    sl.State[ewp.TransformBase],  # bounds
]

# Solver state type varies by solver (e.g., _QuasiNewtonState for BFGS).
# Access NLL via state.f_info.f
SolverState = tp.TypeVar("SolverState")

# Callback signature for interactive fitting: (step_idx, y, state) -> None
# Matches the signature pattern used by solver.step and solver.terminate
Callback: tp.TypeAlias = tp.Callable[[int, sl.State, SolverState], None]


def _reconstruct_full_state(
    free_state: sl.State[V],
    *,
    args: Args,
) -> sl.State[V]:
    """Reconstruct full parameter pytree from free state and Args."""
    (fixed_state, bounds) = args

    # Combine partitions back together (still in unbounded space)
    full_state_transformed = sl.combine_partitions(fixed_state, free_state)

    # Transform back to bounded space for NLL evaluation
    return ewp.wrap(full_state_transformed, bounds.mapping)


class FitResult(eqx.Module, tp.Generic[V]):
    """Result of a fit operation."""

    params: PyTree[V]  #: Fitted parameter state.
    nll: jax.Array  #: Negative log-likelihood at the optimum.
    success: jax.Array  #: Whether the optimisation converged.
    solver_result: PyTree  #: Raw solver result.


def _minimize(
    wrapped_nll: tp.Callable[[sl.State[V], Args], float],
    solver: optx.AbstractMinimiser,
    y0: sl.State[V],
    args: Args,
    *,
    max_steps: int,
    **minimise_kwargs,
) -> optx.Solution:
    """Non-interactive minimization using optx.minimise."""
    return optx.minimise(
        wrapped_nll,
        solver,
        y0=y0,
        args=args,
        max_steps=max_steps,
        **minimise_kwargs,
    )


class _ProgressUpdater:
    """Progress bar updater that can adjust total on completion."""

    def __init__(self, pbar):
        self._pbar = pbar

    def update(self, step: int, nll_value: float) -> None:
        self._pbar.n = step
        self._pbar.set_postfix(NLL=f"{nll_value:.6f}")
        self._pbar.refresh()

    def finalize(self, final_step: int, nll_value: float) -> None:
        """Set total to actual steps taken and complete the bar."""
        self._pbar.total = final_step
        self._pbar.n = final_step
        self._pbar.set_postfix(NLL=f"{nll_value:.6f}")
        self._pbar.refresh()


@contextlib.contextmanager
def _make_progress_context(
    enabled: bool,
    max_steps: int,
) -> tp.Generator[_ProgressUpdater | None, None, None]:
    """Create progress bar context manager.

    Yields a _ProgressUpdater when enabled, else None.
    """
    if not enabled:
        yield None
        return

    from tqdm import tqdm

    pbar = tqdm(total=max_steps, desc="Minimizing", unit="step")

    try:
        yield _ProgressUpdater(pbar)
    finally:
        pbar.close()


def _iminimize(
    wrapped_nll: tp.Callable[[sl.State[V], Args], float],
    solver: optx.AbstractMinimiser,
    y0: sl.State[V],
    args: Args,
    *,
    max_steps: int,
    progress: bool,
    callback: Callback | None,
    solver_options: dict[str, tp.Any] | None = None,
) -> optx.Solution:
    """Interactive minimization with step-by-step iteration and progress bar."""
    # Convert y0 leaves to JAX arrays (required for solver.init which calls tree_full_like)
    y0 = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), y0)

    # Wrap nll to return (f, aux) tuple and ensure outputs are arrays
    def fn_with_aux(y, fn_args):
        result = wrapped_nll(y, fn_args)
        return jnp.asarray(result), None

    # Infer output structure for solver initialization via eval_shape
    f_struct = jax.eval_shape(lambda: fn_with_aux(y0, args)[0])
    aux_struct = None
    options = solver_options if solver_options is not None else {}
    tags: frozenset[object] = frozenset()

    # Initialize solver state
    state = solver.init(fn_with_aux, y0, args, options, f_struct, aux_struct, tags)
    y = y0
    aux = None
    done, result = solver.terminate(fn_with_aux, y, args, options, state, tags)

    # JIT the hot path for performance
    step = eqx.filter_jit(
        eqx.Partial(solver.step, fn=fn_with_aux, args=args, options=options, tags=tags)
    )
    terminate = eqx.filter_jit(
        eqx.Partial(
            solver.terminate, fn=fn_with_aux, args=args, options=options, tags=tags
        )
    )

    iteration = 0
    with _make_progress_context(progress, max_steps) as updater:
        while not done and iteration < max_steps:
            # Call user callback with current state (user can access state.f_info.f for NLL)
            if callback is not None:
                callback(iteration, y, state)

            # Perform one solver step
            y, state, aux = step(y=y, state=state)
            iteration += 1

            # Check termination
            done, result = terminate(y=y, state=state)

            # Update progress bar with current NLL
            if updater is not None:
                current_nll = float(state.f_info.f)
                updater.update(iteration, current_nll)

        # Final progress bar update - set total to actual steps taken
        if updater is not None:
            updater.finalize(iteration, float(state.f_info.f))

    # Postprocess
    y_final, aux_final, stats = solver.postprocess(
        fn_with_aux, y, aux, args, options, state, tags, result
    )

    # Include iteration info in stats
    stats["num_steps"] = jnp.asarray(iteration)
    stats["max_steps"] = max_steps

    return optx.Solution(
        value=y_final,
        result=result,
        aux=aux_final,
        stats=stats,
        state=state,
    )


def _fit(
    nll_fn: tp.Callable[[PyTree[V]], float],
    params: sl.State[V],
    *,
    fixed: sl.State[V | EllipsisType] | None = None,
    bounds: sl.State[ewp.TransformBase] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    interactive: bool = False,
    max_steps: int = 256,
    progress: bool = True,
    callback: Callback | None = None,
    solver_options: dict[str, tp.Any] | None = None,
    **minimise_kwargs,
) -> FitResult[V]:
    """Internal fit implementation that handles both interactive and non-interactive modes.

    This function performs shared setup (validation, partitioning, transforms)
    then dispatches to _minimize() or _iminimize() based on the `interactive` flag.
    """
    # Validate inputs
    if not isinstance(params, sl.State):
        raise TypeError("params must be a State")

    # Normalize fixed and bounds inputs
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

    # Apply bounds transformations (convert to unbounded space)
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
    args: Args = (fixed_state, bounds)

    # Wrap nll to only take free parameters
    def wrapped_nll(new_state, fn_args):
        full_state = _reconstruct_full_state(new_state, args=fn_args)
        return nll_fn(full_state.to_pytree())

    # Set up solver
    if solver is None:
        solver = optx.BFGS(rtol=1e-5, atol=1e-5)

    # Dispatch to appropriate backend
    if interactive:
        solution = _iminimize(
            wrapped_nll,
            solver,
            free_state,
            args,
            max_steps=max_steps,
            progress=progress,
            callback=callback,
            solver_options=solver_options,
        )
    else:
        solution = _minimize(
            wrapped_nll,
            solver,
            free_state,
            args,
            max_steps=max_steps,
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


def fit(
    nll_fn: tp.Callable[[PyTree[V]], float],
    params: sl.State[V],
    *,
    fixed: sl.State[V | EllipsisType] | None = None,
    bounds: sl.State[ewp.TransformBase] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    max_steps: int = 256,
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
        max_steps: Maximum number of optimization steps. Defaults to 256.
        **minimise_kwargs: Additional keyword arguments forwarded to
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
    return _fit(
        nll_fn,
        params,
        fixed=fixed,
        bounds=bounds,
        solver=solver,
        interactive=False,
        max_steps=max_steps,
        **minimise_kwargs,
    )


def ifit(
    nll_fn: tp.Callable[[PyTree[V]], float],
    params: sl.State[V],
    *,
    fixed: sl.State[V | EllipsisType] | None = None,
    bounds: sl.State[ewp.TransformBase] | None = None,
    solver: optx.AbstractMinimiser | None = None,
    max_steps: int = 256,
    progress: bool = True,
    callback: Callback | None = None,
    solver_options: dict[str, tp.Any] | None = None,
) -> FitResult[V]:
    """Perform an interactive maximum-likelihood fit with progress bar and callbacks.

    This function is similar to :func:`fit` but provides interactive features:
    a rich progress bar displayed during optimization, and an optional callback
    function invoked at each iteration.

    Note:
        This function is not JIT-compatible due to Python side effects in the
        iteration loop. Use :func:`fit` if you need JIT compilation.

    Args:
        nll_fn: Callable returning the scalar NLL. It must accept the parameter
            pytree as its first argument.
        params: Initial parameter values organised as a state.
        fixed: Optional state of canonicalized keys to fixed values for
            identifying parameters that should remain unchanged during the fit.
        bounds: Optional state of :class:`~everwillow.parameters.transforms.TransformBase`
            instances for parameter bounds.
        solver: :class:`optimistix.AbstractMinimiser` instance to use. Defaults to
            :class:`optimistix.BFGS`.
        max_steps: Maximum number of optimization steps. Defaults to 256.
        progress: Whether to display a rich progress bar. Defaults to True.
        callback: Optional function called each iteration with signature
            ``(step_idx: int, y: State, state: SolverState) -> None``.
            This matches the pattern used by ``solver.step`` and ``solver.terminate``.
            The NLL value can be accessed via ``state.f_info.f``.
        solver_options: Optional dict of solver-specific options passed to ``solver.init``.

    Returns:
        :class:`FitResult` containing the fitted parameters and diagnostics.

    Examples:
        >>> import everwillow as ew
        >>> import everwillow.statelib as sl

        >>> # Interactive fit with progress bar
        >>> def my_nll(params):
        ...     return (params["mu"] - 2)**2 + (params["sigma"] - 1)**2
        >>> initial_params = sl.State.from_pytree({"mu": 0.0, "sigma": 0.5})
        >>> result = ew.ifit(my_nll, initial_params)

        >>> # With custom callback to record history
        >>> history = []
        >>> def record_history(step, y, state):
        ...     history.append({"step": step, "nll": float(state.f_info.f)})
        >>> result = ew.ifit(my_nll, initial_params, callback=record_history)

        >>> # Disable progress bar
        >>> result = ew.ifit(my_nll, initial_params, progress=False)
    """
    return _fit(
        nll_fn,
        params,
        fixed=fixed,
        bounds=bounds,
        solver=solver,
        interactive=True,
        max_steps=max_steps,
        progress=progress,
        callback=callback,
        solver_options=solver_options,
    )
