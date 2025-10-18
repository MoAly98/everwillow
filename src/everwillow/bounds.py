"""Parameter bounds transformations for constrained optimization.

This module provides transformation functions to convert between bounded and unbounded
parameter spaces, enabling optimization with box constraints using unconstrained solvers.

All transformations are JIT-safe and use clipping to handle edge cases gracefully.
"""

from __future__ import annotations

__all__ = [
    "transform_to_unbounded",
    "transform_to_bounded",
    "create_bounds_transforms",
    "validate_bounds",
]

import typing as tp

import jax.numpy as jnp
from jaxtyping import Array, Float

import everwillow.statelib as sl

# Type alias for bounds specification
BoundSpec = tuple[float | None, float | None] | None


def _logit(x: Float[Array, ""]) -> Float[Array, ""]:
    """Compute logit function: log(x / (1 - x)).

    JIT-safe implementation that handles edge cases.
    """
    return jnp.log(x) - jnp.log1p(-x)


def _sigmoid(x: Float[Array, ""]) -> Float[Array, ""]:
    """Compute sigmoid function: 1 / (1 + exp(-x)).

    Numerically stable implementation that avoids overflow.
    """
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x)),
    )


def transform_to_unbounded(
    value: float | Float[Array, ""],
    lower: float | None,
    upper: float | None,
) -> float | Float[Array, ""]:
    """Transform a bounded parameter value to unbounded space.

    This function is JIT-safe and uses clipping to handle values near boundaries.

    Args:
        value: Parameter value in bounded space.
        lower: Lower bound (None for no lower bound).
        upper: Upper bound (None for no upper bound).

    Returns:
        Transformed value in unbounded space.

    Examples:
        >>> # Bounded on both sides [0, 1]
        >>> transform_to_unbounded(0.5, 0.0, 1.0)  # doctest: +SKIP
        Array(0., dtype=float32)

        >>> # Lower bound only [0, ∞)
        >>> transform_to_unbounded(1.0, 0.0, None)  # doctest: +SKIP
        Array(0., dtype=float32)

        >>> # Upper bound only (-∞, 1]
        >>> transform_to_unbounded(0.0, None, 1.0)  # doctest: +SKIP
        Array(0., dtype=float32)
    """
    if lower is not None and upper is not None:
        # Both bounds: use logit of scaled value
        # Clip to avoid numerical issues at boundaries
        scaled = (value - lower) / (upper - lower)
        scaled = jnp.clip(scaled, 1e-10, 1.0 - 1e-10)
        return _logit(scaled)
    elif lower is not None:
        # Lower bound only: use log
        # Ensure positive argument for log
        safe_diff = jnp.maximum(value - lower, 1e-10)
        return jnp.log(safe_diff)
    elif upper is not None:
        # Upper bound only: use log of distance from upper
        # Ensure positive argument for log
        safe_diff = jnp.maximum(upper - value, 1e-10)
        return jnp.log(safe_diff)
    else:
        # No bounds: identity
        return value


def transform_to_bounded(
    value: float | Float[Array, ""],
    lower: float | None,
    upper: float | None,
) -> float | Float[Array, ""]:
    """Transform an unbounded parameter value back to bounded space.

    This function is JIT-safe.

    Args:
        value: Parameter value in unbounded space.
        lower: Lower bound (None for no lower bound).
        upper: Upper bound (None for no upper bound).

    Returns:
        Transformed value in bounded space.

    Examples:
        >>> # Bounded on both sides [0, 1]
        >>> transform_to_bounded(0.0, 0.0, 1.0)  # doctest: +SKIP
        Array(0.5, dtype=float32)

        >>> # Lower bound only [0, ∞)
        >>> transform_to_bounded(0.0, 0.0, None)  # doctest: +SKIP
        Array(1., dtype=float32)

        >>> # Upper bound only (-∞, 1]
        >>> transform_to_bounded(0.0, None, 1.0)  # doctest: +SKIP
        Array(1., dtype=float32)
    """
    if lower is not None and upper is not None:
        # Both bounds: use sigmoid to scale to [lower, upper]
        scaled = _sigmoid(value)
        return lower + (upper - lower) * scaled
    elif lower is not None:
        # Lower bound only: use exp
        return lower + jnp.exp(value)
    elif upper is not None:
        # Upper bound only: use exp of negative
        return upper - jnp.exp(value)
    else:
        # No bounds: identity
        return value


def validate_bounds(
    params: tp.Any,
    bounds: tp.Mapping[str | tuple[tp.Any, ...], BoundSpec],
) -> None:
    """Validate that parameter values satisfy their bounds.

    This is a non-JIT function that checks concrete values. Call this before
    JIT-compiling fit functions if you want to verify initial values.

    Args:
        params: Parameter pytree to validate.
        bounds: Bounds specification (same format as for fit).

    Raises:
        ValueError: If any parameter violates its bounds.
        TypeError: If bounds specification is malformed.
        KeyError: If a parameter name in bounds is not found in params.

    Examples:
        >>> params = {"mu": 1.0, "sigma": 0.5}
        >>> bounds = {"mu": (0.0, 5.0), "sigma": (0.0, None)}
        >>> validate_bounds(params, bounds)  # No error if valid

        >>> params = {"mu": -1.0, "sigma": 0.5}
        >>> validate_bounds(params, bounds)  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: Parameter 'mu' value -1.0 violates lower bound 0.0
    """
    state = sl.FlatState.from_pytree(params)

    for param_spec, bound_spec in bounds.items():
        if bound_spec is None or bound_spec == (None, None):
            continue

        if not isinstance(bound_spec, tuple) or len(bound_spec) != 2:
            msg = f"Bound specification must be (lower, upper) tuple, got {bound_spec}"
            raise TypeError(msg)

        lower, upper = bound_spec

        # Validate bounds themselves
        if lower is not None and upper is not None and lower >= upper:
            msg = f"Lower bound {lower} must be < upper bound {upper}"
            raise ValueError(msg)

        # Resolve parameter name to values
        if isinstance(param_spec, str):
            matched_keys = [
                key for key in state.raw_mapping if key and key[-1] == param_spec
            ]
            if not matched_keys:
                msg = f"Parameter '{param_spec}' not found in params"
                raise KeyError(msg)
        else:
            key = tuple(param_spec)
            if key not in state.raw_mapping:
                msg = f"Parameter key {key} not found in params"
                raise KeyError(msg)
            matched_keys = [key]

        # Check each matched parameter
        for key in matched_keys:
            value = state[key]
            param_name = param_spec if isinstance(param_spec, str) else str(key)

            if lower is not None and value <= lower:
                msg = f"Parameter '{param_name}' value {value} violates lower bound {lower}"
                raise ValueError(msg)

            if upper is not None and value >= upper:
                msg = f"Parameter '{param_name}' value {value} violates upper bound {upper}"
                raise ValueError(msg)


def create_bounds_transforms(
    state: sl.FlatState[tp.Any],
    bounds: tp.Mapping[str | tuple[tp.Any, ...], BoundSpec],
) -> tuple[
    dict[tuple[tp.Any, ...], sl.Transform],
    dict[tuple[tp.Any, ...], sl.Transform],
]:
    """Create forward and inverse transformation dictionaries from bounds specification.

    Args:
        state: FlatState containing the parameters to be bounded.
        bounds: Mapping from parameter names (str) or canonical key tuples to
            (lower, upper) bound specifications. Each bound can be:

            - ``(lower, upper)``: bounded on both sides
            - ``(lower, None)``: lower bound only
            - ``(None, upper)``: upper bound only
            - ``None`` or ``(None, None)``: no bounds

    Returns:
        Tuple of (forward_transforms, inverse_transforms) where:

        - forward_transforms: Transforms from bounded to unbounded space
        - inverse_transforms: Transforms from unbounded to bounded space

    Note:
        Key validation is handled by :func:`everwillow.statelib.transform.apply_transformations`.
        Invalid keys will raise KeyError when transforms are applied.

    Examples:
        >>> state = sl.FlatState.from_pytree({"mu": 1.0, "sigma": 0.5})
        >>> bounds = {"mu": (0.0, 5.0), "sigma": (0.0, None)}
        >>> fwd, inv = create_bounds_transforms(state, bounds)
    """
    forward_transforms: dict[tuple[tp.Any, ...], sl.Transform] = {}
    inverse_transforms: dict[tuple[tp.Any, ...], sl.Transform] = {}

    # Process each bound specification
    for param_spec, bound_spec in bounds.items():
        # Skip if no bounds specified
        if bound_spec is None or bound_spec == (None, None):
            continue

        lower, upper = bound_spec

        # Resolve parameter name to canonical keys
        if isinstance(param_spec, str):
            # Find all keys matching this name
            matched_keys = [
                key for key in state.raw_mapping if key and key[-1] == param_spec
            ]
        else:
            # Direct key specification
            matched_keys = [tuple(param_spec)]

        # Create transforms for each matched key
        for key in matched_keys:
            # Forward transform: bounded -> unbounded
            def make_forward_fn(lb, ub):
                def forward_fn(_key, value):
                    return transform_to_unbounded(value, lb, ub)

                return forward_fn

            forward_transforms[key] = sl.Transform(
                new_key=key, value_fn=make_forward_fn(lower, upper)
            )

            # Inverse transform: unbounded -> bounded
            def make_inverse_fn(lb, ub):
                def inverse_fn(_key, value):
                    return transform_to_bounded(value, lb, ub)

                return inverse_fn

            inverse_transforms[key] = sl.Transform(
                new_key=key, value_fn=make_inverse_fn(lower, upper)
            )

    return forward_transforms, inverse_transforms
