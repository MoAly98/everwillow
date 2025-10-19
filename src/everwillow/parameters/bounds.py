"""Parameter bounds transformations for constrained optimization.

This module provides transformation functions to convert between bounded and unbounded
parameter spaces, enabling optimization with box constraints using unconstrained solvers.

All transformations are JIT-safe and use clipping to handle edge cases gracefully.
"""

from __future__ import annotations

__all__ = [
    "match_bounds_to_state",
    "apply_bounds_transform",
]

import typing as tp

import everwillow.statelib as sl
from everwillow.parameters.transforms import AbstractParameterTransformation

# Type alias for bounds specification
TransformSpec = AbstractParameterTransformation | None


def _resolve_keys(
    state: sl.FlatState[tp.Any],
    spec: str | sl.KeyPath,
) -> list[sl.KeyPath]:
    """Resolve parameter specification to canonical keys in the state.

    Args:
        state: FlatState containing parameters.
        spec: Parameter name (str) or canonical key tuple.

    Returns:
        List of matching canonical keys.
    """
    if isinstance(spec, str):
        # Find all keys matching this name
        matched_keys = [
            key for key in state.raw_mapping if key and key[-1] == spec
        ]
    elif isinstance(spec, tuple):
        # Direct key specification
        key: sl.KeyPath = tuple(spec)
        if key not in state.raw_mapping:
            raise KeyError(f"Parameter key {key} not found in state")
        matched_keys = [key]
    else:
        raise TypeError(f"Invalid parameter specification type: {type(spec)}")
    return matched_keys


def match_bounds_to_state(
    state: sl.FlatState[tp.Any],
    bounds: tp.Mapping[str | sl.KeyPath, TransformSpec],
) -> dict[sl.KeyPath, AbstractParameterTransformation]:
    """Match parameter bounds to their state representations.

    Args:
        state (sl.FlatState[tp.Any]): The state to match.
        bounds (tp.Mapping[str  |  sl.KeyPath, TransformSpec]): The bounds specifications.

    Raises:
        TypeError: If the bounds specifications are invalid.

    Returns:
        dict[sl.KeyPath, AbstractParameterTransformation]: A mapping of state keys to their corresponding transformations.
    """
    transform_map: dict[sl.KeyPath, AbstractParameterTransformation] = {}
    for spec, transform in bounds.items():
        if transform is None:   continue
        if not isinstance(transform, AbstractParameterTransformation):
            raise TypeError(f"Expected AbstractParameterTransformation, got {type(transform)}")
        # resolve spec → canonical keys (similar to validate_bounds)
        resolved_keys = _resolve_keys(state, spec)  # you can inline this logic
        for key in resolved_keys:
            transform_map[key] = transform
    return transform_map


def apply_bounds_transform(
    state: sl.FlatState[tp.Any],
    bounds: tp.Mapping[str | sl.KeyPath, TransformSpec],
) -> tuple[sl.FlatState[tp.Any], dict[sl.KeyPath, sl.Transform[tp.Any]], dict[sl.KeyPath, sl.Transform[tp.Any]]]:
    """Applies bounds transformations to the state.

    Args:
        state (sl.FlatState[tp.Any]): The state to transform.
        bounds (tp.Mapping[str  |  tuple[tp.Any, ...], TransformSpec]): The bounds specifications.

    Returns:
        tuple[sl.FlatState[tp.Any], dict[sl.KeyPath, sl.Transform[tp.Any]], dict[sl.KeyPath, sl.Transform[tp.Any]]]: The transformed state, unwrap transforms, and wrap transforms.
    """
    transform_map = match_bounds_to_state(state, bounds)
    if not transform_map:
        return state, {}, {}

    def make_unwrap(transform):
        def unwrap_fn(_key, value):
            return transform.unwrap(value)
        return unwrap_fn

    def make_wrap(transform):
        def wrap_fn(_key, value):
            return transform.wrap(value)
        return wrap_fn

    unwrap = {
        key: sl.Transform(new_key=key, value_fn=make_unwrap(transform))
        for key, transform in transform_map.items()
    }
    wrap = {
        key: sl.Transform(new_key=key, value_fn=make_wrap(transform))
        for key, transform in transform_map.items()
    }

    unwrapped_state = sl.transform.apply_transformations(state, unwrap)
    return unwrapped_state, unwrap, wrap