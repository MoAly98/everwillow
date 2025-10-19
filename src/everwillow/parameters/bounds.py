"""Wire parameter transforms to flattened state representations.

The helpers here take user-provided transform instances (e.g. ``MinuitTransform``)
and align them with canonical keys inside a :class:`~everwillow.statelib.state.FlatState`.
Optimisation code can then call :func:`apply_bounds_transform` to obtain:

1. an "unwrapped" state where all targeted values live in an unconstrained space
2. two dictionaries of :class:`~everwillow.statelib.transform.Transform` objects
   that apply ``unwrap``/``wrap`` on demand

Example
-------
>>> state = sl.FlatState.from_pytree({"mu": 0.3})
>>> transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
>>> unwrapped, unwrap_map, wrap_map = apply_bounds_transform(state, {"mu": transform})
>>> list(unwrap_map)
[('mu',)]
>>> jnp.isclose(sl.transform.apply_transformations(unwrapped, wrap_map)[("mu",)], 0.3)
Array(True, dtype=bool)
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
    """Resolve a parameter specifier to canonical keys in ``state``.

    Raises
    ------
    KeyError
        If a tuple specifier does not exist in ``state``.
    TypeError
        If ``spec`` is neither a ``str`` nor a tuple key.
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
    """Return a mapping from canonical keys to user-supplied transforms.

    Parameters
    ----------
    state
        Flattened parameter state.
    bounds
        Mapping from parameter names/keys to transform instances (or ``None`` to skip).
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
    """Apply ``bounds`` to ``state``, returning the unwrapped state and helper maps.

    Returns a tuple ``(unwrapped_state, unwrap_map, wrap_map)`` where:

    - ``unwrapped_state`` is the result of applying ``transform.unwrap`` to each matched key.
    - ``unwrap_map`` can be reused to unwrap additional states (same structure).
    - ``wrap_map`` reverses the transformation (good for post-optimisation wrapping).
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
