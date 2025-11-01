"""
Example
-------
>>> state = sl.State.from_pytree({"mu": 0.3})
>>> transform = transforms.MinuitTransform(lower=0.0, upper=1.0)
>>> unwrapped = unwrap(state, {("mu",): transform})
>>> jnp.isclose(wrap(unwrapped, {("mu",): transform)[("mu",)], state[("mu",)])
Array(True, dtype=bool)
"""

from __future__ import annotations

__all__ = ["unwrap", "wrap"]

import typing as tp

from everwillow.parameters.transforms import AbstractParameterTransformation
from everwillow.statelib import K, State, T


def unwrap(
    state: State[T],
    transform_mapping: tp.Mapping[K, AbstractParameterTransformation],
) -> State[T]:
    if not transform_mapping:
        return state

    if missing := set(transform_mapping.keys()) - set(state.mapping.keys()):
        msg = f"Transform mapping contains keys not in state: {missing}"
        raise KeyError(msg)

    new_mapping = dict(state.mapping)
    for key, transform in transform_mapping.items():
        if key in new_mapping:
            new_mapping[key] = transform.unwrap(new_mapping[key])
    return State(new_mapping, treedef=state.treedef)


def wrap(
    state: State[T],
    transform_mapping: tp.Mapping[K, AbstractParameterTransformation],
) -> State[T]:
    if not transform_mapping:
        return state

    if missing := set(transform_mapping) - set(state.mapping):
        msg = f"Transform mapping contains keys not in state: {missing}"
        raise KeyError(msg)

    new_mapping = dict(state.mapping)
    for key, transform in transform_mapping.items():
        if key in new_mapping:
            new_mapping[key] = transform.wrap(new_mapping[key])
    return State(new_mapping, treedef=state.treedef)
