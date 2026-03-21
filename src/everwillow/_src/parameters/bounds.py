"""
Example
-------
>>> import jax.numpy as jnp
>>> from everwillow import statelib as sl
>>> from everwillow.parameters import transforms
>>> state = sl.State.from_pytree({"mu": 0.3})
>>> transform_map = {("mu",): transforms.MinuitTransform(lower=0.0, upper=1.0)}
>>> unwrapped = unwrap(state, transform_map)
>>> jnp.isclose(wrap(unwrapped, transform_map)[("mu",)], state[("mu",)])
Array(True, dtype=bool)
"""

from __future__ import annotations

__all__ = ["unwrap", "wrap"]

import typing as tp

from everwillow._src.parameters.transforms import TransformBase
from everwillow._src.statelib import K, State, V


def unwrap(
    state: State[V],
    transform_mapping: tp.Mapping[K, TransformBase],
) -> State[V]:
    if not transform_mapping:
        return state

    if missing := set(transform_mapping.keys()) - set(state.mapping.keys()):
        msg = f"Transform mapping contains keys not in state: {missing}"
        raise KeyError(msg)

    new_mapping = dict(state.mapping)
    for key, transform in transform_mapping.items():
        new_mapping[key] = transform.unwrap(new_mapping[key])
    return State(new_mapping, treedefmeta=state.treedefmeta)


def wrap(
    state: State[V],
    transform_mapping: tp.Mapping[K, TransformBase],
) -> State[V]:
    if not transform_mapping:
        return state

    if missing := set(transform_mapping) - set(state.mapping):
        msg = f"Transform mapping contains keys not in state: {missing}"
        raise KeyError(msg)

    new_mapping = dict(state.mapping)
    for key, transform in transform_mapping.items():
        new_mapping[key] = transform.wrap(new_mapping[key])
    return State(new_mapping, treedefmeta=state.treedefmeta)
