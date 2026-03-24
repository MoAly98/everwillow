"""
Example
-------
>>> import jax.numpy as jnp
>>> from everwillow import statelib as sl
>>> from everwillow.parameters import transforms
>>> state = sl.State.from_pytree({"mu": 0.3})
>>> transform_map = {"mu": transforms.MinuitTransform(lower=0.0, upper=1.0)}
>>> unwrapped = unwrap(state, transform_map)
>>> jnp.isclose(wrap(unwrapped, transform_map)["mu"], state["mu"])
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
    """Transform parameter values from bounded to unconstrained space.

    Applies each transform's ``unwrap`` method to the corresponding
    parameter in ``state``. Parameters not in ``transform_mapping``
    are left unchanged.

    Args:
        state: Parameter state with bounded values.
        transform_mapping: Maps parameter keys to their transforms.

    Returns:
        New state with unconstrained (internal) parameter values.

    Raises:
        KeyError: If ``transform_mapping`` contains keys not in ``state``.
    """
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
    """Transform parameter values from unconstrained back to bounded space.

    Applies each transform's ``wrap`` method to the corresponding
    parameter in ``state``. Parameters not in ``transform_mapping``
    are left unchanged.

    Args:
        state: Parameter state with unconstrained (internal) values.
        transform_mapping: Maps parameter keys to their transforms.

    Returns:
        New state with bounded (external) parameter values.

    Raises:
        KeyError: If ``transform_mapping`` contains keys not in ``state``.
    """
    if not transform_mapping:
        return state

    if missing := set(transform_mapping) - set(state.mapping):
        msg = f"Transform mapping contains keys not in state: {missing}"
        raise KeyError(msg)

    new_mapping = dict(state.mapping)
    for key, transform in transform_mapping.items():
        new_mapping[key] = transform.wrap(new_mapping[key])
    return State(new_mapping, treedefmeta=state.treedefmeta)
