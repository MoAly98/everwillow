"""State transformation helpers for :mod:`everwillow.statelib.state`."""

from __future__ import annotations

__all__ = ["Transform", "apply_transformations"]

import dataclasses
import typing as tp

from everwillow.statelib.state import K, State, V


def _identity(key: K, value: V) -> V:
    """Return the value unchanged.

    Args:
        key: Original key for the value (unused).
        value: Value associated with ``key``.

    Returns:
        The unmodified ``value``.
    """
    del key  # unused in default identity
    return value


@dataclasses.dataclass(frozen=True)
class Transform(tp.Generic[V]):
    """Describe how a single key/value pair should be rewritten.

    Examples:
        >>> transform = Transform(new_key=("scale",), value_fn=lambda _k, v: 2 * v)
        >>> transform.new_key
        ('scale',)
    """

    new_key: K  #: Replacement key tuple used in the transformed state.
    value_fn: tp.Callable[[K, V], V] = dataclasses.field(
        default=_identity
    )  #: Callable applied to derive the transformed value.


def apply_transformations(
    state: State[V],
    transformations: tp.Mapping[K, Transform[V]],
) -> State[V]:
    """Rewrite selected entries in a ``State``.

    Args:
        state: Original state whose segments will be updated.
        transformations: Mapping from existing keys to ``Transform`` objects.

    Returns:
        New ``State`` instance containing the transformed key/value pairs.

    Raises:
        TypeError: If ``state`` is not a ``State`` instance.
        KeyError: If transformations reference keys not present in ``state``.
        ValueError: If multiple transformations target the same destination key.

    Examples:
        >>> base = State.from_pytree({"a": 1, "b": 2})
        >>> transform = {("a",): Transform(new_key=("alpha",))}
        >>> apply_transformations(base, transform).to_dict()
        {('alpha',): 1, ('b',): 2}
    """
    if not isinstance(state, State):
        message = "'state' must be a State instance"  # type: ignore[unreachable]
        raise TypeError(message)

    if len(transformations) == 0:
        return state

    new_data: tp.MutableMapping[K, V] = {}
    for key, value in state.items():
        if key in transformations:
            transform = transformations[key]
            new_key = transform.new_key
            new_value = transform.value_fn(key, value)
            if new_key in new_data:
                message = f"multiple transformations target the same key: {new_key}"
                raise ValueError(message)
            new_data[new_key] = new_value
        else:
            new_data[key] = value

    return State(mapping=new_data, treedef=state.treedef)
