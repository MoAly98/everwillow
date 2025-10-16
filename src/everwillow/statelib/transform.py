from __future__ import annotations

import dataclasses
import typing as tp
from typing import TYPE_CHECKING

from .key_paths import KeyPath, derive_key_path, ensure_public_key
from .state import FlatState, _validate_state

if TYPE_CHECKING:
    from .state import SegmentRecord

ValueT = tp.TypeVar("ValueT")


def _identity(key: KeyPath, value: ValueT) -> ValueT:
    del key  # unused in default identity
    return value


@dataclasses.dataclass(frozen=True)
class Transform(tp.Generic[ValueT]):
    """Describe how a single key/value pair should be rewritten."""

    new_key: KeyPath
    value_fn: tp.Callable[[KeyPath, ValueT], ValueT] = dataclasses.field(
        default=_identity
    )


def apply_transformations(
    state: FlatState[ValueT],
    transformations: tp.Mapping[KeyPath, Transform[ValueT]],
) -> FlatState[ValueT]:
    """Rewrite selected entries in a FlatState."""
    if not isinstance(state, FlatState):
        raise TypeError("'state' must be a FlatState instance")
    normalized_transformations: dict[KeyPath, Transform[ValueT]] = {
        ensure_public_key(key): transform for key, transform in transformations.items()
    }
    state_keys = set(state.keys())
    if set(normalized_transformations.keys()) - state_keys:
        missing_keys = set(normalized_transformations.keys()) - state_keys
        raise KeyError(f"Transformations contain keys not in state: {missing_keys!r}")

    flat_state = state.copy()

    new_records: dict[object, SegmentRecord[ValueT]] = {}
    for segment_id in flat_state._segment_order:
        record = flat_state._segments[segment_id]
        updated_keys: set[KeyPath] = set()
        new_values: dict[KeyPath, ValueT] = {}
        new_key_paths: dict[KeyPath, KeyPath] = {}
        for key, value in record.values.items():
            transform = normalized_transformations.get(key)
            if transform is None:
                target_key = key
                transformed_value = value
                key_path = record.key_paths[key]
            else:
                target_key = ensure_public_key(transform.new_key)
                transformed_value = transform.value_fn(key, value)
                key_path = derive_key_path(
                    target_key,
                    template=record.key_paths.get(key),
                )
            if target_key in new_values:
                raise ValueError(
                    "Transformations produce duplicate target key "
                    f"{target_key!r}; ensure new_key values are unique per segment"
                )
            updated_keys.add(target_key)
            new_values[target_key] = transformed_value
            new_key_paths[target_key] = key_path
        record_cls = type(record)
        new_records[segment_id] = record_cls(
            record.treedef,
            frozenset(updated_keys),
            new_values,
            new_key_paths,
        )

    flat_state._segments = new_records
    flat_state._rebuild_mapping()

    _validate_state(flat_state)
    return flat_state
