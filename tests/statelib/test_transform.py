"""Tests for :mod:`everwillow.statelib.transform`."""

from __future__ import annotations

import pytest

import everwillow.statelib as sl


def test_apply_transformations_rewrites_keys_and_values() -> None:
    """Applying transformations updates both keys and values."""
    state = sl.State.from_pytree({"a": 1, "b": 2})
    transforms = {
        "a": sl.Transform(new_key="alpha", value_fn=lambda _k, v: v + 1),
        "b": sl.Transform(new_key="beta", value_fn=lambda _k, v: v * 2),
    }

    transformed = sl.apply_transformations(state, transforms)

    assert dict(transformed.mapping) == {"alpha": 2, "beta": 4}
    assert transformed.treedefmeta.treedef == state.treedefmeta.treedef
    assert transformed.treedefmeta.keys == ("alpha", "beta")
    assert state.treedefmeta.keys == ("a", "b")
    assert transformed.to_pytree() == {"a": 2, "b": 4}


def test_apply_transformations_rejects_duplicate_targets() -> None:
    """Multiple transforms targeting the same destination raise ``ValueError``."""
    state = sl.State.from_pytree({"a": 1, "b": 2})
    transforms = {
        "a": sl.Transform(new_key="shared"),
        "b": sl.Transform(new_key="shared"),
    }

    with pytest.raises(ValueError, match="same key"):
        sl.apply_transformations(state, transforms)


def test_apply_transformations_requires_state_instance() -> None:
    """Only ``State`` instances are accepted."""
    transforms = {"a": sl.Transform(new_key="a")}
    with pytest.raises(TypeError, match="must be a State"):
        sl.apply_transformations({"a": 1}, transforms)


def test_apply_transformations_empty_returns_original() -> None:
    """An empty transforms dict returns the original state unchanged."""
    state = sl.State.from_pytree({"a": 1, "b": 2})

    result = sl.apply_transformations(state, {})
    assert result is state


def test_apply_transformations_partial_transform() -> None:
    """Transforming only some keys leaves others unchanged."""
    state = sl.State.from_pytree({"a": 1, "b": 2, "c": 3})
    transforms = {"a": sl.Transform(new_key="alpha")}

    result = sl.apply_transformations(state, transforms)

    assert dict(result.mapping) == {"alpha": 1, "b": 2, "c": 3}


def test_apply_transformations_missing_key_raises() -> None:
    """Referencing a key not in the state raises KeyError."""
    state = sl.State.from_pytree({"a": 1})
    transforms = {"missing": sl.Transform(new_key="x")}

    with pytest.raises(KeyError, match="not present in state"):
        sl.apply_transformations(state, transforms)


def test_apply_transformations_rejects_collision_with_untransformed_key() -> None:
    """Transform producing a key that collides with untransformed key raises ValueError."""
    state = sl.State.from_pytree({"a": 1, "b": 2})
    # Transform 'a' to 'b', but 'b' already exists as untransformed
    transforms = {"a": sl.Transform(new_key="b")}

    with pytest.raises(ValueError, match="duplicate key"):
        sl.apply_transformations(state, transforms)
