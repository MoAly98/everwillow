"""Tests for :mod:`everwillow.statelib.transform`."""

from __future__ import annotations

import typing as tp

import pytest

import everwillow.statelib as sl

FState: tp.TypeAlias = sl.State[float]
TMapping: tp.TypeAlias = tp.Mapping[sl.K, sl.Transform[float]]


def test_apply_transformations_rewrites_keys_and_values() -> None:
    """Applying transformations updates both keys and values."""
    state: FState = sl.State.from_pytree({"a": 1, "b": 2})
    transforms: TMapping = {
        ("a",): sl.Transform(new_key=("alpha",), value_fn=lambda _k, v: v + 1),
        ("b",): sl.Transform(new_key=("beta",), value_fn=lambda _k, v: v * 2),
    }

    transformed = sl.apply_transformations(state, transforms)

    assert dict(transformed.mapping) == {("alpha",): 2, ("beta",): 4}
    assert transformed.treedefmeta == state.treedefmeta


def test_apply_transformations_rejects_duplicate_targets() -> None:
    """Multiple transforms targeting the same destination raise ``ValueError``."""
    state: FState = sl.State.from_pytree({"a": 1, "b": 2})
    transforms: TMapping = {
        ("a",): sl.Transform(new_key=("shared",)),
        ("b",): sl.Transform(new_key=("shared",)),
    }

    with pytest.raises(ValueError, match="same key"):
        sl.apply_transformations(state, transforms)


def test_apply_transformations_requires_state_instance() -> None:
    """Only ``State`` instances are accepted."""
    transforms: TMapping = {("a",): sl.Transform(new_key=("a",))}
    with pytest.raises(TypeError, match="must be a State"):
        sl.apply_transformations({"a": 1}, transforms)  # type: ignore[arg-type]
