"""Tests for :mod:`everwillow.statelib.model`."""

from __future__ import annotations

import typing as tp

import pytest

import everwillow.statelib as sl

FState: tp.TypeAlias = sl.State[float]


def test_model_requires_state_instance() -> None:
    """``Model`` only accepts ``State`` inputs."""
    model = sl.Model(logpdf=lambda tree: tree["a"])

    state: FState = sl.State.from_pytree({"a": 3.0})
    assert model(state) == 3.0

    with pytest.raises(TypeError, match="must be a State"):
        model({"a": 3.0})  # type: ignore[arg-type]


def test_combined_model_sums_components() -> None:
    """Combined models evaluate each segment and sum the results."""
    model_x = sl.Model(logpdf=lambda tree: tree["x"])
    model_y = sl.Model(logpdf=lambda tree: tree["y"][0])

    state_x: FState = sl.State.from_pytree({"x": 1.0})
    state_y: FState = sl.State.from_pytree({"y": (2.0, 3.0)})

    merged_mapping, metadata = sl.merge(state_x, state_y)
    combined = sl.CombinedModel.combine(model_x, model_y)

    expected = model_x(state_x) + model_y(state_y)
    assert combined(merged_mapping, metadata) == expected


def test_combined_model_validates_metadata_type() -> None:
    """Metadata must be produced by :func:`everwillow.statelib.state.merge`."""
    model = sl.Model(logpdf=lambda tree: tree["a"])
    state: FState = sl.State.from_pytree({"a": 1.0})
    merged_mapping, _ = sl.merge(state)

    combined = sl.CombinedModel.combine(model)
    with pytest.raises(TypeError, match="MergeMeta"):
        combined(merged_mapping, merge_metadata=None)  # type: ignore[arg-type]


def test_combined_model_detects_segment_mismatch() -> None:
    """Mismatch between models and merged segments raises ``ValueError``."""
    model_x = sl.Model(logpdf=lambda tree: tree["x"])
    model_y = sl.Model(logpdf=lambda tree: tree["y"])

    state_x: FState = sl.State.from_pytree({"x": 1.0})
    merged_mapping, metadata = sl.merge(state_x)

    combined = sl.CombinedModel.combine(model_x, model_y)
    with pytest.raises(ValueError, match="Expected 2 states"):
        combined(merged_mapping, metadata)
