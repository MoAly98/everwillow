"""Tests for :mod:`everwillow.statelib.model`."""

from __future__ import annotations

import pytest

import everwillow.statelib as sl


class TestModel:
    """Unit tests around the ``Model`` wrapper."""

    def test_accepts_flat_state_and_pytree(self) -> None:
        """A ``Model`` can consume either a ``FlatState`` or a raw pytree."""

        def logpdf(tree: dict[str, object]) -> int:
            return tree["a"] + tree["b"]["c"]  # type: ignore[index]

        model = sl.Model(logpdf=logpdf)
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": {"c": 2}})

        assert model(state) == 3
        assert model({"a": 1, "b": {"c": 2}}) == 3


class TestCombinedModel:
    """Behavioural checks for ``CombinedModel`` aggregation."""

    def test_sums_component_models(self) -> None:
        """Evaluating a combined model sums the individual contributions."""

        model1 = sl.Model(logpdf=lambda tree: tree["x"])
        model2 = sl.Model(logpdf=lambda tree: tree["y"][0])  # type: ignore[index]

        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"y": (2, 3)})

        merged = sl.merge_states(state1, state2)
        combined = sl.CombinedModel.combine(model1, model2)

        assert combined(merged) == model1(state1) + model2(state2)

    def test_requires_matching_state_count(self) -> None:
        """Mismatch between segments and component models raises ``ValueError``."""

        model1 = sl.Model(logpdf=lambda tree: tree["x"])
        model2 = sl.Model(logpdf=lambda tree: tree["y"][0])  # type: ignore[index]
        state: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1})

        combined = sl.CombinedModel.combine(model1, model2)

        with pytest.raises(ValueError, match="Expected 2 states"):
            combined(state)

    def test_rejects_non_flat_state(self) -> None:
        """Passing a plain mapping instead of ``FlatState`` raises ``TypeError``."""

        model1 = sl.Model(logpdf=lambda tree: tree["x"])
        model2 = sl.Model(logpdf=lambda tree: tree["y"][0])  # type: ignore[index]

        state1: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1})
        state2: sl.FlatState[int] = sl.FlatState.from_pytree({"y": (2, 3)})

        merged = sl.merge_states(state1, state2)
        combined = sl.CombinedModel.combine(model1, model2)
        with pytest.raises(TypeError, match="parameters must be a FlatState"):
            combined(merged.to_dict())  # type: ignore[arg-type]
