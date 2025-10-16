from __future__ import annotations

import pytest

import statelib as sl


def test_model_call_accepts_flat_state_and_pytree() -> None:
    def logpdf(tree):
        return tree["a"] + tree["b"]["c"]

    model = sl.Model(logpdf=logpdf)
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"a": 1, "b": {"c": 2}})

    assert model(state) == 3
    assert model({"a": 1, "b": {"c": 2}}) == 3


def test_combined_model_sums_component_models() -> None:
    model1 = sl.Model(logpdf=lambda tree: tree["x"])
    model2 = sl.Model(logpdf=lambda tree: tree["y"][0])

    state1: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1})
    state2: sl.FlatState[int] = sl.FlatState.from_pytree({"y": (2, 3)})

    merged = sl.merge_states(state1, state2)
    combined = sl.CombinedModel.combine(model1, model2)

    assert combined(merged) == model1(state1) + model2(state2)


def test_combined_model_requires_matching_state_count() -> None:
    model1 = sl.Model(logpdf=lambda tree: tree["x"])
    model2 = sl.Model(logpdf=lambda tree: tree["y"][0])
    state: sl.FlatState[int] = sl.FlatState.from_pytree({"x": 1})

    combined = sl.CombinedModel.combine(model1, model2)

    with pytest.raises(ValueError):
        combined(state)
