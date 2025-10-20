"""Unit tests for everwillow.inference.utils module.

Tests cover:
- _resolve_keys() for parameter name resolution
- _build_param_updates() for creating state updates
- _prepare_fixed_param_state() for fixed parameter fit preparation
"""

from __future__ import annotations

import pytest

from everwillow.inference.utils import (
    _build_param_updates,  # noqa: PLC2701
    _prepare_fixed_param_state,  # noqa: PLC2701
    _resolve_keys,  # noqa: PLC2701
)
from everwillow.statelib import FlatState


class TestResolveKeys:
    """Tests for _resolve_keys function."""

    def test_resolve_string_names(self):
        """Test resolving simple string parameter names."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0, "sigma": 2.0})

        keys = _resolve_keys(state, ["mu"])

        assert keys == {("mu",)}

    def test_resolve_multiple_string_names(self):
        """Test resolving multiple string parameter names."""
        state: FlatState = FlatState.from_pytree(
            {"mu": 1.0, "sigma": 2.0, "background": 100.0}
        )

        keys = _resolve_keys(state, ["mu", "sigma"])

        assert keys == {("mu",), ("sigma",)}

    def test_resolve_tuple_keys(self):
        """Test resolving exact tuple keys."""
        state: FlatState = FlatState.from_pytree({"model": {"mu": 1.0, "sigma": 2.0}})

        keys = _resolve_keys(state, [("model", "mu")])

        assert keys == {("model", "mu")}

    def test_resolve_mixed_strings_and_tuples(self):
        """Test resolving mix of string names and tuple keys."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0, "model": {"sigma": 2.0}})

        keys = _resolve_keys(state, ["mu", ("model", "sigma")])

        assert keys == {("mu",), ("model", "sigma")}

    def test_resolve_nested_structure(self):
        """Test resolving parameters in deeply nested structures."""
        state: FlatState = FlatState.from_pytree({"level1": {"level2": {"mu": 1.0}}})

        keys = _resolve_keys(state, ["mu"])

        assert keys == {("level1", "level2", "mu")}

    def test_error_on_missing_string_name(self):
        """Test that KeyError is raised for non-existent string name."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0})

        with pytest.raises(KeyError, match="Parameter not found in state: nonexistent"):
            _resolve_keys(state, ["nonexistent"])

    def test_error_on_missing_tuple_key(self):
        """Test that KeyError is raised for non-existent tuple key."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0})

        with pytest.raises(
            KeyError, match=r"Parameter not found in state: \('model', 'mu'\)"
        ):
            _resolve_keys(state, [("model", "mu")])

    def test_error_on_ambiguous_name(self):
        """Test that ValueError is raised when string name matches multiple keys."""
        state: FlatState = FlatState.from_pytree(
            {"model1": {"mu": 1.0}, "model2": {"mu": 2.0}}
        )

        with pytest.raises(ValueError, match="Ambiguous parameter name 'mu'"):
            _resolve_keys(state, ["mu"])

    def test_ambiguous_error_suggests_full_key(self):
        """Test that ambiguous error message suggests using full tuple key."""
        state: FlatState = FlatState.from_pytree(
            {"model1": {"mu": 1.0}, "model2": {"mu": 2.0}}
        )

        with pytest.raises(ValueError, match="Use the full tuple key to disambiguate"):
            _resolve_keys(state, ["mu"])

    def test_empty_names_returns_empty_set(self):
        """Test that empty input returns empty set."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0})

        keys = _resolve_keys(state, [])

        assert keys == set()


class TestBuildParamUpdates:
    """Tests for _build_param_updates function."""

    def test_build_updates_single_param(self):
        """Test building updates for a single parameter."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0, "sigma": 2.0})

        keys, updates = _build_param_updates(state, {"mu": 5.0})

        assert keys == {("mu",)}
        assert updates == {("mu",): 5.0}

    def test_build_updates_multiple_params(self):
        """Test building updates for multiple parameters."""
        state: FlatState = FlatState.from_pytree(
            {"mu": 1.0, "sigma": 2.0, "background": 100.0}
        )

        keys, updates = _build_param_updates(state, {"mu": 5.0, "sigma": 3.0})

        assert keys == {("mu",), ("sigma",)}
        assert updates == {("mu",): 5.0, ("sigma",): 3.0}

    def test_build_updates_nested_structure(self):
        """Test building updates for parameters in nested structure."""
        state: FlatState = FlatState.from_pytree({"model": {"mu": 1.0, "sigma": 2.0}})

        keys, updates = _build_param_updates(state, {"mu": 5.0})

        assert keys == {("model", "mu")}
        assert updates == {("model", "mu"): 5.0}

    def test_build_updates_preserves_value_types(self):
        """Test that update values preserve their types."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0, "count": 10})

        _keys, updates = _build_param_updates(state, {"mu": 5.0, "count": 20})

        assert updates["mu",] == 5.0
        assert updates["count",] == 20
        assert isinstance(updates["mu",], float)
        assert isinstance(updates["count",], int)

    def test_build_updates_error_on_missing_param(self):
        """Test that KeyError is raised for non-existent parameter."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0})

        with pytest.raises(KeyError, match="Parameter not found in state: sigma"):
            _build_param_updates(state, {"sigma": 2.0})

    def test_build_updates_error_on_ambiguous_param(self):
        """Test that ValueError is raised for ambiguous parameter name."""
        state: FlatState = FlatState.from_pytree(
            {"model1": {"mu": 1.0}, "model2": {"mu": 2.0}}
        )

        with pytest.raises(ValueError, match="Ambiguous parameter name 'mu'"):
            _build_param_updates(state, {"mu": 5.0})

    def test_build_updates_empty_dict(self):
        """Test that empty param_values returns empty keys and updates."""
        state: FlatState = FlatState.from_pytree({"mu": 1.0})

        keys, updates = _build_param_updates(state, {})

        assert keys == set()
        assert updates == {}

    def test_build_updates_can_be_used_with_update_state(self):
        """Test that returned updates work with update_state."""
        from everwillow.statelib import update_state

        state: FlatState = FlatState.from_pytree({"mu": 1.0, "sigma": 2.0})

        _keys, updates = _build_param_updates(state, {"mu": 5.0})
        updated_state = update_state(state, updates)

        assert updated_state["mu",] == 5.0
        assert updated_state["sigma",] == 2.0  # Unchanged


class TestPrepareFixedParamState:
    """Tests for _prepare_fixed_param_state function."""

    def test_prepare_single_param(self):
        """Test preparing state with single fixed parameter."""
        params = {"mu": 1.0, "sigma": 2.0, "background": 100.0}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, None, None
        )

        # Check updated pytree has new value
        assert updated_pytree["mu"] == 5.0
        assert updated_pytree["sigma"] == 2.0
        assert updated_pytree["background"] == 100.0

        # Check fixed keys includes only param_values key
        assert set(fixed_keys) == {("mu",)}

    def test_prepare_multiple_params(self):
        """Test preparing state with multiple fixed parameters."""
        params = {"mu": 1.0, "sigma": 2.0, "background": 100.0}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0, "sigma": 3.0}, None, None
        )

        # Check updated pytree
        assert updated_pytree["mu"] == 5.0
        assert updated_pytree["sigma"] == 3.0
        assert updated_pytree["background"] == 100.0

        # Check fixed keys
        assert set(fixed_keys) == {("mu",), ("sigma",)}

    def test_prepare_with_additional_fixed(self):
        """Test preparing state with param_values and additional fixed list."""
        params = {"mu": 1.0, "sigma": 2.0, "background": 100.0}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, ["background"], None
        )

        # Check updated pytree
        assert updated_pytree["mu"] == 5.0
        assert updated_pytree["sigma"] == 2.0
        assert updated_pytree["background"] == 100.0

        # Check fixed keys includes both param_values and fixed list
        assert set(fixed_keys) == {("mu",), ("background",)}

    def test_prepare_with_fixed_predicate(self):
        """Test preparing state with fixed_predicate."""
        params = {"mu": 1.0, "sigma": 2.0, "background": 100.0}

        # Predicate: fix all params with values >= 100
        def is_large(key, value):
            return value >= 100.0

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, None, is_large
        )

        # Check updated pytree
        assert updated_pytree["mu"] == 5.0
        assert updated_pytree["background"] == 100.0

        # Check fixed keys includes param_values and predicate matches
        assert set(fixed_keys) == {("mu",), ("background",)}

    def test_prepare_with_all_fixed_mechanisms(self):
        """Test preparing with param_values, fixed list, and fixed_predicate."""
        params = {"mu": 1.0, "sigma": 2.0, "background": 100.0, "scale": 50.0}

        # Predicate: fix params with values >= 50
        def is_moderate(key, value):
            return value >= 50.0

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, ["sigma"], is_moderate
        )

        # Check updated pytree
        assert updated_pytree["mu"] == 5.0
        assert updated_pytree["sigma"] == 2.0

        # Fixed keys should include:
        # - mu (from param_values)
        # - sigma (from fixed list)
        # - scale and background (from predicate)
        assert set(fixed_keys) == {("mu",), ("sigma",), ("background",), ("scale",)}

    def test_prepare_nested_params(self):
        """Test preparing state with nested parameter structure."""
        params = {"model": {"mu": 1.0, "sigma": 2.0}}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, None, None
        )

        # Check nested structure preserved
        assert updated_pytree["model"]["mu"] == 5.0
        assert updated_pytree["model"]["sigma"] == 2.0

        # Check fixed keys
        assert set(fixed_keys) == {("model", "mu")}

    def test_prepare_empty_param_values(self):
        """Test preparing with empty param_values dict."""
        params = {"mu": 1.0, "sigma": 2.0}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {}, ["mu"], None
        )

        # Check pytree unchanged
        assert updated_pytree["mu"] == 1.0
        assert updated_pytree["sigma"] == 2.0

        # Check only fixed list is used
        assert set(fixed_keys) == {("mu",)}

    def test_prepare_error_on_missing_param(self):
        """Test that error is raised for non-existent parameter in param_values."""
        params = {"mu": 1.0}

        with pytest.raises(KeyError, match="Parameter not found in state: nonexistent"):
            _prepare_fixed_param_state(params, {"nonexistent": 5.0}, None, None)

    def test_prepare_return_types(self):
        """Test that return types are correct (pytree and list)."""
        params = {"mu": 1.0, "sigma": 2.0}

        updated_pytree, fixed_keys = _prepare_fixed_param_state(
            params, {"mu": 5.0}, None, None
        )

        # Check types
        assert isinstance(updated_pytree, dict)
        assert isinstance(fixed_keys, list)
        assert all(isinstance(k, tuple) for k in fixed_keys)
