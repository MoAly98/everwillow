"""Utility functions for parameter resolution and state manipulation."""

from __future__ import annotations

import typing as tp

import everwillow.statelib as sl


def _resolve_keys(
    state: sl.FlatState[tp.Any],
    names: tp.Iterable[str | tuple[tp.Any, ...]],
) -> set[tuple[tp.Any, ...]]:
    """Resolve parameter names or tuples to canonical key tuples.

    String names are matched against the final element of each key
    (e.g., ``"mu"`` matches ``("model", "mu")``). Tuple entries must
    match exactly.

    Args:
        state: ``FlatState`` to search for matching keys.
        names: Parameter identifiers (strings or tuples).

    Returns:
        Set of canonical key tuples.

    Raises:
        KeyError: If any name cannot be located in ``state``.
        ValueError: If a string name matches multiple keys (ambiguous).
    """
    keys: set[tuple[tp.Any, ...]] = set()

    for entry in names:
        if isinstance(entry, tuple):
            key = tuple(entry)
            if key not in state.raw_mapping:
                message = f"Parameter not found in state: {key}"
                raise KeyError(message)
            keys.add(key)
        else:
            matches = [key for key in state.raw_mapping if key and key[-1] == entry]
            if not matches:
                message = f"Parameter not found in state: {entry}"
                raise KeyError(message)
            if len(matches) > 1:
                matches_str = ", ".join(str(k) for k in matches)
                message = (
                    f"Ambiguous parameter name '{entry}' matches multiple keys: "
                    f"{matches_str}. Use the full tuple key to disambiguate."
                )
                raise ValueError(message)
            keys.add(matches[0])

    return keys


def _build_param_updates(
    state: sl.FlatState[tp.Any],
    param_values: dict[str, tp.Any],
) -> tuple[set[tuple[tp.Any, ...]], dict[tuple[tp.Any, ...], tp.Any]]:
    """Build canonical parameter updates from name-value pairs.

    Resolves parameter names to canonical keys and constructs the update
    dictionary for :func:`everwillow.statelib.state.update_state`.

    Args:
        state: ``FlatState`` derived from the caller's parameter pytree.
        param_values: Mapping of parameter names to values.

    Returns:
        Tuple of (resolved keys, update dictionary ready for ``update_state``).
    """
    keys = _resolve_keys(state, param_values.keys())
    updates = {key: param_values[key[-1]] for key in keys}
    return keys, updates


def _prepare_fixed_param_state(
    params: tp.Any,
    param_values: dict[str, tp.Any],
    fixed: tp.Sequence[str | tuple[tp.Any, ...]] | None,
    fixed_predicate: tp.Callable[[tuple[tp.Any, ...], tp.Any], bool] | None,
) -> tuple[tp.Any, list[tuple[tp.Any, ...]]]:
    """Prepare parameter state for fixed parameter fits.

    Applies ``param_values`` to the initial parameter pytree, identifies all
    parameters that should be fixed (from ``param_values``, ``fixed``, and
    ``fixed_predicate``), and returns both the updated pytree and the combined
    list of fixed keys.

    Args:
        params: Initial parameter pytree.
        param_values: Values to inject and fix during optimization.
        fixed: Additional parameter names/keys to hold fixed.
        fixed_predicate: Optional callable to identify additional fixed parameters.

    Returns:
        Tuple of (updated_pytree, fixed_keys_list).
    """
    # Convert to FlatState for manipulation
    param_state = sl.FlatState.from_pytree(params)

    # Build parameter updates from provided values
    name_keys, updates = _build_param_updates(param_state, param_values)

    updated_state = sl.update_state(param_state, updates)

    # Identify additional fixed parameters
    user_fixed_keys = _resolve_keys(updated_state, fixed) if fixed else set()
    if fixed_predicate is not None:
        user_fixed_keys |= {
            key for key, value in updated_state.raw_mapping.items()
            if fixed_predicate(key, value)
        }
    combined_keys = user_fixed_keys | name_keys
    fixed_sequence = [tuple(key) for key in combined_keys]

    # Convert back to pytree
    updated_pytree = updated_state.to_pytree()

    return updated_pytree, fixed_sequence
