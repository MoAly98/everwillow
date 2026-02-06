"""Runnable overview of everwillow.statelib.

Demonstrates all major features of the statelib module:
State creation, inspection, updates, transformations,
merge/split, partition/combine, and the end-to-end workflow.

Run with::

    uv run python examples/statelib_overview.py
"""

from __future__ import annotations

import everwillow.statelib as sl

SEPARATOR = "=" * 60


def section_quick_start() -> None:
    print(SEPARATOR)
    print("1. Quick Start")
    print(SEPARATOR)

    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    state = sl.State.from_pytree(tree)

    print(f"  Input tree:       {tree}")
    print(f"  Flattened (dict): {state.to_dict()}")
    print(f"  State repr:       {state!r}")
    print(f"  Round-trip:       {state.to_pytree()}")
    print()

    assert state.to_dict() == {("a",): 1, ("b", "c"): 2, ("b", "d"): 3}
    assert state.to_pytree() == tree


def section_mapping_interface() -> None:
    print(SEPARATOR)
    print("2. Mapping Interface")
    print(SEPARATOR)

    state = sl.State.from_pytree({"x": {"y": 3.0}})

    print(f"  Contains ('x', 'y'): {('x', 'y') in state}")
    print(f"  state['x', 'y']:     {state['x', 'y']}")
    print(f"  keys:                {list(state.keys())}")
    print()


def section_updates() -> None:
    print(SEPARATOR)
    print("3. Safe Updates")
    print(SEPARATOR)

    state = sl.State.from_pytree({"a": 1.0, "b": 2.0})
    updated = sl.update(state, updates={("b",): 99.0})

    print(f"  Original: {state.to_dict()}")
    print(f"  Updated:  {updated.to_dict()}")

    # unknown keys raise KeyError
    try:
        sl.update(state, updates={("missing",): 0.0})
    except KeyError as exc:
        print(f"  KeyError for missing key: {exc}")
    print()


def section_merge_split() -> None:
    print(SEPARATOR)
    print("4. Merge and Split")
    print(SEPARATOR)

    state_a = sl.State.from_pytree({"a": 1.0})
    state_b = sl.State.from_pytree({"b": {"c": 2.0}})

    merged = sl.merge(state_a, state_b)
    print(f"  state_a: {state_a.to_dict()}")
    print(f"  state_b: {state_b.to_dict()}")
    print(f"  merged:  {merged.to_dict()}")

    restored_a, restored_b = sl.split(merged)
    print(f"  split[0] pytree: {restored_a.to_pytree()}")
    print(f"  split[1] pytree: {restored_b.to_pytree()}")
    print()

    assert restored_a.to_pytree() == {"a": 1.0}
    assert restored_b.to_pytree() == {"b": {"c": 2.0}}


def section_partition_combine() -> None:
    print(SEPARATOR)
    print("5. Partition and Combine")
    print(SEPARATOR)

    state = sl.State.from_pytree({"a": {"x": 1.0}, "b": 2.0})
    left, right = sl.partition(state, predicate=lambda key, _value: "a" in key)

    print(f"  Original:        {state.to_dict()}")
    print(f"  Left  (all):     {left.to_dict()}")
    print(f"  Left  (notnone): {dict(left.notnone)}")
    print(f"  Right (all):     {right.to_dict()}")
    print(f"  Right (notnone): {dict(right.notnone)}")

    combined = sl.combine_partitions(left, right)
    print(f"  Recombined:      {combined.to_pytree()}")
    print()

    assert combined.to_pytree() == state.to_pytree()


def section_transformations() -> None:
    print(SEPARATOR)
    print("6. Transformations")
    print(SEPARATOR)

    state = sl.State.from_pytree({"a": 1, "b": {"c": 2, "d": 3}})
    transformations = {
        ("b", "c"): sl.Transform(new_key=("beta",), value_fn=lambda _k, v: v * 10),
    }
    rewritten = sl.apply_transformations(state, transformations)

    print(f"  Original:  {state.to_dict()}")
    print(f"  Rewritten: {rewritten.to_dict()}")
    print()

    assert rewritten.to_dict()["beta",] == 20


def section_end_to_end() -> None:
    print(SEPARATOR)
    print("7. End-to-End Example")
    print(SEPARATOR)

    # 1. Convert three pytrees into State instances.
    tree_a = {"a": {"b": 1.0}}
    tree_b = {"c": {"d": 5.0}, "e": {"f": 3.0}}
    tree_c = {"c": {"d": 7.0}, "g": {"h": 4.0}}

    state_a = sl.State.from_pytree(tree_a)
    state_b = sl.State.from_pytree(tree_b)
    state_c = sl.State.from_pytree(tree_c)

    print(f"  state_a: {state_a.to_dict()}")
    print(f"  state_b: {state_b.to_dict()}")
    print(f"  state_c: {state_c.to_dict()}")

    # 2. Transform keys so overlapping entries align (same name).
    # We're correlating ("c", "d") in state_b with ("g", "h") in state_c.
    aligned_b = sl.apply_transformations(
        state_b,
        {("c", "d"): sl.Transform(new_key=("correlated",))},
    )
    aligned_c = sl.apply_transformations(
        state_c,
        {("g", "h"): sl.Transform(new_key=("correlated",))},
    )

    assert ("correlated",) in aligned_b
    assert ("correlated",) in aligned_c

    print(f"  aligned_b: {aligned_b.to_dict()}")
    print(f"  aligned_c: {aligned_c.to_dict()}")

    # 3. Merge the aligned states into a single combined State.
    merged = sl.merge(state_a, aligned_b, aligned_c)
    print(f"  merged: {merged.to_dict()}")

    # 4. Partition the merged state.
    first_part, second_part = sl.partition(merged, predicate=lambda key, _: "a" in key)
    print(f"  partition[0] (notnone): {dict(first_part.notnone)}")
    print(f"  partition[1] (notnone): {dict(second_part.notnone)}")

    # 5. Recombine the partitions.
    recombined = sl.combine_partitions(first_part, second_part)

    # 6. Split back into the original segments.
    seg_a, seg_b, seg_c = sl.split(recombined)
    print(f"  seg_a pytree: {seg_a.to_pytree()}")
    print(f"  seg_b pytree: {seg_b.to_pytree()}")
    print(f"  seg_c pytree: {seg_c.to_pytree()}")
    print()

    # 7. Verify structure is preserved (overlapping keys get merged value).
    assert seg_a.to_pytree() == tree_a
    assert seg_b.to_pytree() == {"c": {"d": 4.0}, "e": {"f": 3.0}}
    assert seg_c.to_pytree() == tree_c


def main() -> None:
    section_quick_start()
    section_mapping_interface()
    section_updates()
    section_merge_split()
    section_partition_combine()
    section_transformations()
    section_end_to_end()


if __name__ == "__main__":
    main()
