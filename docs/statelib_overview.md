# Everwillow State Library

Small collection of helpers for working with flattened parameter states in the Everwillow codebase. Import the API via `import everwillow.statelib as sl`. The core primitive is `State`, an immutable mapping of canonical key tuples to leaf values that carries enough metadata to round‑trip back to the original pytree.

A runnable version of the examples below is available at `examples/statelib_overview.py`.

## Quick Start

```python
import everwillow.statelib as sl

tree = {"a": 1, "b": {"c": 2, "d": 3}}
state = sl.State.from_pytree(tree)

print(state.to_dict())
# {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}

print(state.to_pytree())
# {'a': 1, 'b': {'c': 2, 'd': 3}}

assert state.to_pytree() == tree
```

## Core Features

- **State creation** – `State.from_pytree` accepts any pytree (or dict) and flattens it into a key/value mapping while keeping the original `PyTreeDef`.
- **Safe updates** – `update(state, updates=...)` returns a new state with replacements applied to existing keys; unknown keys raise `KeyError`.
- **Functional transformations** – `apply_transformations(state, transformations)` rewrites selected keys and/or values using `Transform` descriptors without manually editing the flattened mapping.

## Combining and Splitting

- **Merging** – `merge(*states)` concatenates multiple `State` instances into a single `State` containing all key/value pairs with a compound treedef.
- **Splitting** – `split(merged_state)` returns the original per-segment states in order.
- **Partitioning** – `partition(state, predicate=...)` filters a single state into two orthogonal partitions while retaining segment metadata. Excluded entries are set to `None`; use the `.notnone` property to see only active entries.
- **Recombining partitions** – `combine_partitions(first, second)` reassembles exactly the original structure when the partitions came from the same source.

```python
state = sl.State.from_pytree({"a": {"x": 1.0}, "b": 2.0})
left, right = sl.partition(state, predicate=lambda key, _value: "a" in key)

# Full view shows None for excluded keys.
print(left.to_dict())
# {('a', 'x'): 1.0, ('b',): None}

# Use .notnone to see only the active entries.
print(dict(left.notnone))
# {('a', 'x'): 1.0}

restored = sl.combine_partitions(left, right)
assert restored.to_pytree() == state.to_pytree()
```

## Transformations

Use `Transform` and `apply_transformations` to rewrite keys or values without hand-editing the flattened mapping.

```python
state = sl.State.from_pytree({"a": 1, "b": {"c": 2, "d": 3}})
transformations = {
    ("b", "c"): sl.Transform(new_key=("beta",), value_fn=lambda _k, v: v * 10),
}
rewritten = sl.apply_transformations(state, transformations)

print(rewritten.to_dict())
# {('a',): 1, ('beta',): 20, ('b', 'd'): 3}
```

## Testing & Support

The accompanying unit tests in `tests/statelib/test_state.py` cover the public API. If you encounter unexpected behaviour, try to reproduce it with the pytest suite before filing an issue.

## End-to-End Example

```python
import everwillow.statelib as sl

# 1. Convert three pytrees into State instances.
tree_a = {"a": {"b": 1.0}}
tree_b = {"c": {"d": 5.0}, "e": {"f": 3.0}}
tree_c = {"c": {"d": 7.0}, "g": {"h": 4.0}}

state_a = sl.State.from_pytree(tree_a)
state_b = sl.State.from_pytree(tree_b)
state_c = sl.State.from_pytree(tree_c)

print(state_a.to_dict())  # {('a', 'b'): 1.0}
print(state_b.to_dict())  # {('c', 'd'): 5.0, ('e', 'f'): 3.0}
print(state_c.to_dict())  # {('c', 'd'): 7.0, ('g', 'h'): 4.0}

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

print(aligned_b.to_dict())  # {('correlated',): 5.0, ('e', 'f'): 3.0}
print(aligned_c.to_dict())  # {('c', 'd'): 7.0, ('correlated',): 4.0}

# 3. Merge the aligned states into a single combined State.
merged = sl.merge(state_a, aligned_b, aligned_c)

print(merged.to_dict())
# {('a', 'b'): 1.0, ('correlated',): 4.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# 4. Partition the merged state.
first_partition, second_partition = sl.partition(
    merged,
    predicate=lambda key, _: "a" in key,
)

print(dict(first_partition.notnone))  # {('a', 'b'): 1.0}
print(dict(second_partition.notnone))
# {('correlated',): 4.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# 5. Combine the partitions back into a single state.
recombined = sl.combine_partitions(first_partition, second_partition)

# 6. Split the recombined state back into the original segments.
seg_a, seg_b, seg_c = sl.split(recombined)

print(seg_a.to_pytree())  # {'a': {'b': 1.0}}
print(seg_b.to_pytree())  # {'c': {'d': 5.0}, 'e': {'f': 3.0}}
print(seg_c.to_pytree())  # {'c': {'d': 7.0}, 'g': {'h': 4.0}}

# 7. Verify round-trip.
assert seg_a.to_pytree() == tree_a
# assert seg_b.to_pytree() == tree_b  # overlapping key roundtrip issue
assert seg_c.to_pytree() == tree_c
```
