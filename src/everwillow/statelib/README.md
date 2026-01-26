# Everwillow State Library

Small collection of helpers for working with flattened parameter states in the Everwillow codebase. Import the API via `import everwillow.statelib as sl`. The core primitive is `State`, an immutable mapping of canonical keys to leaf values that carries enough metadata to round-trip back to the original pytree.

## End-to-End Example

```python
import everwillow.statelib as sl

tree_a = {"a": {"b": 1.0}}
tree_b = {"c": {"d": 5.0}, "e": {"f": 3.0}}
tree_c = {"c": {"d": 7.0}, "g": {"h": 4.0}}

state_a = sl.State.from_pytree(tree_a)
state_b = sl.State.from_pytree(tree_b)
state_c = sl.State.from_pytree(tree_c)

print(state_a.mapping)
# {('a', 'b'): 1.0}
print(state_b.mapping)
# {('c', 'd'): 5.0, ('e', 'f'): 3.0}
print(state_c.mapping)
# {('c', 'd'): 7.0, ('g', 'h'): 4.0}

# Align overlapping leaves by rewriting keys.
aligned_b = sl.apply_transformations(
    state_b,
    {("c", "d"): sl.Transform(new_key=("correlated",))},
)
aligned_c = sl.apply_transformations(
    state_c,
    {("g", "h"): sl.Transform(new_key=("correlated",))},
)

assert ("correlated",) in aligned_b.mapping
assert ("correlated",) in aligned_c.mapping

print(aligned_b.mapping)
# {('correlated',): 5.0, ('e', 'f'): 3.0}
print(aligned_c.mapping)
# {('c', 'd'): 7.0, ('correlated',): 4.0}

# Merge the aligned states into one mapping. Earlier segments overwrite earlier keys (ChainMap semantics).
merged_mapping, metadata = sl.merge(state_a, aligned_b, aligned_c)
print(merged_mapping)
# FrozenChainMap({('c', 'd'): 7.0, ('correlated',): 5.0, ('e', 'f'): 3.0, ('a', 'b'): 1.0})

# Split the merged mapping back into the original states.
seg_a, seg_b, seg_c = sl.split(merged_mapping, metadata)
print(seg_a.to_pytree())
# {'a': {'b': 1.0}}
print(seg_b.to_pytree())
# {'c': {'d': 4.0}, 'e': {'f': 3.0}}
print(seg_c.to_pytree())
# {'c': {'d': 7.0}, 'g': {'h': 4.0}}

# Partition a state and recombine it.
# partition() works on State objects and returns State objects
# with None values for excluded keys.
state_to_partition = sl.State.from_pytree({"a": {"b": 1.0}, "c": {"d": 7.0}})
first_partition, second_partition = sl.partition(
    state_to_partition,
    predicate=lambda key, _: "a" in key,
)

# Each partition is a State with None for excluded keys
print({k: v for k, v in first_partition.items() if v is not None})
# {('a', 'b'): 1.0}
print({k: v for k, v in second_partition.items() if v is not None})
# {('c', 'd'): 7.0}

recombined = sl.combine_partitions(first_partition, second_partition)
# recombined is a State with the original structure
assert recombined.to_pytree() == state_to_partition.to_pytree()
```

## Quick Start

```python
import everwillow.statelib as sl

tree = {"a": 1, "b": {"c": 2, "d": 3}}
state = sl.State.from_pytree(tree)

# Inspect flattened entries.
assert dict(state.mapping) == {
    ("a",): 1,
    ("b", "c"): 2,
    ("b", "d"): 3,
}

# Round-trip to the original structure.
assert state.to_pytree() == tree
```

## Core Features

- **State creation** – `State.from_pytree` accepts any pytree (or an existing `State`) and flattens it into a key/value mapping while keeping the original `PyTreeDef`.
- **Safe updates** – `sl.update(state, updates)` returns a new state with replacements applied to existing keys; unknown keys raise `KeyError`.
- **Canonical keys** – `sl.canonicalize_key(path)` converts the key objects emitted by JAX into plain tuples (or joined strings when `sep=` is provided).

## Combining and Splitting

- **Merging** – `sl.merge(*states)` concatenates multiple `State` instances into one mapping and returns the metadata required to split it.
- **Splitting** – `sl.split(mapping, metadata)` rebuilds the original per-state segments in order.
- **Partitioning** – `sl.partition(state, predicate=...)` produces two `State` objects with the same `treedefmeta`. Excluded keys are set to `None` in each partition.
- **Recombining partitions** – `sl.combine_partitions(left, right)` reassembles exactly the original state as long as the partitions share the same `treedefmeta`.

```python
flags, rest = sl.partition(
    state,
    predicate=lambda key, _value: key[0] == "flags",
)

# Each partition has None for excluded keys
print({k: v for k, v in flags.items() if v is not None})
# {('flags', ...): ...}

restored = sl.combine_partitions(flags, rest)
assert restored.to_pytree() == state.to_pytree()
```

## Transformations

Use `Transform` and `apply_transformations` to rewrite keys or values without hand-editing the flattened mapping.

```python
transformations = {
    ("b", "c"): sl.Transform(new_key=("beta",), value_fn=lambda _k, v: v * 10),
}
rewritten = sl.apply_transformations(state, transformations)
assert dict(rewritten.mapping)[("beta",)] == 20
```

## Testing & Support

The accompanying unit tests in `tests/statelib` cover the public API. If you encounter unexpected behaviour, try to reproduce it with the pytest suite before filing an issue.
