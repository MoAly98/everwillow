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

print(state_a.mapping)  # {('a', 'b'): 1.0}
print(state_b.mapping)  # {('c', 'd'): 5.0, ('e', 'f'): 3.0}
print(state_c.mapping)  # {('c', 'd'): 7.0, ('g', 'h'): 4.0}

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

print(aligned_b.mapping)  # {('correlated',): 5.0, ('e', 'f'): 3.0}
print(aligned_c.mapping)  # {('c', 'd'): 7.0, ('correlated',): 4.0}

# Merge the aligned states into one mapping. Later segments overwrite earlier keys.
merged_mapping, metadata = sl.merge(state_a, aligned_b, aligned_c)
print(merged_mapping)
# {('a', 'b'): 1.0, ('correlated',): 4.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# Split the merged mapping back into the original states.
seg_a, seg_b, seg_c = sl.split(merged_mapping, metadata)
print(seg_a.to_pytree())  # {'a': {'b': 1.0}}
print(seg_b.to_pytree())  # {'c': {'d': 4.0}, 'e': {'f': 3.0}}
print(seg_c.to_pytree())  # {'c': {'d': 7.0}, 'g': {'h': 4.0}}

# Partition the merged mapping and recombine it.
first_partition, second_partition = sl.partition(
    merged_mapping,
    predicate=lambda key, _: "a" in key,
)

print(dict(first_partition.mapping))  # {('a', 'b'): 1.0}
print(
    dict(second_partition.mapping)
)  # {('correlated',): 4.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

recombined = sl.combine_partitions(first_partition, second_partition)
print(recombined)
# {('a', 'b'): 1.0, ('correlated',): 4.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# Equality holds because both partitions came from the same mapping.
assert recombined == merged_mapping
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
- **Partitioning** – `sl.partition(mapping, predicate=...)` produces two `PartitionedMapping` objects keyed by the same origin id. Each partition intentionally lacks some leaves, so calling `to_pytree()` directly will raise a `ValueError`.
- **Recombining partitions** – `sl.combine_partitions(left, right)` reassembles exactly the original mapping as long as the partitions share the same origin.

```python
flags, rest = sl.partition(
    state.mapping,
    predicate=lambda key, _value: key[0] == "flags",
)

try:
    sl.State(flags.mapping, treedef=state.treedef).to_pytree()
except ValueError:
    pass  # partitions are intentionally incomplete

restored = sl.combine_partitions(flags, rest)
print(restored)
# {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}
assert restored == state.mapping
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

## Models

- `Model(logpdf=...)` wraps a log-density callable so it can consume either pytree inputs or `State` instances.
- `CombinedModel.combine(model_a, model_b, ...)` builds a model that expects a merged mapping with one segment per component and returns the sum of their evaluations.

```python
model = sl.Model(logpdf=lambda tree: -tree["a"] ** 2)
combo = sl.CombinedModel.combine(model, model)

merged_mapping, metadata = sl.merge(
    sl.State.from_pytree({"a": 1}),
    sl.State.from_pytree({"a": 2}),
)
assert combo(merged_mapping, metadata) == sum(
    model(segment) for segment in sl.split(merged_mapping, metadata)
)
```

## Testing & Support

The accompanying unit tests in `tests/statelib` cover the public API. If you encounter unexpected behaviour, try to reproduce it with the pytest suite before filing an issue.
