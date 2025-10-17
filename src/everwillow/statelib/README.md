# Everwillow State Library

Small collection of helpers for working with flattened parameter states in the Everwillow codebase. Import the API via `import everwillow.statelib as sl`. The core primitive is `FlatState`, an immutable mapping of canonical key tuples to leaf values that carries enough metadata to round‑trip back to the original pytree.

## End-to-End Example

```python
import everwillow.statelib as sl

# 1. Convert three pytrees into FlatState instances.
tree_a = {"a": {"b": 1.0}}
tree_b = {"c": {"d": 5.0}, "e": {"f": 3.0}}
tree_c = {"c": {"d": 7.0}, "g": {"h": 4.0}}

state_a = sl.FlatState.from_pytree(tree_a)
state_b = sl.FlatState.from_pytree(tree_b)
state_c = sl.FlatState.from_pytree(tree_c)

print(state_a.raw_mapping)  # -> {('a', 'b'): 1.0}
print(state_b.raw_mapping)  # -> {('c', 'd'): 5.0, ('e', 'f'): 3.0}
print(state_c.raw_mapping)  # -> {('c', 'd'): 7.0, ('g', 'h'): 4.0}

# 2. Transform keys so overlapping entries align (same name) -> we're correlating `("c", "d")` with `("g", "h")` here.
aligned_b = sl.apply_transformations(
    state_b,
    {
        ("c", "d"): sl.Transform(new_key=("correlated",))
    },
)
aligned_c = sl.apply_transformations(
    state_c,
    {
        ("g", "h"): sl.Transform(new_key=("correlated",))
    },
)

# Both transformed states now expose the shared key/value pair.
assert ("correlated",) in aligned_b.raw_mapping

print(aligned_b.raw_mapping)
# -> {('correlated',): 0.0, ('e', 'f'): 3.0}
print(aligned_c.raw_mapping)
# -> {('c', 'd'): 7.0, ('correlated',): 0.0}

# 3. Merge the aligned states into a single combined FlatState.
merged = sl.merge_states(state_a, aligned_b, aligned_c)

print(dict(merged.raw_mapping))
# -> {('a', 'b'): 1.0, ('correlated',): 0.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# 4. Partition the merged state into two orthogonal partitions.
first_partition, second_partition = sl.partition_state(
    merged,
    predicate=lambda key, _: "a" in key,
)

print(dict(first_partition.raw_mapping))   # -> {('a', 'b'): 1.0}
print(dict(second_partition.raw_mapping))  # -> {('correlated',): 0.0, ('e', 'f'): 3.0, ('c', 'd'): 7.0}

# 5. Combine the partitions back into a single state.
recombined = sl.combine_partitions(first_partition, second_partition)

# 6. Split the recombined state back into the original segments.
seg_a, seg_b, seg_c = sl.split_state(recombined)

print(tuple(seg.to_pytree() for seg in (seg_a, seg_b, seg_c)))
# -> ({'a': {'b': 1.0}}, {'c': {'d': 0.0}, 'e': {'f': 3.0}}, {'c': {'d': 7.0}, 'g': {'h': 0.0}})

# 7. Convert segments back to pytrees and verify they match the originals.
assert seg_a.to_pytree() == tree_a
assert seg_b.to_pytree() == tree_b
assert seg_c.to_pytree() == tree_c
```

## Quick Start

```python
import everwillow.statelib as sl

tree = {"a": 1, "b": {"c": 2, "d": 3}}
state = sl.FlatState.from_pytree(tree)

# Inspect flattened entries.
assert dict(state.raw_mapping) == {
    ("a",): 1,
    ("b", "c"): 2,
    ("b", "d"): 3,
}

# Round-trip to the original structure.
assert state.to_pytree() == tree
```

## Core Features

- **FlatState creation** – `FlatState.from_pytree` accepts any pytree (or an existing `FlatState`) and flattens it into a key/value mapping while keeping the original `PyTreeDef`.
- **Safe updates** – `update_state(state, updates)` returns a new state with replacements applied to existing keys; unknown keys raise `KeyError`.
- **Functional mapping** – `map_state(fn, state)` applies a function to every `(key, value)` pair and yields a new `FlatState` with the transformed values.

## Combining and Splitting

- **Merging** – `merge_states(*states)` concatenates multiple `FlatState` instances into one with distinct segments. Later segments shadow overlapping keys from earlier ones; normalize values first if you need them to agree.
- **Splitting** – `split_state(state)` returns the original per-segment states in order.
- **Partitioning** – `partition_state(state, keys=..., predicate=...)` filters a single state into two orthogonal partitions while retaining segment metadata. Each partition intentionally lacks some leaves, so calling `to_pytree()` will raise a guidance error.
- **Recombining partitions** – `combine_partitions(first, second)` reassembles exactly the original structure when the partitions came from the same source.
- **Partition detection** – `FlatState.is_partitioned` is a cached property that flags states missing leaves from any segment, allowing callers to guard operations that require a complete pytree.

```python
flags, rest = sl.partition_state(
    state,
    predicate=lambda key, _value: key[0] == "flags",
)

try:
    flags.to_pytree()
except ValueError:
    pass  # partitions are intentionally incomplete

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
assert dict(rewritten.raw_mapping)[("beta",)] == 20
```

## Models

- `Model(logpdf=...)` wraps a log-density callable so it can consume either pytree inputs or `FlatState` instances.
- `CombinedModel.combine(model_a, model_b, ...)` builds a model that expects a merged `FlatState` with one segment per component and returns the sum of their evaluations.

```python
model = sl.Model(logpdf=lambda tree: -tree["a"] ** 2)
combo = sl.CombinedModel.combine(model, model)

state = sl.merge_states(
    sl.FlatState.from_pytree({"a": 1}),
    sl.FlatState.from_pytree({"a": 2}),
)
assert combo(state) == sum(model(segment) for segment in sl.split_state(state))
```

## Testing & Support

The accompanying unit tests in `tests/test_statelib_state.py` cover the public API. If you encounter unexpected behaviour, try to reproduce it with the pytest suite before filing an issue.
