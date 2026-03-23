# Parameter Management

Everwillow's `statelib` module provides immutable containers for managing model parameters. You wrap parameter dicts in a `State` to pass them to `fit()`, `uncertainties()`, and other APIs. The `State` flattens nested structures into a flat mapping with dot-separated string keys, tracks the original structure for round-tripping, and provides utilities for updating and partitioning parameters.

## Creating and inspecting a State

```python
import everwillow.statelib as sl

# Create from any dict (or nested dict)
state = sl.State.from_pytree({"a": 1.0, "b": {"c": 2.0, "d": 3.0}})

# Flat view with string keys
print(state.to_dict())
# {'a': 1.0, 'b.c': 2.0, 'b.d': 3.0}

# Access by string key
print(state["b.c"])  # 2.0

# Round-trip back to original structure
print(state.to_pytree())
# {'a': 1.0, 'b': {'c': 2.0, 'd': 3.0}}
```

The default separator is `"."`. Override it with `sep=` or use `sep=None` for tuple keys:

```python
state_tuple = sl.State.from_pytree({"a": {"b": 1.0}}, sep=None)
print(list(state_tuple.keys()))  # [('a', 'b')]

state_slash = sl.State.from_pytree({"a": {"b": 1.0}}, sep="/")
print(list(state_slash.keys()))  # ['a/b']
```

## Updating parameters

`update()` returns a new `State` with specific values replaced  - the original is unchanged:

```python
modified = sl.update(state, updates={"a": 99.0})
print(modified["a"])  # 99.0
print(state["a"])  # 1.0 (original unchanged)
```

## Partitioning and recombining

Split a `State` into two groups using a predicate. Excluded entries become `None`; use `.notnone` to see only the active ones:

```python
left, right = sl.partition(
    state,
    predicate=lambda key, _: key.startswith("b"),
)

print(left.notnone)
# {'b.c': 2.0, 'b.d': 3.0}

print(right.notnone)
# {'a': 1.0}

# Recombine into the original state
restored = sl.combine_partitions(left, right)
assert restored.to_pytree() == state.to_pytree()
```

## Merging and splitting

`merge()` combines multiple states into one. `split()` reverses it:

```python
state_x = sl.State.from_pytree({"x": 1.0})
state_y = sl.State.from_pytree({"y": 2.0, "z": 3.0})

merged = sl.merge(state_x, state_y)
print(merged.to_dict())
# {'x': 1.0, 'y': 2.0, 'z': 3.0}

# to_pytree() returns a tuple  - one entry per original state
print(merged.to_pytree())
# ({'x': 1.0}, {'y': 2.0, 'z': 3.0})

# split() recovers individual states
part_x, part_y = sl.split(merged)
print(part_x.to_pytree())  # {'x': 1.0}
print(part_y.to_pytree())  # {'y': 2.0, 'z': 3.0}
```

When states share overlapping keys (e.g. after renaming), the merged state holds a single value and `to_pytree()` propagates it to each sub-pytree. See [Combining Models](combining-models.md) for the full pattern.

## Renaming keys

`apply_transformations()` renames keys in a state. The original treedef is preserved, so `to_pytree()` still produces the original structure:

```python
state = sl.State.from_pytree({"m": 125.0, "scale": 1.0})
renamed = sl.apply_transformations(state, {"m": sl.Transform(new_key="mass")})

print(renamed.to_dict())
# {'mass': 125.0, 'scale': 1.0}

# to_pytree() still uses the original key
print(renamed.to_pytree())
# {'m': 125.0, 'scale': 1.0}
```

`Transform` also accepts a `value_fn` to transform the value at the same time:

```python
scaled = sl.apply_transformations(
    state,
    {"m": sl.Transform(new_key="mass_gev", value_fn=lambda _k, v: v * 1000)},
)
print(scaled["mass_gev"])  # 125000.0
```
