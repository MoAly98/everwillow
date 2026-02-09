# State Library

The `statelib` module provides immutable state containers for working with JAX pytrees. Import via `import everwillow.statelib as sl`.

## Quick Start

::::{tab-set}
:::{tab-item} JAX
```python
import jax.numpy as jnp
import everwillow as ew
import everwillow.statelib as sl


# Counting model: signal + two backgrounds with shape/norm modifiers
def nll(params, obs):
    mu, norm1, norm2, shape1 = (
        params["mu"],
        params["norm1"],
        params["norm2"],
        params["shape1"],
    )
    # Signal + background expectation
    signal = mu * 3.0
    bkg1 = jnp.exp(norm1 * jnp.log(1.1)) * (10.0 + shape1 * 2.0)
    bkg2 = jnp.exp(norm2 * jnp.log(1.05)) * (20.0 + shape1 * 3.0)
    n_exp = signal + bkg1 + bkg2

    # Poisson + Gaussian constraints
    poisson = n_exp - obs["n"] * jnp.log(n_exp)
    constraints = 0.5 * (norm1**2 + norm2**2 + shape1**2)
    return poisson + constraints


params = sl.State.from_pytree({"mu": 1.0, "norm1": 0.0, "norm2": 0.0, "shape1": 0.0})
result = ew.fit(nll, params, {"n": 37.0})
print(result.params)
# {'mu': 2.33, 'norm1': ~0, 'norm2': ~0, 'shape1': ~0}
```
:::
:::{tab-item} State Basics
```python
import everwillow.statelib as sl

# Create state from any pytree
tree = {"a": 1, "b": {"c": 2, "d": 3}}
state = sl.State.from_pytree(tree)

# Access as flat mapping with tuple keys
print(state.to_dict())
# {('a',): 1, ('b', 'c'): 2, ('b', 'd'): 3}

# Round-trip back to original structure
assert state.to_pytree() == tree
```
:::
::::

## Core Operations

::::{grid} 2
:gutter: 3

:::{grid-item-card} Create
:class-header: bg-primary text-white

```python
# From any pytree
state = sl.State.from_pytree({"x": 1.0})

# Access values
state[("x",)]  # 1.0
state["x",]  # 1.0 (shorthand)
```
:::

:::{grid-item-card} Update
:class-header: bg-primary text-white

```python
# Immutable updates (returns new state)
new = sl.update(state, {("x",): 2.0})

# Original unchanged
assert state[("x",)] == 1.0
assert new[("x",)] == 2.0
```
:::

::::

## Combining States

::::{grid} 1
:gutter: 3

:::{grid-item-card} Merge and Split
:class-header: bg-info text-white

Combine multiple states into one, then split back:

```python
state_a = sl.State.from_pytree({"a": 1.0})
state_b = sl.State.from_pytree({"b": 2.0})

# Merge into single state
merged = sl.merge(state_a, state_b)
print(merged.to_dict())
# {('a',): 1.0, ('b',): 2.0}

# Split back to original segments
restored_a, restored_b = sl.split(merged)
assert restored_a.to_pytree() == {"a": 1.0}
```

**Tip:** When states share keys, the **last value wins** during merge. After split, all segments see the merged value.
:::

::::

## Partitioning

::::{grid} 1
:gutter: 3

:::{grid-item-card} Partition and Combine
:class-header: bg-success text-white

Split a state by predicate, keeping structure metadata:

```python
state = sl.State.from_pytree({"a": 1.0, "b": 2.0, "c": 3.0})

# Partition by key
left, right = sl.partition(state, predicate=lambda key, value: key == ("a",))

# Excluded keys become None
print(dict(left.notnone))  # {('a',): 1.0}
print(dict(right.notnone))  # {('b',): 2.0, ('c',): 3.0}

# Recombine perfectly
restored = sl.combine_partitions(left, right)
assert restored.to_pytree() == state.to_pytree()
```

:::

::::

## Transformations

::::{grid} 1
:gutter: 3

:::{grid-item-card} Rewrite Keys and Values
:class-header: bg-warning text-dark

Rename keys or transform values without manual mapping:

```python
state = sl.State.from_pytree({"old_name": 10})

transformed = sl.apply_transformations(
    state,
    {("old_name",): sl.Transform(new_key=("new_name",), value_fn=lambda k, v: v * 2)},
)

print(transformed.to_dict())
# {('new_name',): 20}
```

See {doc}`parameter_transforms` for parameter bounds and optimization transforms.
:::

::::

## API Quick Reference

| Function | Purpose |
|----------|---------|
| `State.from_pytree(tree)` | Create state from pytree |
| `state.to_pytree()` | Reconstruct original pytree |
| `state.to_dict()` | Get flat {key: value} dict |
| `update(state, updates)` | Return new state with updates |
| `merge(*states)` | Combine multiple states |
| `split(merged)` | Split back to original segments |
| `partition(state, predicate)` | Split by predicate |
| `combine_partitions(a, b)` | Recombine partitions |
| `apply_transformations(state, transforms)` | Rewrite keys/values |

See {doc}`api/statelib/state` for complete documentation.

## Next Steps

::::{grid} 1
:gutter: 2

:::{grid-item-card} 📖 API Reference
:link: api/statelib/state
:link-type: doc

Complete function and class documentation
:::

:::{grid-item-card} 🔬 Fitting Guide
:link: introduction
:link-type: doc

See how statelib powers the fitting API
:::

::::
