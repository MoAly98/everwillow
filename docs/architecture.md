# Architecture Overview

Everwillow is intentionally small: the public API is only a handful of
functions, and most of the heavy lifting happens in `statelib`.
This page explains how the pieces fit together and how `FlatState` works under the hood.

## Core Modules

:::::{grid} 1
:gutter: 3

::::{grid-item-card} 🗂️ statelib.state
:class-header: bg-primary text-white

The heart of everwillow's parameter management.

**Key Components:**
- {class}`~everwillow.statelib.state.FlatState` - Immutable pytree wrapper
- {func}`~everwillow.statelib.state.partition_state` - Split into orthogonal subsets
- {func}`~everwillow.statelib.state.update_state` - Safe value updates
- {func}`~everwillow.statelib.state.merge_states` - Combine multiple states

:::{dropdown} How FlatState Works 🔍
:color: info

`FlatState` uses a **segment-based architecture**:

1. **Segments** - Each pytree you flatten becomes a segment with its own ID
2. **ChainMap** - Segments are layered using Python's `ChainMap` for efficient lookups
3. **Metadata** - Each segment stores:
   - Original `PyTreeDef` for reconstruction
   - Owned keys (which parameters belong to this segment)
   - Key paths (JAX's internal representation)
   - Key order (for deterministic flattening)

```python
# Single segment (most common case)
state = FlatState.from_pytree({"a": 1, "b": 2})
state.n_internal_states  # 1

# Multiple segments (from merging)
merged = merge_states(state1, state2, state3)
merged.n_internal_states  # 3
```

Later segments **shadow** earlier ones for overlapping keys!
:::

:::{dropdown} Partitioning Deep Dive 🔍
:color: info

When you call `partition_state`, everwillow:

1. **Identifies keys** - Using predicate function or explicit key set
2. **Creates two new FlatStates** - Both retain segment metadata
3. **Marks as partitioned** - `.is_partitioned` becomes `True`
4. **Blocks reconstruction** - `.to_pytree()` raises error until combined

```python
state = FlatState.from_pytree({"a": 1, "b": 2, "c": 3})

selected, remainder = partition_state(state, predicate=lambda k, v: k[0] in {"a", "c"})

# selected: {"a": 1, "c": 3}
# remainder: {"b": 2}
# Both remember they came from the same source!

restored = combine_partitions(selected, remainder)
assert restored.to_pytree() == {"a": 1, "b": 2, "c": 3}
```
:::

::::

::::{grid-item-card} 🔄 statelib.transform
:class-header: bg-info text-white

Rewrites keys and values without manual manipulation.

**Key Components:**
- {class}`~everwillow.statelib.transform.Transform` - Declarative rewrite rules
- {func}`~everwillow.statelib.transform.apply_transformations` - Apply transformations

:::{dropdown} Transformation Examples 🔍
:color: info

Useful for aligning parameter names across different models:

```python
from everwillow.statelib import Transform, apply_transformations

state = FlatState.from_pytree({"mu_signal": 5.0, "bkg": 10.0})

# Rename keys and scale values
transforms = {
    ("mu_signal",): Transform(
        new_key=("mu",), value_fn=lambda k, v: v * 2  # Scale by 2
    ),
    ("bkg",): Transform(new_key=("background",)),
}

transformed = apply_transformations(state, transforms)
# Result: {"mu": 10.0, "background": 10.0}
```

This is especially useful when combining models from different libraries!
:::

::::

::::{grid-item-card} 📊 statelib.model
:class-header: bg-success text-white

Lightweight wrappers around log-density functions.

**Key Components:**
- {class}`~everwillow.statelib.model.Model` - Wraps a logpdf callable
- {class}`~everwillow.statelib.model.CombinedModel` - Combines multiple models

:::{dropdown} Model Composition 🔍
:color: info

Perfect for multi-region fits or conditional models:

```python
from everwillow.statelib import Model, CombinedModel

# Define models for different regions
region_a_model = Model(logpdf=lambda params: -params["a"] ** 2)
region_b_model = Model(logpdf=lambda params: -params["b"] ** 2)

# Combine them
combined = CombinedModel.combine(region_a_model, region_b_model)

# Expects merged FlatState with one segment per model
state = merge_states(FlatState.from_pytree({"a": 1}), FlatState.from_pytree({"b": 2}))

total_logpdf = combined(state)  # Sum of both models
```
:::

::::

::::{grid-item-card} 🎯 fitting
:class-header: bg-warning

User-facing API for maximum likelihood fits.

**Key Components:**
- {func}`~everwillow.fitting.fit` - Unconditional MLE
- {func}`~everwillow.fitting.fixed_param_fit` - Profile likelihood
- {class}`~everwillow.fitting.FitResult` - Results container

:::{dropdown} How fitting.fit() Works 🔍
:color: warning

The `fit()` function orchestrates everything:

```python
def fit(nll_fn, params, fixed=None, **kwargs):
    # 1. Convert to FlatState
    state = FlatState.from_pytree(params)

    # 2. Partition if needed
    if fixed:
        fixed_state, free_state = partition_state(...)

    # 3. Wrap NLL to work with flat arrays
    def wrapped_nll(free_values_array):
        # Update free state
        updated = update_state(free_state, dict(zip(keys, free_values_array)))
        # Combine with fixed
        full = combine_partitions(fixed_state, updated)
        # Call user's NLL
        return nll_fn(full.to_pytree())

    # 4. Optimize with optimistix
    solution = optimistix.minimise(wrapped_nll, ...)

    # 5. Reconstruct and return
    return FitResult(params=fitted_pytree, nll=..., ...)
```

Your NLL function **always** receives the full pytree structure!
:::

::::

:::::

## Data Flow

Here's what happens when you call `fit()`:

```{figure} images/fit_data_flow.svg
:alt: fit() data flow diagram
:align: center
:width: 90%
```

The optimization loop repeatedly: updates free parameters, combines with fixed parameters, reconstructs the full pytree, calls your NLL function, computes gradients, and updates. Your NLL function always receives the complete parameter pytree.

## Integration Points

::::{grid} 1
:gutter: 3

:::{grid-item-card} 🔌 Model Library Integration
:class-header: bg-dark text-white

Everwillow deliberately avoids opinions about modeling libraries. As long as you can provide:

1. **A differentiable NLL function**
2. **Parameters as a pytree**

You can use everwillow! The {doc}`quickstart` shows adapters for `pyhs3`, `evermore`, and `pyhf`.

**Integration Pattern:**
```python
# 1. Extract parameters from your model as a pytree
params = model.to_pytree()  # or build dict manually

# 2. Define NLL that reconstructs model state
def nll(params):
    # Rebuild your model's internal state
    model_state = reconstruct_model(params)
    # Call model's loss
    return model.compute_nll(model_state)


# 3. Fit!
result = ew.fit(nll, params, fixed=[...])
```
:::

:::{grid-item-card} ⚡ Optimizer Integration
:class-header: bg-secondary text-white

Currently uses [optimistix](https://docs.kidger.site/optimistix/) (BFGS by default), but the architecture supports other optimizers:

**Requirements for optimizer:**
- Accept callable `f(x) -> scalar`
- Accept initial values as JAX array
- Return optimized values

**Custom Solver:**
```python
import optimistix as optx

custom_solver = optx.BFGS(rtol=1e-6, atol=1e-6)
result = ew.fit(nll, params, solver=custom_solver)
```
:::

::::

## Design Principles

When extending everwillow or building on top of it, keep these principles in mind:

::::{tab-set}

:::{tab-item} Pytrees Everywhere

**At the boundary, use pytrees:**
- Accept pytrees as input
- Return pytrees as output
- Convert to `FlatState` internally only when needed

```python
# Good
def my_function(params: PyTree) -> PyTree:
    state = FlatState.from_pytree(params)
    # ... work with state ...
    return state.to_pytree()


# Bad
def my_function(state: FlatState) -> FlatState:
    # Users shouldn't manage FlatState directly
    ...
```
:::

:::{tab-item} Immutability

**FlatState instances are immutable:**
- Operations return new instances
- No in-place mutations
- Cheap to copy (shallow copy of metadata)

```python
# Good
updated = update_state(state, {("a",): 42})

# Bad (doesn't exist!)
state.set_value(("a",), 42)  # ❌ No such method
```
:::

:::{tab-item} Delegation

**Use statelib helpers instead of reimplementing:**

```python
# Good
from everwillow.statelib import partition_state

fixed, free = partition_state(state, keys=fixed_keys)

# Bad - reinventing the wheel
fixed_dict = {k: v for k, v in state.items() if k in fixed_keys}
# You'll lose segment metadata! ❌
```
:::

:::{tab-item} Segment Awareness

**Understand single vs multi-segment states:**

```python
# Single segment (most common)
state = FlatState.from_pytree({"a": 1})
state.n_internal_states  # 1
state.to_pytree()  # ✅ Works!

# Multiple segments
merged = merge_states(state1, state2)
merged.n_internal_states  # 2
merged.to_pytree()  # ❌ Raises ValueError!

# Need to split first
seg1, seg2 = split_state(merged)
seg1.to_pytree()  # ✅ Works!
```
:::

::::

## Next Steps

::::{grid} 1
:gutter: 2

:::{grid-item-card} 👥 Introduction
:link: introduction
:link-type: doc

High-level workflows and common patterns
:::

:::{grid-item-card} 📚 Statelib Details
:link: statelib_overview
:link-type: doc

Complete statelib documentation with examples
:::

:::{grid-item-card} 📖 API Reference
:link: api/index
:link-type: doc

Full function and class documentation
:::

::::

## High-Level Design

```{figure} images/statelib_design.svg
:alt: Statelib architecture
:align: center
:width: 80%
```
