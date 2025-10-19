# Introduction

Everwillow is a statistical inference library for high-energy physics built on JAX pytrees and optimistix optimizers. This guide explains how to use everwillow's fitting API, the first of several planned inference tools.

## Overview

Everwillow provides a simple workflow for statistical inference tasks. Currently, the library focuses on parameter fitting—taking your model parameters (organized as a pytree) and finding values that minimize your negative log-likelihood. Future releases will add profile likelihood scans, hypothesis tests, and limit setting.

The library handles all the complexity of parameter management internally while giving you a clean, flexible API.

## The Fitting Workflow

Here's how everwillow processes a fit request:

```{figure} images/user_workflow.svg
:alt: Everwillow fitting workflow
:align: center
:width: 85%
```

Let's break down each step:

::::{grid} 1
:gutter: 3

:::{grid-item-card} 1. Your Input
:class-header: bg-light

You provide:
- A **parameter pytree** (dict, nested dict, custom class, etc.)
- A **negative log-likelihood function** that takes your parameters
- Optionally: which parameters to **fix** during fitting

```python
params = {"mu": 1.0, "sigma": 0.5, "background": 100}


def nll(params):
    return compute_loss(params)


result = ew.fit(nll, params, fixed=["background"])
```
:::

:::{grid-item-card} 2. Internal Conversion
:class-header: bg-light

Everwillow converts your pytree to a `FlatState`:
- Flattens nested structures into key-value pairs
- Preserves structure information for reconstruction
- Each parameter gets a unique key path like `("mu",)` or `("level1", "sigma")`

This happens automatically - you never need to create a `FlatState` yourself!
:::

:::{grid-item-card} 3. Partitioning (Optional)
:class-header: bg-light

If you specified fixed parameters, everwillow:
- Splits the state into **free** and **fixed** partitions
- Only the free parameters will be optimized
- Fixed parameters maintain their exact values

This is useful for profile likelihood scans and conditional fits.
:::

:::{grid-item-card} 4. Optimization
:class-header: bg-light

The optimizer (default: BFGS) works only on free parameters:
- Wrapped to accept flat arrays
- Automatically reconstructs full pytree for your NLL function
- Efficient gradient-based optimization via JAX

Your NLL function always receives the complete parameter pytree!
:::

:::{grid-item-card} Bounds Transform (Optional)
:class-header: bg-light

If you specified bounds on parameters, everwillow transparently manages transformations:
- **Unwrap**: Initial bounded parameters → unbounded space for optimization
- **Optimize**: Optimizer works in unbounded space (no constraint violations)
- **Wrap**: Before each NLL call, transform back to bounded space
- **Gradients**: Flow back through differentiable transformations

Your NLL function always receives parameters in the **original bounded space**.
:::

::::

## State Management with FlatState

Behind the scenes, everwillow uses `FlatState` to manage your parameters efficiently.

::::{grid} 1
:gutter: 3

:::{grid-item-card} ✨ What is FlatState?
:class-header: bg-primary text-white

A `FlatState` is an immutable container that:
- **Flattens** any pytree into canonical key-value pairs
- **Remembers** the original structure for reconstruction
- **Supports** partitioning into orthogonal subsets
- **Enables** safe, efficient parameter updates

Think of it as a smart dictionary that knows how to rebuild your original data structure.
:::

:::{grid-item-card} 🔑 Key Features
:class-header: bg-info text-white

**Pytree Round-tripping**
```python
state = FlatState.from_pytree({"a": 1, "b": {"c": 2}})
tree = state.to_pytree()  # Perfect reconstruction
```

**Partitioning**
```python
free, fixed = partition_state(state, keys={("a",)})
# free has {"a": 1}, fixed has {"b": {"c": 2}}
```

**Safe Updates**
```python
updated = update_state(state, {("a",): 42})
# Original state unchanged, new state has a=42
```
:::

::::

## Parameter Bounds via Transformations

When you specify bounds on parameters, everwillow uses a transformation strategy to keep the optimizer working in unbounded space while ensuring your NLL function receives parameters within the specified bounds.

::::{grid} 1
:gutter: 3

:::{grid-item-card} 🎯 Why Transform to Unbounded Space?
:class-header: bg-primary text-white

Gradient-based optimizers (like BFGS) work best when they can explore the full parameter space without hitting hard boundaries. By transforming bounded parameters to an unbounded representation:

- The optimizer never encounters constraint violations
- Gradients remain well-behaved near boundaries
- No special constraint-handling logic is needed

Your NLL function always receives parameters in the **original bounded space**, so you never need to think about the transformations.
:::

:::{grid-item-card} 🔄 The Wrap/Unwrap Pattern
:class-header: bg-info text-white

Everwillow automatically manages the transformation lifecycle:

1. **Unwrap**: Convert your bounded initial parameters to unbounded space
2. **Optimize**: The optimizer works entirely in unbounded space
3. **Wrap**: Before each NLL evaluation, transform back to bounded space
4. **Repeat**: Gradients flow back through the transformations

This happens transparently—you just pass `bounds` to the fit function.
:::

:::{grid-item-card} 📐 Transformation Types
:class-header: bg-success text-white

Different bound types use different transformations:

- **Finite bounds** `(a, b)`: Logit/sigmoid family keeps values in `(a, b)`
- **Lower bound** `(a, None)`: Logarithmic transform ensures values stay above `a`
- **Upper bound** `(None, b)`: Negative logarithmic transform keeps values below `b`
- **No bounds** `(None, None)`: Identity (no transformation)

All transformations are differentiable and JAX-compatible.
:::

::::

### Using Bounds in Practice

```python
import everwillow as ew

def nll(params):
    return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2

# Specify bounds as a pytree matching your parameters
result = ew.fit(
    nll,
    params={"mu": 0.0, "sigma": 0.5},
    bounds={"mu": (0.0, 5.0), "sigma": (0.0, None)},  # sigma must be positive
)

# Result parameters are in the original bounded space
print(result.params)  # {'mu': 2.0, 'sigma': 1.0}
```

The transformation machinery ensures that:
- The optimizer never proposes `mu < 0` or `mu > 5`
- The optimizer never proposes `sigma ≤ 0`
- Your NLL function always receives valid parameter values
- Gradients account for the transformation Jacobian

### Caveats

:::{admonition} Important Considerations
:class: warning

**Initial values must satisfy bounds**: Your starting parameters must already lie within the specified bounds. The unwrap transformation will fail if initial values violate the constraints.

**Steep gradients near boundaries**: Transformations like logit can produce very steep gradients when parameters approach bounds. This may slow convergence or cause numerical instability in some cases.

**Hard bounds vs. soft constraints**: Bounds enforce hard limits, which is different from soft constraints like Gaussian priors. If the true parameter value naturally lies at or very near a boundary, bounds may not be the appropriate tool—consider penalty terms in your NLL instead.

**Asymptotic approach to boundaries**: The transformations are asymptotic, meaning the optimizer can approach but never exactly reach a boundary. If you need a parameter to sit exactly at a bound, you should fix it to that value rather than using bounds.
:::

## Common Patterns

### Basic Unconditional Fit

Fit all parameters to minimize the NLL:

```python
import everwillow as ew


def my_nll(params):
    return (params["mu"] - 2.0) ** 2 + (params["sigma"] - 1.0) ** 2


result = ew.fit(my_nll, {"mu": 0.0, "sigma": 0.5})
print(result.params)  # {"mu": ~2.0, "sigma": ~1.0}
```

### Fixing Parameters

Hold specific parameters constant:

```python
result = ew.fit(
    my_nll,
    {"mu": 0.0, "sigma": 0.5, "background": 100},
    fixed=["background"],  # Background stays at 100
)
```

### Profile Likelihood

Scan over values of one parameter while optimizing others:

```python
mu_values = [1.0, 1.5, 2.0, 2.5]
nll_profile = []

for mu_val in mu_values:
    result = ew.fixed_param_fit(
        {"mu": mu_val},  # Fix mu to this value
        my_nll,
        initial_params,
    )
    nll_profile.append(result.nll)
```

### Passing Additional Data

Your NLL function can accept extra arguments:

```python
def nll_with_data(params, observed_data, templates):
    expected = params["mu"] * templates["signal"] + templates["background"]
    return poisson_nll(observed_data, expected)


result = ew.fit(
    nll_with_data, initial_params, args=(data, templates)  # Forwarded to NLL function
)
```

## Advanced: Nested Parameters

Everwillow handles arbitrary pytree structures:

```{figure} images/nested_params.svg
:alt: Handling nested parameter pytrees
:align: center
:width: 100%
```

You can fix parameters by:
- **Name**: `fixed=["background"]` matches any key ending with `"background"`
- **Full path**: `fixed=[("model", "physics", "mu")]` matches exact location
- **Predicate**: `fixed_predicate=lambda k, v: "nuisance" in k`

## Next Steps

::::{grid} 1
:gutter: 2

:::{grid-item-card} 📚 State Management
:link: statelib_overview
:link-type: doc

Deep dive into `FlatState` and the statelib utilities
:::

:::{grid-item-card} 🏗️ Architecture
:link: architecture
:link-type: doc

Learn how the pieces fit together
:::

:::{grid-item-card} 📖 API Reference
:link: api/index
:link-type: doc

Complete function and class documentation
:::

::::
