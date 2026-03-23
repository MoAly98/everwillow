# Combining Models

A common pattern in HEP is combining independent statistical models that share a parameter of interest but use different naming conventions. This page walks through the full workflow: aligning parameter names, merging states, and running a joint fit.

## Setup: two independent measurements

Two experiments each measure a particle mass with their own calibration uncertainties:

```python
import jax
import jax.numpy as jnp
import everwillow as ew
import everwillow.statelib as sl
from everwillow.uncertainty import uncertainties

jax.config.update("jax_enable_x64", True)


# Experiment A: measures "mass" with a scale uncertainty
def nll_a(params, obs):
    mass = params["mass"]
    scale = params["scale_a"]
    pred = mass * scale
    return (
        0.5 * ((pred - obs["m_a"]) / obs["err_a"]) ** 2
        + 0.5 * (scale - 1.0) ** 2 / 0.02**2
    )


# Experiment B: measures the same quantity but calls it "m"
def nll_b(params, obs):
    mass = params["m"]
    scale = params["scale_b"]
    pred = mass * scale
    return (
        0.5 * ((pred - obs["m_b"]) / obs["err_b"]) ** 2
        + 0.5 * (scale - 1.0) ** 2 / 0.03**2
    )
```

Each model was developed independently, so the shared parameter has a different name:

```python
state_a = sl.State.from_pytree({"mass": 125.0, "scale_a": 1.0})
state_b = sl.State.from_pytree({"m": 125.0, "scale_b": 1.0})
```

## Aligning parameter names

Use `apply_transformations()` to rename `"m"` → `"mass"` in model B:

```python
state_b_aligned = sl.apply_transformations(
    state_b,
    {"m": sl.Transform(new_key="mass")},
)
```

The renamed state remembers the original key. When you call `to_pytree()`, model B still receives `{"m": ...}`  - the rename is transparent to the NLL function.

## Combined fit

Before fitting, call `prepare()` to merge the states and build a combined NLL that dispatches each sub-pytree to the correct model. This is necessary because `fit()` takes a single NLL and a single state  - `prepare()` produces both from the individual pieces:

```python
obs = {"m_a": 125.5, "err_a": 1.5, "m_b": 124.8, "err_b": 2.0}

combined_nll, combined = ew.prepare([nll_a, nll_b], [state_a, state_b_aligned])

result = ew.fit(combined_nll, combined, obs)
print(result.params.to_pytree())
# ({'mass': 125.27, 'scale_a': 1.0}, {'m': 125.27, 'scale_b': 1.0})
#           ^ same value in both models
```

To recover the individual states after fitting, use `split()`:

```python
state_a_fitted, state_b_fitted = sl.split(result.params)
print(state_a_fitted.to_pytree())  # {'mass': 125.27, 'scale_a': 1.0}
print(state_b_fitted.to_pytree())  # {'m': 125.27, 'scale_b': 1.0}
```

## Comparing uncertainties

The combined measurement has a smaller uncertainty on `"mass"` than either experiment alone:

```python
# Individual fits
result_a = ew.fit(nll_a, state_a, obs)
result_b = ew.fit(nll_b, state_b, obs)

unc_a = uncertainties(nll_a, result_a.params, obs)
unc_b = uncertainties(nll_b, result_b.params, obs)
unc_combined = uncertainties(combined_nll, result.params, obs)

print(f"σ(mass) exp A only:  {float(unc_a['mass']):.3f}")
print(f"σ(mass) exp B only:  {float(unc_b['m']):.3f}")
print(f"σ(mass) combined:    {float(unc_combined['mass']):.3f}")
# σ(mass) exp A only:  1.500
# σ(mass) exp B only:  2.000
# σ(mass) combined:    1.200
```

The combined constraint is tighter than either individual measurement  - the merged state ensures both models are optimized with a single shared mass parameter.
