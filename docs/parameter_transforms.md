# Parameter Transforms

Everwillow expresses parameter bounds through explicit transform objects. Each
transform maps a user-facing "bounded" value into an unconstrained real space
that the optimizer can work with, and provides the inverse mapping to return
results to the bounded domain.

## Why transforms?

optimizers such as quasi-Newton or gradient-based methods expect parameters to
live in an unconstrained space. By wrapping a bounded parameter with a
transformation, Everwillow can run the optimizer in a stable domain while
ensuring that all calls to your loss function still observe the requested
bounds. This also keeps gradients well-defined and allows the optimizer to
approach a boundary asymptotically without ever leaving the valid region.

## Built-in transforms

The available transforms live in `everwillow.parameters.transforms`:

| Transform | Purpose | Constructor Arguments |
|-----------|---------|------------------------|
| `MinuitTransform` | Finite lower/upper bounds using the Minuit sine mapping. | `lower`, `upper` (both finite) |
| `SigmoidTransform` | Alternative finite-bound mapping using the logistic/logit pair. | `lower`, `upper` (both finite) |
| `OneSidedLogTransform` | Single-sided bounds via a log/exponential mapping. | `bound`, `direction="lower"|"upper"` |
| `SoftPlusTransform` | Enforce positivity without an explicit bound. | none |

All transforms subclass `TransformBase`, so you can mix them
freely in dictionaries keyed by leaf names or canonical key paths.

## Creating custom transforms

To define a new transform, subclass
``everwillow.parameters.transforms.TransformBase`` and
implement ``unwrap`` and ``wrap`` using JAX-compatible operations. Mark
static configuration (such as bounds) with ``equinox.field(static=True)``
so that transform instances behave as pytrees.

```python
import equinox as eqx
import jax.numpy as jnp
from everwillow.parameters.transforms import TransformBase


class SquareTransform(TransformBase):
    scale: float = eqx.field(static=True)

    def unwrap(self, value):
        value = jnp.asarray(value)
        return jnp.sqrt(value / self.scale)

    def wrap(self, value):
        value = jnp.asarray(value)
        return self.scale * value**2
```

Expose the transform by passing instances in the ``bounds`` mapping or by
following the workflow described in :ref:`transforms-direct`.

## Using transforms with `fit`

Pass transform instances via the `bounds` argument. Each transform will be used
to unwrap the corresponding value before optimisation, and wrapped again in the
final result:

```python
import everwillow as ew
from everwillow.parameters.transforms import (
    MinuitTransform,
    OneSidedLogTransform,
    SoftPlusTransform,
)


def nll(params):
    return (params["mu"] - 2) ** 2 + (params["sigma"] - 1) ** 2 + params["beta"] ** 2


result = ew.fit(
    nll,
    {"mu": 0.2, "sigma": 0.5, "beta": 0.1},
    bounds={
        "mu": MinuitTransform(lower=0.0, upper=5.0),
        "sigma": OneSidedLogTransform(bound=0.0, direction="lower"),
        "beta": SoftPlusTransform(),  # enforce beta >= 0
    },
)
```

The `bounds` dictionary accepts either leaf names (strings) or canonical key
paths (`tuple` instances) to support complex pytree layouts.

## Applying transforms to a flat state

.. _transforms-direct:

### Using transforms directly

If you need to manage optimisation loops yourself, the bounds helpers
operate on :class:`everwillow.statelib.state.FlatState` objects. They return
transformation maps that are compatible with
:func:`everwillow.statelib.transform.apply_transformations`.

For advanced use cases—such as custom optimisation loops—you can work directly
with `everwillow.parameters.bounds`:

- `match_bounds_to_state(state, bounds)` resolves names/keys to a mapping of
  canonical key paths to transform instances.
- `apply_bounds_transform(state, bounds)` unwraps a `FlatState` and returns both
  the transformed state and the corresponding wrap/unwrap transformation maps
  (compatible with `everwillow.statelib.transform.apply_transformations`).

```python
import everwillow.statelib as sl
from everwillow.parameters import bounds
from everwillow.parameters.transforms import MinuitTransform

state = sl.FlatState.from_pytree({"mu": 0.3})
unwrapped_state, unwrap_map, wrap_map = bounds.apply_bounds_transform(
    state, {"mu": MinuitTransform(lower=0.0, upper=1.0)}
)

# run optimizer in unwrapped space ...
# then wrap results back:
wrapped_state = sl.transform.apply_transformations(unwrapped_state, wrap_map)
```

These helpers provide the same behaviour as `fit`, while giving you full
control over when unwrapping and wrapping occur.
