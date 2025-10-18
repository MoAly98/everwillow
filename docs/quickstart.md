# Quickstart

This tutorial walks through a simple counting experiment implemented with three
JAX-supporting modelling libraries: [`pyhs3`](https://github.com/scipp-atlas/pyhs3),
[`evermore`](https://github.com/pfackeldey/evermore), and
[`pyhf`](https://github.com/scikit-hep/pyhf). In each case we:

1. Build the likelihood with the library’s native abstractions.
2. Expose a pytree of parameters that everwillow can optimise.
3. Run {func}`everwillow.fitting.fit` to obtain the best-fit values.

## The counting model

We use a single-bin signal-plus-background measurement with one constrained
nuisance parameter.

- Observed events: {math}`n_\text{obs} = 52`.
- Signal template: {math}`s = 18` expected events.
- Background expectation {math}`b_\text{norm}` constrained by an auxiliary
  measurement {math}`b_\text{aux} = 34` with Gaussian width {math}`\sigma_b = 6`.
- Parameters to estimate: signal strength {math}`\mu` and the nuisance scale
  {math}`b_\text{norm}`.
- Negative log-likelihood:

  ```{math}
  -\log L(\mu, b_\text{norm}) = \lambda - n_\text{obs} \log \lambda
  + \frac{(b_\text{norm} - b_\text{aux})^2}{2 \sigma_b^2},
  ```

  where {math}`\lambda = \mu s + b_\text{norm}`.

The tabs below show the full, runnable example in each toolkit. All snippets use
``everwillow`` to perform the final optimisation.

::::{tab-set}
:::{tab-item} pyhs3
```python
import jax
import jax.numpy as jnp
import pyhs3
import everwillow as ew
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify


def to_jax(model: pyhs3.Model, *, name: str):
    """Return the inputs and JAX callable for the given distribution."""
    dist = model.distributions[name]
    inputs = [var for var in graph_inputs([dist]) if var.name]
    function_graph = FunctionGraph(inputs=inputs, outputs=[dist], clone=True)
    mode.JAX.optimizer.rewrite(function_graph)
    return inputs, jax_funcify(function_graph)


signal = 18.0
bkg_aux = 34.0
bkg_sigma = 6.0
n_obs = 52.0

poisson = PoissonDist(name="counts", x="n_obs", mean="n_expected")
constraint = GaussianDist(
    name="constraint",
    x="bkg_aux",
    mean="bkg_norm",
    sigma=bkg_sigma,
)
combined = ProductDist(name="model", factors=["counts", "constraint"])

model = pyhs3.Model(
    distributions={
        "counts": poisson,
        "constraint": constraint,
        "model": combined,
    },
    functions={
        "n_expected": GenericFunction(
            name="n_expected",
            expression="mu * signal + bkg_norm",
        )
    },
    metadata=Metadata(hs3_version="0.2"),
)

workspace = pyhs3.Workspace(
    metadata=model.metadata,
    distributions=list(model.distributions.values()),
    functions=list(model.functions.values()),
    parameter_points=[
        ParameterSet(
            name="initial",
            parameters=[
                ParameterPoint(name="mu", value=1.0),
                ParameterPoint(name="bkg_norm", value=bkg_aux),
                ParameterPoint(name="signal", value=signal),
                ParameterPoint(name="n_obs", value=n_obs),
                ParameterPoint(name="bkg_aux", value=bkg_aux),
            ],
        )
    ],
)

inputs, likelihood = to_jax(workspace.model(), name="model")


def nll(params):
    """Negative log-likelihood constructed from the JAXified graph."""
    args = [params[var.name] for var in inputs]
    return -jnp.log(likelihood(*args))


initial = {
    "mu": 1.0,
    "bkg_norm": bkg_aux,
    "signal": signal,
    "n_obs": n_obs,
    "bkg_aux": bkg_aux,
}

result = ew.fit(nll, initial, fixed=["signal", "n_obs", "bkg_aux"])
print(result.params)
```
:::
:::{tab-item} evermore
```python
import jax.numpy as jnp
import evermore as evm
import everwillow as ew

signal = 18.0
bkg_aux = 34.0
bkg_sigma = 6.0
n_obs = 52.0

model = evm.Model(
    parameters={
        "mu": evm.Parameter(value=1.0, name="mu"),
        "bkg_norm": evm.Parameter(
            value=bkg_aux,
            name="bkg_norm",
            prior=evm.pdf.Normal(mean=bkg_aux, width=bkg_sigma),
        ),
    },
    data={"observed": n_obs, "signal": signal},
)


def nll(params):
    mu = params["mu"]
    bkg = params["bkg_norm"]
    expected = mu * signal + bkg
    poisson = expected - model.data["observed"] * jnp.log(expected)
    constraint = ((bkg - bkg_aux) / bkg_sigma) ** 2
    return poisson + constraint


initial = model.to_pytree()
result = ew.fit(nll, initial, fixed=["signal", "observed"])
print(result.params)
```
:::
:::{tab-item} pyhf
```python
import jax.numpy as jnp
import pyhf
import everwillow as ew

signal = [18.0]
bkg = [34.0]
bkg_uncorr = [6.0]
n_obs = jnp.array([52.0])

model = pyhf.simplemodels.correlated_background(
    signal=signal,
    bkg=bkg,
    bkg_uncorr=bkg_uncorr,
)

par_order = model.config.par_order
initial = dict(zip(par_order, model.config.suggested_init()))
auxdata = jnp.array(model.config.auxdata)


def nll(params):
    parameter_vector = jnp.array([params[name] for name in par_order])
    data = jnp.concatenate((n_obs, auxdata))
    return float(-model.logpdf(data, parameter_vector))


result = ew.fit(nll, initial, fixed=["gamma_bkg"])
print(result.params)
```
:::
::::

## Next steps

- Explore the {doc}`statelib_overview` for a refresher on the state-management
  helpers used by the fitting API.
- Head over to the {doc}`architecture` notes when you need to extend the core
  modules or integrate new optimisers.
- Review the {doc}`api/index` for the complete API reference used throughout
  these examples.
