<div align="center">
<tr>
<td width="200px">
<img src="https://raw.githubusercontent.com/MoAly98/everwillow/main/images/logo.svg" alt="everwillow logo" width="180">
</td>

</tr>
</div>

##
everwillow is a statistical inference library for high-energy physics built on JAX pytrees and optimistix optimizers. It provides tools for fitting, profiling, and hypothesis testing with flexible parameter handling and parameter bounds via transformations. It works with any JAX-based statistical model.

## Installation

```bash
pip install everwillow
```

or with uv:

```bash
uv add everwillow
```

From source:

```bash
git clone https://github.com/MoAly98/everwillow.git
cd everwillow
uv sync
```

## Example

A Poisson counting experiment: define a model, fit it, compute CLs p-values, and find a 95% CL upper limit — both with asymptotic formulas and toys.

```python
import jax
import jax.numpy as jnp

import everwillow as ew
import everwillow.statelib as sl
from everwillow.hypotest.calculators import AsymptoticCalculator, HypoTestCalculator
from everwillow.hypotest.distributions import (
    QTildeAsymptotic,
    SimpleEmpiricalDistribution,
)
from everwillow.hypotest.test_statistics import QTilde
from everwillow.hypotest.toys import ToyGenerator
from everwillow.hypotest.upper_limit import upper_limit

jax.config.update("jax_enable_x64", True)

# --- Model ---

signal, background = 10.0, 5.0


def nll(params, observation):
    mu = params["mu"]
    expected = mu * signal + background
    return expected - observation["n"] * jnp.log(expected)


def predict(params_state):
    mu = params_state.to_pytree()["mu"]
    return {"n": mu * signal + background}


params = sl.State.from_pytree({"mu": 1.0})
observed = {"n": 12.0}

# --- Fit ---

result = ew.fit(nll_fn=nll, params=params, observation=observed)
print(result.params.to_pytree())
# {'mu': Array(0.7, dtype=float64)}

# --- Asymptotic hypothesis test ---

calc = AsymptoticCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    predict_fn=predict,
    test_statistic=QTilde(),
    distribution=QTildeAsymptotic(),
)
result = calc.test(1.0)
print(f"CLs: {calc.cls(result):.4f}")
# CLs: 0.2140


def cls_objective(poi):
    return calc.cls(calc.test(poi))


limit = upper_limit(cls_objective, bounds=(0.0, 5.0), level=0.05)
print(f"95% CL upper limit (asymptotic): {float(limit):.4f}")
# 95% CL upper limit (asymptotic): 1.3673

# --- Toy-based hypothesis test ---

toy_gen = ToyGenerator(test_statistic=QTilde(), ntoys=5000)
toys = toy_gen.generate(
    nll,
    params,
    observed,
    "mu",
    poi_null=1.0,
    poi_alt=0.0,
    key=jax.random.key(42),
    predict_fn=predict,
)
dist = SimpleEmpiricalDistribution.from_toys(toys)

toy_calc = HypoTestCalculator(
    nll_fn=nll,
    params=params,
    observation=observed,
    poi_key="mu",
    test_statistic=QTilde(),
    distribution=dist,
)
toy_result = toy_calc.test(1.0)
print(f"CLs (toys): {toy_calc.cls(toy_result):.4f}")


def toy_cls_objective(poi):
    return jnp.float64(toy_calc.cls(toy_calc.test(poi)))


toy_limit = upper_limit(toy_cls_objective, bounds=(0.0, 5.0), level=0.05)
print(f"95% CL upper limit (toys): {float(toy_limit):.4f}")
```

## Documentation

- [Quickstart](https://everwillow.readthedocs.io/en/latest/quickstart.html)
- [API Reference](https://everwillow.readthedocs.io/en/latest/api/)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow.

## License

everwillow is distributed under the [BSD-3-Clause License](LICENSE).
