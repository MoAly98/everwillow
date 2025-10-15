"""Test everwillow with pyhs3 model - self-contained."""

from collections.abc import Callable
import rich
import jax
import jax.numpy as jnp

import pyhs3
from pyhs3.typing.aliases import TensorVar
from pyhs3.distributions import PoissonDist, GaussianDist, ProductDist
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.metadata import Metadata
from pyhs3.functions import GenericFunction
from pyhs3.data import PointData
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew


def get_jaxified_graph(
    model: pyhs3.Model,
    name: str,
) -> tuple[list[TensorVar], Callable]:
    """Convert PyTensor graph to JAX function."""
    # 1. Get the distribution by name
    dist = model.distributions[name]

    # 2. Extract all named parameters
    inputs = [var for var in graph_inputs([dist]) if var.name is not None]

    # 3. Create a function graph with those inputs and the dist as output
    fgraph = FunctionGraph(inputs=inputs, outputs=[dist], clone=True)

    # 4. Optimize the graph for JAX
    mode.JAX.optimizer.rewrite(fgraph)

    # 5. Convert to actual JAX function
    return inputs, jax_funcify(fgraph)


# ===== Step 1: Build pyhs3 model =====
rich.print("="*60)
rich.print("Building pyhs3 model")
rich.print("="*60)

# Create metadata
metadata = Metadata(hs3_version="0.2")

# Create distributions
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",
    mean="n_expected"
)

beta_constraint = GaussianDist(
    name="beta_constraint",
    x="a_beta",
    mean="nu",
    sigma=1.0
)

combined = ProductDist(
    type="product_dist",
    name="model",
    factors=["main_poisson", "beta_constraint"]
)

# Define functions for computed values
background_modified_func = GenericFunction(
    type="generic_function",
    name="background_modified",
    expression="background * (1 + nu)"
)

n_expected_func = GenericFunction(
    type="generic_function",
    name="n_expected",
    expression="mu * signal + bkg_norm * background_modified"
)

# Create parameter points (only for actual parameters, not data or templates)
params = ParameterSet(
    name="default_values",
    parameters=[
        # Free parameters (will vary during fit)
        ParameterPoint(name="mu", value=1.0),
        ParameterPoint(name="bkg_norm", value=1.0),
        ParameterPoint(name="nu", value=0.0),
    ]
)

# Create data (observed values and templates)
data = [
    PointData(name="n_obs", value=90.0),
    PointData(name="a_beta", value=0.0),
    PointData(name="signal", value=5.0),
    PointData(name="background", value=50.0),
]

# Create workspace
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[main_poisson, beta_constraint, combined],
    functions=[background_modified_func, n_expected_func],
    parameter_points=[params],
    data=data,
)

# Get the model and transpile to JAX
model = ws.model()
inputs, jaxified = get_jaxified_graph(model, "model")

rich.print(f"Model has {len(inputs)} input parameters: {[inp.name for inp in inputs]}")

# ===== Step 2: Extract parameters as pytree =====
params_set = ws.parameter_points[0]
all_params = {p.name: float(p.value) for p in params_set.parameters}

# Extract data values separately (these are fixed and referenced by the model)
data_values = {data_point.name: float(data_point.value) for data_point in ws.data}

rich.print("\nInitial parameters:")
rich.print(all_params)
rich.print("\nData values:")
rich.print(data_values)

# ===== Step 3: Define NLL function for everwillow =====
def nll_fn(params):
    """
    Negative log-likelihood that takes a pytree of parameters.

    This is what everwillow expects: a function that takes any pytree
    and returns a scalar NLL value.
    """
    # Combine params with data values for the jaxified function
    all_values = {**params, **data_values}

    # Call jaxified with parameters in correct order (as expected by pyhs3 graph)
    param_values = [all_values[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]

    return -jnp.log(prob)

# Test the NLL function
initial_nll = nll_fn(all_params)
rich.print(f"NLL at initial params: {initial_nll}")

# ===== Step 4: Run unconditional fit with everwillow =====
rich.print("\n" + "="*60)
rich.print("Test 1: Unconditional fit (free: mu, bkg_norm, nu)")
rich.print("="*60)

result = ew.fit(
    nll_fn,
    all_params,
    fixed=[],  # No fixed parameters - all three (mu, bkg_norm, nu) are free
    max_steps=100
)

rich.print("\nFitted parameters:", result.params)
rich.print(f"NLL at minimum: {result.nll}")
rich.print(f"Fitted mu: {result.params['mu']}")
rich.print(f"Fitted bkg_norm: {result.params['bkg_norm']}")
rich.print(f"Fitted nu: {result.params['nu']}")

# ===== Step 5: Run fixed parameter fit (profile likelihood) =====
rich.print("\n" + "="*60)
rich.print("Test 2: Fixed parameter fit (mu=1.5)")
rich.print("="*60)

result_fixed = ew.fixed_param_fit(
    {"mu": 1.5},
    nll_fn,
    all_params,
    fixed=[]  # No additional fixed parameters beyond mu
)

rich.print("\nFitted parameters:", result_fixed.params)
rich.print(f"NLL at mu=1.5: {result_fixed.nll}")
rich.print(f"Δ(NLL) = {result_fixed.nll - result.nll}")

rich.print("\n✅ All tests with pyhs3 model completed!")
