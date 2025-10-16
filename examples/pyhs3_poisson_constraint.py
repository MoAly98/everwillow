"""Poisson with Gaussian Constraint - everwillow fit example.

This example demonstrates using everwillow to fit a pyhs3 model with:
- A Poisson distribution for the main observable
- A Gaussian constraint on a nuisance parameter
- Background normalization with uncertainty

Key features demonstrated:
1. Building a pyhs3 model using PointData to separate parameters from data
2. Converting PyTensor graph to JAX function
3. Fitting with everwillow's fit() function (unconditional fit)
4. Using everwillow's fixed_param_fit() for profile likelihood
"""

from collections.abc import Callable

import jax.numpy as jnp
import pyhs3
import rich
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.typing.aliases import TensorVar
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew

# ============================================================================
# UTILITY: Convert PyTensor graph to JAX function
# ============================================================================


def get_jaxified_graph(
    model: pyhs3.Model,
    name: str,
) -> tuple[list[TensorVar], Callable]:
    """Convert PyTensor graph to JAX function.

    Args:
        model: The pyhs3 Model containing distributions
        name: Name of the distribution to convert

    Returns:
        Tuple of (input_variables, jax_function)
    """
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


# ============================================================================
# STEP 1: Build pyhs3 model
# ============================================================================

rich.print("=" * 60)
rich.print("Building pyhs3 model")
rich.print("=" * 60)

# Create metadata for HS3 version
metadata = Metadata(hs3_version="0.2")

# Main Poisson distribution for observed counts
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",  # Observed counts
    mean="n_expected",  # Expected counts (computed from function)
)

# Gaussian constraint on nuisance parameter (auxiliary measurement)
beta_constraint = GaussianDist(
    name="beta_constraint",
    x="a_beta",  # Auxiliary observable
    mean="nu",  # Nuisance parameter
    sigma=1.0,  # Constraint width
)

# Combined likelihood = main * constraint
combined = ProductDist(
    type="product_dist", name="model", factors=["main_poisson", "beta_constraint"]
)

# Background with uncertainty: background * (1 + nu)
background_modified_func = GenericFunction(
    type="generic_function",
    name="background_modified",
    expression="background * (1 + nu)",
)

# Expected counts: signal contribution + background contribution
n_expected_func = GenericFunction(
    type="generic_function",
    name="n_expected",
    expression="mu * signal + bkg_norm * background_modified",
)

# Define parameters (physics parameters to fit)
params = ParameterSet(
    name="default_values",
    parameters=[
        # Free parameters (will vary during fit)
        ParameterPoint(name="mu", value=1.0),  # Signal strength
        ParameterPoint(name="bkg_norm", value=1.0),  # Background norm factor
        ParameterPoint(name="nu", value=0.0),  # Nuisance parameter
    ],
)

# Define data (observed values and templates - these stay fixed)
data = [
    PointData(name="n_obs", value=90.0),  # Main observable
    PointData(name="a_beta", value=0.0),  # Auxiliary measurement
    PointData(name="signal", value=5.0),  # Signal template
    PointData(name="background", value=50.0),  # Background template
]

# Create workspace combining all components
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[main_poisson, beta_constraint, combined],
    functions=[background_modified_func, n_expected_func],
    parameter_points=[params],
    data=data,
)


# ============================================================================
# STEP 2: Build model and transpile to JAX
# ============================================================================

model = ws.model()
inputs, jaxified = get_jaxified_graph(model, "model")

rich.print(f"Model has {len(inputs)} input parameters: {[inp.name for inp in inputs]}")


# ============================================================================
# STEP 3: Extract parameters as pytree
# ============================================================================

# Extract parameters (values to fit)
params_set = ws.parameter_points[0]
all_params = {p.name: float(p.value) for p in params_set.parameters}

# Extract data values separately (these are fixed and referenced by the model)
data_values = {data_point.name: float(data_point.value) for data_point in ws.data}

rich.print("\nInitial parameters:")
rich.print(all_params)
rich.print("\nData values:")
rich.print(data_values)


# ============================================================================
# STEP 4: Define NLL function for everwillow
# ============================================================================


def nll_fn(params):
    """Negative log-likelihood function for everwillow.

    everwillow expects a function that takes a pytree of parameters
    and returns a scalar NLL value.

    Args:
        params: Dict of parameters to fit

    Returns:
        Negative log-likelihood value
    """
    # Combine parameters with data values for the jaxified function
    all_values = {**params, **data_values}

    # Call jaxified with parameters in correct order (as expected by pyhs3 graph)
    param_values = [all_values[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]

    return -jnp.log(prob)


# Test the NLL function at initial parameters
initial_nll = nll_fn(all_params)
rich.print(f"NLL at initial params: {initial_nll}")


# ============================================================================
# STEP 5: Run unconditional fit with everwillow
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 1: Unconditional fit (free: mu, bkg_norm, nu)")
rich.print("=" * 60)

result = ew.fit(
    nll_fn,
    all_params,
    fixed=[],  # No fixed parameters - all three are free
    max_steps=100,
)

rich.print("\nFitted parameters:", result.params)
rich.print(f"NLL at minimum: {result.nll}")
rich.print(f"Fitted mu: {result.params['mu']}")
rich.print(f"Fitted bkg_norm: {result.params['bkg_norm']}")
rich.print(f"Fitted nu: {result.params['nu']}")


# ============================================================================
# STEP 6: Run fixed parameter fit (profile likelihood)
# ============================================================================

rich.print("\n" + "=" * 60)
rich.print("Test 2: Fixed parameter fit (mu=1.5)")
rich.print("=" * 60)

result_fixed = ew.fixed_param_fit(
    {"mu": 1.5},  # Fix mu to this value
    nll_fn,
    all_params,
    fixed=[],  # No additional fixed parameters beyond mu
)

rich.print("\nFitted parameters:", result_fixed.params)
rich.print(f"NLL at mu=1.5: {result_fixed.nll}")
rich.print(f"Δ(NLL) = {result_fixed.nll - result.nll}")

rich.print("\n✅ All tests with pyhs3 model completed!")
