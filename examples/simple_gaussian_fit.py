"""Simple Gaussian Fit Example.

This example demonstrates a basic statistical model with:
- A Poisson distribution for the main observable
- A Gaussian constraint on a nuisance parameter
- Background normalization with uncertainty

The model structure:
    likelihood = Poisson(n_obs | n_expected) × Gaussian(a_beta | nu, sigma=1)

where n_expected = mu * signal + bkg_norm * background * (1 + nu)

This showcases:
1. Building a pyhs3 model with distributions and functions
2. Converting PyTensor graph to JAX
3. Fitting with optimistix
"""

from collections.abc import Callable
import rich

import pyhs3
from pyhs3.typing.aliases import TensorVar
from pyhs3.distributions import PoissonDist, GaussianDist, ProductDist
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.metadata import Metadata
from pyhs3.functions import GenericFunction
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify
import jax
import jax.numpy as jnp


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
# STEP 1: Define the statistical model structure
# ============================================================================

# Create metadata for HS3 version
metadata = Metadata(hs3_version="0.2")

# Main Poisson distribution for observed counts
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",              # Observed counts
    mean="n_expected"       # Expected counts (computed from function)
)

# Gaussian constraint on nuisance parameter (auxiliary measurement)
beta_constraint = GaussianDist(
    name="beta_constraint",
    x="a_beta",             # Auxiliary observable
    mean="nu",              # Nuisance parameter
    sigma=1.0               # Constraint width
)

# Combined likelihood = main × constraint
config = {
    "type": "product_dist",
    "name": "model",
    "factors": ["main_poisson", "beta_constraint"]
}
combined = ProductDist(**config)


# ============================================================================
# STEP 2: Define computed quantities (functions)
# ============================================================================

# Background with uncertainty: background * (1 + nu)
background_modified_config = {
    "type": "generic_function",
    "name": "background_modified",
    "expression": "background * (1 + nu)"
}
background_modified_func = GenericFunction(**background_modified_config)

# Expected counts: signal contribution + background contribution
n_expected_config = {
    "type": "generic_function",
    "name": "n_expected",
    "expression": "mu * signal + bkg_norm * background_modified"
}
n_expected_func = GenericFunction(**n_expected_config)


# ============================================================================
# STEP 3: Create parameter points and workspace
# ============================================================================

# Define all parameters with initial/fixed values
params = ParameterSet(
    name="default_values",
    parameters=[
        # === Free parameters (will vary during fit) ===
        ParameterPoint(name="mu", value=1.0),           # Signal strength
        ParameterPoint(name="bkg_norm", value=1.0),     # Background norm factor
        ParameterPoint(name="nu", value=0.0),           # Nuisance parameter

        # === Fixed - observed data ===
        ParameterPoint(name="n_obs", value=90),         # Main observable
        ParameterPoint(name="a_beta", value=0.0),       # Auxiliary measurement

        # === Fixed - templates ===
        ParameterPoint(name="signal", value=5.0),       # Signal template
        ParameterPoint(name="background", value=50.0),  # Background template
    ]
)

# Create workspace combining all components
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[main_poisson, beta_constraint, combined],
    functions=[background_modified_func, n_expected_func],
    parameter_points=[params],
)


# ============================================================================
# STEP 4: Build model and transpile to JAX
# ============================================================================

model = ws.model()

print("=" * 60)
print("Model Summary:")
print("=" * 60)
rich.print(model)

# Convert PyTensor graph to JAX function
inputs, jaxified = get_jaxified_graph(model, "model")

rich.print(f"\nInput parameters: {[inp.name for inp in inputs]}")
rich.print(f"Jaxified function: {jaxified}")


# ============================================================================
# STEP 5: Extract parameters and separate free vs fixed
# ============================================================================

# Extract all parameters from pyhs3 as pytree
params_set = ws.parameter_points[0]
all_params = jax.tree.map(jnp.asarray, {p.name: p.value for p in params_set.parameters})
rich.print("\nAll parameters from pyhs3:", all_params)

# Split into free parameters (to fit) and fixed parameters (data + templates)
free_param_names = ["mu", "bkg_norm", "nu"]
fixed_params = {k: v for k, v in all_params.items() if k not in free_param_names}
init_free_params = {k: v for k, v in all_params.items() if k in free_param_names}

rich.print("\nFixed parameters:", fixed_params)
rich.print("Initial free parameters:", init_free_params)

# Debug: verify parameter ordering
rich.print("\n=== Debugging Info ===")
rich.print(f"Input parameter names: {[inp.name for inp in inputs]}")
rich.print(f"Input parameter types: {[type(inp) for inp in inputs]}")


# ============================================================================
# STEP 6: Define negative log-likelihood function
# ============================================================================

def neg_loglikelihood(free_params, fixed_params):
    """Negative log-likelihood function.

    The computations for background_modified and n_expected are handled
    automatically by the pyhs3 functions compiled into the JAX graph.

    Args:
        free_params: Dict of free parameters {"mu": ..., "bkg_norm": ..., "nu": ...}
        fixed_params: Dict of fixed parameters (data and templates)

    Returns:
        Negative log-likelihood value
    """
    # Merge all parameters
    all_params_merged = {
        **fixed_params,
        **free_params,
    }

    # Call jaxified function with parameters in correct order
    param_values = [all_params_merged[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]

    return -jnp.log(prob)


# ============================================================================
# STEP 7: Fit with optimistix
# ============================================================================

import optimistix as optx

print("\n" + "=" * 60)
print("Fitting with optimistix...")
print("=" * 60)

solver = optx.BFGS(rtol=1e-5, atol=1e-5)
solution = optx.minimise(
    neg_loglikelihood,
    solver,
    y0=init_free_params,
    args=fixed_params
)

# Display results
fitted_params = solution.value
rich.print("\nFitted parameters:", fitted_params)
rich.print(f"Fitted mu: {fitted_params['mu']}")
rich.print(f"NLL at minimum: {neg_loglikelihood(fitted_params, fixed_params)}")
