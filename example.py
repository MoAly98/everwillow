from collections.abc import Callable
import rich

import pyhs3
from pyhs3.typing.aliases import TensorVar
from pyhs3.distributions import PoissonDist, GaussianDist, ProductDist
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pyhs3.domains import ProductDomain, Axis
from pyhs3.metadata import Metadata
from pyhs3.functions import GenericFunction
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify
import jax
import jax.numpy as jnp

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

# Step 1: Create metadata
metadata = Metadata(hs3_version="0.2")

# Step 2: Create distributions
# Single Poisson distribution
main_poisson = PoissonDist(
    name="main_poisson",
    x="n_obs",
    mean="n_expected"
)

beta_constraint = GaussianDist(
    name="beta_constraint",
    x="a_beta",
    mean="nu",
    sigma=1.0)

config = {
      "type": "product_dist",
      "name": "model",
      "factors": ["main_poisson", "beta_constraint"]
  }
combined = ProductDist(**config)

# Define functions for computed values
background_modified_config = {
    "type": "generic_function",
    "name": "background_modified",
    "expression": "background * (1 + nu)"
}
background_modified_func = GenericFunction(**background_modified_config)

n_expected_config = {
    "type": "generic_function",
    "name": "n_expected",
    "expression": "mu * signal + bkg_norm * background_modified"
}
n_expected_func = GenericFunction(**n_expected_config)

# Step 3: Create parameter points and workspace
# ALL parameters in pyhs3
params = ParameterSet(
    name="default_values",
    parameters=[
        # Free parameters (will vary during fit)
        ParameterPoint(name="mu", value=1.0),           # signal strength
        ParameterPoint(name="bkg_norm", value=1.0),     # background normfactor
        ParameterPoint(name="nu", value=0.0),         # nuisance parameter for bkg uncertainty

        # Fixed - observed data
        ParameterPoint(name="n_obs", value=90),         # main observable
        ParameterPoint(name="a_beta", value=0.0),       # auxiliary data for beta constraint

        # Fixed - templates
        ParameterPoint(name="signal", value=5.0),
        ParameterPoint(name="background", value=50.0),
    ]
)
# Step 4: Create workspace (with all distributions and parameter points
ws = pyhs3.Workspace(
    metadata=metadata,
    distributions=[main_poisson, beta_constraint, combined],
    functions=[background_modified_func, n_expected_func],
    parameter_points=[params],
)

# Get the model
model = ws.model()

print("="*40)
print("Model Summary:")
print("="*40)
rich.print(model)

# Step 5: transpile into JAX
inputs, jaxified = get_jaxified_graph(model, "model")

# Print what we got
rich.print(f"\nInput parameters\n {[inp.name for inp in inputs]}")
rich.print(f"Jaxified function\n {jaxified}")

# ===== Extract parameters from pyhs3 into pytree =====
params_set = ws.parameter_points[0]
all_params = jax.tree.map(jnp.asarray, {p.name: p.value for p in params_set.parameters})
rich.print("\nAll parameters from pyhs3:", all_params)

# ===== Split into free vs fixed =====
free_param_names = ["mu", "bkg_norm", "nu"]
fixed_params = {k: v for k, v in all_params.items() if k not in free_param_names}
init_free_params = {k: v for k, v in all_params.items() if k in free_param_names}

rich.print("\nFixed parameters:", fixed_params)
rich.print("Initial free parameters:", init_free_params)

# Right after transpilation, print this:
rich.print("\n=== Debugging Info ===")
rich.print(f"Input parameter names: {[inp.name for inp in inputs]}")
rich.print(f"Input parameter types: {[type(inp) for inp in inputs]}")

# ===== Build neg_loglikelihood wrapper =====
#@jax.jit
def neg_loglikelihood(free_params, fixed_params):
    """
    Negative log-likelihood with background modifiers.

    The computations for background_modified and n_expected are now
    handled by pyhs3 functions in the graph.

    Args:
        free_params: {"mu": ..., "bkg_norm": ..., "beta": ...}
        fixed_params: all the fixed parameters
    """
    # Merge all parameters
    all_params_merged = {
        **fixed_params,
        **free_params,
    }

    # Call jaxified with parameters in correct order
    # The graph will compute background_modified and n_expected automatically
    param_values = [all_params_merged[inp.name] for inp in inputs]
    prob = jaxified(*param_values)[0]

    return -jnp.log(prob)

import optimistix as optx

solver = optx.BFGS(rtol=1e-5, atol=1e-5)
solution = optx.minimise(
    neg_loglikelihood,  # Your function
    solver,
    y0=init_free_params,   # {"mu": 1.0}
    args=fixed_params
)

# Get fitted parameters
fitted_params = solution.value
rich.print("\nFitted parameters:", fitted_params)
rich.print(f"Fitted mu: {fitted_params['mu']}")
rich.print(f"NLL at minimum: {neg_loglikelihood(fitted_params, fixed_params)}")