"""Standalone pyhs3 counting experiment example."""

import jax.numpy as jnp
import pyhs3
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet
from pytensor.compile import mode
from pytensor.graph.basic import graph_inputs
from pytensor.graph.fg import FunctionGraph
from pytensor.link.jax.dispatch import jax_funcify

import everwillow as ew


def jaxify_distribution(model, distribution_name):
    """Convert a PyTensor distribution graph into a JAX-callable function."""
    distribution = model.distributions[distribution_name]
    inputs = [var for var in graph_inputs([distribution]) if var.name is not None]
    function_graph = FunctionGraph(inputs=inputs, outputs=[distribution], clone=True)
    mode.JAX.optimizer.rewrite(function_graph)
    return inputs, jax_funcify(function_graph)


# Build the workspace
workspace = pyhs3.Workspace(
    metadata=Metadata(hs3_version="0.2"),
    distributions=[
        PoissonDist(name="main_poisson", x="n_obs", mean="n_expected"),
        GaussianDist(name="norm1_constraint", x="a_norm1", mean="norm1", sigma=1.0),
        GaussianDist(name="norm2_constraint", x="a_norm2", mean="norm2", sigma=1.0),
        GaussianDist(name="shape1_constraint", x="a_shape1", mean="shape1", sigma=1.0),
        ProductDist(
            type="product_dist",
            name="model",
            factors=[
                "main_poisson",
                "norm1_constraint",
                "norm2_constraint",
                "shape1_constraint",
            ],
        ),
    ],
    functions=[
        GenericFunction(name="signal_expected", expression="mu * signal_nominal"),
        GenericFunction(
            name="bkg1_lnN_factor",
            expression="exp(norm1 * log(1.1))",
        ),
        GenericFunction(
            name="bkg1_shape_interp",
            expression="bkg1_nominal + shape1 * (bkg1_shape_up - bkg1_nominal)",
        ),
        GenericFunction(
            name="bkg1_expected", expression="bkg1_lnN_factor * bkg1_shape_interp"
        ),
        GenericFunction(
            name="bkg2_lnN_factor",
            expression="exp(norm2 * log(1.05))",
        ),
        GenericFunction(
            name="bkg2_shape_interp",
            expression="bkg2_nominal + shape1 * (bkg2_shape_up - bkg2_nominal)",
        ),
        GenericFunction(
            name="bkg2_expected", expression="bkg2_lnN_factor * bkg2_shape_interp"
        ),
        GenericFunction(
            name="n_expected",
            expression="signal_expected + bkg1_expected + bkg2_expected",
        ),
    ],
    parameter_points=[
        ParameterSet(
            name="default_values",
            parameters=[
                ParameterPoint(name="mu", value=1.0),
                ParameterPoint(name="norm1", value=0.0),
                ParameterPoint(name="norm2", value=0.0),
                ParameterPoint(name="shape1", value=0.0),
            ],
        )
    ],
    data=[
        PointData(name="n_obs", value=37.0),
        PointData(name="a_norm1", value=0.0),
        PointData(name="a_norm2", value=0.0),
        PointData(name="a_shape1", value=0.0),
        PointData(name="signal_nominal", value=3.0),
        PointData(name="bkg1_nominal", value=10.0),
        PointData(name="bkg1_shape_up", value=12.0),
        PointData(name="bkg2_nominal", value=20.0),
        PointData(name="bkg2_shape_up", value=23.0),
    ],
)

# Extract model and convert to JAX
model = workspace.model()
inputs, jaxified = jaxify_distribution(model, "model")

# Build initial parameters
initial = {
    point.name: float(point.value) for point in workspace.parameter_points[0].parameters
}
fixed_values = {point.name: float(point.value) for point in workspace.data}


# Define NLL
def nll(params):
    merged = {**fixed_values, **params}
    ordered = [merged[var.name] for var in inputs]
    probability = jaxified(*ordered)[0]
    return -jnp.log(jnp.asarray(probability))


# Perform the fit
result = ew.ifit(nll, initial)

print(result.params)
