"""pyhs3 implementation of the 1sig_2bkg_1nf_1ss_1ns example."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import jax.numpy as jnp
import pyhs3
from pyhs3.data import PointData
from pyhs3.distributions import GaussianDist, PoissonDist, ProductDist
from pyhs3.functions import GenericFunction
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterSet

from .model_config import (
    DEFAULT_DATA,
    ModelData,
    default_initial_params,
    expected_components,
    gaussian_constraint_width,
)
from .utils import jaxify_distribution


@dataclass(frozen=True)
class Pyhs3Setup:
    """Container describing the pyhs3 model and helper hooks."""

    negative_log_likelihood: Callable[[Mapping[str, float]], float]
    initial_params: dict[str, float]
    model: pyhs3.Model
    data_values: dict[str, float]
    inputs: list


def build_pyhs3_setup(data: ModelData = DEFAULT_DATA) -> Pyhs3Setup:
    """Create the pyhs3 workspace and return an everwillow-ready NLL."""

    metadata = Metadata(hs3_version="0.2")

    main_poisson = PoissonDist(
        name="main_poisson",
        x="n_obs",
        mean="n_expected",
    )

    norm1_constraint = GaussianDist(
        name="norm1_constraint",
        x="a_norm1",
        mean="norm1",
        sigma=gaussian_constraint_width(),
    )

    norm2_constraint = GaussianDist(
        name="norm2_constraint",
        x="a_norm2",
        mean="norm2",
        sigma=gaussian_constraint_width(),
    )

    shape1_constraint = GaussianDist(
        name="shape1_constraint",
        x="a_shape1",
        mean="shape1",
        sigma=gaussian_constraint_width(),
    )

    combined = ProductDist(
        type="product_dist",
        name="model",
        factors=[
            "main_poisson",
            "norm1_constraint",
            "norm2_constraint",
            "shape1_constraint",
        ],
    )

    signal_expected_func = GenericFunction(
        type="generic_function",
        name="signal_expected",
        expression="mu * signal_nominal",
    )

    bkg1_lnN_func = GenericFunction(
        type="generic_function",
        name="bkg1_lnN_factor",
        expression="exp(norm1 * (log(1.1)**((norm1 + abs(norm1))/(2*abs(norm1) + 1e-10)) * log(1.0/0.9)**((abs(norm1) - norm1)/(2*abs(norm1) + 1e-10))))",
    )

    bkg1_shape_interp_func = GenericFunction(
        type="generic_function",
        name="bkg1_shape_interp",
        expression="bkg1_nominal + shape1 * (bkg1_shape_up - bkg1_nominal)",
    )

    bkg1_expected_func = GenericFunction(
        type="generic_function",
        name="bkg1_expected",
        expression="bkg1_lnN_factor * bkg1_shape_interp",
    )

    bkg2_lnN_func = GenericFunction(
        type="generic_function",
        name="bkg2_lnN_factor",
        expression="exp(norm2 * (log(1.05)**((norm2 + abs(norm2))/(2*abs(norm2) + 1e-10)) * log(1.0/0.95)**((abs(norm2) - norm2)/(2*abs(norm2) + 1e-10))))",
    )

    bkg2_shape_interp_func = GenericFunction(
        type="generic_function",
        name="bkg2_shape_interp",
        expression="bkg2_nominal + shape1 * (bkg2_shape_up - bkg2_nominal)",
    )

    bkg2_expected_func = GenericFunction(
        type="generic_function",
        name="bkg2_expected",
        expression="bkg2_lnN_factor * bkg2_shape_interp",
    )

    n_expected_func = GenericFunction(
        type="generic_function",
        name="n_expected",
        expression="signal_expected + bkg1_expected + bkg2_expected",
    )

    parameters = ParameterSet(
        name="default_values",
        parameters=[
            ParameterPoint(name=name, value=value)
            for name, value in default_initial_params().items()
        ],
    )

    data_points = [
        PointData(name="n_obs", value=data.observed),
        PointData(name="a_norm1", value=0.0),
        PointData(name="a_norm2", value=0.0),
        PointData(name="a_shape1", value=0.0),
        PointData(name="signal_nominal", value=data.signal_nominal),
        PointData(name="bkg1_nominal", value=data.bkg1_nominal),
        PointData(name="bkg1_shape_up", value=data.bkg1_shape_up),
        PointData(name="bkg2_nominal", value=data.bkg2_nominal),
        PointData(name="bkg2_shape_up", value=data.bkg2_shape_up),
    ]

    workspace = pyhs3.Workspace(
        metadata=metadata,
        distributions=[
            main_poisson,
            norm1_constraint,
            norm2_constraint,
            shape1_constraint,
            combined,
        ],
        functions=[
            signal_expected_func,
            bkg1_lnN_func,
            bkg1_shape_interp_func,
            bkg1_expected_func,
            bkg2_lnN_func,
            bkg2_shape_interp_func,
            bkg2_expected_func,
            n_expected_func,
        ],
        parameter_points=[parameters],
        data=data_points,
    )

    model = workspace.model()
    inputs, jaxified = jaxify_distribution(model, "model")

    initial_params = {
        point.name: float(point.value) for point in parameters.parameters
    }
    data_values = {point.name: float(point.value) for point in data_points}

    def negative_log_likelihood(params: Mapping[str, float]) -> float:
        merged = {**data_values, **params}
        ordered = [merged[var.name] for var in inputs]
        probability = jaxified(*ordered)[0]
        return -jnp.log(jnp.asarray(probability))

    return Pyhs3Setup(
        negative_log_likelihood=negative_log_likelihood,
        initial_params=initial_params,
        model=model,
        data_values=data_values,
        inputs=inputs,
    )


def summarise_pyhs3_fit(
    params: Mapping[str, float],
    data: ModelData = DEFAULT_DATA,
) -> dict[str, float]:
    """Return component yields derived from the fitted parameters."""

    return expected_components(params, data=data)
