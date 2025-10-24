"""Minimal wrapper around the evermore reference example."""

from collections.abc import Mapping
from functools import partial
from typing import NamedTuple

import evermore as evm
import jax
import jax.numpy as jnp
import optimistix as optx
from flax import nnx
from model_config import DEFAULT_DATA, ModelData, expected_components

import everwillow as ew


class Params(NamedTuple):
    mu: evm.Parameter
    norm1: evm.NormalParameter
    norm2: evm.NormalParameter
    shape1: evm.NormalParameter


def build_components(
    data: ModelData = DEFAULT_DATA,
) -> tuple[Params, dict[str, dict[str, jnp.ndarray]], jnp.ndarray]:
    """Return the Params, histogram templates, and observed data."""

    params = Params(
        mu=evm.Parameter(name="mu"),
        norm1=evm.NormalParameter(name="norm1"),
        norm2=evm.NormalParameter(name="norm2"),
        shape1=evm.NormalParameter(name="shape1"),
    )

    hists = {
        "nominal": {
            "signal": jnp.array([data.signal_nominal]),
            "bkg1": jnp.array([data.bkg1_nominal]),
            "bkg2": jnp.array([data.bkg2_nominal]),
        },
        "shape_up": {
            "bkg1": jnp.array([data.bkg1_shape_up]),
            "bkg2": jnp.array([data.bkg2_shape_up]),
        },
        "shape_down": {
            "bkg1": jnp.array([data.bkg1_shape_down]),
            "bkg2": jnp.array([data.bkg2_shape_down]),
        },
    }

    observation = jnp.array([data.observed])
    return params, hists, observation


def model(params: Params, hists: dict[str, dict[str, jnp.ndarray]]):
    expectations: dict[str, jnp.ndarray] = {}

    sig_mod = params.mu.scale()
    expectations["signal"] = sig_mod(hists["nominal"]["signal"])

    bkg1_lnN = params.norm1.scale_log(up=jnp.array([1.1]), down=jnp.array([0.9]))
    bkg1_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg1"],
        down_template=hists["shape_down"]["bkg1"],
    )
    expectations["bkg1"] = (bkg1_lnN @ bkg1_shape)(hists["nominal"]["bkg1"])

    bkg2_lnN = params.norm2.scale_log(up=jnp.array([1.05]), down=jnp.array([0.95]))
    bkg2_shape = params.shape1.morphing(
        up_template=hists["shape_up"]["bkg2"],
        down_template=hists["shape_down"]["bkg2"],
    )
    expectations["bkg2"] = (bkg2_lnN @ bkg2_shape)(hists["nominal"]["bkg2"])

    return expectations


@nnx.jit
def loss(
    dynamic: Params,
    args: tuple,
) -> jnp.ndarray:
    graphdef, static, hists, observation = args
    params = nnx.merge(graphdef, dynamic, static)
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    log_prob = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    log_prob += evm.util.sum_over_leaves(jax.tree.map(jnp.sum, constraints))
    return -jnp.sum(log_prob)


def fit_with_optimistix(
    components,
    max_steps: int = 10_000,
):
    params, hists, observation = components
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        loss,
        solver,
        dynamic,
        has_aux=False,
        args=(graphdef, static, hists, observation),
        max_steps=max_steps,
    )

    best = result.value.to_pure_dict()
    nll = result.state.f_info.f

    return best, nll


def fit_with_everwillow(
    components,
    max_steps: int = 150,
):
    params, hists, observation = components
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_parameter, ...)

    args = (graphdef, static, hists, observation)

    result = ew.fit(
        partial(loss, args=args),
        params=dynamic,
        max_steps=max_steps,
    )

    return result.params.to_pure_dict(), result.nll


def summarise_evermore_fit(params: Mapping[str, float], data: ModelData = DEFAULT_DATA):
    return expected_components(params, data=data)
