"""Minimal wrapper around the evermore reference example."""

from __future__ import annotations

from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

import evermore as evm

from .model_config import DEFAULT_DATA, ModelData, expected_components

jax.config.update("jax_enable_x64", True)


class Params(eqx.Module):
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


@eqx.filter_jit
def loss(
    dynamic: Params,
    static: Params,
    hists: dict[str, dict[str, jnp.ndarray]],
    observation: jnp.ndarray,
) -> jnp.ndarray:
    params = evm.tree.combine(dynamic, static)
    expectations = model(params, hists)
    constraints = evm.loss.get_log_probs(params)
    log_prob = (
        evm.pdf.PoissonContinuous(evm.util.sum_over_leaves(expectations))
        .log_prob(observation)
        .sum()
    )
    log_prob += evm.util.sum_over_leaves(constraints)
    return -jnp.sum(log_prob)


def fit_with_optimistix(
    data: ModelData = DEFAULT_DATA,
    max_steps: int = 10_000,
):
    params, hists, observation = build_components(data)
    dynamic, static = evm.tree.partition(params)

    def optx_loss(dynamic_params, args):
        static_params, hists_, obs_ = args
        return loss(dynamic_params, static_params, hists_, obs_)

    solver = optx.BFGS(rtol=1e-5, atol=1e-7)
    result = optx.minimise(
        optx_loss,
        solver,
        dynamic,
        has_aux=False,
        args=(static, hists, observation),
        max_steps=max_steps,
    )

    best = evm.tree.combine(result.value, static)
    nll = float(loss(result.value, static, hists, observation))

    return (
        {
            "mu": float(best.mu.value),
            "norm1": float(best.norm1.value),
            "norm2": float(best.norm2.value),
            "shape1": float(best.shape1.value),
        },
        nll,
    )


def fit_with_everwillow(
    data: ModelData = DEFAULT_DATA,
    max_steps: int = 150,
):
    import everwillow as ew

    params, hists, observation = build_components(data)
    dynamic, static = evm.tree.partition(params)

    def nll(param_dict: Mapping[str, float]) -> jnp.ndarray:
        updated = evm.tree.update_values(
            dynamic,
            values=Params(
                mu=param_dict["mu"],
                norm1=param_dict["norm1"],
                norm2=param_dict["norm2"],
                shape1=param_dict["shape1"],
            ),
        )
        return loss(updated, static, hists, observation)

    result = ew.fit(
        nll,
        {
            "mu": float(params.mu.value),
            "norm1": float(params.norm1.value),
            "norm2": float(params.norm2.value),
            "shape1": float(params.shape1.value),
        },
        max_steps=max_steps,
    )

    return dict(result.params), float(result.nll)


def summarise_evermore_fit(params: Mapping[str, float], data: ModelData = DEFAULT_DATA):
    return expected_components(params, data=data)
