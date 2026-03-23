"""Standalone pyhf counting experiment example."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pyhf

import everwillow as ew

jax.config.update("jax_enable_x64", True)
pyhf.set_backend("jax")

# Build the workspace
spec = {
    "channels": [
        {
            "name": "singlebin",
            "samples": [
                {
                    "name": "signal",
                    "data": [3.0],
                    "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                },
                {
                    "name": "bkg1",
                    "data": [10.0],
                    "modifiers": [
                        {
                            "name": "norm1",
                            "type": "normsys",
                            "data": {"hi": 1.1, "lo": 0.9},
                        },
                        {
                            "name": "shape1",
                            "type": "histosys",
                            "data": {"hi_data": [12.0], "lo_data": [8.0]},
                        },
                    ],
                },
                {
                    "name": "bkg2",
                    "data": [20.0],
                    "modifiers": [
                        {
                            "name": "norm2",
                            "type": "normsys",
                            "data": {"hi": 1.05, "lo": 0.95},
                        },
                        {
                            "name": "shape1",
                            "type": "histosys",
                            "data": {"hi_data": [23.0], "lo_data": [19.0]},
                        },
                    ],
                },
            ],
        }
    ],
    "observations": [{"name": "singlebin", "data": [37.0]}],
    "measurements": [
        {
            "name": "Measurement",
            "config": {
                "poi": "mu",
                "parameters": [],
            },
        }
    ],
    "version": "1.0.0",
}

workspace = pyhf.Workspace(spec)
model = workspace.model()

# Get parameter order and initial values
parameter_order = model.config.par_order
initial_dict = dict(zip(parameter_order, model.config.suggested_init(), strict=False))
observation = workspace.data(model, include_auxdata=True)


# Define NLL
def nll(params, obs):
    parameter_vector = jnp.asarray([params[name] for name in parameter_order])
    logpdf = model.logpdf(parameter_vector, obs)
    return -2 * logpdf[0]


# Perform the fit
result = ew.fit(nll, initial_dict, observation)

print(result.params)
#  {
#   'mu': Array(2.33333334, dtype=float64),
#   'norm1': Array(-3.37256584e-07, dtype=float64),
#   'norm2': Array(4.53219024e-07, dtype=float64),
#   'shape1': Array(-2.25047663e-08, dtype=float64),
# }
