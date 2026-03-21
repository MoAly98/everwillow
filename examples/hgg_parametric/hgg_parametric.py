"""H→γγ parametric model example with fitting and limit setting.

This example demonstrates:
1. Building a parametric model using evermore for parameters and paramore for PDFs
2. Fitting to data using everwillow
3. Computing upper limits on signal strength using everwillow hypotest

The model includes:
- Signal: Gaussian PDF for Higgs mass peak
- Background: Exponential PDF
- Systematics: Photon ID and JEC uncertainties applied via evermore modifiers

Data is downloaded from the paramore repository if not present locally.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import evermore as evm
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import paramore as pm
from flax import nnx

import everwillow as ew
import everwillow.statelib as sl
from everwillow.hypotest import HypoTestCalculator, QTilde, QTildeAsymptotic
from everwillow.hypotest.upper_limit import expected_upper_limit
from everwillow.parameters.transforms import MinuitTransform
from everwillow.uncertainty import correlation_matrix, uncertainties

jax.config.update("jax_enable_x64", True)

# Data files from paramore repository
DATA_BASE_URL = "https://raw.githubusercontent.com/maxgalli/jax_parametric_models/master/examples/samples"
DATA_FILES = ["mc_part1.parquet", "data_part1.parquet"]


def download_data(data_dir: Path) -> bool:
    """Download sample data from paramore repository.

    Args:
        data_dir: Directory to save data files.

    Returns:
        True if download successful, False otherwise.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    for filename in DATA_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            continue

        url = f"{DATA_BASE_URL}/{filename}"
        print(f"Downloading {filename}...")

        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return False

    return True


# ============================================================================
# Parameters PyTree
# ============================================================================


class Params(nnx.Pytree):
    """Container for all physics parameters."""

    def __init__(
        self,
        higgs_mass: evm.Parameter,
        d_higgs_mass: evm.Parameter,
        higgs_width: evm.Parameter,
        lamb: evm.Parameter,
        bkg_norm: evm.Parameter,
        mu: evm.Parameter,
        phoid_syst: evm.NormalParameter,
        jec_syst: evm.NormalParameter,
        nuisance_scale: evm.NormalParameter,
        nuisance_smear: evm.NormalParameter,
    ):
        self.higgs_mass = higgs_mass
        self.d_higgs_mass = d_higgs_mass
        self.higgs_width = higgs_width
        self.lamb = lamb
        self.bkg_norm = bkg_norm
        self.mu = mu
        self.phoid_syst = phoid_syst
        self.jec_syst = jec_syst
        self.nuisance_scale = nuisance_scale
        self.nuisance_smear = nuisance_smear


# ============================================================================
# Model builder
# ============================================================================


def build_nll(
    params: Params,
    observation: dict[str, jax.Array],
    *,
    xs_ggH: float,
    br_hgg: float,
    eff: float,
    lumi: float,
    mass_lower: float,
    mass_upper: float,
) -> jax.Array:
    """Compute extended negative log-likelihood.

    Args:
        params: Model parameters.
        observation: Dict with 'data' key containing observed mass values.
        xs_ggH: ggH cross-section in pb.
        br_hgg: H→γγ branching ratio.
        eff: Signal efficiency.
        lumi: Integrated luminosity in pb^-1.
        mass_lower: Lower mass bound for PDFs.
        mass_upper: Upper mass bound for PDFs.

    Returns:
        Extended NLL value.
    """
    data = observation["data"]

    # Signal PDF parameters with nuisance effects
    signal_mu = (params.higgs_mass.get_value() + params.d_higgs_mass.get_value()) * (
        1.0 + 0.003 * params.nuisance_scale.get_value()
    )
    signal_sigma = params.higgs_width.get_value() * (
        1.0 + 0.045 * params.nuisance_smear.get_value()
    )

    # Signal PDF: Gaussian centered at Higgs mass
    signal_pdf = pm.Gaussian(
        mu=signal_mu,
        sigma=signal_sigma,
        lower=mass_lower,
        upper=mass_upper,
    )

    # Background PDF: Exponential
    background_pdf = pm.Exponential(
        lambd=params.lamb.get_value(),
        lower=mass_lower,
        upper=mass_upper,
    )

    # Signal rate with modifiers
    signal_rate_base = params.mu.get_value() * xs_ggH * br_hgg * eff * lumi

    # Apply systematics via evermore modifiers
    phoid_modifier = pm.SymmLogNormalModifier(parameter=params.phoid_syst, kappa=1.05)
    jec_modifier = pm.AsymmetricLogNormalModifier(
        parameter=params.jec_syst,
        kappa_up=1.056,
        kappa_down=0.951,
    )
    composed_modifier = pm.ComposedModifier(phoid_modifier, jec_modifier)

    # Apply modifiers to signal rate
    signal_rate_param = evm.Parameter(value=signal_rate_base, name="signal_rate_base")
    signal_rate_with_modifiers = composed_modifier.apply(signal_rate_param)
    signal_rate = signal_rate_with_modifiers.get_value()

    bkg_rate = params.bkg_norm.get_value()

    # Build combined PDF
    sum_pdf = pm.SumPDF(
        pdfs=[signal_pdf, background_pdf],
        extended_vals=[signal_rate, bkg_rate],
        lower=mass_lower,
        upper=mass_upper,
    )

    # Compute extended NLL including constraint terms
    nll = pm.create_extended_nll(params, sum_pdf, data)

    return jnp.squeeze(nll)


def build_sb_pdf(
    params: Params,
    *,
    xs_ggH: float,
    br_hgg: float,
    eff: float,
    lumi: float,
    mass_lower: float,
    mass_upper: float,
) -> pm.SumPDF:
    """Build the S+B PDF for sampling Asimov data."""
    signal_mu = params.higgs_mass.get_value() + params.d_higgs_mass.get_value()
    signal_sigma = params.higgs_width.get_value()

    signal_pdf = pm.Gaussian(
        mu=signal_mu,
        sigma=signal_sigma,
        lower=mass_lower,
        upper=mass_upper,
    )
    background_pdf = pm.Exponential(
        lambd=params.lamb.get_value(),
        lower=mass_lower,
        upper=mass_upper,
    )

    signal_rate = params.mu.get_value() * xs_ggH * br_hgg * eff * lumi
    bkg_rate = params.bkg_norm.get_value()

    return pm.SumPDF(
        pdfs=[signal_pdf, background_pdf],
        extended_vals=[signal_rate, bkg_rate],
        lower=mass_lower,
        upper=mass_upper,
    )


# ============================================================================
# Main script
# ============================================================================


def main():
    # Load data from paramore examples
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "samples"

    # Download sample data if not present
    if not download_data(data_dir):
        print("Failed to download sample data.")
        return

    # Physics constants
    xs_ggH = 48.58  # pb
    br_hgg = 0.0027
    lumi = 138000.0  # pb^-1

    # Load MC for efficiency calculation
    df_mc = pd.read_parquet(data_dir / "mc_part1.parquet")
    sumw = df_mc["weight"].sum()
    eff = sumw / (xs_ggH * br_hgg)

    # Load observed data
    df_data = pd.read_parquet(data_dir / "data_part1.parquet")
    data = jnp.array(df_data["CMS_hgg_mass"].values)

    # Mass window
    mass_lower = 100.0
    mass_upper = 180.0

    # ========================================================================
    # Define parameters using evermore
    # ========================================================================

    params = Params(
        higgs_mass=evm.Parameter(value=125.0, name="higgs_mass", frozen=True),
        d_higgs_mass=evm.Parameter(value=0.000848571, name="d_higgs_mass", frozen=True),
        higgs_width=evm.Parameter(value=1.99705, name="higgs_width", frozen=True),
        lamb=evm.Parameter(value=0.1, name="lamb"),
        bkg_norm=evm.Parameter(value=float(len(df_data)), name="bkg_norm"),
        mu=evm.Parameter(value=1.0, name="mu"),
        phoid_syst=evm.NormalParameter(value=0.0, name="phoid_syst"),
        jec_syst=evm.NormalParameter(value=0.0, name="jec_syst"),
        nuisance_scale=evm.NormalParameter(value=0.0, name="nuisance_scale"),
        nuisance_smear=evm.NormalParameter(value=0.0, name="nuisance_smear"),
    )

    # Split into dynamic (to be fitted) and static (frozen) parts
    graphdef, dynamic, static = nnx.split(params, evm.filter.is_dynamic_parameter, ...)

    # ========================================================================
    # Create NLL function for everwillow
    # ========================================================================

    observation = {"data": data}

    def nll_fn(dynamic: nnx.State, obs: dict) -> jax.Array:
        """NLL function with everwillow signature: nll(params, observation).

        param_values is a simple dict of values: {'mu': 1.0, 'lamb': 0.1, ...}
        We reconstruct the nnx.State with Parameters inside the function.
        """

        full_params = nnx.merge(graphdef, dynamic, static)

        return build_nll(
            full_params,
            obs,
            xs_ggH=xs_ggH,
            br_hgg=br_hgg,
            eff=eff,
            lumi=lumi,
            mass_lower=mass_lower,
            mass_upper=mass_upper,
        )

    # ========================================================================
    # Fit the model using everwillow
    # ========================================================================

    print("=" * 60)
    print("Fitting H->gammagamma model")
    print("=" * 60)

    init_state = sl.State.from_pytree(dynamic)

    # Add bounds to constrain parameters
    bounds = sl.State.from_pytree(
        {
            ("mu", "value"): MinuitTransform(lower=0.0, upper=10.0),
            ("lamb", "value"): MinuitTransform(lower=0.0, upper=1.0),
            ("bkg_norm", "value"): MinuitTransform(lower=1000.0, upper=50000.0),
            ("phoid_syst", "value"): MinuitTransform(lower=-5.0, upper=5.0),
            ("jec_syst", "value"): MinuitTransform(lower=-5.0, upper=5.0),
            ("nuisance_scale", "value"): MinuitTransform(lower=-5.0, upper=5.0),
            ("nuisance_smear", "value"): MinuitTransform(lower=-5.0, upper=5.0),
        }
    )

    @jax.jit
    def fun():
        return ew.fit(
            nll_fn, init_state, observation, bounds=bounds, max_steps=2000, throw=False
        )

    result = fun()

    print(f"  NLL = {float(result.nll):.2f}")
    print(f"  Converged: {bool(result.success)}")

    print("\nFit Results:")
    print("-" * 40)
    # Compute parameter uncertainties
    fitted_state = sl.State.from_pytree(result.params)
    param_uncertainties = uncertainties(nll_fn, fitted_state, observation)
    unc_pytree = param_uncertainties.to_pytree()

    print("\nFitted Parameters:")
    print("-" * 40)
    for name in [
        "mu",
        "lamb",
        "bkg_norm",
        "phoid_syst",
        "jec_syst",
        "nuisance_scale",
        "nuisance_smear",
    ]:
        val = float(result.params[name])
        err = float(unc_pytree[name])
        print(f"  {name} = {val:.4f} ± {err:.4f}")

    # Plot correlation matrix
    corr = correlation_matrix(nll_fn, fitted_state, observation)
    param_names = list(fitted_state.to_pytree().keys())

    _fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_yticklabels(param_names)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Parameter Correlation Matrix")
    plt.tight_layout()
    plt.savefig(base_dir / "hgg_correlation.png", dpi=150)
    plt.close()
    print(f"\nSaved correlation plot to {base_dir / 'hgg_correlation.png'}")

    # ========================================================================
    # Generate Asimov dataset from fitted S+B PDF
    # ========================================================================

    # Reconstruct fitted params for building the PDF
    fitted_params_dict = {}
    for path_tuple, param in dynamic.flat_state():
        name = path_tuple[0]
        fitted_params_dict[name] = param.replace(value=result.params[name])
    fitted_dynamic = nnx.State(fitted_params_dict)
    fitted_params = nnx.merge(graphdef, fitted_dynamic, static)

    # Build S+B PDF and sample Asimov data
    sb_pdf = build_sb_pdf(
        fitted_params,
        xs_ggH=xs_ggH,
        br_hgg=br_hgg,
        eff=eff,
        lumi=lumi,
        mass_lower=mass_lower,
        mass_upper=mass_upper,
    )
    asimov_key = jax.random.PRNGKey(42)
    asimov_masses = sb_pdf.sample(asimov_key, 1)  # samples expected total
    asimov_observation = {"data": asimov_masses}
    print(f"\nGenerated Asimov dataset with {len(asimov_masses)} events")

    # Plot data vs Asimov
    bins = 40
    _fig, ax = plt.subplots()
    ax.hist(
        data, bins=bins, range=(mass_lower, mass_upper), histtype="step", label="Data"
    )
    ax.hist(
        asimov_masses,
        bins=bins,
        range=(mass_lower, mass_upper),
        histtype="step",
        label="Asimov (S+B)",
    )
    ax.set_xlabel("$m_{\\gamma\\gamma}$ [GeV]")
    ax.set_ylabel("Events")
    ax.legend()
    plt.savefig(base_dir / "hgg_data_asimov.png", dpi=150)
    plt.close()
    print(f"Saved plot to {base_dir / 'hgg_data_asimov.png'}")

    # ========================================================================
    # Compute upper limit on signal strength
    # ========================================================================

    print("\n" + "=" * 60)
    print("Computing 95% CL upper limit on mu")
    print("=" * 60)

    # Use fitted nuisance parameters as starting point
    fitted_state = sl.State.from_pytree(result.params)

    # Create calculator
    calc = HypoTestCalculator(test_statistic=QTilde(), distribution=QTildeAsymptotic())

    # Define function that returns HypoTestResult for a given mu value
    def calc_fn(mu_test: float):
        return calc(
            nll_fn,
            fitted_state,
            observation,
            poi_key=("mu", "value"),
            poi_test=mu_test,
            bounds=bounds,
            asimov_observation=asimov_observation,
        )

    @jax.jit
    def compute_limits():
        return expected_upper_limit(
            calc_fn,
            bounds=(0.01, 5.0),
            level=0.05,
            max_steps=100,
        )

    limit_result = compute_limits()

    print("\nUpper Limits (95% CL):")
    print("-" * 40)
    print(f"  Observed:    {float(limit_result.observed):.3f}")
    print(f"  Expected:    {float(limit_result.expected.median):.3f}")
    print(f"    -2 sigma:  {float(limit_result.expected.minus_2sigma):.3f}")
    print(f"    -1 sigma:  {float(limit_result.expected.minus_1sigma):.3f}")
    print(f"    +1 sigma:  {float(limit_result.expected.plus_1sigma):.3f}")
    print(f"    +2 sigma:  {float(limit_result.expected.plus_2sigma):.3f}")


if __name__ == "__main__":
    main()
