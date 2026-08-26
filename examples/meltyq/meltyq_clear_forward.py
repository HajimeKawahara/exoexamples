"""Minimal clear-atmosphere forward model inspired by MELTYQ.

This example connects ExoGibbs, ExoEOS, ExoJAX, and external ExoPie without
introducing package-specific adapters in ExoExamples. It is a compact
forward-model demonstration, not a reproduction of the MELTYQ retrieval.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from exopie import get_radius
from exoeos import (
    AdditiveVolumeCompositeDensityProvider,
    ChabrierDebrasTableLoader,
    DensityComponent,
    FixedCompositionDensityProvider,
    IdealEOS,
    TPHelmholtzDensityProvider,
    ZhangDuanEOS,
)
from exogibbs.api import gas, magma_gas
from exogibbs.interop.exoeos import make_pure_lnphi_func
from exogibbs.presets.magma_gas import (
    MELTYQ_SPECIES,
    MeltyqMagmaGasInputs,
    prepare_meltyq_problem,
)
from exogibbs.presets.fastchem4 import chemsetup
from exojax.atm.atmprof import (
    hydrostatic_radius_profile,
    pressure_layer_logspace_from_boundaries,
)
from exojax.database.molinfo import mean_molmass_manual
from exojax.utils.constants import ME, RE


PLANET_MASS_EARTH = 4.0
ROCKY_CORE_MASS_FRACTION = 0.33

PRESSURE_MELT_BAR = 7000.0
TEMPERATURE_MELT_K = 1700.0
PRESSURE_BASE_BAR = 10.0
TEMPERATURE_BASE_K = 1000.0
PRESSURE_TOP_BAR = 1.0e-10

OXYGEN_FUGACITY_BAR = 1.0e-10
CO_MELT_MOLE_RATIO = 5.0e-5
N_MELT_MOLE_RATIO = 1.0e-4
HELIUM_MASS_FRACTION = 0.275

DEEP_NLAYER = 64
UPPER_NLAYER = 40
WAVENUMBER_MIN = 4355.0
WAVENUMBER_MAX = 4365.0
WAVENUMBER_POINTS = 2048

DATABASE_DIRECTORY = Path(__file__).resolve().parents[2] / ".database"

SPECIES_MAP = {
    "He": "He1",
    "H2O": "H2O1",
    "CO": "C1O1",
    "CO2": "C1O2",
    "CH4": "C1H4",
    "NH3": "H3N1",
}

CANONICAL_BY_SOURCE = {
    "H2": "H2",
    "O2": "O2",
    "H2O1": "H2O",
    "C1O1": "CO",
    "C1O2": "CO2",
    "C1H4": "CH4",
}


def require_convergence(label, converged):
    """Raise when a scalar or profile solver did not converge."""

    if not bool(np.all(np.asarray(converged))):
        raise RuntimeError(f"{label} did not converge.")


def project_h_he_for_density(mole_fractions, molar_masses_kg_mol):
    """Project the H2-He component to fixed Y while preserving its mass."""

    h2_index = MELTYQ_SPECIES.index("H2")
    he_index = MELTYQ_SPECIES.index("He")
    h_he_mass = (
        mole_fractions[h2_index] * molar_masses_kg_mol[h2_index]
        + mole_fractions[he_index] * molar_masses_kg_mol[he_index]
    )
    projected = mole_fractions.at[h2_index].set(
        (1.0 - HELIUM_MASS_FRACTION)
        * h_he_mass
        / molar_masses_kg_mol[h2_index]
    )
    projected = projected.at[he_index].set(
        HELIUM_MASS_FRACTION
        * h_he_mass
        / molar_masses_kg_mol[he_index]
    )
    return projected / jnp.sum(projected)


def make_density_provider(molar_masses_kg_mol, pure_eos):
    """Build the MELTYQ additive-volume density closure."""

    species_index = {name: index for index, name in enumerate(MELTYQ_SPECIES)}

    def masses(*names):
        return jnp.asarray(
            [molar_masses_kg_mol[species_index[name]] for name in names]
        )

    h_he_provider = FixedCompositionDensityProvider(
        eos=ChabrierDebrasTableLoader(variant="Y0275").load(),
        molar_masses=masses("H2", "He"),
        expected_mass_fractions=jnp.asarray(
            [1.0 - HELIUM_MASS_FRACTION, HELIUM_MASS_FRACTION]
        ),
        composition_rtol=1.0e-6,
    )
    components = [DensityComponent(("H2", "He"), h_he_provider)]
    for species in ("O2", "H2O", "CO", "CO2", "CH4"):
        components.append(
            DensityComponent(
                (species,),
                TPHelmholtzDensityProvider(
                    eos=pure_eos[species],
                    molar_masses=masses(species),
                ),
            )
        )
    components.append(
        DensityComponent(
            ("N2", "NH3"),
            TPHelmholtzDensityProvider(
                eos=IdealEOS(),
                molar_masses=masses("N2", "NH3"),
            ),
        )
    )
    return AdditiveVolumeCompositeDensityProvider(
        species=MELTYQ_SPECIES,
        components=components,
    )


def plot_result(
    pressure_deep_bar,
    temperature_deep_k,
    mole_fractions_deep,
    density_deep_kg_m3,
    radius_boundaries_cm,
    pressure_boundaries_bar,
    wavenumber,
    transit_radius_rearth,
):
    """Plot the coupled structure and clear transmission spectrum."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(temperature_deep_k, pressure_deep_bar)
    axes[0, 0].set(xlabel="Temperature (K)", ylabel="Pressure (bar)")

    for index, species in enumerate(MELTYQ_SPECIES):
        axes[0, 1].plot(
            np.clip(mole_fractions_deep[:, index], 1.0e-30, None),
            pressure_deep_bar,
            label=species,
        )
    axes[0, 1].set(
        xscale="log",
        xlabel="Mole fraction",
        ylabel="Pressure (bar)",
    )
    axes[0, 1].legend(ncol=3, fontsize=8)

    axes[1, 0].plot(density_deep_kg_m3, pressure_deep_bar, color="tab:blue")
    axes[1, 0].set(
        xscale="log",
        xlabel="Mass density (kg m$^{-3}$)",
        ylabel="Pressure (bar)",
    )
    radius_axis = axes[1, 0].twiny()
    radius_axis.plot(
        radius_boundaries_cm / RE,
        pressure_boundaries_bar,
        color="tab:orange",
    )
    radius_axis.set_xlabel("Radius ($R_\\oplus$)", color="tab:orange")

    wavelength_micron = 1.0e4 / np.asarray(wavenumber)
    axes[1, 1].plot(
        wavelength_micron[::-1],
        np.asarray(transit_radius_rearth)[::-1],
    )
    axes[1, 1].set(
        xlabel="Wavelength ($\\mu$m)",
        ylabel="Transit radius ($R_\\oplus$)",
    )

    for axis in axes[:, :1].flat:
        axis.set_yscale("log")
        axis.invert_yaxis()
    axes[0, 1].set_yscale("log")
    axes[0, 1].invert_yaxis()
    radius_axis.set_yscale("log")

    figure.suptitle("Minimal MELTYQ-like clear forward model")
    figure.tight_layout()
    plt.show()


def main():
    """Run the coupled magma, deep-atmosphere, and transmission model."""

    molar_masses_g_mol = jnp.asarray(
        [mean_molmass_manual(species) for species in MELTYQ_SPECIES]
    )
    molar_masses_kg_mol = molar_masses_g_mol * 1.0e-3

    rocky_radius_rearth = float(
        np.asarray(
            get_radius(
                PLANET_MASS_EARTH,
                cmf=ROCKY_CORE_MASS_FRACTION,
                wmf=None,
                amf=None,
                xSi=0.0,
                xFe=0.0,
            )
        ).squeeze()
    )
    planet_mass_g = PLANET_MASS_EARTH * ME
    rocky_radius_cm = rocky_radius_rearth * RE

    source_setup = chemsetup(silent=True)
    pure_eos = {
        species: ZhangDuanEOS.from_species((species,))
        for species in ("H2", "O2", "H2O", "CO", "CO2", "CH4")
    }
    source_lnphi = make_pure_lnphi_func(
        source_species=source_setup.species,
        eos_by_species={
            source: pure_eos[canonical]
            for source, canonical in CANONICAL_BY_SOURCE.items()
        },
        unspecified_species="ideal",
    )
    problem = prepare_meltyq_problem(
        source_setup,
        lnphi_func=source_lnphi,
        species_map=SPECIES_MAP,
    )
    boundary = magma_gas.solve(
        problem,
        temperature_k=TEMPERATURE_MELT_K,
        pressure_bar=PRESSURE_MELT_BAR,
        model_inputs=MeltyqMagmaGasInputs(
            oxygen_fugacity_bar=OXYGEN_FUGACITY_BAR,
            co_melt_mole_ratio=CO_MELT_MOLE_RATIO,
            n_melt_mole_ratio=N_MELT_MOLE_RATIO,
        ),
    )
    require_convergence("Magma-gas boundary", boundary.diagnostics.converged)

    (
        pressure_deep_bar,
        _,
        _,
        pressure_boundaries_bar,
    ) = pressure_layer_logspace_from_boundaries(
        np.log10(PRESSURE_BASE_BAR),
        np.log10(PRESSURE_MELT_BAR),
        DEEP_NLAYER,
        numpy=True,
    )
    temperature_deep_k = np.interp(
        np.log10(pressure_deep_bar),
        [np.log10(PRESSURE_BASE_BAR), np.log10(PRESSURE_MELT_BAR)],
        [TEMPERATURE_BASE_K, TEMPERATURE_MELT_K],
    )
    pressure_deep_bar = jnp.asarray(pressure_deep_bar)
    temperature_deep_k = jnp.asarray(temperature_deep_k)

    boundary_init = gas.EquilibriumInit(
        ln_nk=boundary.gas.equilibrium.ln_n,
        ln_ntot=jnp.log(boundary.gas.equilibrium.ntot),
    )
    deep_equilibrium, deep_diagnostics = gas.solve_profile(
        problem.setup,
        temperature_deep_k,
        pressure_deep_bar,
        boundary.element_abundances,
        init=boundary_init,
        options=gas.EquilibriumOptions(method="scan_hot_from_bottom"),
        return_diagnostics=True,
        lnphi_func=problem.lnphi_func,
    )
    require_convergence(
        "Deep-atmosphere equilibrium",
        deep_diagnostics["converged"],
    )

    base_init = gas.EquilibriumInit(
        ln_nk=deep_equilibrium.ln_n[0],
        ln_ntot=jnp.log(deep_equilibrium.ntot[0]),
    )
    base_equilibrium, base_diagnostics = gas.solve(
        problem.setup,
        TEMPERATURE_BASE_K,
        PRESSURE_BASE_BAR,
        boundary.element_abundances,
        init=base_init,
        return_diagnostics=True,
        lnphi_func=problem.lnphi_func,
    )
    require_convergence("10-bar equilibrium", base_diagnostics["converged"])

    density_provider = make_density_provider(molar_masses_kg_mol, pure_eos)
    density_provider.validate_species(MELTYQ_SPECIES)
    density_composition = jax.vmap(
        lambda composition: project_h_he_for_density(
            composition,
            molar_masses_kg_mol,
        )
    )(deep_equilibrium.x)
    density_deep_kg_m3 = jax.vmap(density_provider.mass_density_tp)(
        temperature_deep_k,
        pressure_deep_bar * 1.0e5,
        density_composition,
    )
    if not np.all(np.isfinite(np.asarray(density_deep_kg_m3))):
        raise RuntimeError("Deep-atmosphere density contains non-finite values.")
    if not np.all(np.asarray(density_deep_kg_m3) > 0.0):
        raise RuntimeError("Deep-atmosphere density must be positive.")

    radius_boundaries_cm, gravity_boundaries_cgs = hydrostatic_radius_profile(
        jnp.asarray(pressure_boundaries_bar),
        density_deep_kg_m3 * 1.0e-3,
        planet_mass_g,
        rocky_radius_cm,
    )
    radius_base_cm = radius_boundaries_cm[0]
    gravity_base_cgs = gravity_boundaries_cgs[0]
    radius_values = np.asarray(radius_boundaries_cm)
    gravity_values = np.asarray(gravity_boundaries_cgs)
    if not np.all(np.isfinite(radius_values)) or not np.all(radius_values > 0.0):
        raise RuntimeError("Deep-atmosphere radius must be finite and positive.")
    if not np.all(np.diff(radius_values) < 0.0):
        raise RuntimeError("Deep-atmosphere radius must decrease with depth.")
    if not np.all(np.isfinite(gravity_values)) or not np.all(gravity_values > 0.0):
        raise RuntimeError("Deep-atmosphere gravity must be finite and positive.")
    if not float(radius_base_cm > rocky_radius_cm):
        raise RuntimeError("The 10-bar radius must exceed the rocky radius.")

    from exojax.database import CdbCIA, MdbExomol
    from exojax.opacity import OpaCIA, OpaPremodit
    from exojax.rt import ArtTransPure
    from exojax.utils.grids import wavenumber_grid

    DATABASE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    nu_grid, _, spectral_resolution = wavenumber_grid(
        WAVENUMBER_MIN,
        WAVENUMBER_MAX,
        WAVENUMBER_POINTS,
        unit="cm-1",
        xsmode="premodit",
    )
    art = ArtTransPure.from_pressure_boundaries(
        pressure_top_boundary=PRESSURE_TOP_BAR,
        pressure_btm_boundary=PRESSURE_BASE_BAR,
        nlayer=UPPER_NLAYER,
        integration="simpson",
        nu_grid=nu_grid,
    )
    upper_temperature_k = jnp.full_like(
        jnp.asarray(art.pressure),
        TEMPERATURE_BASE_K,
    )
    upper_mean_molecular_weight = jnp.full_like(
        upper_temperature_k,
        jnp.sum(base_equilibrium.x * molar_masses_g_mol),
    )
    upper_gravity_cgs = art.gravity_profile(
        upper_temperature_k,
        upper_mean_molecular_weight,
        radius_base_cm,
        gravity_base_cgs,
    )

    mdb_co = MdbExomol(
        str(DATABASE_DIRECTORY / "CO" / "12C-16O" / "Li2015"),
        nurange=nu_grid,
        broadf=False,
        broadf_download=False,
        gpu_transfer=False,
    )
    opa_co = OpaPremodit(
        mdb_co,
        nu_grid,
        auto_trange=[0.9 * TEMPERATURE_BASE_K, 1.1 * TEMPERATURE_BASE_K],
    )
    cross_section_co = opa_co.xsmatrix(upper_temperature_k, art.pressure)
    co_index = MELTYQ_SPECIES.index("CO")
    h2_index = MELTYQ_SPECIES.index("H2")
    vmr_co = jnp.full_like(upper_temperature_k, base_equilibrium.x[co_index])
    vmr_h2 = jnp.full_like(upper_temperature_k, base_equilibrium.x[h2_index])
    mmr_co = vmr_co * opa_co.molmass / upper_mean_molecular_weight
    optical_depth_co = art.opacity_profile_xs(
        cross_section_co,
        mmr_co,
        opa_co.molmass,
        upper_gravity_cgs,
    )

    cdb_h2_h2 = CdbCIA(
        str(DATABASE_DIRECTORY / "H2-H2_2011.cia"),
        nurange=nu_grid,
    )
    opa_h2_h2 = OpaCIA(
        cdb_h2_h2,
        nu_grid=nu_grid,
        wavenumber_interpolation="interp",
    )
    optical_depth_h2_h2 = art.opacity_profile_cia(
        opa_h2_h2.logacia_matrix(upper_temperature_k),
        upper_temperature_k,
        vmr_h2,
        vmr_h2,
        upper_mean_molecular_weight,
        upper_gravity_cgs,
    )
    transit_radius_squared = art.run(
        optical_depth_co + optical_depth_h2_h2,
        upper_temperature_k,
        upper_mean_molecular_weight,
        radius_base_cm,
        gravity_base_cgs,
    )
    transit_radius_rearth = radius_base_cm * jnp.sqrt(
        transit_radius_squared
    ) / RE
    if not np.all(np.isfinite(np.asarray(transit_radius_rearth))):
        raise RuntimeError("Transmission spectrum contains non-finite values.")
    if float(jnp.min(transit_radius_rearth)) < float(radius_base_cm / RE):
        raise RuntimeError("Transit radius must not be smaller than its base.")

    print("MELTYQ-like clear forward model")
    print(f"  Rocky radius:       {rocky_radius_rearth:.6f} R_earth")
    print(f"  Radius at 10 bar:   {float(radius_base_cm / RE):.6f} R_earth")
    print(
        "  Deep thickness:     "
        f"{float((radius_base_cm - rocky_radius_cm) * 1.0e-5):.2f} km"
    )
    print(
        "  Deep density range: "
        f"{float(jnp.min(density_deep_kg_m3)):.3g}--"
        f"{float(jnp.max(density_deep_kg_m3)):.3g} kg m^-3"
    )
    print(
        "  Solver iterations:  "
        f"boundary={int(boundary.diagnostics.iterations)}, "
        f"deep_max={int(jnp.max(deep_diagnostics['n_iter']))}, "
        f"base={int(base_diagnostics['n_iter'])}"
    )
    print(f"  Spectral resolution: {spectral_resolution:.0f}")
    print("  Mole fractions at 10 bar:")
    for species, abundance in zip(MELTYQ_SPECIES, base_equilibrium.x):
        print(f"    {species:>3s}: {float(abundance):.6e}")

    plot_result(
        np.asarray(pressure_deep_bar),
        np.asarray(temperature_deep_k),
        np.asarray(deep_equilibrium.x),
        np.asarray(density_deep_kg_m3),
        np.asarray(radius_boundaries_cm),
        np.asarray(pressure_boundaries_bar),
        nu_grid,
        transit_radius_rearth,
    )


if __name__ == "__main__":
    main()
