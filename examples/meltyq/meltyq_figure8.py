"""Recompute the quantities shown in MELTYQ Figure 8.

The script evaluates the published one-parameter sweeps with ExoGibbs,
ExoEOS, ExoJAX, and ExoPie.  It writes solver diagnostics and numerical
results before plotting, so missing curve segments remain explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax

DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / ".cache" / "jax"
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
)
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        str(DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY),
    )
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from exopie import get_radius
from exoeos import ZhangDuanEOS
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

if __package__:
    from .meltyq_clear_forward import (
        CANONICAL_BY_SOURCE,
        SPECIES_MAP,
        make_density_provider,
        project_h_he_for_density,
    )
else:
    from meltyq_clear_forward import (
        CANONICAL_BY_SOURCE,
        SPECIES_MAP,
        make_density_provider,
        project_h_he_for_density,
    )


PAPER_DOI = "https://doi.org/10.3847/1538-4357/ae6917"
PLANET_MASS_EARTH = 4.0
ROCKY_CORE_MASS_FRACTION = 0.33
PRESSURE_BASE_BAR = 10.0
DEEP_NLAYER = 64

PLOTTED_SPECIES = (
    "H2",
    "O2",
    "H2O",
    "CO",
    "CO2",
    "CH4",
    "N2",
    "NH3",
)

SPECIES_COLORS = {
    "H2": "#8a00ff",
    "O2": "#009e73",
    "H2O": "#56b4e9",
    "CO": "#e69f00",
    "CO2": "#f0e442",
    "CH4": "#0072b2",
    "N2": "#a52a2a",
    "NH3": "#f4a3a8",
}


@dataclass(frozen=True)
class Figure8Parameters:
    """Inputs varied in MELTYQ Figure 8."""

    pressure_melt_bar: float = 1.0e4
    temperature_melt_k: float = 3000.0
    oxygen_fugacity_bar: float = 1.0e-5
    co_melt_mole_ratio: float = 1.0e-5
    n_melt_mole_ratio: float = 1.0e-6
    temperature_base_k: float = 1000.0


@dataclass(frozen=True)
class SweepSpec:
    """Definition of one published Figure 8 panel."""

    name: str
    title: str
    field: str
    values: tuple[float, ...]
    xlabel: str
    plot_log10: bool


@dataclass
class ModelContext:
    """Static objects shared by every sweep point."""

    problem: object
    solve_boundary: object
    solve_structure: object
    planet_mass_earth: float
    rocky_core_mass_fraction: float
    rocky_radius_rearth: float
    magma_options: magma_gas.MagmaGasOptions
    deep_nlayer: int
    thermochemistry_probe_change: float
    thermochemistry_clipped: bool


@dataclass
class PointResult:
    """Figure 8 observables and diagnostics for one input point."""

    parameters: Figure8Parameters
    melt_mole_fractions: np.ndarray
    base_mole_fractions: np.ndarray
    radius_base_rearth: float
    boundary_converged: bool
    boundary_outer_converged: bool
    boundary_inner_converged: bool
    boundary_iterations: int
    boundary_residual_norm: float
    boundary_root: np.ndarray
    boundary_residual: np.ndarray
    profile_converged: bool
    profile_max_iterations: int
    base_converged: bool
    base_iterations: int
    error: str


@dataclass(frozen=True)
class SweepPoint:
    """A result associated with its panel coordinate."""

    sweep: SweepSpec
    requested_value: float
    result: PointResult


BASELINE = Figure8Parameters()

SWEEPS = (
    SweepSpec(
        name="pressure_melt",
        title=r"(a) Dependence on $P_{\mathrm{melt}}$",
        field="pressure_melt_bar",
        values=tuple(np.logspace(3.0, 5.0, 13)),
        xlabel=r"Log($P_{\mathrm{melt}}$ [bar])",
        plot_log10=True,
    ),
    SweepSpec(
        name="temperature_melt",
        title=r"(b) Dependence on $T_{\mathrm{melt}}$",
        field="temperature_melt_k",
        values=tuple(np.arange(2000.0, 4001.0, 100.0)),
        xlabel=r"$T_{\mathrm{melt}}$ [K]",
        plot_log10=False,
    ),
    SweepSpec(
        name="oxygen_fugacity",
        title=r"(c) Dependence on $f_{\mathrm{O_2}}$",
        field="oxygen_fugacity_bar",
        values=tuple(np.logspace(-10.0, 0.0, 11)),
        xlabel=r"Log($f_{\mathrm{O_2}}$ [bar])",
        plot_log10=True,
    ),
    SweepSpec(
        name="co_melt",
        title="(d) Dependence on CO fraction in melt",
        field="co_melt_mole_ratio",
        values=tuple(np.logspace(-10.0, -4.0, 7)),
        xlabel=r"Log($\mathrm{CO}_{\mathrm{melt}}$)",
        plot_log10=True,
    ),
    SweepSpec(
        name="n_melt",
        title=r"(e) Dependence on N$_2$ fraction in melt",
        field="n_melt_mole_ratio",
        values=tuple(np.logspace(-10.0, -4.0, 7)),
        xlabel=r"Log($N_{\mathrm{melt}}$)",
        plot_log10=True,
    ),
    SweepSpec(
        name="temperature_base",
        title=r"(f) Dependence on $T_b$",
        field="temperature_base_k",
        values=tuple(np.arange(300.0, 2001.0, 100.0)),
        xlabel=r"$T_b$ [K]",
        plot_log10=False,
    ),
)


def _empty_result(parameters, *, error=""):
    species_count = len(MELTYQ_SPECIES)
    return PointResult(
        parameters=parameters,
        melt_mole_fractions=np.full(species_count, np.nan),
        base_mole_fractions=np.full(species_count, np.nan),
        radius_base_rearth=np.nan,
        boundary_converged=False,
        boundary_outer_converged=False,
        boundary_inner_converged=False,
        boundary_iterations=0,
        boundary_residual_norm=np.nan,
        boundary_root=np.full(4, np.nan),
        boundary_residual=np.full(4, np.nan),
        profile_converged=False,
        profile_max_iterations=0,
        base_converged=False,
        base_iterations=0,
        error=error,
    )


def _require_positive_integer(label, value):
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise ValueError(f"{label} must be a positive integer.")


def _validated_mole_fractions(label, values):
    values = np.asarray(values)
    if values.shape != (len(MELTYQ_SPECIES),):
        raise RuntimeError(f"{label} has an unexpected species shape.")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise RuntimeError(f"{label} must be finite and non-negative.")
    if not np.isclose(np.sum(values), 1.0, rtol=1.0e-8, atol=1.0e-10):
        raise RuntimeError(f"{label} must sum to one.")
    return values


def build_context(
    *,
    max_outer_iterations=100,
    deep_nlayer=DEEP_NLAYER,
    planet_mass_earth=PLANET_MASS_EARTH,
    rocky_core_mass_fraction=ROCKY_CORE_MASS_FRACTION,
    rocky_radius_rearth=None,
):
    """Build package-owned providers used by the Figure 8 workflow.

    The optional rocky-radius override lets another workflow supply a radius
    relation while retaining the same deep-atmosphere solver.
    """

    planet_mass_earth = float(planet_mass_earth)
    rocky_core_mass_fraction = float(rocky_core_mass_fraction)
    if not np.isfinite(planet_mass_earth) or planet_mass_earth <= 0.0:
        raise ValueError("planet_mass_earth must be finite and positive.")
    if (
        not np.isfinite(rocky_core_mass_fraction)
        or not 0.0 <= rocky_core_mass_fraction <= 1.0
    ):
        raise ValueError("rocky_core_mass_fraction must lie in [0, 1].")
    _require_positive_integer("max_outer_iterations", max_outer_iterations)
    _require_positive_integer("deep_nlayer", deep_nlayer)
    max_outer_iterations = int(max_outer_iterations)
    deep_nlayer = int(deep_nlayer)

    molar_masses_g_mol = jnp.asarray(
        [mean_molmass_manual(species) for species in MELTYQ_SPECIES]
    )
    molar_masses_kg_mol = molar_masses_g_mol * 1.0e-3
    pure_eos = {
        species: ZhangDuanEOS.from_species((species,))
        for species in ("H2", "O2", "H2O", "CO", "CO2", "CH4")
    }
    source_setup = chemsetup(silent=True)
    source_names = tuple(source_setup.species)
    selected_names = tuple(
        SPECIES_MAP.get(species, species) for species in MELTYQ_SPECIES
    )
    selected_indices = [source_names.index(name) for name in selected_names]
    hvector_1500 = np.asarray(source_setup.hvector_func(1500.0))[
        selected_indices
    ]
    hvector_3000 = np.asarray(source_setup.hvector_func(3000.0))[
        selected_indices
    ]
    if np.any(~np.isfinite(hvector_1500)) or np.any(
        ~np.isfinite(hvector_3000)
    ):
        raise RuntimeError("Thermochemistry probe contains non-finite values.")
    thermochemistry_probe_change = float(
        np.max(np.abs(hvector_3000 - hvector_1500))
    )
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
    density_provider = make_density_provider(
        molar_masses_kg_mol,
        pure_eos,
    )
    density_provider.validate_species(MELTYQ_SPECIES)
    if rocky_radius_rearth is None:
        rocky_radius_rearth = float(
            np.asarray(
                get_radius(
                    planet_mass_earth,
                    cmf=rocky_core_mass_fraction,
                    wmf=None,
                    amf=None,
                    xSi=0.0,
                    xFe=0.0,
                )
            ).squeeze()
        )
    else:
        rocky_radius_rearth = float(rocky_radius_rearth)
    if not np.isfinite(rocky_radius_rearth) or rocky_radius_rearth <= 0.0:
        raise ValueError(
            "rocky_radius_rearth must be finite and positive when provided."
        )
    magma_options = magma_gas.MagmaGasOptions(
        max_iter=max_outer_iterations,
    )
    profile_options = gas.EquilibriumOptions(
        method="scan_hot_from_bottom",
    )

    def solve_boundary(
        temperature_k,
        pressure_bar,
        oxygen_fugacity_bar,
        co_melt_mole_ratio,
        n_melt_mole_ratio,
        root_variables,
    ):
        return magma_gas.solve(
            problem,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            model_inputs=MeltyqMagmaGasInputs(
                oxygen_fugacity_bar=oxygen_fugacity_bar,
                co_melt_mole_ratio=co_melt_mole_ratio,
                n_melt_mole_ratio=n_melt_mole_ratio,
            ),
            init=magma_gas.MagmaGasInit(root_variables=root_variables),
            options=magma_options,
        )

    def solve_structure(
        temperature_deep_k,
        pressure_deep_bar,
        pressure_boundaries_bar,
        temperature_base_k,
        element_abundances,
        boundary_ln_n,
        boundary_ntot,
    ):
        boundary_init = gas.EquilibriumInit(
            ln_nk=boundary_ln_n,
            ln_ntot=jnp.log(boundary_ntot),
        )
        deep_equilibrium, deep_diagnostics = gas.solve_profile(
            problem.setup,
            temperature_deep_k,
            pressure_deep_bar,
            element_abundances,
            init=boundary_init,
            options=profile_options,
            return_diagnostics=True,
            lnphi_func=problem.lnphi_func,
        )
        base_init = gas.EquilibriumInit(
            ln_nk=deep_equilibrium.ln_n[0],
            ln_ntot=jnp.log(deep_equilibrium.ntot[0]),
        )
        base_equilibrium, base_diagnostics = gas.solve(
            problem.setup,
            temperature_base_k,
            PRESSURE_BASE_BAR,
            element_abundances,
            init=base_init,
            return_diagnostics=True,
            lnphi_func=problem.lnphi_func,
        )
        density_composition = jax.vmap(
            lambda composition: project_h_he_for_density(
                composition,
                molar_masses_kg_mol,
            )
        )(deep_equilibrium.x)
        density_deep_kg_m3 = jax.vmap(
            density_provider.mass_density_tp
        )(
            temperature_deep_k,
            pressure_deep_bar * 1.0e5,
            density_composition,
        )
        radius_boundaries_cm, _ = hydrostatic_radius_profile(
            pressure_boundaries_bar,
            density_deep_kg_m3 * 1.0e-3,
            planet_mass_earth * ME,
            rocky_radius_rearth * RE,
        )
        return (
            deep_diagnostics,
            base_equilibrium,
            base_diagnostics,
            density_deep_kg_m3,
            radius_boundaries_cm,
        )

    return ModelContext(
        problem=problem,
        solve_boundary=jax.jit(solve_boundary),
        solve_structure=jax.jit(solve_structure),
        planet_mass_earth=planet_mass_earth,
        rocky_core_mass_fraction=rocky_core_mass_fraction,
        rocky_radius_rearth=rocky_radius_rearth,
        magma_options=magma_options,
        deep_nlayer=deep_nlayer,
        thermochemistry_probe_change=thermochemistry_probe_change,
        thermochemistry_clipped=thermochemistry_probe_change == 0.0,
    )


def _boundary_result(parameters, boundary):
    diagnostics = boundary.diagnostics
    result = _empty_result(parameters)
    result.boundary_converged = bool(diagnostics.converged)
    result.boundary_outer_converged = bool(diagnostics.outer_converged)
    result.boundary_inner_converged = bool(diagnostics.inner_converged)
    result.boundary_iterations = int(diagnostics.iterations)
    result.boundary_residual_norm = float(diagnostics.residual_norm)
    result.boundary_root = np.asarray(boundary.root_variables)
    result.boundary_residual = np.asarray(diagnostics.residual)
    if result.boundary_converged:
        if (
            result.boundary_root.shape != (4,)
            or result.boundary_residual.shape != (4,)
            or np.any(~np.isfinite(result.boundary_root))
            or np.any(~np.isfinite(result.boundary_residual))
        ):
            raise RuntimeError(
                "Converged magma-gas root diagnostics must be finite."
            )
        result.melt_mole_fractions = _validated_mole_fractions(
            "Magma-surface gas mole fractions",
            boundary.gas.equilibrium.x,
        )
    return result


def solve_point(context, parameters, *, init=None):
    """Evaluate one Figure 8 point through the public package APIs."""

    try:
        solver_dtype = context.problem.setup.formula_matrix.dtype
        boundary_arguments = tuple(
            jnp.asarray(value, dtype=solver_dtype)
            for value in (
                parameters.temperature_melt_k,
                parameters.pressure_melt_bar,
                parameters.oxygen_fugacity_bar,
                parameters.co_melt_mole_ratio,
                parameters.n_melt_mole_ratio,
            )
        )
        if init is None:
            conditions = magma_gas.MagmaGasConditions(
                temperature_k=boundary_arguments[0],
                pressure_bar=boundary_arguments[1],
                model_inputs=MeltyqMagmaGasInputs(
                    oxygen_fugacity_bar=boundary_arguments[2],
                    co_melt_mole_ratio=boundary_arguments[3],
                    n_melt_mole_ratio=boundary_arguments[4],
                ),
            )
            root_variables = context.problem.model.initial_root(conditions)
        else:
            root_variables = init.root_variables
        root_variables = jnp.asarray(root_variables, dtype=solver_dtype)
        boundary = context.solve_boundary(
            *boundary_arguments,
            root_variables,
        )
        result = _boundary_result(parameters, boundary)
    except Exception as exception:
        return _empty_result(parameters, error=str(exception)), None

    if not result.boundary_converged:
        result.error = "Magma-gas boundary did not converge."
        return result, None

    next_init = magma_gas.MagmaGasInit(
        root_variables=boundary.root_variables,
    )
    try:
        (
            pressure_deep_bar,
            _,
            _,
            pressure_boundaries_bar,
        ) = pressure_layer_logspace_from_boundaries(
            np.log10(PRESSURE_BASE_BAR),
            np.log10(parameters.pressure_melt_bar),
            context.deep_nlayer,
            numpy=True,
        )
        temperature_deep_k = np.interp(
            np.log10(pressure_deep_bar),
            [
                np.log10(PRESSURE_BASE_BAR),
                np.log10(parameters.pressure_melt_bar),
            ],
            [
                parameters.temperature_base_k,
                parameters.temperature_melt_k,
            ],
        )
        pressure_deep_bar = jnp.asarray(pressure_deep_bar)
        temperature_deep_k = jnp.asarray(temperature_deep_k)
        (
            deep_diagnostics,
            base_equilibrium,
            base_diagnostics,
            density_deep_kg_m3,
            radius_boundaries_cm,
        ) = context.solve_structure(
            temperature_deep_k,
            pressure_deep_bar,
            jnp.asarray(pressure_boundaries_bar),
            jnp.asarray(parameters.temperature_base_k, dtype=solver_dtype),
            boundary.element_abundances,
            boundary.gas.equilibrium.ln_n,
            boundary.gas.equilibrium.ntot,
        )
        result.profile_converged = bool(
            np.all(np.asarray(deep_diagnostics["converged"]))
        )
        result.profile_max_iterations = int(
            jnp.max(deep_diagnostics["n_iter"])
        )
        if not result.profile_converged:
            result.error = "Deep gas-equilibrium profile did not converge."
            return result, next_init

        result.base_converged = bool(base_diagnostics["converged"])
        result.base_iterations = int(base_diagnostics["n_iter"])
        if not result.base_converged:
            result.error = "Exact 10-bar gas equilibrium did not converge."
            return result, next_init
        result.base_mole_fractions = _validated_mole_fractions(
            "10-bar gas mole fractions",
            base_equilibrium.x,
        )

        density_values = np.asarray(density_deep_kg_m3)
        if not np.all(np.isfinite(density_values)):
            raise RuntimeError("Density contains non-finite values.")
        if not np.all(density_values > 0.0):
            raise RuntimeError("Density must be positive.")

        radius_values = np.asarray(radius_boundaries_cm)
        if radius_values.shape != (context.deep_nlayer + 1,):
            raise RuntimeError("Hydrostatic radius has an unexpected shape.")
        if np.any(~np.isfinite(radius_values)) or np.any(radius_values <= 0.0):
            raise RuntimeError("Hydrostatic radius must be finite and positive.")
        if np.any(np.diff(radius_values) >= 0.0):
            raise RuntimeError("Hydrostatic radius must decrease with depth.")
        radius_base_rearth = float(radius_values[0] / RE)
        if not np.isfinite(radius_base_rearth):
            raise RuntimeError("10-bar radius is not finite.")
        if radius_base_rearth <= context.rocky_radius_rearth:
            raise RuntimeError("10-bar radius must exceed the rocky radius.")
        result.radius_base_rearth = radius_base_rearth
        return result, next_init
    except Exception as exception:
        result.error = str(exception)
        return result, next_init


def _parameter_key(parameters):
    return tuple(asdict(parameters).values())


def _baseline_index(sweep):
    baseline_value = getattr(BASELINE, sweep.field)
    values = np.asarray(sweep.values)
    index = int(np.argmin(np.abs(values - baseline_value)))
    if not np.isclose(values[index], baseline_value):
        raise RuntimeError(f"{sweep.name} does not contain its baseline.")
    return index


def run_sweeps(context):
    """Run all panels from the common baseline in both directions."""

    cache = {}

    def cached_solve(parameters, init):
        key = _parameter_key(parameters)
        if key not in cache:
            print(
                "Solving "
                f"P={parameters.pressure_melt_bar:.6g} bar, "
                f"Tmelt={parameters.temperature_melt_k:.6g} K, "
                f"fO2={parameters.oxygen_fugacity_bar:.6g} bar, "
                f"CO={parameters.co_melt_mole_ratio:.6g}, "
                f"N={parameters.n_melt_mole_ratio:.6g}, "
                f"Tb={parameters.temperature_base_k:.6g} K"
            )
            cache[key] = solve_point(context, parameters, init=init)
        return cache[key]

    baseline_result, baseline_init = cached_solve(BASELINE, None)
    if baseline_init is None:
        return [], cache, baseline_result

    runs = []
    for sweep in SWEEPS:
        baseline_index = _baseline_index(sweep)
        panel_results = [None] * len(sweep.values)
        panel_results[baseline_index] = SweepPoint(
            sweep=sweep,
            requested_value=sweep.values[baseline_index],
            result=baseline_result,
        )
        for indices in (
            range(baseline_index - 1, -1, -1),
            range(baseline_index + 1, len(sweep.values)),
        ):
            continuation_init = baseline_init
            for index in indices:
                value = sweep.values[index]
                parameters = replace(BASELINE, **{sweep.field: value})
                result, candidate_init = cached_solve(
                    parameters,
                    continuation_init,
                )
                panel_results[index] = SweepPoint(
                    sweep=sweep,
                    requested_value=value,
                    result=result,
                )
                if candidate_init is not None:
                    continuation_init = candidate_init
        runs.append(panel_results)
    return runs, cache, baseline_result


def _result_row(sweep_point, context):
    result = sweep_point.result
    row = {
        "sweep": sweep_point.sweep.name,
        "sweep_value": sweep_point.requested_value,
        **asdict(result.parameters),
        "rocky_radius_rearth": context.rocky_radius_rearth,
        "radius_base_rearth": result.radius_base_rearth,
        "boundary_converged": result.boundary_converged,
        "boundary_outer_converged": result.boundary_outer_converged,
        "boundary_inner_converged": result.boundary_inner_converged,
        "boundary_iterations": result.boundary_iterations,
        "boundary_residual_norm": result.boundary_residual_norm,
        "profile_converged": result.profile_converged,
        "profile_max_iterations": result.profile_max_iterations,
        "base_converged": result.base_converged,
        "base_iterations": result.base_iterations,
        "error": result.error,
    }
    for index, value in enumerate(result.boundary_root):
        row[f"boundary_root_{index}"] = value
    for index, value in enumerate(result.boundary_residual):
        row[f"boundary_residual_{index}"] = value
    for index, species in enumerate(MELTYQ_SPECIES):
        row[f"x_melt_{species}"] = result.melt_mole_fractions[index]
        row[f"x_10bar_{species}"] = result.base_mole_fractions[index]
    return row


def write_results_csv(path, runs, context, baseline_result=None):
    """Write plot values and diagnostics in a machine-readable table."""

    rows = [
        _result_row(point, context)
        for panel_results in runs
        for point in panel_results
    ]
    if not rows and baseline_result is not None:
        baseline_sweep = SweepSpec(
            name="baseline",
            title="baseline",
            field="pressure_melt_bar",
            values=(BASELINE.pressure_melt_bar,),
            xlabel="pressure_melt_bar",
            plot_log10=False,
        )
        rows = [
            _result_row(
                SweepPoint(
                    sweep=baseline_sweep,
                    requested_value=BASELINE.pressure_melt_bar,
                    result=baseline_result,
                ),
                context,
            )
        ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path, context):
    """Record the fixed conditions and inferred published grids."""

    payload = {
        "paper": PAPER_DOI,
        "baseline": asdict(BASELINE),
        "planet_mass_earth": context.planet_mass_earth,
        "rocky_core_mass_fraction": context.rocky_core_mass_fraction,
        "rocky_radius_rearth": context.rocky_radius_rearth,
        "pressure_base_bar": PRESSURE_BASE_BAR,
        "deep_nlayer": context.deep_nlayer,
        "max_outer_iterations": context.magma_options.max_iter,
        "melt_input_basis": {
            "co_melt_mole_ratio": (
                "ExoGibbs elemental-C dilute mole ratio"
            ),
            "n_melt_mole_ratio": (
                "ExoGibbs atomic-N dilute mole ratio"
            ),
        },
        "thermochemistry_probe": {
            "provider": "exogibbs.presets.fastchem4.chemsetup",
            "selected_species": list(MELTYQ_SPECIES),
            "temperatures_k": [1500.0, 3000.0],
            "max_abs_hvector_change": (
                context.thermochemistry_probe_change
            ),
            "clipped": context.thermochemistry_clipped,
        },
        "sweeps": {
            sweep.name: list(sweep.values) for sweep in SWEEPS
        },
        "sampling_note": (
            "Sweep sampling was inferred from the published vector figure; "
            "the paper does not specify the numerical grid."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plot_results(path, runs, *, show=False):
    """Plot the six panels with the line conventions of Figure 8."""

    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    species_index = {
        species: index for index, species in enumerate(MELTYQ_SPECIES)
    }
    for axis, panel_results in zip(axes.flat, runs):
        sweep = panel_results[0].sweep
        raw_x = np.asarray(
            [point.requested_value for point in panel_results]
        )
        x = np.log10(raw_x) if sweep.plot_log10 else raw_x
        for species in PLOTTED_SPECIES:
            index = species_index[species]
            melt = np.asarray(
                [
                    point.result.melt_mole_fractions[index]
                    for point in panel_results
                ]
            )
            base = np.asarray(
                [
                    point.result.base_mole_fractions[index]
                    for point in panel_results
                ]
            )
            axis.plot(
                x,
                np.log10(np.clip(melt, 1.0e-10, 1.0)),
                color=SPECIES_COLORS[species],
                label=species,
            )
            axis.plot(
                x,
                np.log10(np.clip(base, 1.0e-10, 1.0)),
                color=SPECIES_COLORS[species],
                linestyle=":",
            )
        axis.set(
            title=sweep.title,
            xlabel=sweep.xlabel,
            ylabel="Log(Mole fraction)",
            ylim=(-10.0, 0.0),
        )
        radius_axis = axis.twinx()
        radius_axis.plot(
            x,
            [point.result.radius_base_rearth for point in panel_results],
            color="0.65",
            linewidth=2.0,
        )
        radius_axis.set(
            ylabel=r"Radius at 10 bar [$R_\oplus$]",
            ylim=(1.4, 2.4),
        )
        axis.legend(loc="lower left", ncol=2, fontsize=7)
    figure.suptitle("MELTYQ Figure 8 cross-package comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def parse_args():
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/meltyq_figure8"),
    )
    parser.add_argument("--deep-layers", type=int, default=DEEP_NLAYER)
    parser.add_argument("--max-outer-iterations", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    """Run the published Figure 8 sweep definitions."""

    args = parse_args()
    if args.deep_layers <= 0:
        raise ValueError("--deep-layers must be positive.")
    if args.max_outer_iterations <= 0:
        raise ValueError("--max-outer-iterations must be positive.")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    context = build_context(
        max_outer_iterations=args.max_outer_iterations,
        deep_nlayer=args.deep_layers,
    )
    write_metadata(args.output_directory / "metadata.json", context)
    if context.thermochemistry_clipped:
        raise RuntimeError(
            "The selected thermochemistry is unchanged between 1500 and "
            "3000 K. Figure 8 requires temperature-dependent data over this "
            "range."
        )
    runs, cache, baseline_result = run_sweeps(context)
    csv_path = args.output_directory / "results.csv"
    write_results_csv(
        csv_path,
        runs,
        context,
        baseline_result=baseline_result,
    )
    if not runs:
        raise RuntimeError(
            "Figure 8 baseline did not converge; diagnostics were written to "
            f"{csv_path}."
        )
    figure_path = args.output_directory / "figure8.png"
    plot_results(figure_path, runs, show=args.show)
    failures = [
        point
        for panel_results in runs
        for point in panel_results
        if not (
            point.result.boundary_converged
            and point.result.profile_converged
            and point.result.base_converged
            and np.isfinite(point.result.radius_base_rearth)
        )
    ]
    print(
        f"Completed {len(cache)} unique points; "
        f"{len(failures)} plotted entries are incomplete."
    )
    print(f"Results: {csv_path}")
    print(f"Figure:  {figure_path}")
    if failures and not args.allow_incomplete:
        raise RuntimeError(
            "Some Figure 8 points did not converge; rerun with "
            "--allow-incomplete to accept plot gaps."
        )


if __name__ == "__main__":
    main()
