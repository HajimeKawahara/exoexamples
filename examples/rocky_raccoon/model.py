"""Coupled chemistry-structure model for Rocky Raccoon-like experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from exogibbs.api.condensate import CondensateEquilibriumOptions

from .chemistry import (
    PAPER_EXTRAPOLATED,
    STRICT_VALIDITY,
    RainoutLayerTransition,
    ValidityMode,
    load_ion_inclusive_fastchem4_setup,
    solve_rainout_layer,
    subset_condensate_chemical_setup,
)
from .paper import (
    PAPER_GAS_SPECIES,
    PAPER_FIGURE_CASES,
    PaperFigureCase,
    paper_gamma,
)
from .physics import (
    EARTH_MASS_KG,
    EARTH_RADIUS_M,
    ConstantTransport,
    IdealGasMixture,
    RaccoonLikeComposition,
    elemental_inventory,
    rocky_core_radius_m,
)
from .structure import (
    AcceptedLayerCallback,
    ChemistryState,
    LayerState,
    StructureCallbacks,
    StructureInputs,
    StructureProfile,
    integrate_structure,
)


PresetName = Literal["oxygen_poor", "oxygen_rich", "oxygen_poor_sio"]

PRESET_CASE_IDENTIFIERS = {
    "oxygen_poor": "figure2_mg_si_o_1_1_3",
    "oxygen_rich": "figure2_mg_si_o_1_1_4",
    "oxygen_poor_sio": "figure5_mg_si_o_1_1_3_sio_on",
}


@dataclass(frozen=True)
class RaccoonLikeConfig:
    """Inputs for one explicitly approximate raccoon-like column."""

    preset: PresetName = "oxygen_poor"
    validity_mode: ValidityMode = PAPER_EXTRAPOLATED
    core_mass_earth: float = 4.0
    temperature_base_k: float = 4000.0
    pressure_base_bar: float = 2.0e5
    pressure_top_bar: float = 1.0e-3
    luminosity_w: float = 1.0e16
    pressure_ratio: float = 0.99
    transit_pressure_bar: float = 2.0e-2
    composition: RaccoonLikeComposition | None = None
    transport: ConstantTransport = field(default_factory=ConstantTransport)
    max_steps: int = 10_000
    chemistry_diagnostics: bool = False


@dataclass(frozen=True)
class RaccoonLikeContext:
    """Static provider objects reused across fixed-boundary column runs."""

    config: RaccoonLikeConfig
    case: PaperFigureCase
    chemistry_setup: object
    ideal_gas: IdealGasMixture
    composition: RaccoonLikeComposition
    element_inventory: np.ndarray
    core_mass_kg: float
    radius_base_m: float
    callbacks: StructureCallbacks


@dataclass(frozen=True)
class ColumnMetrics:
    """Scalar diagnostics derived from one accepted structure profile."""

    envelope_mass_kg: float
    hydrogen_mass_kg: float
    hydrogen_to_core_mass_ratio: float
    transit_radius_m: float
    outer_rcb_radius_m: float | None
    top_temperature_k: float
    convective_layer_count: int
    nonconvective_layer_count: int

    @property
    def transit_radius_earth(self) -> float:
        return self.transit_radius_m / EARTH_RADIUS_M

    @property
    def outer_rcb_radius_earth(self) -> float | None:
        if self.outer_rcb_radius_m is None:
            return None
        return self.outer_rcb_radius_m / EARTH_RADIUS_M


@dataclass(frozen=True)
class RaccoonLikeResult:
    """One coupled column and its scalar diagnostics."""

    context: RaccoonLikeContext
    profile: StructureProfile
    metrics: ColumnMetrics


def _case_by_identifier(identifier: str) -> PaperFigureCase:
    matches = tuple(case for case in PAPER_FIGURE_CASES if case.identifier == identifier)
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate paper-policy case {identifier!r}.")
    return matches[0]


def preset_composition(preset: PresetName) -> RaccoonLikeComposition:
    """Return the default basal composition implied by one named preset."""

    if preset not in PRESET_CASE_IDENTIFIERS:
        raise ValueError(f"Unknown raccoon-like preset {preset!r}.")
    case = _case_by_identifier(PRESET_CASE_IDENTIFIERS[preset])
    prescribed_ratio = dict(case.prescribed_number_ratio)
    silicon_ratio = float(prescribed_ratio["Si"])
    defaults = RaccoonLikeComposition()
    return RaccoonLikeComposition(
        silicon_per_hydrogen=defaults.silicon_per_hydrogen,
        magnesium_to_silicon=(
            float(prescribed_ratio.get("Mg", 0.0)) / silicon_ratio
        ),
        oxygen_to_silicon=float(prescribed_ratio["O"]) / silicon_ratio,
        carbon_per_hydrogen=defaults.carbon_per_hydrogen,
    )


def _validate_config(config: RaccoonLikeConfig) -> None:
    if config.preset not in PRESET_CASE_IDENTIFIERS:
        raise ValueError(f"Unknown raccoon-like preset {config.preset!r}.")
    if config.validity_mode not in {PAPER_EXTRAPOLATED, STRICT_VALIDITY}:
        raise ValueError(f"Unknown validity mode {config.validity_mode!r}.")
    for name in (
        "core_mass_earth",
        "temperature_base_k",
        "pressure_base_bar",
        "pressure_top_bar",
        "luminosity_w",
        "transit_pressure_bar",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if config.pressure_top_bar >= config.pressure_base_bar:
        raise ValueError("pressure_top_bar must be below pressure_base_bar.")
    if not 0.0 < config.pressure_ratio < 1.0:
        raise ValueError("pressure_ratio must lie strictly between zero and one.")
    if (
        not isinstance(config.max_steps, (int, np.integer))
        or isinstance(config.max_steps, (bool, np.bool_))
        or config.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer.")
    if not (
        config.pressure_top_bar
        <= config.transit_pressure_bar
        <= config.pressure_base_bar
    ):
        raise ValueError("transit_pressure_bar must lie inside the column.")


def _compatible_reported_gas_species(
    source,
    physical_elements: tuple[str, ...],
) -> tuple[str, ...]:
    """Filter the explicit Appendix list without admitting catalog extras."""

    allowed_elements = set(physical_elements) | {"e-"}
    excluded_rows = tuple(
        index
        for index, name in enumerate(source.elements)
        if name not in allowed_elements
    )
    matrix = np.asarray(source.formula_matrix)
    selected = []
    for name in PAPER_GAS_SPECIES:
        column = source.gas_species.index(name)
        if not excluded_rows or not np.any(matrix[excluded_rows, column] != 0.0):
            selected.append(name)
    return tuple(selected)


def solver_gas_species(source, case: PaperFigureCase) -> tuple[str, ...]:
    """Return the compatible paper-reported gases without extra references."""

    return _compatible_reported_gas_species(source, case.physical_elements)


def build_context(config: RaccoonLikeConfig = RaccoonLikeConfig()) -> RaccoonLikeContext:
    """Build static ExoGibbs and ExoEOS providers for one run family."""

    _validate_config(config)
    case = _case_by_identifier(PRESET_CASE_IDENTIFIERS[config.preset])
    source = load_ion_inclusive_fastchem4_setup()
    chemistry_setup = subset_condensate_chemical_setup(
        source,
        gas_species_names=solver_gas_species(source, case),
        condensate_species_names=case.condensate_species,
        physical_element_names=case.physical_elements,
        validity_mode=config.validity_mode,
    )
    ideal_gas = IdealGasMixture.from_chemical_setup(
        chemistry_setup,
        paper_gamma,
    )
    effective_composition = config.composition or preset_composition(
        config.preset
    )
    inventory = elemental_inventory(
        chemistry_setup.elements,
        effective_composition,
    )
    core_mass_kg = config.core_mass_earth * EARTH_MASS_KG
    radius_base_m = rocky_core_radius_m(core_mass_kg)
    options = CondensateEquilibriumOptions(
        rainout=True,
        profile_method="scan_hot_from_bottom",
        return_diagnostics=config.chemistry_diagnostics,
    )

    def equilibrium(
        temperature_k: float,
        pressure_bar: float,
        incoming_inventory: np.ndarray,
        previous_chemistry: ChemistryState | None,
    ) -> ChemistryState:
        previous_transition = None
        if previous_chemistry is not None:
            previous_transition = _transition(previous_chemistry)
        try:
            transition = solve_rainout_layer(
                chemistry_setup,
                temperature_k,
                pressure_bar,
                incoming_inventory,
                options=options,
                previous_transition=previous_transition,
                return_diagnostics=config.chemistry_diagnostics,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "ExoGibbs rainout failed at "
                f"T={temperature_k:.17g} K, P={pressure_bar:.17g} bar: "
                f"{error}"
            ) from error
        if not transition.equilibrium.converged:
            raise RuntimeError(
                "ExoGibbs did not converge at "
                f"T={temperature_k:g} K, P={pressure_bar:g} bar."
            )
        mole_fractions = np.asarray(transition.equilibrium.gas_x, dtype=float)
        return ChemistryState(
            mean_molar_mass_kg_mol=ideal_gas.mean_molar_mass_kg_mol(
                mole_fractions
            ),
            remaining_inventory=np.asarray(transition.inventory_out, dtype=float),
            payload=transition,
        )

    def eos_state(
        pressure_bar: float,
        temperature_k: float,
        chemistry: ChemistryState,
    ):
        transition = _transition(chemistry)
        return ideal_gas.state(
            temperature_k,
            pressure_bar,
            transition.equilibrium.gas_x,
        )

    def mass_density(
        pressure_bar: float,
        temperature_k: float,
        chemistry: ChemistryState,
    ) -> float:
        return float(
            eos_state(pressure_bar, temperature_k, chemistry).mass_density
        )

    def convective_gradient(layer: LayerState, luminosity_w: float) -> float:
        del luminosity_w
        return float(
            eos_state(
                layer.pressure_bar,
                layer.temperature_k,
                layer.chemistry,
            ).adiabatic_gradient
        )

    def nonconvective_gradient(layer: LayerState, luminosity_w: float) -> float:
        return config.transport.nonconvective_log_temperature_gradient(
            pressure_bar=layer.pressure_bar,
            temperature_k=layer.temperature_k,
            mass_density_kg_m3=layer.mass_density_kg_m3,
            luminosity_w=luminosity_w,
            planet_mass_kg=core_mass_kg,
        )

    return RaccoonLikeContext(
        config=config,
        case=case,
        chemistry_setup=chemistry_setup,
        ideal_gas=ideal_gas,
        composition=effective_composition,
        element_inventory=inventory,
        core_mass_kg=core_mass_kg,
        radius_base_m=radius_base_m,
        callbacks=StructureCallbacks(
            equilibrium=equilibrium,
            convective_log_temperature_gradient=convective_gradient,
            nonconvective_log_temperature_gradient=nonconvective_gradient,
            mass_density=mass_density,
        ),
    )


def _transition(chemistry: ChemistryState) -> RainoutLayerTransition:
    if not isinstance(chemistry.payload, RainoutLayerTransition):
        raise TypeError("The real model requires a RainoutLayerTransition payload.")
    return chemistry.payload


def run_column(
    context: RaccoonLikeContext,
    *,
    pressure_base_bar: float | None = None,
    luminosity_w: float | None = None,
    accepted_layer_callback: AcceptedLayerCallback | None = None,
) -> RaccoonLikeResult:
    """Run one coupled fixed-pressure, fixed-luminosity column."""

    config = context.config
    structure_inputs = StructureInputs(
        pressure_base_bar=(
            config.pressure_base_bar
            if pressure_base_bar is None
            else float(pressure_base_bar)
        ),
        pressure_top_bar=config.pressure_top_bar,
        temperature_base_k=config.temperature_base_k,
        radius_base_m=context.radius_base_m,
        planet_mass_kg=context.core_mass_kg,
        luminosity_w=(
            config.luminosity_w if luminosity_w is None else float(luminosity_w)
        ),
        element_inventory=context.element_inventory,
        pressure_ratio=config.pressure_ratio,
        max_steps=config.max_steps,
    )
    profile = integrate_structure(
        structure_inputs,
        context.callbacks,
        accepted_layer_callback=accepted_layer_callback,
    )
    return RaccoonLikeResult(
        context=context,
        profile=profile,
        metrics=_column_metrics(context, profile),
    )


def _column_metrics(
    context: RaccoonLikeContext,
    profile: StructureProfile,
) -> ColumnMetrics:
    shell_mass = profile.shell_mass_kg
    hydrogen_fraction_by_layer = np.asarray(
        [
            context.ideal_gas.hydrogen_mass_fraction(
                _transition(layer.chemistry).equilibrium.gas_x
            )
            for layer in profile.layers[:-1]
        ]
    )
    envelope_mass = float(np.sum(shell_mass))
    hydrogen_mass = float(np.sum(shell_mass * hydrogen_fraction_by_layer))
    transit_radius = radius_at_pressure(
        profile,
        context.config.transit_pressure_bar,
    )
    outer_rcb = outer_rcb_radius_m(profile)
    transport = profile.transport[1:]
    return ColumnMetrics(
        envelope_mass_kg=envelope_mass,
        hydrogen_mass_kg=hydrogen_mass,
        hydrogen_to_core_mass_ratio=hydrogen_mass / context.core_mass_kg,
        transit_radius_m=transit_radius,
        outer_rcb_radius_m=outer_rcb,
        top_temperature_k=float(profile.temperature_k[-1]),
        convective_layer_count=transport.count("convective"),
        nonconvective_layer_count=transport.count("nonconvective"),
    )


def radius_at_pressure(profile: StructureProfile, pressure_bar: float) -> float:
    """Log-pressure interpolate radius inside an accepted profile."""

    pressure = float(pressure_bar)
    if pressure < profile.pressure_bar[-1] or pressure > profile.pressure_bar[0]:
        raise ValueError("Requested pressure lies outside the structure profile.")
    return float(
        np.interp(
            np.log(pressure),
            np.log(profile.pressure_bar[::-1]),
            profile.radius_m[::-1],
        )
    )


def outer_rcb_radius_m(profile: StructureProfile) -> float | None:
    """Return the outermost accepted convective-to-nonconvective boundary."""

    transitions = [
        index
        for index in range(2, len(profile.layers))
        if profile.layers[index - 1].transport == "convective"
        and profile.layers[index].transport == "nonconvective"
    ]
    if not transitions:
        return None
    return float(profile.layers[transitions[-1]].radius_m)


__all__ = (
    "ColumnMetrics",
    "PRESET_CASE_IDENTIFIERS",
    "RaccoonLikeConfig",
    "RaccoonLikeContext",
    "RaccoonLikeResult",
    "build_context",
    "outer_rcb_radius_m",
    "preset_composition",
    "radius_at_pressure",
    "run_column",
    "solver_gas_species",
)
