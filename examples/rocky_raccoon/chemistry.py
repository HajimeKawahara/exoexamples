"""Rocky Raccoon adapters for ExoGibbs condensate chemistry."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from exogibbs.api.chemistry import ChemicalSetup
from exogibbs.api.condensate import (
    CondensateChemicalSetup,
    CondensateEquilibriumInit,
    CondensateEquilibriumOptions,
    CondensateEquilibriumPoint,
    CondensateEquilibriumResult,
    build_condensate_chemical_setup,
    regauge_gas_only_warm_start,
    solve_profile,
)
from exogibbs.presets.fastchem4_cond import condensate_chemical_setup

from .paper import (
    FASTCHEM4_CONDENSATE_CATALOG,
    FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG,
)

ROCKY_RACCOON_PHYSICAL_ELEMENTS = ("H", "Mg", "Si", "O", "C")
CHARGE_ELEMENT = "e-"
PAPER_EXTRAPOLATED = "paper_extrapolated"
STRICT_VALIDITY = "strict_validity"
ORIGINAL_VALIDITY_METADATA_KEY = (
    "rocky_raccoon_original_temperature_validity_upper"
)
ValidityMode = Literal["paper_extrapolated", "strict_validity"]


@dataclass(frozen=True)
class RainoutLayerTransition:
    """An accepted one-layer equilibrium and its proposed rainout inventory."""

    temperature_k: float
    pressure_bar: float
    inventory_in: jax.Array
    equilibrium: CondensateEquilibriumResult
    inventory_out: jax.Array


def load_ion_inclusive_fastchem4_setup() -> CondensateChemicalSetup:
    """Load the packaged FastChem4 gas-condensate catalog including ions."""

    return condensate_chemical_setup(
        gas_path=FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG,
        condensate_path=FASTCHEM4_CONDENSATE_CATALOG,
        silent=True,
    )


def _ordered_indices(
    requested_names: Sequence[str],
    available_names: Sequence[str],
    label: str,
) -> tuple[int, ...]:
    requested = tuple(requested_names)
    if not requested:
        raise ValueError(f"{label} must not be empty.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"{label} must not contain duplicate names.")
    available_index = {name: index for index, name in enumerate(available_names)}
    missing = tuple(name for name in requested if name not in available_index)
    if missing:
        raise ValueError(f"Unknown {label}: {missing!r}.")
    return tuple(available_index[name] for name in requested)


def _subset_chemical_setup(
    source: ChemicalSetup,
    *,
    element_names: Sequence[str],
    species_names: Sequence[str],
    species_label: str,
) -> ChemicalSetup:
    if source.elements is None or source.species is None:
        raise ValueError("The source ChemicalSetup must name elements and species.")

    source_elements = tuple(source.elements)
    source_species = tuple(source.species)
    selected_elements = tuple(element_names)
    selected_species = tuple(species_names)
    element_indices = _ordered_indices(
        selected_elements,
        source_elements,
        "elements",
    )
    species_indices = _ordered_indices(
        selected_species,
        source_species,
        species_label,
    )

    formula_matrix = np.asarray(source.formula_matrix)
    excluded_rows = tuple(
        index
        for index, name in enumerate(source_elements)
        if name not in set(selected_elements)
    )
    if excluded_rows:
        excluded_stoichiometry = formula_matrix[
            np.ix_(excluded_rows, species_indices)
        ]
        incompatible = np.any(excluded_stoichiometry != 0, axis=0)
        if np.any(incompatible):
            names = tuple(
                selected_species[index] for index in np.flatnonzero(incompatible)
            )
            raise ValueError(
                f"Selected {species_label} use excluded elements: {names!r}."
            )

    element_index_array = jnp.asarray(element_indices, dtype=jnp.int32)
    species_index_array = jnp.asarray(species_indices, dtype=jnp.int32)
    reduced_formula_matrix = jnp.take(
        jnp.take(source.formula_matrix, element_index_array, axis=0),
        species_index_array,
        axis=1,
    )

    def reduced_hvector(temperature):
        return jnp.take(
            source.hvector_func(temperature),
            species_index_array,
            axis=-1,
        )

    element_vector_reference = source.element_vector_reference
    if element_vector_reference is not None:
        element_vector_reference = jnp.take(
            element_vector_reference,
            element_index_array,
        )

    validity_upper = source.temperature_validity_upper
    if validity_upper is not None:
        validity_upper = tuple(validity_upper[index] for index in species_indices)

    metadata = dict(source.metadata or {})
    if validity_upper is not None and "temperature_validity_upper" in metadata:
        metadata["temperature_validity_upper"] = validity_upper

    return ChemicalSetup(
        formula_matrix=reduced_formula_matrix,
        hvector_func=reduced_hvector,
        elements=selected_elements,
        species=selected_species,
        element_vector_reference=element_vector_reference,
        metadata=metadata,
        temperature_validity_upper=validity_upper,
    )


def clone_condensate_validity(
    setup: CondensateChemicalSetup,
    mode: ValidityMode,
) -> CondensateChemicalSetup:
    """Clone a setup with paper-extrapolated or strict condensate validity."""

    if mode not in {PAPER_EXTRAPOLATED, STRICT_VALIDITY}:
        raise ValueError(
            "mode must be 'paper_extrapolated' or 'strict_validity'."
        )

    condensates = setup.condensate_setup
    metadata = dict(condensates.metadata or {})
    original_upper = condensates.temperature_validity_upper
    if original_upper is None:
        original_upper = metadata.get(ORIGINAL_VALIDITY_METADATA_KEY)
    if original_upper is None:
        original_upper = metadata.get("temperature_validity_upper")
    if original_upper is not None:
        original_upper = tuple(float(value) for value in original_upper)

    if mode == STRICT_VALIDITY:
        if original_upper is None:
            raise ValueError("Strict validity requires one upper bound per condensate.")
        active_upper = original_upper
        metadata["temperature_validity_upper"] = original_upper
    else:
        active_upper = None
        metadata.pop("temperature_validity_upper", None)

    if original_upper is not None:
        metadata[ORIGINAL_VALIDITY_METADATA_KEY] = original_upper
    metadata["rocky_raccoon_validity_mode"] = mode
    metadata["rocky_raccoon_validity_scope"] = "condensates_only"
    cloned_condensates = replace(
        condensates,
        metadata=metadata,
        temperature_validity_upper=active_upper,
    )
    return build_condensate_chemical_setup(
        gas_setup=setup.gas_setup,
        condensate_setup=cloned_condensates,
    )


def subset_condensate_chemical_setup(
    source: CondensateChemicalSetup,
    *,
    gas_species_names: Sequence[str],
    condensate_species_names: Sequence[str],
    physical_element_names: Sequence[str] = ROCKY_RACCOON_PHYSICAL_ELEMENTS,
    validity_mode: ValidityMode = PAPER_EXTRAPOLATED,
) -> CondensateChemicalSetup:
    """Build an exact, explicitly named Rocky Raccoon chemistry network."""

    physical_elements = tuple(physical_element_names)
    if CHARGE_ELEMENT in physical_elements:
        raise ValueError("physical_element_names must not include the charge row.")
    element_names = physical_elements + (CHARGE_ELEMENT,)
    gas_setup = _subset_chemical_setup(
        source.gas_setup,
        element_names=element_names,
        species_names=gas_species_names,
        species_label="gas species",
    )
    condensate_setup = _subset_chemical_setup(
        source.condensate_setup,
        element_names=element_names,
        species_names=condensate_species_names,
        species_label="condensate species",
    )
    subset = build_condensate_chemical_setup(
        gas_setup=gas_setup,
        condensate_setup=condensate_setup,
    )
    return clone_condensate_validity(subset, validity_mode)


def build_rocky_raccoon_chemistry(
    *,
    gas_species_names: Sequence[str],
    condensate_species_names: Sequence[str],
    physical_element_names: Sequence[str] = ROCKY_RACCOON_PHYSICAL_ELEMENTS,
    validity_mode: ValidityMode = PAPER_EXTRAPOLATED,
) -> CondensateChemicalSetup:
    """Load ion-inclusive FastChem4 data and select the requested exact network."""

    return subset_condensate_chemical_setup(
        load_ion_inclusive_fastchem4_setup(),
        gas_species_names=gas_species_names,
        condensate_species_names=condensate_species_names,
        physical_element_names=physical_element_names,
        validity_mode=validity_mode,
    )


def _gas_only_warm_start(
    setup: CondensateChemicalSetup,
    transition: RainoutLayerTransition,
    incoming_inventory: jax.Array,
) -> CondensateEquilibriumInit:
    """Put an accepted gas composition in the current caller amount gauge."""

    previous_inventory_out = np.asarray(
        jax.device_get(transition.inventory_out),
    )
    target_inventory = np.asarray(jax.device_get(incoming_inventory))
    if (
        previous_inventory_out.shape != target_inventory.shape
        or not np.array_equal(previous_inventory_out, target_inventory)
    ):
        raise ValueError(
            "incoming_inventory must equal previous_transition.inventory_out."
        )

    initial = regauge_gas_only_warm_start(
        setup,
        transition.equilibrium.gas_ln_n,
        incoming_inventory,
    )
    return replace(
        initial,
        inventory_bridge_origin=CondensateEquilibriumPoint(
            temperature=float(transition.temperature_k),
            pressure=float(transition.pressure_bar),
            element_inventory=jnp.asarray(transition.inventory_in).copy(),
        ),
    )


def solve_rainout_layer(
    setup: CondensateChemicalSetup,
    temperature_k: float,
    pressure_bar: float,
    incoming_inventory,
    *,
    pref_bar: float = 1.0,
    options: CondensateEquilibriumOptions | None = None,
    previous_transition: RainoutLayerTransition | None = None,
    return_diagnostics: bool = False,
) -> RainoutLayerTransition:
    """Evaluate one transactional rainout candidate without mutating its input.

    A previous accepted transition supplies a gas-state hint and its source
    problem as numerical provenance.  ExoGibbs still owns fallback and
    condensate-support discovery; no condensate amount, support, or rainout
    result is propagated between structure branches.
    """

    temperature = float(temperature_k)
    pressure = float(pressure_bar)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be finite and positive.")
    if not math.isfinite(pressure) or pressure <= 0.0:
        raise ValueError("pressure_bar must be finite and positive.")

    dtype = jnp.result_type(
        setup.formula_matrix,
        jnp.asarray(incoming_inventory),
        jnp.float32,
    )
    inventory_in = jnp.asarray(incoming_inventory, dtype=dtype).copy()
    expected_shape = (len(setup.elements),)
    if inventory_in.shape != expected_shape:
        raise ValueError(
            "incoming_inventory must have one value per setup element: "
            f"expected {expected_shape}, got {inventory_in.shape}."
        )

    active_options = options or CondensateEquilibriumOptions(
        rainout=True,
        profile_method="scan_hot_from_bottom",
    )
    if not active_options.rainout:
        raise ValueError("A rainout layer requires options.rainout=True.")

    warm_start = None
    if previous_transition is not None:
        warm_start = _gas_only_warm_start(
            setup,
            previous_transition,
            inventory_in,
        )

    profile = solve_profile(
        setup,
        T=jnp.asarray([temperature], dtype=dtype),
        P=jnp.asarray([pressure], dtype=dtype),
        b=inventory_in.copy(),
        Pref=pref_bar,
        init=(warm_start,),
        options=active_options,
        return_diagnostics=return_diagnostics,
    )
    inventory_out_profile = profile.rainout_element_inventory_out
    if (
        not profile.rainout
        or len(profile.layers) != 1
        or inventory_out_profile is None
    ):
        raise RuntimeError("ExoGibbs did not return one rainout profile layer.")
    inventory_out = jnp.asarray(inventory_out_profile, dtype=dtype)
    if inventory_out.shape != (1, len(setup.elements)):
        raise RuntimeError(
            "ExoGibbs returned an unexpected rainout inventory shape: "
            f"{inventory_out.shape}."
        )

    return RainoutLayerTransition(
        temperature_k=temperature,
        pressure_bar=pressure,
        inventory_in=inventory_in,
        equilibrium=profile.layers[0],
        inventory_out=inventory_out[0].copy(),
    )


__all__ = (
    "CHARGE_ELEMENT",
    "ORIGINAL_VALIDITY_METADATA_KEY",
    "PAPER_EXTRAPOLATED",
    "ROCKY_RACCOON_PHYSICAL_ELEMENTS",
    "STRICT_VALIDITY",
    "RainoutLayerTransition",
    "ValidityMode",
    "build_rocky_raccoon_chemistry",
    "clone_condensate_validity",
    "load_ion_inclusive_fastchem4_setup",
    "solve_rainout_layer",
    "subset_condensate_chemical_setup",
)
