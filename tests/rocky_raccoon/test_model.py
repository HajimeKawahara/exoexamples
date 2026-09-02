from types import SimpleNamespace

import numpy as np
import pytest

from examples.rocky_raccoon import model
from examples.rocky_raccoon.chemistry import RainoutLayerTransition
from examples.rocky_raccoon.model import (
    RaccoonLikeConfig,
    build_context,
    outer_rcb_radius_m,
    radius_at_pressure,
    solver_gas_species,
)
from examples.rocky_raccoon.paper import PAPER_GAS_SPECIES
from examples.rocky_raccoon.structure import (
    ChemistryState,
    LayerState,
    StructureInputs,
    StructureProfile,
)


@pytest.fixture(scope="module")
def oxygen_poor_context():
    return build_context(RaccoonLikeConfig(preset="oxygen_poor"))


def test_default_context_is_an_explicit_raccoon_like_network(
    oxygen_poor_context,
):
    context = oxygen_poor_context
    setup = context.chemistry_setup

    assert context.case.identifier == "figure2_mg_si_o_1_1_3"
    assert setup.elements == ("H", "Mg", "Si", "O", "C", "e-")
    assert len(setup.gas_species) == 70
    assert setup.gas_species == PAPER_GAS_SPECIES
    assert np.linalg.matrix_rank(np.asarray(setup.formula_matrix)) == len(
        setup.elements
    )
    assert np.count_nonzero(
        np.asarray(setup.formula_matrix)[setup.elements.index("e-")]
    ) > 0
    assert len(setup.condensate_species) == 14
    assert "SiO(s)" not in setup.condensate_species

    values = dict(zip(setup.elements, context.element_inventory))
    assert values["Mg"] / values["Si"] == pytest.approx(1.0)
    assert values["O"] / values["Si"] == pytest.approx(3.0)
    assert values["e-"] == 0.0
    assert context.composition.oxygen_to_silicon == pytest.approx(3.0)


def test_explicit_composition_overrides_the_preset_ratio():
    from examples.rocky_raccoon.physics import RaccoonLikeComposition

    context = build_context(
        RaccoonLikeConfig(
            preset="oxygen_rich",
            composition=RaccoonLikeComposition(
                magnesium_to_silicon=2.0,
                oxygen_to_silicon=5.0,
            ),
        )
    )
    values = dict(
        zip(context.chemistry_setup.elements, context.element_inventory)
    )

    assert values["Mg"] / values["Si"] == pytest.approx(2.0)
    assert values["O"] / values["Si"] == pytest.approx(5.0)


def test_context_rejects_nonpositive_max_steps_before_provider_loading():
    with pytest.raises(ValueError, match="max_steps must be a positive integer"):
        build_context(RaccoonLikeConfig(max_steps=0))


def test_sio_preset_changes_only_the_condensate_catalog(oxygen_poor_context):
    sio_context = build_context(RaccoonLikeConfig(preset="oxygen_poor_sio"))

    assert sio_context.chemistry_setup.gas_species == (
        oxygen_poor_context.chemistry_setup.gas_species
    )
    assert set(sio_context.chemistry_setup.condensate_species) - set(
        oxygen_poor_context.chemistry_setup.condensate_species
    ) == {"SiO(s)"}


def test_solver_gas_species_filters_the_three_element_case(
    oxygen_poor_context,
):
    from examples.rocky_raccoon.paper import FIGURE_1_CASES

    species = solver_gas_species(
        oxygen_poor_context.chemistry_setup,
        FIGURE_1_CASES[0],
    )

    assert species
    assert not {"H1", "Si1", "O1", "e1-"} & set(species)
    assert not any("Mg" in name or "C" in name for name in species)


def test_radius_and_outer_rcb_diagnostics_use_accepted_layers():
    inventory = np.asarray([1.0])
    chemistry = ChemistryState(2.0e-3, inventory)
    inputs = StructureInputs(
        pressure_base_bar=100.0,
        pressure_top_bar=0.1,
        temperature_base_k=1000.0,
        radius_base_m=1.0,
        planet_mass_kg=1.0,
        luminosity_w=1.0,
        element_inventory=inventory,
    )
    layers = tuple(
        LayerState(
            pressure_bar=pressure,
            temperature_k=1000.0,
            radius_m=radius,
            mass_density_kg_m3=1.0,
            chemistry=chemistry,
            transport=transport,
        )
        for pressure, radius, transport in (
            (100.0, 1.0, "base"),
            (10.0, 2.0, "convective"),
            (1.0, 3.0, "convective"),
            (0.1, 4.0, "nonconvective"),
        )
    )
    profile = StructureProfile(inputs, layers, ())

    assert radius_at_pressure(profile, 10.0) == pytest.approx(2.0)
    assert radius_at_pressure(profile, np.sqrt(10.0)) == pytest.approx(2.5)
    assert outer_rcb_radius_m(profile) == pytest.approx(4.0)


def test_equilibrium_callback_preserves_the_previous_transition(
    monkeypatch,
    oxygen_poor_context,
):
    context = oxygen_poor_context
    inventory = np.asarray(context.element_inventory).copy()
    previous = RainoutLayerTransition(
        temperature_k=1200.0,
        pressure_bar=100.0,
        inventory_in=inventory.copy(),
        equilibrium=SimpleNamespace(),
        inventory_out=inventory.copy(),
    )
    observed = []

    def fake_solve_rainout_layer(
        setup,
        temperature_k,
        pressure_bar,
        incoming_inventory,
        **kwargs,
    ):
        observed.append(kwargs["previous_transition"])
        gas_x = np.full(
            len(setup.gas_species),
            1.0 / len(setup.gas_species),
        )
        return RainoutLayerTransition(
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            inventory_in=np.asarray(incoming_inventory).copy(),
            equilibrium=SimpleNamespace(converged=True, gas_x=gas_x),
            inventory_out=np.asarray(incoming_inventory).copy(),
        )

    monkeypatch.setattr(model, "solve_rainout_layer", fake_solve_rainout_layer)

    context.callbacks.equilibrium(
        1100.0,
        90.0,
        inventory,
        ChemistryState(2.0e-3, inventory.copy(), payload=previous),
    )

    assert observed == [previous]
