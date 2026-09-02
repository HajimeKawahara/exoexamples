from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.constants import R

from examples.rocky_raccoon.physics import (
    BAR_TO_PA,
    ConstantTransport,
    EARTH_MASS_KG,
    EARTH_RADIUS_M,
    IdealGasMixture,
    RaccoonLikeComposition,
    elemental_inventory,
    rocky_core_radius_m,
)


def _h2_water_setup():
    return SimpleNamespace(
        elements=("H", "O", "e-"),
        gas_species=("H2", "H2O1"),
        formula_matrix=jnp.asarray(
            [
                [2.0, 2.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_raccoon_like_inventory_preserves_requested_ratios_and_zero_charge():
    composition = RaccoonLikeComposition(
        silicon_per_hydrogen=0.02,
        magnesium_to_silicon=1.5,
        oxygen_to_silicon=4.0,
        carbon_per_hydrogen=3.0e-4,
    )
    elements = ("H", "Mg", "Si", "O", "C", "e-")

    inventory = elemental_inventory(elements, composition)

    values = dict(zip(elements, inventory))
    assert np.sum(inventory) == pytest.approx(1.0)
    assert values["Si"] / values["H"] == pytest.approx(0.02)
    assert values["Mg"] / values["Si"] == pytest.approx(1.5)
    assert values["O"] / values["Si"] == pytest.approx(4.0)
    assert values["C"] / values["H"] == pytest.approx(3.0e-4)
    assert values["e-"] == 0.0


def test_ideal_gas_provider_uses_exoeos_state_and_species_gamma():
    mixture = IdealGasMixture.from_chemical_setup(
        _h2_water_setup(),
        gamma=lambda species: 4.0 / 3.0 if species == "H2O1" else 1.4,
    )
    fractions = np.asarray([0.75, 0.25])

    state = mixture.state(1000.0, 2.0, fractions)

    mean_mass = mixture.mean_molar_mass_kg_mol(fractions)
    assert float(state.mass_density) == pytest.approx(
        2.0 * BAR_TO_PA * mean_mass / (R * 1000.0)
    )
    expected_cp = 0.75 * R * 1.4 / 0.4 + 0.25 * R * 4.0
    assert float(state.molar_heat_capacity_cp) == pytest.approx(expected_cp)
    assert float(state.adiabatic_gradient) == pytest.approx(R / expected_cp)


def test_hydrogen_mass_fraction_uses_stoichiometry():
    mixture = IdealGasMixture.from_chemical_setup(
        _h2_water_setup(),
        gamma=lambda species: 1.4,
    )

    assert mixture.hydrogen_mass_fraction([1.0, 0.0]) == pytest.approx(1.0)
    water_hydrogen_fraction = mixture.hydrogen_mass_fraction([0.0, 1.0])
    assert water_hydrogen_fraction == pytest.approx(
        2.0 * 1.00794 / (2.0 * 1.00794 + 15.9994)
    )


def test_constant_transport_uses_harmonic_effective_opacity():
    transport = ConstantTransport(
        rosseland_opacity_m2_kg=0.02,
        thermal_conductivity_w_m_k=1000.0,
    )

    conductive = transport.conductive_opacity_m2_kg(2000.0, 10.0)
    effective = transport.effective_opacity_m2_kg(2000.0, 10.0)
    assert effective == pytest.approx(1.0 / (1.0 / 0.02 + 1.0 / conductive))
    assert 0.0 < effective < min(0.02, conductive)
    assert transport.nonconvective_log_temperature_gradient(
        pressure_bar=100.0,
        temperature_k=2000.0,
        mass_density_kg_m3=10.0,
        luminosity_w=1.0e17,
        planet_mass_kg=4.0 * EARTH_MASS_KG,
    ) > 0.0


def test_rocky_core_radius_uses_the_mass_to_one_quarter_law():
    assert rocky_core_radius_m(EARTH_MASS_KG) == pytest.approx(EARTH_RADIUS_M)
    assert rocky_core_radius_m(4.0 * EARTH_MASS_KG) == pytest.approx(
        np.sqrt(2.0) * EARTH_RADIUS_M
    )
