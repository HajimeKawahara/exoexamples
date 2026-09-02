from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import examples.rocky_raccoon.chemistry as chemistry
from examples.rocky_raccoon.paper import (
    FASTCHEM4_CONDENSATE_CATALOG,
    FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG,
)
from exogibbs.api.condensate import CondensateEquilibriumOptions


SMALL_GAS_NETWORK = (
    "H1",
    "Mg1",
    "Si1",
    "O1",
    "C1",
    "e1-",
    "H2",
    "H2O1",
    "Mg1O1",
    "O1Si1",
    "C1O1",
    "H1+",
)
SMALL_CONDENSATE_NETWORK = (
    "Mg(OH)2(s)",
    "C(s)",
    "SiO(s)",
)


@pytest.fixture(scope="module")
def full_fastchem4_setup():
    return chemistry.load_ion_inclusive_fastchem4_setup()


@pytest.fixture(scope="module")
def strict_small_setup(full_fastchem4_setup):
    return chemistry.subset_condensate_chemical_setup(
        full_fastchem4_setup,
        gas_species_names=SMALL_GAS_NETWORK,
        condensate_species_names=SMALL_CONDENSATE_NETWORK,
        validity_mode=chemistry.STRICT_VALIDITY,
    )


def test_loader_uses_the_explicit_fastchem4_catalogs(full_fastchem4_setup):
    assert (
        full_fastchem4_setup.gas_setup.metadata["fastchem_logk_file"]
        == FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG
    )
    assert (
        full_fastchem4_setup.condensate_setup.metadata["fastchem_logk_file"]
        == FASTCHEM4_CONDENSATE_CATALOG
    )
    assert "e1-" in full_fastchem4_setup.gas_species
    assert "H1+" in full_fastchem4_setup.gas_species


def test_exact_subset_preserves_requested_order_and_sources(
    full_fastchem4_setup,
    strict_small_setup,
):
    expected_elements = chemistry.ROCKY_RACCOON_PHYSICAL_ELEMENTS + (
        chemistry.CHARGE_ELEMENT,
    )
    assert strict_small_setup.elements == expected_elements
    assert strict_small_setup.gas_species == SMALL_GAS_NETWORK
    assert strict_small_setup.condensate_species == SMALL_CONDENSATE_NETWORK
    assert strict_small_setup.formula_matrix.shape == (
        len(expected_elements),
        len(SMALL_GAS_NETWORK),
    )
    assert strict_small_setup.formula_matrix_cond.shape == (
        len(expected_elements),
        len(SMALL_CONDENSATE_NETWORK),
    )

    temperature = 1234.0
    gas_indices = [
        full_fastchem4_setup.gas_species.index(name)
        for name in SMALL_GAS_NETWORK
    ]
    condensate_indices = [
        full_fastchem4_setup.condensate_species.index(name)
        for name in SMALL_CONDENSATE_NETWORK
    ]
    element_indices = [
        full_fastchem4_setup.elements.index(name) for name in expected_elements
    ]
    np.testing.assert_array_equal(
        strict_small_setup.formula_matrix,
        np.asarray(full_fastchem4_setup.formula_matrix)[
            np.ix_(element_indices, gas_indices)
        ],
    )
    np.testing.assert_array_equal(
        strict_small_setup.formula_matrix_cond,
        np.asarray(full_fastchem4_setup.formula_matrix_cond)[
            np.ix_(element_indices, condensate_indices)
        ],
    )
    np.testing.assert_allclose(
        strict_small_setup.gas_setup.hvector_func(temperature),
        np.asarray(full_fastchem4_setup.gas_setup.hvector_func(temperature))[
            gas_indices
        ],
    )
    np.testing.assert_allclose(
        strict_small_setup.condensate_setup.hvector_func(temperature),
        np.asarray(
            full_fastchem4_setup.condensate_setup.hvector_func(temperature)
        )[condensate_indices],
    )


def test_exact_subset_rejects_missing_and_out_of_element_species(
    full_fastchem4_setup,
):
    with pytest.raises(ValueError, match="Unknown gas species"):
        chemistry.subset_condensate_chemical_setup(
            full_fastchem4_setup,
            gas_species_names=("not-a-fastchem-species",),
            condensate_species_names=("C(s)",),
        )

    with pytest.raises(ValueError, match="excluded elements"):
        chemistry.subset_condensate_chemical_setup(
            full_fastchem4_setup,
            gas_species_names=("N2",),
            condensate_species_names=("C(s)",),
        )


def test_validity_modes_are_clones_and_keep_original_bounds(strict_small_setup):
    original_upper = strict_small_setup.condensate_setup.temperature_validity_upper
    assert original_upper == (1000.0, 6000.0, 6000.0)

    paper_setup = chemistry.clone_condensate_validity(
        strict_small_setup,
        chemistry.PAPER_EXTRAPOLATED,
    )
    paper_metadata = paper_setup.condensate_setup.metadata
    assert paper_setup is not strict_small_setup
    assert paper_setup.condensate_setup.temperature_validity_upper is None
    assert "temperature_validity_upper" not in paper_metadata
    assert (
        paper_metadata[chemistry.ORIGINAL_VALIDITY_METADATA_KEY]
        == original_upper
    )
    assert paper_metadata["rocky_raccoon_validity_scope"] == "condensates_only"
    assert strict_small_setup.condensate_setup.temperature_validity_upper == (
        1000.0,
        6000.0,
        6000.0,
    )

    restored = chemistry.clone_condensate_validity(
        paper_setup,
        chemistry.STRICT_VALIDITY,
    )
    assert restored.condensate_setup.temperature_validity_upper == original_upper
    np.testing.assert_allclose(
        restored.condensate_setup.hvector_func(4000.0),
        paper_setup.condensate_setup.hvector_func(4000.0),
    )


def test_rainout_candidates_snapshot_the_same_incoming_inventory(
    monkeypatch,
    strict_small_setup,
):
    observed_inputs = []

    def fake_solve_profile(
        setup,
        T,
        P,
        b,
        *,
        Pref,
        init,
        options,
        return_diagnostics,
    ):
        del setup, P, Pref, init, return_diagnostics
        assert options.rainout
        observed_inputs.append(np.asarray(b).copy())
        offset = float(np.asarray(T)[0]) / 1000.0
        return SimpleNamespace(
            rainout=True,
            layers=(SimpleNamespace(converged=True),),
            rainout_element_inventory_out=jnp.asarray([b + offset]),
        )

    monkeypatch.setattr(chemistry, "solve_profile", fake_solve_profile)
    incoming = np.arange(len(strict_small_setup.elements), dtype=float) + 1.0
    expected = incoming.copy()
    options = CondensateEquilibriumOptions(
        rainout=True,
        profile_method="scan_hot_from_bottom",
    )

    convective = chemistry.solve_rainout_layer(
        strict_small_setup,
        1000.0,
        10.0,
        incoming,
        options=options,
    )
    nonconvective = chemistry.solve_rainout_layer(
        strict_small_setup,
        1200.0,
        10.0,
        incoming,
        options=options,
    )

    np.testing.assert_array_equal(incoming, expected)
    np.testing.assert_array_equal(observed_inputs, [expected, expected])
    np.testing.assert_array_equal(convective.inventory_in, expected)
    np.testing.assert_array_equal(nonconvective.inventory_in, expected)
    np.testing.assert_allclose(convective.inventory_out, expected + 1.0)
    np.testing.assert_allclose(nonconvective.inventory_out, expected + 1.2)

    incoming[0] = -999.0
    assert float(convective.inventory_in[0]) == expected[0]
    with pytest.raises(FrozenInstanceError):
        convective.pressure_bar = 20.0


def test_rainout_layer_rejects_nonrainout_options(strict_small_setup):
    with pytest.raises(ValueError, match="rainout=True"):
        chemistry.solve_rainout_layer(
            strict_small_setup,
            1000.0,
            10.0,
            np.ones(len(strict_small_setup.elements)),
            options=CondensateEquilibriumOptions(rainout=False),
        )


def test_previous_transition_builds_independent_gas_only_warm_starts(
    monkeypatch,
    strict_small_setup,
):
    observed_inits = []
    provider_calls = []
    provider = chemistry.regauge_gas_only_warm_start

    def tracked_regauge(setup, gas_ln_n, element_inventory):
        provider_calls.append((setup, gas_ln_n, element_inventory))
        return provider(setup, gas_ln_n, element_inventory)

    def fake_solve_profile(
        setup,
        T,
        P,
        b,
        *,
        Pref,
        init,
        options,
        return_diagnostics,
    ):
        del setup, T, P, Pref, options, return_diagnostics
        observed_inits.append(init[0])
        return SimpleNamespace(
            rainout=True,
            layers=(SimpleNamespace(converged=True),),
            rainout_element_inventory_out=jnp.asarray([b]),
        )

    monkeypatch.setattr(chemistry, "solve_profile", fake_solve_profile)
    monkeypatch.setattr(
        chemistry,
        "regauge_gas_only_warm_start",
        tracked_regauge,
    )
    gas_amounts = np.arange(
        1.0,
        len(strict_small_setup.gas_species) + 1.0,
    )
    inventory = np.asarray([0.7, 0.0, 0.1, 0.15, 0.05, 0.0])
    parent_inventory = np.asarray([0.65, 0.0, 0.12, 0.13, 0.1, 0.0])
    previous = chemistry.RainoutLayerTransition(
        temperature_k=900.0,
        pressure_bar=20.0,
        inventory_in=jnp.asarray(parent_inventory),
        equilibrium=SimpleNamespace(
            gas_ln_n=jnp.log(jnp.asarray(gas_amounts)),
        ),
        inventory_out=jnp.asarray(inventory),
    )

    for temperature in (1000.0, 1100.0):
        chemistry.solve_rainout_layer(
            strict_small_setup,
            temperature,
            10.0,
            inventory,
            previous_transition=previous,
        )

    first, second = observed_inits
    source_logs = np.log(gas_amounts)
    h_index = strict_small_setup.gas_species.index("H1")
    h2_index = strict_small_setup.gas_species.index("H2")

    assert len(provider_calls) == 2
    for setup, gas_ln_n, element_inventory in provider_calls:
        assert setup is strict_small_setup
        np.testing.assert_array_equal(gas_ln_n, previous.equilibrium.gas_ln_n)
        np.testing.assert_array_equal(element_inventory, inventory)
    assert first is not second
    assert second.gas_ln_n is not first.gas_ln_n
    for initial in (first, second):
        warm_logs = np.asarray(initial.gas_ln_n)
        assert warm_logs[h2_index] - warm_logs[h_index] == pytest.approx(
            source_logs[h2_index] - source_logs[h_index],
        )
        assert float(initial.gas_ntot) > 0.0
    assert first.condensate_amounts is None
    assert first.support_indices is None
    assert first.support_amounts is None
    assert first.inventory_bridge_origin is not None
    assert second.inventory_bridge_origin is not None
    assert first.inventory_bridge_origin is not second.inventory_bridge_origin
    assert (
        first.inventory_bridge_origin.element_inventory
        is not second.inventory_bridge_origin.element_inventory
    )
    assert first.inventory_bridge_origin.temperature == 900.0
    assert first.inventory_bridge_origin.pressure == 20.0
    np.testing.assert_array_equal(
        first.inventory_bridge_origin.element_inventory,
        np.asarray(previous.inventory_in),
    )

    nonadjacent_inventory = inventory.copy()
    nonadjacent_inventory[0] -= 0.01
    with pytest.raises(ValueError, match="previous_transition.inventory_out"):
        chemistry.solve_rainout_layer(
            strict_small_setup,
            1000.0,
            10.0,
            nonadjacent_inventory,
            previous_transition=previous,
        )


def test_warm_start_preserves_finite_subfloor_log_abundances(
    strict_small_setup,
):
    inventory = np.asarray([0.7, 0.0, 0.1, 0.15, 0.05, 0.0])
    dominant_index = strict_small_setup.gas_species.index("H1")
    trace_index = strict_small_setup.gas_species.index("H2")
    gas_log_amounts = np.full(
        len(strict_small_setup.gas_species),
        -np.inf,
    )
    gas_log_amounts[dominant_index] = -0.7
    gas_log_amounts[trace_index] = -721.0
    with jax.experimental.enable_x64():
        previous = chemistry.RainoutLayerTransition(
            temperature_k=900.0,
            pressure_bar=20.0,
            inventory_in=jnp.asarray(inventory),
            equilibrium=SimpleNamespace(
                gas_ln_n=jnp.asarray(gas_log_amounts),
            ),
            inventory_out=jnp.asarray(inventory),
        )

        provider_initial = chemistry.regauge_gas_only_warm_start(
            strict_small_setup,
            previous.equilibrium.gas_ln_n,
            jnp.asarray(inventory),
        )
        initial = chemistry._gas_only_warm_start(
            strict_small_setup,
            previous,
            jnp.asarray(inventory),
        )

    warm_logs = np.asarray(initial.gas_ln_n)
    np.testing.assert_array_equal(
        warm_logs,
        np.asarray(provider_initial.gas_ln_n),
    )
    assert warm_logs[trace_index] - warm_logs[dominant_index] == pytest.approx(
        gas_log_amounts[trace_index] - gas_log_amounts[dominant_index]
    )
    assert np.all(np.isfinite(warm_logs))
    assert initial.condensate_amounts is None
    assert initial.support_indices is None
    assert initial.support_amounts is None
    assert initial.inventory_bridge_origin is not None
