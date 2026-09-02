import numpy as np
import pytest

from examples.rocky_raccoon.structure import (
    BAR_PA,
    GRAVITATIONAL_CONSTANT_SI,
    ChemistryState,
    StructureCallbacks,
    StructureInputs,
    integrate_structure,
)


def _inputs(*, pressure_top_bar=0.975, element_inventory=None):
    if element_inventory is None:
        element_inventory = np.array([10.0, 20.0])
    return StructureInputs(
        pressure_base_bar=1.0,
        pressure_top_bar=pressure_top_bar,
        temperature_base_k=1000.0,
        radius_base_m=6.4e6,
        planet_mass_kg=5.97e24,
        luminosity_w=1.0e13,
        element_inventory=element_inventory,
    )


def _constant_chemistry(
    temperature_k,
    pressure_bar,
    inventory,
    previous_chemistry,
):
    del temperature_k, pressure_bar, previous_chemistry
    return ChemistryState(2.0e-3, inventory)


def _constant_density(pressure_bar, temperature_k, chemistry):
    del pressure_bar, temperature_k, chemistry
    return 2.0e-2


def _callbacks(
    *,
    chemistry=_constant_chemistry,
    convective_gradient=0.1,
    nonconvective_gradient=0.3,
    density=_constant_density,
):
    return StructureCallbacks(
        equilibrium=chemistry,
        convective_log_temperature_gradient=(
            lambda layer, luminosity: convective_gradient
        ),
        nonconvective_log_temperature_gradient=(
            lambda layer, luminosity: nonconvective_gradient
        ),
        mass_density=density,
    )


def test_candidates_are_transactional_and_only_selected_inventory_is_committed():
    received = []
    original_inventory = np.array([10.0, 20.0])
    inputs = _inputs(
        pressure_top_bar=0.985,
        element_inventory=original_inventory,
    )

    def mutating_chemistry(
        temperature_k,
        pressure_bar,
        inventory,
        previous_chemistry,
    ):
        incoming = inventory.copy()
        previous_payload = (
            None if previous_chemistry is None else previous_chemistry.payload
        )
        received.append((pressure_bar, incoming, previous_payload))
        inventory[:] = -999.0
        if np.isclose(pressure_bar, 1.0):
            remaining = incoming
            payload = "base"
        elif temperature_k > 998.0:
            remaining = incoming - 1.0
            payload = "convective"
        else:
            remaining = incoming - 3.0
            payload = "nonconvective"
        return ChemistryState(2.0e-3, remaining, payload)

    profile = integrate_structure(
        inputs,
        _callbacks(chemistry=mutating_chemistry),
    )

    np.testing.assert_array_equal(original_inventory, [10.0, 20.0])
    first_step_inputs = [
        values
        for pressure, values, previous in received
        if np.isclose(pressure, 0.99)
    ]
    second_step_inputs = [
        values
        for pressure, values, previous in received
        if np.isclose(pressure, 0.9801)
    ]
    first_step_previous = [
        previous
        for pressure, values, previous in received
        if np.isclose(pressure, 0.99)
    ]
    second_step_previous = [
        previous
        for pressure, values, previous in received
        if np.isclose(pressure, 0.9801)
    ]
    assert len(first_step_inputs) == 2
    assert len(second_step_inputs) == 2
    np.testing.assert_array_equal(first_step_inputs[0], [10.0, 20.0])
    np.testing.assert_array_equal(first_step_inputs[1], [10.0, 20.0])
    np.testing.assert_array_equal(second_step_inputs[0], [9.0, 19.0])
    np.testing.assert_array_equal(second_step_inputs[1], [9.0, 19.0])
    assert received[0][2] is None
    assert first_step_previous == ["base", "base"]
    assert second_step_previous == ["convective", "convective"]
    assert "nonconvective" not in second_step_previous
    np.testing.assert_array_equal(profile.remaining_inventory[-1], [8.0, 18.0])
    assert profile.transport == ("base", "convective", "convective")


def test_accepted_layer_callback_observes_only_committed_states():
    observed = []

    def chemistry(temperature_k, pressure_bar, inventory, previous_chemistry):
        del pressure_bar, previous_chemistry
        payload = "convective" if temperature_k > 998.0 else "nonconvective"
        return ChemistryState(2.0e-3, inventory, payload)

    profile = integrate_structure(
        _inputs(pressure_top_bar=0.985),
        _callbacks(chemistry=chemistry),
        accepted_layer_callback=lambda index, layer: observed.append(
            (index, layer.transport, layer.chemistry.payload)
        ),
    )

    assert [item[:2] for item in observed] == list(
        enumerate(profile.transport)
    )
    assert [item[2] for item in observed[1:]] == [
        "convective",
        "convective",
    ]
    assert all(item[2] != "nonconvective" for item in observed)


def test_accepted_layer_callback_excludes_a_failed_candidate_step():
    observed = []

    def density(pressure_bar, temperature_k, chemistry):
        del temperature_k, chemistry
        if pressure_bar < 1.0:
            raise ValueError("synthetic selected-density failure")
        return 2.0e-2

    with pytest.raises(ValueError, match="selected-density failure"):
        integrate_structure(
            _inputs(pressure_top_bar=0.995),
            _callbacks(density=density),
            accepted_layer_callback=lambda index, layer: observed.append(
                (index, layer.pressure_bar)
            ),
        )

    assert observed == [(0, 1.0)]


def test_pressure_recurrence_keeps_full_one_percent_steps():
    inputs = _inputs(pressure_top_bar=0.975)

    profile = integrate_structure(inputs, _callbacks())

    np.testing.assert_allclose(
        profile.pressure_bar[1:],
        0.99 * profile.pressure_bar[:-1],
        rtol=0.0,
        atol=1.0e-15,
    )
    assert profile.pressure_bar[-2] > inputs.pressure_top_bar
    assert profile.pressure_bar[-1] <= inputs.pressure_top_bar


def test_pressure_ratio_is_configurable_for_coarse_runs():
    inputs = _inputs(pressure_top_bar=0.45)
    inputs = StructureInputs(
        **{
            **inputs.__dict__,
            "pressure_ratio": 0.5,
        }
    )

    profile = integrate_structure(inputs, _callbacks())

    np.testing.assert_allclose(profile.pressure_bar, [1.0, 0.5, 0.25])


def test_equal_equation_1_gradients_select_the_stable_branch():
    profile = integrate_structure(
        _inputs(pressure_top_bar=0.995),
        _callbacks(convective_gradient=0.2, nonconvective_gradient=0.2),
    )

    decision = profile.decisions[0]
    assert decision.convective.equation_1_gradient == pytest.approx(
        decision.nonconvective.equation_1_gradient
    )
    assert decision.selected_transport == "nonconvective"
    assert profile.transport[-1] == "nonconvective"


def test_lower_state_hydrostatic_euler_step_is_monotonic():
    inputs = _inputs(pressure_top_bar=0.995)
    density = 2.0e-2

    profile = integrate_structure(
        inputs,
        _callbacks(density=lambda pressure, temperature, chemistry: density),
    )

    gravity_base = (
        GRAVITATIONAL_CONSTANT_SI
        * inputs.planet_mass_kg
        / inputs.radius_base_m**2
    )
    expected_radius = inputs.radius_base_m + (
        (inputs.pressure_base_bar - 0.99 * inputs.pressure_base_bar)
        * BAR_PA
        / (density * gravity_base)
    )
    assert profile.radius_m[1] == pytest.approx(expected_radius)
    assert np.all(np.diff(profile.radius_m) > 0.0)

    expected_shell_mass = (
        4.0
        * np.pi
        / 3.0
        * density
        * (expected_radius**3 - inputs.radius_base_m**3)
    )
    assert profile.shell_mass_kg[0] == pytest.approx(expected_shell_mass)
