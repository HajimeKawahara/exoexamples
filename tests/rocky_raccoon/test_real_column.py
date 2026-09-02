"""Opt-in integration check against the installed ExoFamily providers."""

import os
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from examples.rocky_raccoon.chemistry import (
    RainoutLayerTransition,
    solve_rainout_layer,
)
from examples.rocky_raccoon.model import (
    RaccoonLikeConfig,
    build_context,
    run_column,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROCKY_RACCOON_INTEGRATION") != "1",
    reason="set RUN_ROCKY_RACCOON_INTEGRATION=1 for the real ExoGibbs solve",
)


@pytest.fixture(scope="module", autouse=True)
def require_x64_integration_runtime():
    """Fail before a provider solve when the integration runtime is float32."""

    assert jax.config.x64_enabled, (
        "Rocky Raccoon integration tests require JAX_ENABLE_X64=1."
    )


STEP_698_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_698_warm_parent.npz"
)
STEP_999_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_999_warm_parent.npz"
)
STEP_1075_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1075_warm_parent.npz"
)
STEP_1076_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1076_warm_parent.npz"
)
STEP_1077_CONVECTIVE_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1077_convective_warm_parent.npz"
)
STEP_1084_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1084_warm_parent.npz"
)
STEP_1186_CONVECTIVE_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1186_convective_warm_parent.npz"
)
STEP_1342_NONCONVECTIVE_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1342_nonconvective_warm_parent.npz"
)
STEP_1372_CONVECTIVE_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1372_convective_warm_parent.npz"
)
STEP_1383_CONVECTIVE_WARM_PARENT = (
    Path(__file__).with_name("data")
    / "default_column_step_1383_convective_warm_parent.npz"
)


def _load_warm_parent_case(
    path: Path,
    context,
) -> tuple[float, float, np.ndarray, RainoutLayerTransition]:
    with np.load(path) as stored:
        if "schema" in stored:
            assert stored["schema"].item() == (
                "rocky_raccoon.warm_parent_case@1"
            )
            assert tuple(stored["element_names"].tolist()) == tuple(
                context.chemistry_setup.elements
            )
            assert tuple(stored["gas_species"].tolist()) == tuple(
                context.chemistry_setup.gas_species
            )
        inventory = np.asarray(stored["target_inventory"]).copy()
        previous = RainoutLayerTransition(
            temperature_k=float(stored["parent_temperature"]),
            pressure_bar=float(stored["parent_pressure"]),
            inventory_in=np.asarray(stored["parent_inventory"]).copy(),
            equilibrium=SimpleNamespace(
                gas_ln_n=np.asarray(stored["parent_gas_ln_n"]).copy(),
            ),
            inventory_out=inventory.copy(),
        )
        return (
            float(stored["target_temperature"]),
            float(stored["target_pressure"]),
            inventory,
            previous,
        )


def _assert_public_transition_budget(context, transition) -> None:
    equilibrium = transition.equilibrium
    assert equilibrium.converged
    reconstructed = (
        np.asarray(context.chemistry_setup.formula_matrix)
        @ np.asarray(equilibrium.gas_n)
        + np.asarray(context.chemistry_setup.formula_matrix_cond)
        @ np.asarray(equilibrium.condensate_amounts)
    )
    expected = np.asarray(transition.inventory_in)
    positive = expected > 0.0
    np.testing.assert_allclose(
        reconstructed[positive],
        expected[positive],
        rtol=1.0e-3,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reconstructed[~positive],
        expected[~positive],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.all(np.asarray(transition.inventory_out) >= 0.0)


def test_three_layer_real_exogibbs_exoeos_column():
    context = build_context(
        RaccoonLikeConfig(
            pressure_top_bar=1.5e5,
            transit_pressure_bar=1.6e5,
            pressure_ratio=0.8,
        )
    )

    result = run_column(context)

    assert len(result.profile.layers) == 3
    assert len(context.chemistry_setup.gas_species) == 70
    for layer in result.profile.layers:
        transition = layer.chemistry.payload
        equilibrium = transition.equilibrium
        _assert_public_transition_budget(context, transition)
        np.testing.assert_allclose(np.sum(equilibrium.gas_x), 1.0, rtol=1.0e-9)


def test_positive_trace_mg_exogibbs_boundary():
    """Keep the resolved positive-trace provider boundary as a hard regression."""

    context = build_context()
    inventory = np.asarray(
        [
            0.9996763591270366,
            2.415508476464756e-12,
            1.824264250909887e-5,
            3.6485287433706685e-5,
            2.689129406051728e-4,
            0.0,
        ]
    )

    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1433.7645951014717,
        pressure_bar=8796.093022208004,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_optimizer_limit_boundary():
    """Keep the resolved optimizer-limit layer as a hard regression."""

    inventory = np.asarray(
        [
            0.9996545882751542,
            6.928615382247818e-12,
            2.5501542246828624e-5,
            5.100309142229654e-5,
            2.68907084246017e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1561.8193557386803,
        pressure_bar=11290.04441816559,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_amount_gauge_boundary():
    """Keep the resolved default-column amount-gauge layer as a hard regression."""

    inventory = np.asarray(
        [
            0.9997209426096871,
            1.964558116000134e-14,
            3.3774855697403563e-6,
            6.7549711591333905e-6,
            2.689249335620061e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1334.4049016146876,
        pressure_bar=6495.780683442079,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_zero_barrier_polish_boundary():
    """Keep the resolved default-column polish boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997253184018701,
            3.774168837574554e-15,
            1.9184958230696084e-6,
            3.83699164992498e-6,
            2.689261106501034e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1269.1589798706555,
        pressure_bar=5643.1822694059156,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_trace_capacity_boundary():
    """Keep the resolved trace-capacity boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997295583246257,
            7.589000706073114e-17,
            5.0480806048163e-7,
            1.0096161210834609e-6,
            2.689272511893248e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1173.1942732095774,
        pressure_bar=4132.5213914599017,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_step_378():
    """Keep the former GPU parity boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997289110989905,
            2.15359688402623e-16,
            7.206079734611716e-7,
            1.4412159471491326e-6,
            2.6892707708562876e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1188.1415292259892,
        pressure_bar=4478.5100542051532,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_step_380():
    """Keep the former CPU parity boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997290935901879,
            1.6631578573605774e-16,
            6.597612108647926e-7,
            1.3195224219399422e-6,
            2.68927126175761e-4,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=1181.3459388985098,
        pressure_bar=4389.3877041264705,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)


def test_resolved_default_column_step_698_warm_parent():
    """Keep the trace-Mg scaling boundary as a warm-parent regression."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_698_WARM_PARENT,
        context,
    )

    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    _assert_public_transition_budget(context, transition)
    assert tuple(transition.equilibrium.condensate_support_indices) == (1, 8)


def test_resolved_default_column_step_702_mixed_charge_budget():
    """Keep the mixed log/linear budget boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997310723415463,
            6.300502379398082e-47,
            3.5801015757132734e-17,
            8.331738184445885e-17,
            0.000268927658459876,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=475.01010900904657,
        pressure_bar=172.55859783339542,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)
    assert tuple(transition.equilibrium.condensate_support_indices) == (1, 8)


def test_resolved_default_column_step_774_support_release():
    """Keep the support-release budget boundary as a hard regression."""

    inventory = np.asarray(
        [
            0.9997310723415466,
            2.139116395339677e-56,
            3.504937031478576e-27,
            1.1715327176146462e-17,
            0.00026892765845987606,
            0.0,
        ]
    )

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=386.57556568831939,
        pressure_bar=83.689430815806617,
        incoming_inventory=inventory,
    )

    _assert_public_transition_budget(context, transition)
    assert tuple(transition.equilibrium.condensate_support_indices) == (1, 8)


@pytest.fixture(scope="module")
def default_column_step_999_transition():
    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_999_WARM_PARENT,
        context,
    )
    return solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )


def test_resolved_default_column_step_999_optimizer_directed_release(
    default_column_step_999_transition,
):
    """Keep the optimizer-directed proper-face boundary as a hard regression."""

    context = build_context()
    transition = default_column_step_999_transition
    _assert_public_transition_budget(context, transition)
    assert tuple(transition.equilibrium.condensate_support_indices) == (1, 8)


def test_default_column_step_1082_inventory_bridge(
    default_column_step_999_transition,
):
    """Keep the target-thermodynamics inventory bridge as a hard regression."""

    context = build_context()
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=157.89357053396711,
        pressure_bar=3.7871329378560565,
        incoming_inventory=np.asarray(
            [
                0.9997310723415466,
                6.831754190721877e-112,
                8.328995133274878e-119,
                1.1715327169136589e-17,
                0.00026892765845987606,
                0.0,
            ]
        ),
        previous_transition=default_column_step_999_transition,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (9,)


def test_default_column_step_1075_log_domain_warm_parent():
    """Keep finite log abundances through the public warm-parent boundary."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1075_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (9,)


def test_default_column_step_1076_positive_boundary_face():
    """Keep the rank-deficient support envelope's positive boundary face."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1076_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (9,)


def test_default_column_step_1077_convective_warm_parent():
    """Keep the first convective candidate after committed layer 1076."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1077_CONVECTIVE_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (9,)


def test_default_column_step_1084_log_domain_warm_parent():
    """Keep finite subfloor gas logs through the default-column boundary."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1084_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (9,)


def test_default_column_step_1186_self_reopening_basic_support():
    """Keep the sibling-basis transition after committed layer 1185."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1186_CONVECTIVE_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (5, 8)


def test_default_column_step_1342_nonconvective_warm_parent():
    """Keep the nonconvective candidate after committed layer 1341."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1342_NONCONVECTIVE_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (1, 5)


def test_default_column_step_1372_convective_warm_parent():
    """Keep the convective candidate after committed layer 1371."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1372_CONVECTIVE_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (5, 1)


def test_default_column_step_1383_convective_warm_parent():
    """Keep the convective candidate after committed layer 1382."""

    context = build_context()
    temperature, pressure, inventory, previous = _load_warm_parent_case(
        STEP_1383_CONVECTIVE_WARM_PARENT,
        context,
    )
    transition = solve_rainout_layer(
        context.chemistry_setup,
        temperature_k=temperature,
        pressure_bar=pressure,
        incoming_inventory=inventory,
        previous_transition=previous,
    )

    equilibrium = transition.equilibrium
    _assert_public_transition_budget(context, transition)
    assert tuple(equilibrium.condensate_support_indices) == (5,)
