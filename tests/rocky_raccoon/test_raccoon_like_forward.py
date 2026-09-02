import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import examples.rocky_raccoon.raccoon_like_forward as forward
from examples.rocky_raccoon.model import (
    ColumnMetrics,
    RaccoonLikeConfig,
    RaccoonLikeResult,
)
from examples.rocky_raccoon.physics import RaccoonLikeComposition
from examples.rocky_raccoon.raccoon_like_forward import (
    CLAIM_STATUS,
    build_parser,
    config_from_args,
    write_outputs,
)
from examples.rocky_raccoon.structure import (
    CandidateState,
    ChemistryState,
    LayerState,
    StepDecision,
    StructureInputs,
    StructureProfile,
)


class _FakeIdealGas:
    def hydrogen_mass_fraction(self, gas_x):
        return float(np.asarray(gas_x)[0])


def _fake_result():
    gas_names = ("H2", "SiO")
    condensate_names = ("MgSiO3(s)", "SiO(s)")
    element_names = ("H", "Si", "e-")
    setup = SimpleNamespace(
        gas_species=gas_names,
        condensate_species=condensate_names,
        elements=element_names,
        condensate_setup=SimpleNamespace(
            metadata={
                "rocky_raccoon_validity_scope": "condensates_only",
                "rocky_raccoon_original_temperature_validity_upper": (
                    2000.0,
                    1800.0,
                ),
            },
            temperature_validity_upper=None,
        ),
    )
    config = RaccoonLikeConfig(pressure_base_bar=1.0, pressure_top_bar=0.9)
    context = SimpleNamespace(
        config=config,
        case=SimpleNamespace(identifier="fake_case"),
        chemistry_setup=setup,
        ideal_gas=_FakeIdealGas(),
        composition=RaccoonLikeComposition(),
        element_inventory=np.asarray([1.0, 0.1, 0.0]),
    )

    def chemistry_state(gas_x, condensates, support, inventory_in, inventory_out):
        gas_n = 0.5 * np.asarray(gas_x)
        equilibrium = SimpleNamespace(
            gas_x=np.asarray(gas_x),
            gas_n=gas_n,
            gas_ln_n=np.log(gas_n),
            gas_ntot=np.sum(gas_n),
            condensate_amounts=np.asarray(condensates),
            condensate_support_indices=np.asarray(support, dtype=int),
            converged=True,
            status="converged",
            acceptance_tier="fake_acceptance",
            selected_route="fake_route",
            diagnostics={"fake": True},
        )
        transition = SimpleNamespace(
            equilibrium=equilibrium,
            inventory_in=np.asarray(inventory_in),
            inventory_out=np.asarray(inventory_out),
        )
        return ChemistryState(
            mean_molar_mass_kg_mol=2.0e-3,
            remaining_inventory=np.asarray(inventory_out),
            payload=transition,
        )

    base_chemistry = chemistry_state(
        [0.8, 0.2],
        [0.0, 0.0],
        [],
        [1.0, 0.1, 0.0],
        [1.0, 0.1, 0.0],
    )
    top_chemistry = chemistry_state(
        [0.7, 0.3],
        [0.04, 0.0],
        [0],
        [1.0, 0.1, 0.0],
        [0.95, 0.08, 0.0],
    )
    inputs = StructureInputs(
        pressure_base_bar=1.0,
        pressure_top_bar=0.9,
        temperature_base_k=1000.0,
        radius_base_m=6.0e6,
        planet_mass_kg=6.0e24,
        luminosity_w=1.0e13,
        element_inventory=np.asarray([1.0, 0.1, 0.0]),
        pressure_ratio=0.9,
    )
    layers = (
        LayerState(1.0, 1000.0, 6.0e6, 0.02, base_chemistry, "base"),
        LayerState(0.9, 980.0, 6.01e6, 0.018, top_chemistry, "convective"),
    )
    convective = CandidateState(
        "convective",
        980.0,
        0.2,
        0.01,
        0.19,
        top_chemistry,
    )
    nonconvective = CandidateState(
        "nonconvective",
        970.0,
        0.3,
        0.02,
        0.28,
        top_chemistry,
    )
    decisions = (
        StepDecision(0.9, convective, nonconvective, "convective"),
    )
    profile = StructureProfile(inputs, layers, decisions)
    shell_mass = float(profile.shell_mass_kg[0])
    metrics = ColumnMetrics(
        envelope_mass_kg=shell_mass,
        hydrogen_mass_kg=0.8 * shell_mass,
        hydrogen_to_core_mass_ratio=1.0e-3,
        transit_radius_m=6.01e6,
        outer_rcb_radius_m=None,
        top_temperature_k=980.0,
        convective_layer_count=1,
        nonconvective_layer_count=0,
    )
    return RaccoonLikeResult(context, profile, metrics)


def test_parser_maps_scientific_knobs_to_model_config(tmp_path):
    assert config_from_args(build_parser().parse_args([])).composition is None

    args = build_parser().parse_args(
        [
            "--preset",
            "oxygen_rich",
            "--validity",
            "strict_validity",
            "--pbase",
            "300000",
            "--pressure-top-bar",
            "10",
            "--transit-pressure-bar",
            "20",
            "--luminosity",
            "2e16",
            "--pressure-ratio",
            "0.95",
            "--max-steps",
            "321",
            "--abundance",
            "0.02",
            "--carbon-per-hydrogen",
            "3e-4",
            "--magnesium-to-silicon",
            "1.2",
            "--oxygen-to-silicon",
            "3.4",
            "--opacity",
            "0.03",
            "--transport",
            "constant",
            "--conductivity",
            "2000",
            "--output-dir",
            str(tmp_path),
            "--accepted-layer-snapshot",
            str(tmp_path / "accepted-layer.npz"),
            "--check-inputs",
        ]
    )
    config = config_from_args(args)

    assert config.preset == "oxygen_rich"
    assert config.validity_mode == "strict_validity"
    assert config.pressure_base_bar == 3.0e5
    assert config.pressure_top_bar == 10.0
    assert config.transit_pressure_bar == 20.0
    assert config.luminosity_w == 2.0e16
    assert config.pressure_ratio == 0.95
    assert config.max_steps == 321
    assert config.composition.silicon_per_hydrogen == 0.02
    assert config.composition.magnesium_to_silicon == 1.2
    assert config.composition.oxygen_to_silicon == 3.4
    assert config.composition.carbon_per_hydrogen == 3.0e-4
    assert config.transport.rosseland_opacity_m2_kg == 0.03
    assert config.transport.thermal_conductivity_w_m_k == 2000.0
    assert args.transport == "constant"
    assert args.output_directory == tmp_path
    assert args.accepted_layer_snapshot == tmp_path / "accepted-layer.npz"
    assert args.check_inputs


def test_partial_composition_override_inherits_the_selected_preset():
    args = build_parser().parse_args(
        [
            "--preset",
            "oxygen_rich",
            "--carbon-per-hydrogen",
            "3e-4",
        ]
    )

    composition = config_from_args(args).composition

    assert composition.silicon_per_hydrogen == 1.0e-2
    assert composition.magnesium_to_silicon == 1.0
    assert composition.oxygen_to_silicon == 4.0
    assert composition.carbon_per_hydrogen == 3.0e-4


def test_fake_profile_writer_emits_named_science_and_metadata(tmp_path):
    output_directory = tmp_path / "forward"
    summary = write_outputs(
        output_directory,
        _fake_result(),
        versions={"jax": "test-version"},
        source_report={"schema_version": "test-source"},
    )

    with (output_directory / "profiles.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    stored_summary = json.loads(
        (output_directory / "summary.json").read_text(encoding="utf-8")
    )

    assert len(rows) == 2
    assert rows[1]["transport"] == "convective"
    assert float(rows[1]["convective_equation_1_gradient"]) == 0.19
    assert float(rows[1]["nonconvective_equation_1_gradient"]) == 0.28
    assert float(rows[1]["gas_x[H2]"]) == 0.7
    assert float(rows[1]["condensate_amount[MgSiO3(s)]"]) == 0.04
    assert rows[1]["condensate_support[MgSiO3(s)]"] == "1"
    assert float(rows[1]["normalized_inventory_in[H]"]) == 1.0
    assert float(rows[1]["normalized_inventory_out[Si]"]) == 0.08
    assert rows[0]["shell_mass_kg"]
    assert float(rows[0]["hydrogen_mass_fraction"]) == 0.8
    assert rows[1]["shell_mass_kg"] == ""

    assert summary["claim_status"] == CLAIM_STATUS
    assert stored_summary["claim_status"] == CLAIM_STATUS
    assert summary["species"]["gas"] == ["H2", "SiO"]
    assert summary["validity"]["scope"] == "condensates_only"
    assert summary["package_versions"] == {"jax": "test-version"}
    assert summary["source_provenance"] == {
        "schema_version": "test-source"
    }
    assert summary["solver_diagnostics"]["all_equilibria_converged"]
    assert summary["solver_diagnostics"]["phase_support_change_count"] == 1
    assert summary["solver_diagnostics"]["mode"] == "fixed_boundary"
    assert summary["element_inventory"]["charge_constraint_inventory"] == 0.0
    assert (output_directory / "profile.png").read_bytes().startswith(b"\x89PNG")


def test_accepted_layer_snapshot_atomically_keeps_the_latest_layer(tmp_path):
    result = _fake_result()
    expected_gas_logs = np.asarray([-721.0, -np.inf])
    result.profile.layers[1].chemistry.payload.equilibrium.gas_ln_n = (
        expected_gas_logs
    )
    snapshot_path = tmp_path / "diagnostics" / "accepted-layer.npz"
    callback = forward.accepted_layer_snapshot_callback(
        snapshot_path,
        result.context,
        source_report={"schema_version": "test-source"},
    )

    callback(0, result.profile.layers[0])
    callback(1, result.profile.layers[1])

    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        assert snapshot["schema"].item() == (
            forward.ACCEPTED_LAYER_SNAPSHOT_SCHEMA
        )
        assert snapshot["layer_index"].item() == 1
        assert snapshot["pressure_bar"].item() == 0.9
        assert snapshot["transport"].item() == "convective"
        np.testing.assert_array_equal(
            snapshot["element_names"],
            ["H", "Si", "e-"],
        )
        np.testing.assert_allclose(snapshot["gas_n"], [0.35, 0.15])
        np.testing.assert_array_equal(snapshot["gas_ln_n"], expected_gas_logs)
        np.testing.assert_array_equal(snapshot["inventory_in"], [1.0, 0.1, 0.0])
        np.testing.assert_array_equal(snapshot["inventory_out"], [0.95, 0.08, 0.0])
        np.testing.assert_array_equal(
            snapshot["condensate_support_indices"], [0]
        )
        assert json.loads(snapshot["source_provenance_json"].item()) == {
            "schema_version": "test-source"
        }
    assert not list(snapshot_path.parent.glob(".*.tmp"))


def test_accepted_layer_snapshot_keeps_previous_file_after_write_failure(
    tmp_path,
    monkeypatch,
):
    result = _fake_result()
    snapshot_path = tmp_path / "diagnostics" / "accepted-layer.npz"
    snapshot_path.parent.mkdir(parents=True)
    previous_contents = b"previous accepted layer"
    snapshot_path.write_bytes(previous_contents)

    def fail_save(stream, **_arrays):
        stream.write(b"partial replacement")
        raise OSError("synthetic snapshot failure")

    monkeypatch.setattr(forward.np, "savez_compressed", fail_save)
    callback = forward.accepted_layer_snapshot_callback(
        snapshot_path,
        result.context,
        source_report={"schema_version": "test-source"},
    )

    with pytest.raises(OSError, match="synthetic snapshot failure"):
        callback(0, result.profile.layers[0])

    assert snapshot_path.read_bytes() == previous_contents
    assert not list(snapshot_path.parent.glob(".*.tmp"))


def test_main_keeps_the_last_snapshot_when_a_later_step_fails(
    tmp_path,
    monkeypatch,
):
    result = _fake_result()

    monkeypatch.setattr(forward, "build_context", lambda _config: result.context)
    monkeypatch.setattr(
        forward,
        "package_versions",
        lambda: {"python": "test-version"},
    )
    monkeypatch.setattr(
        forward,
        "source_provenance",
        lambda: {"schema_version": "test-source"},
    )

    def fail_after_two_layers(context, *, accepted_layer_callback):
        assert context is result.context
        accepted_layer_callback(0, result.profile.layers[0])
        accepted_layer_callback(1, result.profile.layers[1])
        raise RuntimeError("synthetic upper-layer failure")

    monkeypatch.setattr(forward, "run_column", fail_after_two_layers)
    output_directory = tmp_path / "failed"
    snapshot_path = output_directory / "accepted-layer.npz"

    exit_code = forward.main(
        [
            "--output-dir",
            str(output_directory),
            "--accepted-layer-snapshot",
            str(snapshot_path),
        ]
    )

    assert exit_code == 1
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        assert snapshot["layer_index"].item() == 1
        assert snapshot["pressure_bar"].item() == 0.9


def test_main_records_a_failed_run_without_claiming_completion(
    tmp_path,
    monkeypatch,
):
    def fail(_args, **_kwargs):
        raise RuntimeError("synthetic depleted-element failure")

    monkeypatch.setattr(forward, "run", fail)
    output_directory = tmp_path / "failed"

    exit_code = forward.main(
        [
            "--pressure-top-bar",
            "150000",
            "--transit-pressure-bar",
            "160000",
            "--output-dir",
            str(output_directory),
        ]
    )
    status = json.loads(
        (output_directory / "run_status.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "depleted-element" in status["error"]
    assert status["source_provenance"]["schema_version"] == (
        forward.SOURCE_PROVENANCE_SCHEMA
    )


@pytest.mark.parametrize("artifact_name", sorted(forward.MANAGED_OUTPUT_FILENAMES))
def test_main_rejects_snapshot_colliding_with_managed_output(
    tmp_path,
    monkeypatch,
    artifact_name,
):
    output_directory = tmp_path / "forward"
    snapshot_path = output_directory / "unused" / ".." / artifact_name

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The column must not run for colliding paths.")

    monkeypatch.setattr(forward, "run", fail_if_called)

    exit_code = forward.main(
        [
            "--output-dir",
            str(output_directory),
            "--accepted-layer-snapshot",
            str(snapshot_path),
        ]
    )

    status = json.loads(
        (output_directory / "run_status.json").read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert "managed output artifact" in status["error"]


def test_main_records_config_construction_failure(tmp_path):
    output_directory = tmp_path / "invalid"

    exit_code = forward.main(
        [
            "--opacity",
            "0",
            "--output-dir",
            str(output_directory),
        ]
    )
    status = json.loads(
        (output_directory / "run_status.json").read_text(encoding="utf-8")
    )

    assert exit_code == 1
    assert status["status"] == "failed"
    assert status["error_type"] == "ValueError"
    assert status["config"]["unvalidated_cli_arguments"][
        "rosseland_opacity_m2_kg"
    ] == 0.0
