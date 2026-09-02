"""Contracts for the standalone Rocky Raccoon paper comparison."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.rocky_raccoon import paper_comparison as comparison
from examples.rocky_raccoon.raccoon_like_forward import CLAIM_STATUS


ELEMENTS = ("H", "O", "e-")
GASES = ("H2", "H2O1")
CONDENSATES = ("H2O(s)",)
GAS_FORMULA = np.asarray(
    [
        [2.0, 2.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]
)
CONDENSATE_FORMULA = np.asarray([[2.0], [1.0], [0.0]])


def _synthetic_arrays():
    pressure = np.asarray([1.0, 0.1])
    temperature = np.asarray([1000.0, 900.0])
    gas_x = np.asarray([[0.75, 0.25], [0.6, 0.4]])
    gas_total = np.asarray([0.4, 0.3])
    condensates = np.asarray([[0.1], [0.05]])
    inventory = (
        gas_total[:, None] * (gas_x @ GAS_FORMULA.T)
        + condensates @ CONDENSATE_FORMULA.T
    )
    return pressure, temperature, gas_x, gas_total, condensates, inventory


def _fake_setup():
    return SimpleNamespace(
        elements=ELEMENTS,
        gas_species=GASES,
        condensate_species=CONDENSATES,
        formula_matrix=GAS_FORMULA,
        formula_matrix_cond=CONDENSATE_FORMULA,
    )


def _write_fake_run(
    tmp_path,
    *,
    top_transport="nonconvective",
    config_overrides=None,
    composition_overrides=None,
):
    directory = tmp_path / "completed"
    directory.mkdir()
    pressure, temperature, gas_x, _gas_total, condensates, inventory = (
        _synthetic_arrays()
    )
    config = {
        "preset": "oxygen_poor",
        "validity_mode": "paper_extrapolated",
        "core_mass_earth": 4.0,
        "temperature_base_k": 4000.0,
        "pressure_base_bar": 1.0,
        "pressure_top_bar": 1.0e-3,
        "transit_pressure_bar": 2.0e-2,
        "pressure_ratio": 0.99,
        "luminosity_w": 1.0e16,
    }
    config.update(config_overrides or {})
    composition = {
        "silicon_per_hydrogen": 1.0e-2,
        "magnesium_to_silicon": 1.0,
        "oxygen_to_silicon": 3.0,
        "carbon_per_hydrogen": 2.69e-4,
    }
    composition.update(composition_overrides or {})
    status = {
        "status": "completed",
        "claim_status": CLAIM_STATUS,
        "config": config,
    }
    summary = {
        "claim_status": CLAIM_STATUS,
        "config": config,
        "effective_composition": composition,
        "case_identifier": "figure2_mg_si_o_1_1_3",
        "species": {
            "elements": list(ELEMENTS),
            "gas": list(GASES),
            "condensates": list(CONDENSATES),
        },
        "metrics": {
            "outer_rcb_radius_earth": 1.5,
            "transit_radius_earth": 2.0,
            "top_temperature_k": 900.0,
            "envelope_mass_kg": 1.0e23,
            "hydrogen_mass_kg": 8.0e22,
        },
        "solver_diagnostics": {
            "mode": "fixed_boundary",
            "all_equilibria_converged": True,
            "accepted_layer_count": 2,
        },
        "source_provenance": {"schema_version": "test-source"},
    }
    (directory / "run_status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    (directory / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    fields = [
        "pressure_bar",
        "temperature_k",
        "transport",
        *[f"gas_x[{name}]" for name in GASES],
        *[f"condensate_amount[{name}]" for name in CONDENSATES],
        *[f"normalized_inventory_in[{name}]" for name in ELEMENTS],
    ]
    with (directory / "profiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(2):
            row = {
                "pressure_bar": pressure[index],
                "temperature_k": temperature[index],
                "transport": "base" if index == 0 else top_transport,
            }
            row.update(
                {
                    f"gas_x[{name}]": gas_x[index, species_index]
                    for species_index, name in enumerate(GASES)
                }
            )
            row.update(
                {
                    f"condensate_amount[{name}]": condensates[
                        index, species_index
                    ]
                    for species_index, name in enumerate(CONDENSATES)
                }
            )
            row.update(
                {
                    f"normalized_inventory_in[{name}]": inventory[
                        index, element_index
                    ]
                    for element_index, name in enumerate(ELEMENTS)
                }
            )
            writer.writerow(row)
    return directory


def _matching_reference(run):
    return comparison.PublishedTemperatureReference(
        case_identifier=run.case.identifier,
        reference_contract="published_vector_plot_digitization",
        pressure_bar=np.asarray([0.1, 0.1, 1.0]),
        temperature_k=np.asarray([900.0, 900.0, 1000.0]),
        segment_index=np.asarray([0, 1, 1]),
        transport=("non_convective", "convective", "convective"),
    )


def test_condensate_density_recovery_audits_every_element():
    pressure, temperature, gas_x, gas_total, condensates, inventory = (
        _synthetic_arrays()
    )

    result = comparison.recover_condensate_number_density_cm3(
        pressure_bar=pressure,
        temperature_k=temperature,
        gas_mixing_ratio=gas_x,
        condensate_amounts=condensates,
        incoming_inventory=inventory,
        gas_formula_matrix=GAS_FORMULA,
        condensate_formula_matrix=CONDENSATE_FORMULA,
        element_names=ELEMENTS,
    )

    np.testing.assert_allclose(result.gas_amount_total, gas_total)
    expected = (
        condensates[:, 0]
        / gas_total
        * pressure
        * 1.0e5
        / (comparison.BOLTZMANN_CONSTANT_J_K * temperature)
        / 1.0e6
    )
    np.testing.assert_allclose(result.number_density_cm3[:, 0], expected)
    assert result.maximum_tolerance_ratio < 1.0


def test_condensate_density_recovery_rejects_non_hydrogen_closure_error():
    pressure, temperature, gas_x, _gas_total, condensates, inventory = (
        _synthetic_arrays()
    )
    invalid_inventory = inventory.copy()
    invalid_inventory[1, 1] += 1.0e-5

    with pytest.raises(ValueError, match="element 'O'"):
        comparison.recover_condensate_number_density_cm3(
            pressure_bar=pressure,
            temperature_k=temperature,
            gas_mixing_ratio=gas_x,
            condensate_amounts=condensates,
            incoming_inventory=invalid_inventory,
            gas_formula_matrix=GAS_FORMULA,
            condensate_formula_matrix=CONDENSATE_FORMULA,
            element_names=ELEMENTS,
        )


def test_completed_run_loader_checks_status_and_recovers_density(tmp_path):
    directory = _write_fake_run(tmp_path)

    run = comparison.load_completed_run(
        directory, chemistry_setup=_fake_setup()
    )

    assert run.case.identifier == "figure2_mg_si_o_1_1_3"
    assert run.condensate_number_density_cm3.shape == (2, 1)
    status_path = directory / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["status"] = "failed"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    with pytest.raises(ValueError, match="not a completed run"):
        comparison.load_completed_run(
            directory, chemistry_setup=_fake_setup()
        )


@pytest.mark.parametrize(
    ("config_overrides", "composition_overrides", "message"),
    [
        ({"transit_pressure_bar": 1.0}, None, "transit_pressure_bar"),
        ({"core_mass_earth": 5.0}, None, "core_mass_earth"),
        (None, {"oxygen_to_silicon": 4.0}, "oxygen_to_silicon"),
    ],
)
def test_completed_run_loader_rejects_mismatched_paper_inputs(
    tmp_path,
    config_overrides,
    composition_overrides,
    message,
):
    directory = _write_fake_run(
        tmp_path,
        config_overrides=config_overrides,
        composition_overrides=composition_overrides,
    )

    with pytest.raises(ValueError, match=message):
        comparison.load_completed_run(
            directory,
            chemistry_setup=_fake_setup(),
        )


def test_temperature_error_uses_uniform_log_pressure_grid_and_deduplicates(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path), chemistry_setup=_fake_setup()
    )

    result = comparison.compare_temperature_profile(
        run, _matching_reference(run), grid_size=17
    )

    assert result["metric_contract"]["coordinate"] == (
        "uniform_log10_pressure_bar"
    )
    assert result["metric_contract"]["grid_point_count"] == 17
    assert result["metric_contract"]["vertex_weighting"] is False
    assert result["reference_duplicate_pressure_count"] == 1
    assert result["temperature_error_k"] == {
        "rmse": pytest.approx(0.0),
        "mae": pytest.approx(0.0),
        "sampled_maximum_absolute": pytest.approx(0.0),
    }


def test_report_and_three_row_figure_record_targets_and_missing_reference(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path), chemistry_setup=_fake_setup()
    )
    output = tmp_path / "comparison"

    report = comparison.write_comparison(output, (run, run), {})

    case = report["cases"][0]
    assert report["claim_status"] == comparison.COMPARISON_CLAIM_STATUS
    assert case["radius_comparison"]["outer_rcb"] == {
        "availability": "available_top_connected_nonconvective_region",
        "top_transport": "nonconvective",
        "paper_rearth": 1.63,
        "model_rearth": 1.5,
        "model_minus_paper_rearth": pytest.approx(-0.13),
        "fractional_difference": pytest.approx(-0.13 / 1.63),
    }
    assert case["temperature_profile_comparison"]["availability"] == (
        "missing_for_case"
    )
    assert case["gas_profile_contract"]["normalization"] == (
        "sum_over_explicit_solver_gas_species"
    )
    assert (output / comparison.OUTPUT_FIGURE_NAME).read_bytes().startswith(
        b"\x89PNG"
    )
    stored = json.loads(
        (output / comparison.OUTPUT_REPORT_NAME).read_text(encoding="utf-8")
    )
    assert len(stored["cases"]) == 2


def test_profile_figure_uses_pressure_as_an_upward_decreasing_vertical_axis(
    monkeypatch,
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path), chemistry_setup=_fake_setup()
    )
    original_close = comparison.plt.close
    monkeypatch.setattr(comparison.plt, "close", lambda _figure: None)

    comparison.plot_comparison(tmp_path / "profile.png", (run,), {})

    figure = comparison.plt.gcf()
    gas_axis, condensate_axis, temperature_axis = figure.axes
    assert all(
        axis.get_yscale() == "log"
        and axis.get_ylim()[0] > axis.get_ylim()[1]
        for axis in figure.axes
    )
    assert gas_axis.get_shared_y_axes().joined(gas_axis, temperature_axis)
    assert gas_axis.get_xscale() == "log"
    assert condensate_axis.get_xscale() == "log"
    assert temperature_axis.get_xscale() == "linear"
    assert gas_axis.get_xlabel() == "Model gas mixing ratio"
    assert temperature_axis.get_xlabel() == "Temperature (K)"
    original_close(figure)


def test_detached_nonconvective_transition_is_not_compared_as_paper_rcb(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path, top_transport="convective"),
        chemistry_setup=_fake_setup(),
    )

    outer_rcb = comparison.build_comparison_report((run,), {})["cases"][0][
        "radius_comparison"
    ]["outer_rcb"]

    assert outer_rcb["availability"] == (
        "unavailable_without_top_connected_nonconvective_region"
    )
    assert outer_rcb["model_rearth"] is None
    assert outer_rcb["fractional_difference"] is None


def test_repeatable_run_directory_parser():
    args = comparison.build_parser().parse_args(
        [
            "--run-directory",
            "first",
            "--run-directory",
            "second",
            "--output-dir",
            "comparison",
        ]
    )

    assert args.run_directory == [comparison.Path("first"), comparison.Path("second")]
    assert args.temperature_reference == comparison.DEFAULT_TEMPERATURE_REFERENCE


def test_cli_reports_provider_runtime_error_without_traceback(
    monkeypatch,
    capsys,
    tmp_path,
):
    def fail(_path):
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(comparison, "load_completed_run", fail)

    status = comparison.main(
        [
            "--run-directory",
            "input",
            "--output-directory",
            str(tmp_path / "output"),
        ]
    )

    assert status == 1
    assert "backend unavailable" in capsys.readouterr().err


def test_committed_temperature_reference_has_supported_case_contracts():
    references = comparison.load_temperature_references()

    assert "figure2_mg_si_o_1_1_3" in references
    assert "figure2_mg_si_o_1_1_4" in references
    assert all(
        reference.reference_contract == "published_vector_plot_digitization"
        for reference in references.values()
    )
