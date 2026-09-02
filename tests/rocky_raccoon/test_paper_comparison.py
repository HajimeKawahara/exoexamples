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


def _gas_reference(run, species, pressure, mixing_ratio, segments=None):
    pressure = np.asarray(pressure, dtype=float)
    if segments is None:
        segments = np.zeros(pressure.size, dtype=int)
    segments = np.asarray(segments, dtype=int)
    return comparison.PublishedGasReference(
        case_identifier=run.case.identifier,
        species=species,
        reference_contract="published_vector_plot_digitization",
        pressure_bar=pressure,
        mixing_ratio=np.asarray(mixing_ratio, dtype=float),
        segment_index=segments,
        transport=tuple("convective" for _ in pressure),
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


def test_gas_reference_loader_validates_and_groups_curves(tmp_path):
    path = tmp_path / "gas.csv"
    fields = [
        "reference_contract",
        "case_id",
        "quantity",
        "species",
        "segment_index",
        "transport_regime",
        "pressure_bar",
        "mixing_ratio",
        "ignored_metadata",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for species, values in (("H2", (0.8, 0.7)), ("H2O", (0.2, 0.3))):
            for pressure, value in zip((1.0, 0.1), values):
                writer.writerow(
                    {
                        "reference_contract": (
                            "published_vector_plot_digitization"
                        ),
                        "case_id": "figure2_mg_si_o_1_1_3",
                        "quantity": "gas_mixing_ratio",
                        "species": species,
                        "segment_index": 0,
                        "transport_regime": "convective",
                        "pressure_bar": pressure,
                        "mixing_ratio": value,
                        "ignored_metadata": "accepted",
                    }
                )

    references = comparison.load_gas_references(path)

    assert set(references["figure2_mg_si_o_1_1_3"]) == {"H2", "H2O"}
    assert references["figure2_mg_si_o_1_1_3"]["H2O"].mixing_ratio.tolist() == [
        0.2,
        0.3,
    ]


def test_h2_relative_gas_error_uses_visible_segments_without_crossing_gap(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path), chemistry_setup=_fake_setup()
    )
    pressure = np.asarray([1.0, 0.6, 0.2, 0.1])
    log_weight = -np.log10(pressure)
    model_ratio = 10.0 ** (
        np.log10(1.0 / 3.0)
        + log_weight * (np.log10(2.0 / 3.0) - np.log10(1.0 / 3.0))
    )
    references = {
        "H2": _gas_reference(run, "H2", [1.0, 0.1], [0.5, 0.5]),
        "H2O": _gas_reference(
            run,
            "H2O",
            pressure,
            0.5 * model_ratio,
            segments=[0, 0, 1, 1],
        ),
        "H": _gas_reference(run, "H", [1.0, 0.1], [1.0e-4, 1.0e-3]),
    }

    result = comparison.compare_gas_profiles(run, references, grid_size=101)

    species = result["h2_relative_dex_comparison"]["species"]["H2O"]
    assert result["availability"] == "available"
    assert result["paper_only_species"] == ["H"]
    assert 0 < species["sample_count"] < 101
    assert len(species["joint_visible_overlap_pressure_bar_intervals"]) == 2
    expected_coverage = np.log10(1.0 / 0.6) + np.log10(0.2 / 0.1)
    assert species["paper_visible_overlap_dex"] == pytest.approx(
        expected_coverage
    )
    assert species["joint_visible_overlap_dex"] == pytest.approx(
        expected_coverage
    )
    assert species["joint_visible_fraction"] == pytest.approx(1.0)
    assert species["model_below_floor_overlap_dex"] == pytest.approx(0.0)
    assert species["model_below_floor_fraction"] == pytest.approx(0.0)
    assert species["error_dex"] == {
        "rmse": pytest.approx(0.0, abs=1.0e-14),
        "mae": pytest.approx(0.0, abs=1.0e-14),
        "sampled_maximum_absolute": pytest.approx(0.0, abs=1.0e-14),
        "bias": pytest.approx(0.0, abs=1.0e-14),
    }
    assert "H" not in result["h2_relative_dex_comparison"]["species"]
    contract = result["h2_relative_dex_comparison"]["metric_contract"]
    assert contract["model_numerator_minimum_mixing_ratio"] == 1.0e-18
    assert contract["excluded_region_interpretation"] == "censored_not_zero"
    assert contract["model_below_floor_policy"] == (
        "excluded_and_reported_as_censored"
    )


def test_model_ratio_segments_split_at_plot_floor_without_bridging():
    run = SimpleNamespace(
        case=SimpleNamespace(identifier="figure2_mg_si_o_1_1_3"),
        gas_species=("H2", "H2O1"),
        pressure_bar=np.asarray([1.0, 0.1, 0.01, 0.001, 0.0001]),
        gas_mixing_ratio=np.asarray(
            [
                [0.5, 1.0e-10],
                [0.5, 1.0e-18],
                [0.5, 1.0e-19],
                [0.5, 1.0e-18],
                [0.5, 1.0e-10],
            ]
        ),
    )

    segments = comparison._model_log_ratio_segments(run, 1, 0)

    assert len(segments) == 2
    assert [pressure.size for _index, pressure, _ratio in segments] == [2, 2]
    assert all(
        not (pressure[0] < -2.0 < pressure[-1])
        for _index, pressure, _ratio in segments
    )
    references = {
        "H2": _gas_reference(run, "H2", [1.0, 0.0001], [0.5, 0.5]),
        "H2O": _gas_reference(
            run, "H2O", [1.0, 0.0001], [1.0e-10, 1.0e-10]
        ),
    }
    species = comparison.compare_gas_profiles(
        run, references, grid_size=101
    )["h2_relative_dex_comparison"]["species"]["H2O"]
    assert len(species["joint_visible_overlap_pressure_bar_intervals"]) == 2
    assert species["paper_visible_overlap_dex"] == pytest.approx(4.0)
    assert species["joint_visible_overlap_dex"] == pytest.approx(2.0)
    assert species["model_below_floor_overlap_dex"] == pytest.approx(2.0)
    assert species["joint_visible_fraction"] == pytest.approx(0.5)
    assert species["model_below_floor_fraction"] == pytest.approx(0.5)


def test_gas_comparison_reports_paper_coverage_when_model_is_below_floor():
    run = SimpleNamespace(
        case=SimpleNamespace(identifier="figure2_mg_si_o_1_1_3"),
        gas_species=("H2", "H2O1"),
        pressure_bar=np.asarray([1.0, 0.1]),
        gas_mixing_ratio=np.asarray([[0.5, 1.0e-19], [0.5, 1.0e-20]]),
    )
    references = {
        "H2": _gas_reference(run, "H2", [1.0, 0.1], [0.5, 0.5]),
        "H2O": _gas_reference(run, "H2O", [1.0, 0.1], [1.0e-8, 1.0e-9]),
    }

    species = comparison.compare_gas_profiles(run, references)[
        "h2_relative_dex_comparison"
    ]["species"]["H2O"]

    assert species["availability"] == (
        "model_below_comparison_floor_over_paper_visible_coverage"
    )
    assert species["paper_visible_overlap_dex"] == pytest.approx(1.0)
    assert species["joint_visible_overlap_dex"] == pytest.approx(0.0)
    assert species["model_below_floor_overlap_dex"] == pytest.approx(1.0)
    assert species["joint_visible_fraction"] == pytest.approx(0.0)
    assert species["model_below_floor_fraction"] == pytest.approx(1.0)


def test_published_gas_colors_match_audited_vector_rgb():
    assert comparison.PUBLISHED_GAS_COLORS["CH3"] == (
        0.86665344,
        0.62744141,
        0.86665344,
    )
    assert comparison.PUBLISHED_GAS_COLORS["H2"] == (
        0.66273499,
        0.66273499,
        0.66273499,
    )
    assert comparison.PUBLISHED_GAS_COLORS["CH4"] == (
        1.0,
        0.41175842,
        0.70587158,
    )
    assert comparison.PUBLISHED_GAS_COLORS["SiO"] == (
        0.80390930,
        0.36077881,
        0.36077881,
    )


def test_report_and_three_row_figure_record_targets_and_missing_reference(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path), chemistry_setup=_fake_setup()
    )
    output = tmp_path / "comparison"

    report = comparison.write_comparison(output, (run, run), {}, {})

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
    gas_references = {
        run.case.identifier: {
            "H2": _gas_reference(run, "H2", [1.0, 0.1], [0.8, 0.7]),
            "H": _gas_reference(run, "H", [1.0, 0.1], [1.0e-4, 1.0e-3]),
        }
    }

    comparison.plot_comparison(
        tmp_path / "profile.png", (run,), {}, gas_references
    )

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
    assert gas_axis.get_xlabel() == "Gas mixing ratio (raw overlay)"
    assert temperature_axis.get_xlabel() == "Temperature (K)"
    h2_color = comparison.PUBLISHED_GAS_COLORS["H2"]
    assert any(
        line.get_linestyle() == "-" and line.get_color() == h2_color
        for line in gas_axis.lines
    )
    assert any(
        line.get_linestyle() == "--" and line.get_color() == h2_color
        for line in gas_axis.lines
    )
    h2o_color = comparison.PUBLISHED_GAS_COLORS["H2O"]
    assert any(
        line.get_linestyle() == "-" and line.get_color() == h2o_color
        for line in gas_axis.lines
    )
    original_close(figure)


def test_detached_nonconvective_transition_is_not_compared_as_paper_rcb(
    tmp_path,
):
    run = comparison.load_completed_run(
        _write_fake_run(tmp_path, top_transport="convective"),
        chemistry_setup=_fake_setup(),
    )

    outer_rcb = comparison.build_comparison_report((run,), {}, {})["cases"][0][
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
    assert args.gas_reference == comparison.DEFAULT_GAS_REFERENCE


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


def test_committed_gas_reference_has_h2_anchor_and_audited_species():
    references = comparison.load_gas_references()

    assert "figure2_mg_si_o_1_1_3" in references
    assert "figure2_mg_si_o_1_1_4" in references
    assert all("H2" in curves for curves in references.values())
    assert all(
        set(curves) <= set(comparison.PUBLISHED_TO_MODEL_GAS)
        for curves in references.values()
    )
