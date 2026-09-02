"""Unit tests for the Rocky Raccoon vector-reference extractor."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from examples.rocky_raccoon import paper_comparison as comparison


EXTRACTOR_PATH = (
    Path(__file__).parents[2]
    / "docs"
    / "rocky_raccoon"
    / "extract_rocky_raccoon_reference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "extract_rocky_raccoon_reference", EXTRACTOR_PATH
)
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


@pytest.mark.parametrize("panel_spec", extractor.PANEL_SPECS)
def test_pressure_axis_uses_the_printed_logarithmic_ticks(panel_spec):
    assert extractor.pressure_bar_from_vector_x(
        panel_spec.tick_x_logp_5, panel_spec
    ) == pytest.approx(1.0e5)
    assert extractor.pressure_bar_from_vector_x(
        panel_spec.tick_x_logp_minus_3, panel_spec
    ) == pytest.approx(1.0e-3)


def test_temperature_axis_uses_the_printed_linear_ticks():
    assert extractor.temperature_k_from_vector_y(
        extractor.TEMPERATURE_TICK_Y_1000
    ) == pytest.approx(1000.0)
    assert extractor.temperature_k_from_vector_y(
        extractor.TEMPERATURE_TICK_Y_4000
    ) == pytest.approx(4000.0)


def test_gas_axis_uses_the_printed_logarithmic_ticks():
    assert extractor.log10_mixing_ratio_from_vector_y(
        extractor.GAS_TICK_Y_LOG10_0
    ) == pytest.approx(0.0)
    assert extractor.log10_mixing_ratio_from_vector_y(
        extractor.GAS_TICK_Y_LOG10_MINUS_18
    ) == pytest.approx(-18.0)


def test_gas_species_colors_are_exact_and_unique():
    assert len(extractor.GAS_SPECIES_BY_VECTOR_RGB) == 13
    assert len(set(extractor.GAS_SPECIES_BY_VECTOR_RGB.values())) == 13
    assert extractor.GAS_SPECIES_BY_VECTOR_RGB == {
        "86.665344%,62.744141%,86.665344%": "CH3",
        "85.881042%,43.920898%,57.646179%": "C2H2",
        "66.273499%,66.273499%,66.273499%": "H2",
        "100%,64.704895%,0%": "SiH3",
        "100%,41.175842%,70.587158%": "CH4",
        "100%,75.292969%,79.606628%": "C2H4",
        "62.744141%,32.156372%,17.64679%": "Mg(OH)2",
        "85.488892%,64.704895%,12.548828%": "SiH4",
        "0%,50.195312%,50.195312%": "CO",
        "82.743835%,82.743835%,82.743835%": "H",
        "11.764526%,56.469727%,100%": "H2O",
        "80.39093%,36.077881%,36.077881%": "SiO",
        "0%,0%,80.39093%": "CO2",
    }


def test_comparison_colors_match_every_extracted_species_rgb():
    expected = {
        species: tuple(
            float(component.removesuffix("%")) / 100.0
            for component in vector_rgb.split(",")
        )
        for vector_rgb, species in extractor.GAS_SPECIES_BY_VECTOR_RGB.items()
    }

    assert set(comparison.PUBLISHED_GAS_COLORS) == set(expected)
    for species, rgb in expected.items():
        assert comparison.PUBLISHED_GAS_COLORS[species] == pytest.approx(rgb)


def test_straight_path_parser_rejects_bezier_marker_paths():
    assert extractor._straight_path_points("M 1 2 L 3 4") == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]
    assert extractor._straight_path_points(
        "M 1 2 C 1 3 2 4 3 4"
    ) == []


def test_visible_polyline_runs_do_not_bridge_a_censored_gap():
    runs = extractor._visible_polyline_runs(
        [(0.0, 260.0), (2.0, 240.0), (4.0, 240.0), (6.0, 260.0)],
        (0.0, 6.0, 247.0, 343.0),
    )

    assert runs == [
        [(0.0, 260.0), pytest.approx((1.3, 247.0))],
        [pytest.approx((4.7, 247.0)), (6.0, 260.0)],
    ]


def test_temperature_path_identity_uses_color_width_geometry_and_order(
    tmp_path,
):
    panel_spec = extractor.PanelSpec(
        figure=99,
        page=1,
        panel="test",
        case_id="test_case",
        include_sio_s=False,
        x_min=0.0,
        x_max=10.0,
        tick_x_logp_5=0.0,
        tick_x_logp_minus_3=10.0,
        expected_point_counts=(3, 3),
        expected_regimes=("non_convective", "convective"),
    )
    blue = extractor.BLUE
    svg_path = tmp_path / "panel.svg"
    svg_path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg">
        <path style="fill:none;stroke-width:2;stroke:rgb({blue});"
              d="M 5 80 L 7 70 L 9 60"/>
        <path style="fill:none;stroke-width:1;stroke:rgb({blue});"
              d="M 0 120 L 2 110 L 4 90"/>
        <path style="fill:none;stroke-width:1;stroke:rgb({blue});"
              d="M 5 80 L 8 80"/>
        <path style="fill:none;stroke-width:1;stroke:rgb({blue});"
              d="M 3 100 C 3 99 4 99 4 100"/>
        <path style="fill:none;stroke-width:1;stroke:rgb(100%,0%,0%);"
              d="M 0 120 L 2 110 L 4 90"/>
        </svg>''',
        encoding="utf-8",
    )

    assert extractor._temperature_segments(svg_path, panel_spec) == [
        ("non_convective", [(0.0, 120.0), (2.0, 110.0), (4.0, 90.0)]),
        ("convective", [(5.0, 80.0), (7.0, 70.0), (9.0, 60.0)]),
    ]


def test_committed_reference_metadata_matches_the_csv_snapshot():
    data_directory = EXTRACTOR_PATH.parent / "data"
    csv_path = (
        data_directory / "rocky_raccoon_temperature_vector_reference.csv"
    )
    metadata_path = (
        data_directory / "rocky_raccoon_temperature_vector_reference.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    csv_bytes = csv_path.read_bytes()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == metadata["data"]["row_count"] == 244
    assert hashlib.sha256(csv_bytes).hexdigest() == metadata["data"]["sha256"]
    assert {row["case_id"] for row in rows} == {
        "figure2_mg_si_o_1_1_3",
        "figure2_mg_si_o_1_1_4",
        "figure5_mg_si_o_1_1_3_sio_on",
    }


def test_committed_gas_metadata_matches_the_csv_snapshot():
    data_directory = EXTRACTOR_PATH.parent / "data"
    csv_path = data_directory / "rocky_raccoon_gas_vector_reference.csv"
    metadata_path = data_directory / "rocky_raccoon_gas_vector_reference.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    csv_bytes = csv_path.read_bytes()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == metadata["data"]["row_count"] == 2609
    assert hashlib.sha256(csv_bytes).hexdigest() == metadata["data"]["sha256"]
    assert list(rows[0]) == [
        "reference_contract",
        "paper_figure",
        "paper_page",
        "paper_panel",
        "case_id",
        "include_sio_s",
        "quantity",
        "species",
        "segment_index",
        "transport_regime",
        "point_index",
        "vector_x",
        "vector_y",
        "log10_pressure_bar",
        "pressure_bar",
        "log10_mixing_ratio",
        "mixing_ratio",
    ]
    assert {row["quantity"] for row in rows} == {"gas_mixing_ratio"}
    assert {row["transport_regime"] for row in rows} == {
        "convective",
        "non_convective",
    }
    minimum_log_mixing_ratio = min(
        float(row["log10_mixing_ratio"]) for row in rows
    )
    assert minimum_log_mixing_ratio == pytest.approx(-18.0)

    metadata_species = {
        item["species"] for item in metadata["species_color_contract"]
    }
    assert {row["species"] for row in rows} <= metadata_species


def test_committed_gas_segments_are_contiguous_visible_runs():
    csv_path = (
        EXTRACTOR_PATH.parent
        / "data"
        / "rocky_raccoon_gas_vector_reference.csv"
    )
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    segments = {}
    for row in rows:
        key = (row["case_id"], row["species"], int(row["segment_index"]))
        segments.setdefault(key, []).append(row)
    for segment_rows in segments.values():
        assert len(segment_rows) >= 2
        assert [int(row["point_index"]) for row in segment_rows] == list(
            range(len(segment_rows))
        )
        pressures = [float(row["pressure_bar"]) for row in segment_rows]
        assert all(
            pressure_0 > pressure_1
            for pressure_0, pressure_1 in zip(pressures, pressures[1:])
        )
        assert len({row["transport_regime"] for row in segment_rows}) == 1

    # The Figure 2 oxygen-poor C2H2 curve falls below the 10^-18 plot floor
    # and later re-enters.  Its two visible non-convective runs remain split.
    censored_runs = [
        segments[("figure2_mg_si_o_1_1_3", "C2H2", segment_index)]
        for segment_index in (4, 5)
    ]
    assert float(censored_runs[0][-1]["log10_mixing_ratio"]) == pytest.approx(
        -18.0
    )
    assert float(censored_runs[1][0]["log10_mixing_ratio"]) == pytest.approx(
        -18.0
    )
    assert float(censored_runs[0][-1]["pressure_bar"]) > float(
        censored_runs[1][0]["pressure_bar"]
    )
