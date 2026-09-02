"""Unit tests for the Rocky Raccoon vector-reference extractor."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def test_straight_path_parser_rejects_bezier_marker_paths():
    assert extractor._straight_path_points("M 1 2 L 3 4") == [
        (1.0, 2.0),
        (3.0, 4.0),
    ]
    assert extractor._straight_path_points(
        "M 1 2 C 1 3 2 4 3 4"
    ) == []


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
    csv_path = data_directory / "rocky_raccoon_temperature_vector_reference.csv"
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
