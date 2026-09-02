"""Extract published Rocky Raccoon temperature traces from the vector PDF."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
BLUE = "12.156677%,46.665955%,70.587158%"
COLOR_PATTERN = re.compile(r"stroke:rgb\(([^)]+)\)")
WIDTH_PATTERN = re.compile(r"stroke-width:([^;]+)")
POINT_PATTERN = re.compile(r"([ML])\s*(-?[0-9.]+)\s+(-?[0-9.]+)")
COMMAND_PATTERN = re.compile(r"[A-Za-z]")

TEMPERATURE_Y_MIN = 32.320224
TEMPERATURE_Y_MAX = 128.399884
TEMPERATURE_TICK_Y_1000 = 36.686992
TEMPERATURE_TICK_Y_4000 = 124.033116


@dataclass(frozen=True)
class PanelSpec:
    figure: int
    page: int
    panel: str
    case_id: str
    include_sio_s: bool
    x_min: float
    x_max: float
    tick_x_logp_5: float
    tick_x_logp_minus_3: float
    expected_point_counts: tuple[int, ...]
    expected_regimes: tuple[str, ...]


PANEL_SPECS = (
    PanelSpec(
        figure=2,
        page=8,
        panel="left",
        case_id="figure2_mg_si_o_1_1_3",
        include_sio_s=False,
        x_min=56.407892,
        x_max=246.803296,
        tick_x_logp_5=60.659934,
        tick_x_logp_minus_3=246.774615,
        expected_point_counts=(12, 33, 43, 8, 16),
        expected_regimes=(
            "non_convective",
            "convective",
            "non_convective",
            "convective",
            "non_convective",
        ),
    ),
    PanelSpec(
        figure=2,
        page=8,
        panel="right",
        case_id="figure2_mg_si_o_1_1_4",
        include_sio_s=False,
        x_min=265.776152,
        x_max=456.175141,
        tick_x_logp_5=273.011077,
        tick_x_logp_minus_3=456.099852,
        expected_point_counts=(12, 24, 6, 10, 16),
        expected_regimes=(
            "non_convective",
            "convective",
            "non_convective",
            "convective",
            "non_convective",
        ),
    ),
    PanelSpec(
        figure=5,
        page=11,
        panel="right",
        case_id="figure5_mg_si_o_1_1_3_sio_on",
        include_sio_s=True,
        x_min=265.743885,
        x_max=456.107022,
        tick_x_logp_5=270.013853,
        tick_x_logp_minus_3=456.099852,
        expected_point_counts=(38, 10, 16),
        expected_regimes=(
            "non_convective",
            "convective",
            "non_convective",
        ),
    ),
)


def _linear_map(value, source_min, source_max, target_min, target_max):
    fraction = (value - source_min) / (source_max - source_min)
    return target_min + fraction * (target_max - target_min)


def pressure_bar_from_vector_x(vector_x, spec):
    """Map a vector x coordinate through the printed log-pressure ticks."""

    log10_pressure_bar = _linear_map(
        vector_x,
        spec.tick_x_logp_5,
        spec.tick_x_logp_minus_3,
        5.0,
        -3.0,
    )
    return 10.0**log10_pressure_bar


def temperature_k_from_vector_y(vector_y):
    """Map a vector y coordinate through the printed temperature ticks."""

    return _linear_map(
        vector_y,
        TEMPERATURE_TICK_Y_1000,
        TEMPERATURE_TICK_Y_4000,
        1000.0,
        4000.0,
    )


def _straight_path_points(path_data):
    commands = COMMAND_PATTERN.findall(path_data)
    if not commands or set(commands) - {"M", "L"}:
        return []
    points = [
        (float(x), float(y))
        for _, x, y in POINT_PATTERN.findall(path_data)
    ]
    if len(points) != len(commands):
        return []
    return points


def _temperature_segments(svg_path, spec):
    root = ET.parse(svg_path).getroot()
    candidates = []
    tolerance = 1.0e-3
    for path in root.iter(f"{SVG_NAMESPACE}path"):
        style = path.attrib.get("style", "")
        color_match = COLOR_PATTERN.search(style)
        width_match = WIDTH_PATTERN.search(style)
        if color_match is None or color_match.group(1) != BLUE:
            continue
        if width_match is None or width_match.group(1) not in {"1", "2"}:
            continue
        points = _straight_path_points(path.attrib.get("d", ""))
        # Matplotlib transition markers use Bezier paths and legend samples
        # contain only two vertices.  A plotted temperature segment has at
        # least three straight-line vertices inside this panel.
        if len(points) < 3:
            continue
        if not all(
            spec.x_min - tolerance <= x <= spec.x_max + tolerance
            and TEMPERATURE_Y_MIN - tolerance
            <= y
            <= TEMPERATURE_Y_MAX + tolerance
            for x, y in points
        ):
            continue
        if not all(x1 < x2 for (x1, _), (x2, _) in zip(points, points[1:])):
            raise RuntimeError(
                f"Non-monotonic pressure path in Figure {spec.figure} "
                f"{spec.panel} panel."
            )
        regime = (
            "convective"
            if width_match.group(1) == "2"
            else "non_convective"
        )
        candidates.append((points[0][0], regime, points))

    candidates.sort(key=lambda candidate: candidate[0])
    point_counts = tuple(len(points) for _, _, points in candidates)
    regimes = tuple(regime for _, regime, _ in candidates)
    if point_counts != spec.expected_point_counts:
        raise RuntimeError(
            f"Figure {spec.figure} {spec.panel} temperature path changed: "
            f"expected point counts {spec.expected_point_counts}, found "
            f"{point_counts}."
        )
    if regimes != spec.expected_regimes:
        raise RuntimeError(
            f"Figure {spec.figure} {spec.panel} line-width sequence changed: "
            f"expected {spec.expected_regimes}, found {regimes}."
        )
    return [(regime, points) for _, regime, points in candidates]


def _extract_panel(svg_path, spec):
    rows = []
    for segment_index, (regime, points) in enumerate(
        _temperature_segments(svg_path, spec)
    ):
        for point_index, (vector_x, vector_y) in enumerate(points):
            pressure_bar = pressure_bar_from_vector_x(vector_x, spec)
            rows.append(
                {
                    "reference_contract": "published_vector_plot_digitization",
                    "paper_figure": spec.figure,
                    "paper_page": spec.page,
                    "paper_panel": spec.panel,
                    "case_id": spec.case_id,
                    "include_sio_s": str(spec.include_sio_s).lower(),
                    "quantity": "temperature",
                    "segment_index": segment_index,
                    "transport_regime": regime,
                    "point_index": point_index,
                    "vector_x": vector_x,
                    "vector_y": vector_y,
                    "log10_pressure_bar": math.log10(pressure_bar),
                    "pressure_bar": pressure_bar,
                    "temperature_k": temperature_k_from_vector_y(vector_y),
                }
            )
    return rows


def extract_reference(paper_pdf):
    """Return visible temperature vertices from Figures 2 and 5."""

    rows = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        svg_paths_by_page = {}
        for spec in PANEL_SPECS:
            svg_path = svg_paths_by_page.get(spec.page)
            if svg_path is None:
                svg_path = temporary_directory / f"page_{spec.page}.svg"
                subprocess.run(
                    [
                        "pdftocairo",
                        "-f",
                        str(spec.page),
                        "-l",
                        str(spec.page),
                        "-svg",
                        str(paper_pdf),
                        str(svg_path),
                    ],
                    check=True,
                )
                svg_paths_by_page[spec.page] = svg_path
            rows.extend(_extract_panel(svg_path, spec))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_pdf", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.paper_pdf.is_file():
        raise FileNotFoundError(args.paper_pdf)
    rows = extract_reference(args.paper_pdf)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} vector vertices to {args.output_csv}.")


if __name__ == "__main__":
    main()
