"""Extract published Rocky Raccoon traces from the vector PDF."""

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

GAS_Y_MIN = 247.155907
GAS_Y_MAX = 343.239153
GAS_TICK_Y_LOG10_0 = 339.646787
GAS_TICK_Y_LOG10_MINUS_18 = 247.155907

# The paper uses one exact CSS color for each gas in the Figure 2 and Figure 5
# legends.  Keep the PDF strings here rather than matching approximately: an
# incompatible re-rendering must fail instead of silently changing identity.
GAS_SPECIES_BY_VECTOR_RGB = {
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

EXPECTED_VISIBLE_GAS_SPECIES = {
    "figure2_mg_si_o_1_1_3": frozenset(GAS_SPECIES_BY_VECTOR_RGB.values())
    - {"CO2"},
    "figure2_mg_si_o_1_1_4": frozenset(
        {
            "H2", "SiH3", "CH4", "Mg(OH)2", "SiH4", "CO", "H",
            "H2O", "SiO", "CO2",
        }
    ),
    "figure5_mg_si_o_1_1_3_sio_on": frozenset(
        {
            "CH3", "H2", "SiH3", "CH4", "Mg(OH)2", "SiH4", "CO",
            "H", "H2O", "SiO", "CO2",
        }
    ),
}


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


def log10_mixing_ratio_from_vector_y(vector_y):
    """Map a vector y coordinate through the printed gas-panel ticks."""

    return _linear_map(
        vector_y,
        GAS_TICK_Y_LOG10_0,
        GAS_TICK_Y_LOG10_MINUS_18,
        0.0,
        -18.0,
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


def _clip_segment_to_rectangle(point_0, point_1, bounds):
    """Return the visible part of a straight segment, or ``None``."""

    x_min, x_max, y_min, y_max = bounds
    x_0, y_0 = point_0
    dx = point_1[0] - x_0
    dy = point_1[1] - y_0
    lower = 0.0
    upper = 1.0
    for direction, distance in (
        (-dx, x_0 - x_min),
        (dx, x_max - x_0),
        (-dy, y_0 - y_min),
        (dy, y_max - y_0),
    ):
        if direction == 0.0:
            if distance < 0.0:
                return None
            continue
        fraction = distance / direction
        if direction < 0.0:
            lower = max(lower, fraction)
        else:
            upper = min(upper, fraction)
        if lower > upper:
            return None
    return (
        (x_0 + lower * dx, y_0 + lower * dy),
        (x_0 + upper * dx, y_0 + upper * dy),
    )


def _same_point(point_0, point_1):
    return all(
        math.isclose(value_0, value_1, rel_tol=0.0, abs_tol=1.0e-10)
        for value_0, value_1 in zip(point_0, point_1)
    )


def _visible_polyline_runs(points, bounds):
    """Clip a polyline without bridging intervals hidden by plot bounds."""

    runs = []
    current = []
    for point_0, point_1 in zip(points, points[1:]):
        clipped = _clip_segment_to_rectangle(point_0, point_1, bounds)
        if clipped is None:
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        visible_0, visible_1 = clipped
        if current and _same_point(current[-1], visible_0):
            if not _same_point(current[-1], visible_1):
                current.append(visible_1)
        else:
            if len(current) >= 2:
                runs.append(current)
            current = [visible_0]
            if not _same_point(visible_0, visible_1):
                current.append(visible_1)
    if len(current) >= 2:
        runs.append(current)
    return runs


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


def _gas_segments(svg_path, spec):
    root = ET.parse(svg_path).getroot()
    paths_by_species = {
        species: [] for species in GAS_SPECIES_BY_VECTOR_RGB.values()
    }
    tolerance = 1.0e-3
    for path in root.iter(f"{SVG_NAMESPACE}path"):
        style = path.attrib.get("style", "")
        color_match = COLOR_PATTERN.search(style)
        width_match = WIDTH_PATTERN.search(style)
        if color_match is None:
            continue
        species = GAS_SPECIES_BY_VECTOR_RGB.get(color_match.group(1))
        if species is None:
            continue
        if width_match is None or width_match.group(1) not in {"1", "2"}:
            continue
        points = _straight_path_points(path.attrib.get("d", ""))
        if len(points) < 2:
            continue
        if not all(
            spec.x_min - tolerance <= x <= spec.x_max + tolerance
            for x, _ in points
        ):
            continue
        if max(y for _, y in points) < GAS_Y_MIN - tolerance:
            continue
        if min(y for _, y in points) > GAS_Y_MAX + tolerance:
            continue
        adjacent_points = zip(points, points[1:])
        if not all(x_0 < x_1 for (x_0, _), (x_1, _) in adjacent_points):
            raise RuntimeError(
                f"Non-monotonic gas path for {species} in Figure "
                f"{spec.figure} {spec.panel} panel."
            )
        regime = (
            "convective"
            if width_match.group(1) == "2"
            else "non_convective"
        )
        paths_by_species[species].append((points[0][0], regime, points))

    bounds = (spec.x_min, spec.x_max, GAS_Y_MIN, GAS_Y_MAX)
    visible_by_species = {}
    for species, paths in paths_by_species.items():
        paths.sort(key=lambda candidate: candidate[0])
        regimes = tuple(regime for _, regime, _ in paths)
        if paths and regimes != spec.expected_regimes:
            raise RuntimeError(
                f"Figure {spec.figure} {spec.panel} {species} line-width "
                f"sequence changed: expected {spec.expected_regimes}, found "
                f"{regimes}."
            )
        visible_runs = []
        for _, regime, points in paths:
            for run in _visible_polyline_runs(points, bounds):
                visible_runs.append((run[0][0], regime, run))
        if visible_runs:
            visible_runs.sort(key=lambda candidate: candidate[0])
            visible_by_species[species] = [
                (regime, points)
                for _, regime, points in visible_runs
            ]

    visible_species = frozenset(visible_by_species)
    expected_species = EXPECTED_VISIBLE_GAS_SPECIES[spec.case_id]
    if visible_species != expected_species:
        raise RuntimeError(
            f"Figure {spec.figure} {spec.panel} visible gas species changed: "
            f"expected {sorted(expected_species)}, found "
            f"{sorted(visible_species)}."
        )
    return visible_by_species


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


def _extract_gas_panel(svg_path, spec):
    rows = []
    segments_by_species = _gas_segments(svg_path, spec)
    for species in GAS_SPECIES_BY_VECTOR_RGB.values():
        for segment_index, (regime, points) in enumerate(
            segments_by_species.get(species, ())
        ):
            for point_index, (vector_x, vector_y) in enumerate(points):
                pressure_bar = pressure_bar_from_vector_x(vector_x, spec)
                log10_mixing_ratio = log10_mixing_ratio_from_vector_y(
                    vector_y
                )
                rows.append(
                    {
                        "reference_contract": (
                            "published_vector_plot_digitization"
                        ),
                        "paper_figure": spec.figure,
                        "paper_page": spec.page,
                        "paper_panel": spec.panel,
                        "case_id": spec.case_id,
                        "include_sio_s": str(spec.include_sio_s).lower(),
                        "quantity": "gas_mixing_ratio",
                        "species": species,
                        "segment_index": segment_index,
                        "transport_regime": regime,
                        "point_index": point_index,
                        "vector_x": vector_x,
                        "vector_y": vector_y,
                        "log10_pressure_bar": math.log10(pressure_bar),
                        "pressure_bar": pressure_bar,
                        "log10_mixing_ratio": log10_mixing_ratio,
                        "mixing_ratio": 10.0**log10_mixing_ratio,
                    }
                )
    return rows


def _extract_reference(paper_pdf, panel_extractor):
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
            rows.extend(panel_extractor(svg_path, spec))
    return rows


def extract_reference(paper_pdf):
    """Return visible temperature vertices from Figures 2 and 5."""

    return _extract_reference(paper_pdf, _extract_panel)


def extract_gas_reference(paper_pdf):
    """Return visible gas mixing-ratio vertices from Figures 2 and 5."""

    return _extract_reference(paper_pdf, _extract_gas_panel)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_pdf", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--gas-output-csv", type=Path)
    return parser.parse_args()


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not args.paper_pdf.is_file():
        raise FileNotFoundError(args.paper_pdf)
    rows = extract_reference(args.paper_pdf)
    _write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} vector vertices to {args.output_csv}.")
    if args.gas_output_csv is not None:
        gas_rows = extract_gas_reference(args.paper_pdf)
        _write_csv(args.gas_output_csv, gas_rows)
        print(
            f"Wrote {len(gas_rows)} gas vector vertices to "
            f"{args.gas_output_csv}."
        )


if __name__ == "__main__":
    main()
