"""Extract plotted MELTYQ Figure 8 traces from the published vector PDFs."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


PANEL_SPECS = {
    "a": ("f8a_dep_pmelt_t.pdf", "pressure_melt", 3.0, 5.0),
    "b": ("f8b_dep_tmelt_t.pdf", "temperature_melt", 2000.0, 4000.0),
    "c": ("f8c_dep_fo2_t.pdf", "oxygen_fugacity", -10.0, 0.0),
    "d": ("f8d_dep_comelt_t.pdf", "co_melt", -10.0, -4.0),
    "e": ("f8e_dep_nmelt_t.pdf", "n_melt", -10.0, -4.0),
    "f": ("f8f_dep_tb_t.pdf", "temperature_base", 300.0, 2000.0),
}

COLOR_TO_SPECIES = {
    "58.006287%,0%,83.006287%": "H2",
    "0%,61.914062%,44.993591%": "O2",
    "34.008789%,71.09375%,91.014099%": "H2O",
    "90.039062%,61.914062%,0%": "CO",
    "93.943787%,89.0625%,26.000977%": "CO2",
    "0%,44.993591%,69.921875%": "CH4",
    "65.039062%,15.989685%,15.989685%": "N2",
    "100%,75%,75%": "NH3",
    "75%,75%,75%": "radius",
}

SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
COLOR_PATTERN = re.compile(r"stroke:rgb\(([^)]+)\)")
POINT_PATTERN = re.compile(r"([ML])\s*(-?[0-9.]+)\s+(-?[0-9.]+)")

PLOT_X_MIN = 503.984375
PLOT_X_MAX = 2549.492188
PLOT_Y_MIN = 280.0
PLOT_Y_MAX = 2239.492188


def _linear_map(value, source_min, source_max, target_min, target_max):
    fraction = (value - source_min) / (source_max - source_min)
    return target_min + fraction * (target_max - target_min)


def _extract_panel(svg_path, panel, sweep, x_min, x_max):
    root = ET.parse(svg_path).getroot()
    rows = []
    curve_count = 0
    for path in root.iter(f"{SVG_NAMESPACE}path"):
        style = path.attrib.get("style", "")
        color_match = COLOR_PATTERN.search(style)
        if color_match is None or "stroke-width:12.5" not in style:
            continue
        species = COLOR_TO_SPECIES.get(color_match.group(1))
        if species is None:
            continue
        level = "radius"
        if species != "radius":
            level = "base" if "stroke-dasharray" in style else "surface"
        points = [
            (float(x), float(y))
            for _, x, y in POINT_PATTERN.findall(path.attrib["d"])
            if PLOT_X_MIN - 1.0 <= float(x) <= PLOT_X_MAX + 1.0
        ]
        if len(points) < 2:
            raise RuntimeError(
                f"Could not identify the plotted trace for {panel=} "
                f"{species=} {level=}."
            )
        curve_count += 1
        for point_index, (plot_x, plot_y) in enumerate(points):
            x = _linear_map(
                plot_x,
                PLOT_X_MIN,
                PLOT_X_MAX,
                x_min,
                x_max,
            )
            if level == "radius":
                value = _linear_map(
                    plot_y,
                    PLOT_Y_MIN,
                    PLOT_Y_MAX,
                    1.4,
                    2.4,
                )
            else:
                value = _linear_map(
                    plot_y,
                    PLOT_Y_MIN,
                    PLOT_Y_MAX,
                    -10.0,
                    0.0,
                )
            rows.append(
                {
                    "panel": panel,
                    "sweep": sweep,
                    "level": level,
                    "species": species,
                    "point_index": point_index,
                    "x": x,
                    "value": value,
                }
            )
    if curve_count != 16:
        raise RuntimeError(
            f"Expected 16 visible traces in panel {panel}, found "
            f"{curve_count}."
        )
    return rows


def extract_reference(source_directory):
    """Return all visible vector vertices from the six published panels."""

    rows = []
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)
        for panel, (filename, sweep, x_min, x_max) in PANEL_SPECS.items():
            pdf_path = source_directory / filename
            if not pdf_path.is_file():
                raise FileNotFoundError(pdf_path)
            svg_path = temporary_directory / f"figure8_{panel}.svg"
            subprocess.run(
                ["pdftocairo", "-svg", str(pdf_path), str(svg_path)],
                check=True,
            )
            rows.extend(
                _extract_panel(svg_path, panel, sweep, x_min, x_max)
            )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory",
        type=Path,
        help="Directory containing the six f8*_dep_*_t.pdf source files.",
    )
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = extract_reference(args.source_directory)
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
