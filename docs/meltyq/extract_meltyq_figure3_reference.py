"""Digitize visible MELTYQ curves from the published Figure 3 PNG.

The output is a source-artwork measurement, not the authors' intrinsic
numerical spectrum.  Axis calibration and trace bounds are intentionally
specific to ``f3_Combined_k2-18b.png`` from the arXiv source bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_SHA256 = "3ca19cbe480878a8bf67d022cbe2eb6f0caa14733187c6b929a057886575ebe7"
SOURCE_SIZE = (3597, 1494)
REFERENCE_CONTRACT = "published_raster_plot_digitization"
BEST_FIT_CURVE_ID = "meltyq_best_fit_solid_black"

# Pixel centers and printed tick values in the top-left spectral panel.
# Wavelength is logarithmic; transit depth is linear.
WAVELENGTH_TICKS = (
    (382.5, 0.8),
    (580.0, 1.0),
    (1194.0, 2.0),
    (1554.0, 3.0),
    (1808.0, 4.0),
    (2005.0, 5.0),
    (2619.0, 10.0),
)
DEPTH_TICKS = (
    (193.0, 3200.0),
    (335.5, 3100.0),
    (478.0, 3000.0),
    (620.5, 2900.0),
    (763.0, 2800.0),
)

# Pixel centers and printed tick values in the top-right temperature panel.
TEMPERATURE_TICKS = (
    (3030.0, 0.0),
    (3264.0, 2000.0),
    (3498.0, 4000.0),
)
PRESSURE_TICKS = tuple(
    (50.5 + index * 855.0 / 7.0, 10.0 ** (-4.0 + 2.0 * index))
    for index in range(8)
)

TRACE_X_MIN = 270
TRACE_X_MAX = 2848
TRACE_Y_MIN = 420
TRACE_Y_MAX = 710
TRACE_MAX_VERTICAL_STEP = 10
TRACE_PRIMARY_PENALTY = 0.025
TRACE_AUDIT_PENALTIES = (0.015, TRACE_PRIMARY_PENALTY, 0.04)
TRACE_CENTER_HALF_WINDOW = 6
DARK_THRESHOLD = 64
AMBIGUOUS_SPREAD_PIXEL = 2.0
ERROR_BAR_SPAN_PIXEL = 30

COMPONENT_Y_MIN = 350
COMPONENT_Y_MAX = 904
COMPONENT_COLORS = {
    "molecule_CH4": (77, 175, 74),
    "molecule_H2O": (55, 126, 184),
    "molecule_CO": (152, 78, 163),
    "molecule_CO2": (228, 26, 28),
}
ORANGE = (255, 127, 0)
AEROSOL_Y_MIN = 717
AEROSOL_Y_MAX = 721
RAYLEIGH_CIA_EXCLUSION_MIN = 714
RAYLEIGH_CIA_EXCLUSION_MAX = 724

TEMPERATURE_X_MIN = 3040
TEMPERATURE_X_MAX = 3490
TEMPERATURE_Y_MIN = 50
TEMPERATURE_Y_MAX = 905


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear_calibration(ticks, logarithmic_value=False):
    pixel = np.asarray([item[0] for item in ticks], dtype=float)
    value = np.asarray([item[1] for item in ticks], dtype=float)
    if logarithmic_value:
        value = np.log10(value)
    slope, intercept = np.polyfit(pixel, value, 1)
    reconstructed_pixel = (value - intercept) / slope
    maximum_residual_pixel = float(np.max(np.abs(reconstructed_pixel - pixel)))
    if maximum_residual_pixel > 1.1:
        raise RuntimeError(
            "Axis calibration is inconsistent with the printed ticks: "
            f"maximum residual is {maximum_residual_pixel:.3f} pixel."
        )
    return float(slope), float(intercept)


def _ordered_steps(maximum_step: int) -> np.ndarray:
    steps = [0]
    for magnitude in range(1, maximum_step + 1):
        steps.extend((-magnitude, magnitude))
    return np.asarray(steps, dtype=np.int8)


def _trace_dark_curve(rgb: np.ndarray, penalty_per_pixel: float) -> np.ndarray:
    """Return one minimum-cost top-to-bottom pixel index per image column."""

    crop = rgb[
        TRACE_Y_MIN : TRACE_Y_MAX + 1,
        TRACE_X_MIN : TRACE_X_MAX + 1,
    ]
    darkness_cost = np.max(crop, axis=2).astype(float) / 255.0
    height, width = darkness_cost.shape
    steps = _ordered_steps(TRACE_MAX_VERTICAL_STEP)
    transition_penalty = penalty_per_pixel * np.abs(steps)

    previous = darkness_cost[:, 0].copy()
    back_step = np.empty((width, height), dtype=np.int8)
    back_step[0] = 0
    row_index = np.arange(height)

    for column in range(1, width):
        candidates = np.full((steps.size, height), np.inf)
        for step_index, step_value in enumerate(steps):
            step = int(step_value)
            if step < 0:
                candidates[step_index, :step] = (
                    previous[-step:] + transition_penalty[step_index]
                )
            elif step > 0:
                candidates[step_index, step:] = (
                    previous[:-step] + transition_penalty[step_index]
                )
            else:
                candidates[step_index] = previous
        selected = np.argmin(candidates, axis=0)
        previous = (
            darkness_cost[:, column]
            + candidates[selected, row_index]
        )
        back_step[column] = steps[selected]

    trace = np.empty(width, dtype=np.int32)
    trace[-1] = int(np.argmin(previous))
    for column in range(width - 1, 0, -1):
        trace[column - 1] = trace[column] - int(
            back_step[column, trace[column]]
        )
    return trace.astype(float) + TRACE_Y_MIN


def _center_dark_stroke(rgb: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """Center a path on the local dark stroke without following error bars."""

    centered = trace.copy()
    for offset, pixel_x in enumerate(range(TRACE_X_MIN, TRACE_X_MAX + 1)):
        center = int(round(trace[offset]))
        lower = max(TRACE_Y_MIN, center - TRACE_CENTER_HALF_WINDOW)
        upper = min(TRACE_Y_MAX, center + TRACE_CENTER_HALF_WINDOW)
        local_rgb_max = np.max(rgb[lower : upper + 1, pixel_x], axis=1)
        dark_rows = np.flatnonzero(local_rgb_max <= DARK_THRESHOLD) + lower
        if dark_rows.size:
            centered[offset] = float(np.median(dark_rows))
    return centered


FIELDNAMES = (
    "reference_contract",
    "panel",
    "curve_id",
    "extraction_method",
    "visibility",
    "source_rgb",
    "point_index",
    "pixel_x",
    "pixel_y",
    "wavelength_micron",
    "transit_depth_ppm",
    "pressure_pa",
    "temperature_k",
    "trace_spread_pixel",
    "estimated_uncertainty_ppm",
    "estimated_uncertainty_temperature_k",
    "estimated_uncertainty_log10_pressure",
    "matching_pixel_count",
    "interference_span_pixel",
    "ambiguous_algorithm",
    "possible_error_bar_overlap",
)


def _row(**values):
    row = {field: "" for field in FIELDNAMES}
    row.update(values)
    return row


def _best_fit_rows(
    rgb,
    wavelength_slope,
    wavelength_intercept,
    depth_slope,
    depth_intercept,
):
    audit_traces = np.vstack(
        [
            _center_dark_stroke(rgb, _trace_dark_curve(rgb, penalty))
            for penalty in TRACE_AUDIT_PENALTIES
        ]
    )
    primary_index = TRACE_AUDIT_PENALTIES.index(TRACE_PRIMARY_PENALTY)
    pixel_y = audit_traces[primary_index]
    trace_spread = np.ptp(audit_traces, axis=0)
    pixel_x = np.arange(TRACE_X_MIN, TRACE_X_MAX + 1, dtype=int)

    crop_dark = np.max(
        rgb[
            TRACE_Y_MIN : TRACE_Y_MAX + 1,
            TRACE_X_MIN : TRACE_X_MAX + 1,
        ],
        axis=2,
    ) <= DARK_THRESHOLD
    dark_span = np.zeros(pixel_x.size, dtype=int)
    matching_pixel_count = np.zeros(pixel_x.size, dtype=int)
    for column in range(pixel_x.size):
        dark_rows = np.flatnonzero(crop_dark[:, column])
        if dark_rows.size:
            dark_span[column] = int(dark_rows[-1] - dark_rows[0])
        center = int(round(pixel_y[column])) - TRACE_Y_MIN
        lower = max(0, center - TRACE_CENTER_HALF_WINDOW)
        upper = min(crop_dark.shape[0] - 1, center + TRACE_CENTER_HALF_WINDOW)
        matching_pixel_count[column] = int(
            np.count_nonzero(crop_dark[lower : upper + 1, column])
        )

    wavelength_micron = 10.0 ** (
        wavelength_slope * pixel_x + wavelength_intercept
    )
    transit_depth_ppm = depth_slope * pixel_y + depth_intercept
    ppm_per_pixel = abs(depth_slope)
    uncertainty_ppm = np.maximum(2.0, trace_spread) * ppm_per_pixel
    ambiguous = trace_spread > AMBIGUOUS_SPREAD_PIXEL
    possible_error_bar_overlap = dark_span > ERROR_BAR_SPAN_PIXEL

    if not np.all(np.diff(wavelength_micron) > 0.0):
        raise RuntimeError("Digitized wavelength samples are not increasing.")
    if np.any((transit_depth_ppm < 2700.0) | (transit_depth_ppm > 3300.0)):
        raise RuntimeError("Digitized transit depths leave the plotted range.")
    if np.mean(~ambiguous) < 0.85:
        raise RuntimeError("Too few stable columns were recovered from the raster.")

    result = []
    for point_index in range(pixel_x.size):
        result.append(
            _row(
                reference_contract=REFERENCE_CONTRACT,
                panel="spectrum",
                curve_id=BEST_FIT_CURVE_ID,
                extraction_method="dark_continuity_trace",
                visibility="continuity_traced",
                source_rgb="#000000",
                point_index=point_index,
                pixel_x=int(pixel_x[point_index]),
                pixel_y=f"{pixel_y[point_index]:.3f}",
                wavelength_micron=f"{wavelength_micron[point_index]:.9f}",
                transit_depth_ppm=f"{transit_depth_ppm[point_index]:.6f}",
                trace_spread_pixel=f"{trace_spread[point_index]:.3f}",
                estimated_uncertainty_ppm=(
                    f"{uncertainty_ppm[point_index]:.6f}"
                ),
                matching_pixel_count=int(matching_pixel_count[point_index]),
                interference_span_pixel=int(dark_span[point_index]),
                ambiguous_algorithm=str(bool(ambiguous[point_index])).lower(),
                possible_error_bar_overlap=str(
                    bool(possible_error_bar_overlap[point_index])
                ).lower(),
            )
        )
    return result


def _palette_rows(
    rgb,
    curve_id,
    color,
    wavelength_slope,
    wavelength_intercept,
    depth_slope,
    depth_intercept,
    row_filter=None,
):
    result = []
    color = np.asarray(color, dtype=np.uint8)
    ppm_per_pixel = abs(depth_slope)
    for pixel_x in range(TRACE_X_MIN, TRACE_X_MAX + 1):
        column = rgb[COMPONENT_Y_MIN : COMPONENT_Y_MAX + 1, pixel_x]
        pixel_rows = (
            np.flatnonzero(np.all(column == color, axis=1)) + COMPONENT_Y_MIN
        )
        if row_filter is not None:
            pixel_rows = pixel_rows[row_filter(pixel_rows)]
        if not pixel_rows.size:
            continue
        pixel_y = float(np.median(pixel_rows))
        span = int(pixel_rows[-1] - pixel_rows[0])
        wavelength = 10.0 ** (
            wavelength_slope * pixel_x + wavelength_intercept
        )
        depth = depth_slope * pixel_y + depth_intercept
        result.append(
            _row(
                reference_contract=REFERENCE_CONTRACT,
                panel="spectrum",
                curve_id=curve_id,
                extraction_method="exact_palette_median",
                visibility="visible_palette_pixels",
                source_rgb="#" + "".join(f"{value:02x}" for value in color),
                point_index=len(result),
                pixel_x=pixel_x,
                pixel_y=f"{pixel_y:.3f}",
                wavelength_micron=f"{wavelength:.9f}",
                transit_depth_ppm=f"{depth:.6f}",
                trace_spread_pixel=f"{span:.3f}",
                estimated_uncertainty_ppm=(
                    f"{2.0 * ppm_per_pixel:.6f}"
                ),
                matching_pixel_count=int(pixel_rows.size),
                interference_span_pixel=span,
                ambiguous_algorithm="false",
                possible_error_bar_overlap="false",
            )
        )
    return result


def _temperature_profile_rows(
    rgb,
    temperature_slope,
    temperature_intercept,
    log_pressure_slope,
    log_pressure_intercept,
):
    result = []
    temperature_per_pixel = abs(temperature_slope)
    pressure_uncertainty_dex = 0.5 * abs(log_pressure_slope)
    for pixel_y in range(TEMPERATURE_Y_MIN, TEMPERATURE_Y_MAX + 1):
        column = rgb[
            pixel_y,
            TEMPERATURE_X_MIN : TEMPERATURE_X_MAX + 1,
        ]
        pixel_columns = (
            np.flatnonzero(np.max(column, axis=1) <= DARK_THRESHOLD)
            + TEMPERATURE_X_MIN
        )
        if not pixel_columns.size:
            continue
        pixel_x = float(np.median(pixel_columns))
        span = int(pixel_columns[-1] - pixel_columns[0])
        temperature = temperature_slope * pixel_x + temperature_intercept
        pressure = 10.0 ** (
            log_pressure_slope * pixel_y + log_pressure_intercept
        )
        result.append(
            _row(
                reference_contract=REFERENCE_CONTRACT,
                panel="temperature_pressure",
                curve_id="temperature_profile",
                extraction_method="dark_row_median",
                visibility="visible_dark_pixels",
                source_rgb="#000000",
                point_index=len(result),
                pixel_x=f"{pixel_x:.3f}",
                pixel_y=pixel_y,
                pressure_pa=f"{pressure:.9e}",
                temperature_k=f"{temperature:.6f}",
                estimated_uncertainty_temperature_k=(
                    f"{max(2.0, 0.5 * span) * temperature_per_pixel:.6f}"
                ),
                estimated_uncertainty_log10_pressure=(
                    f"{pressure_uncertainty_dex:.9f}"
                ),
                matching_pixel_count=int(pixel_columns.size),
                interference_span_pixel=span,
                ambiguous_algorithm=str(span > 10).lower(),
                possible_error_bar_overlap="false",
            )
        )
    pressure = np.asarray([float(row["pressure_pa"]) for row in result])
    temperature = np.asarray([float(row["temperature_k"]) for row in result])
    if not np.all(np.diff(pressure) > 0.0):
        raise RuntimeError("Digitized temperature-profile pressure is not increasing.")
    if np.any((temperature < 0.0) | (temperature > 4200.0)):
        raise RuntimeError("Digitized temperatures leave the plotted range.")
    return result


def extract_reference(source_png: Path):
    """Return calibrated rows for visible curves in the published raster."""

    if not source_png.is_file():
        raise FileNotFoundError(source_png)
    actual_sha256 = _sha256(source_png)
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(
            "Unexpected Figure 3 source SHA-256: "
            f"expected {SOURCE_SHA256}, found {actual_sha256}."
        )

    with Image.open(source_png) as image:
        if image.size != SOURCE_SIZE:
            raise ValueError(
                f"Unexpected Figure 3 dimensions: expected {SOURCE_SIZE}, "
                f"found {image.size}."
            )
        rgb = np.asarray(image.convert("RGB"))

    wavelength_slope, wavelength_intercept = _linear_calibration(
        WAVELENGTH_TICKS,
        logarithmic_value=True,
    )
    depth_slope, depth_intercept = _linear_calibration(DEPTH_TICKS)
    temperature_slope, temperature_intercept = _linear_calibration(
        TEMPERATURE_TICKS
    )
    log_pressure_slope, log_pressure_intercept = _linear_calibration(
        PRESSURE_TICKS,
        logarithmic_value=True,
    )

    result = _best_fit_rows(
        rgb,
        wavelength_slope,
        wavelength_intercept,
        depth_slope,
        depth_intercept,
    )
    for curve_id, color in COMPONENT_COLORS.items():
        result.extend(
            _palette_rows(
                rgb,
                curve_id,
                color,
                wavelength_slope,
                wavelength_intercept,
                depth_slope,
                depth_intercept,
            )
        )
    result.extend(
        _palette_rows(
            rgb,
            "aerosols",
            ORANGE,
            wavelength_slope,
            wavelength_intercept,
            depth_slope,
            depth_intercept,
            row_filter=lambda rows: (
                (rows >= AEROSOL_Y_MIN) & (rows <= AEROSOL_Y_MAX)
            ),
        )
    )
    result.extend(
        _palette_rows(
            rgb,
            "rayleigh_plus_cia",
            ORANGE,
            wavelength_slope,
            wavelength_intercept,
            depth_slope,
            depth_intercept,
            row_filter=lambda rows: (
                (rows < RAYLEIGH_CIA_EXCLUSION_MIN)
                | (rows > RAYLEIGH_CIA_EXCLUSION_MAX)
            ),
        )
    )
    result.extend(
        _temperature_profile_rows(
            rgb,
            temperature_slope,
            temperature_intercept,
            log_pressure_slope,
            log_pressure_intercept,
        )
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_png", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = extract_reference(args.source_png)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    ambiguous_count = sum(row["ambiguous_algorithm"] == "true" for row in rows)
    overlap_count = sum(
        row["possible_error_bar_overlap"] == "true" for row in rows
    )
    print(
        f"Wrote {len(rows)} raster samples to {args.output_csv}; "
        f"algorithm-ambiguous={ambiguous_count}, "
        f"possible-error-bar-overlap={overlap_count}."
    )


if __name__ == "__main__":
    main()
