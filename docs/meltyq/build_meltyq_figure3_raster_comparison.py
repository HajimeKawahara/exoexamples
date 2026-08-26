"""Build the provisional MELTYQ Figure 3 raster comparison assets.

This script compares the digitized curves visible in the published raster
with the checked-in public CKD demonstration.  It is a plot-coordinate
comparison for discussion, not a goodness-of-fit calculation or a claim that
the public demonstration reproduces the unpublished MELTYQ best fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / ".cache" / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np


DEFAULT_RASTER_REFERENCE = (
    REPOSITORY_ROOT / "docs/meltyq/data/meltyq_figure3_raster_reference.csv"
)
DEFAULT_ARCHIVED_MODEL = (
    REPOSITORY_ROOT / "docs/meltyq/data/meltyq_figure3_public_ckd_model.csv"
)
DEFAULT_ARCHIVED_METADATA = (
    REPOSITORY_ROOT / "docs/meltyq/data/meltyq_figure3_public_ckd_metadata.json"
)
DEFAULT_PUBLIC_CONFIG = (
    REPOSITORY_ROOT
    / "examples/meltyq/configs/meltyq_figure3_public_demo.json"
)
DEFAULT_SUMMARY = (
    REPOSITORY_ROOT
    / "docs/meltyq/data/meltyq_figure3_raster_comparison_summary.json"
)
DEFAULT_JA_FIGURE = (
    REPOSITORY_ROOT
    / "docs/ja/meltyq/meltyq_figure3_forward_comparison_ja_files"
    / "meltyq_figure3_raster_comparison_ja.png"
)
DEFAULT_EN_FIGURE = (
    REPOSITORY_ROOT
    / "docs/en/meltyq/meltyq_figure3_forward_comparison_en_files"
    / "meltyq_figure3_raster_comparison_en.png"
)

BLACK_CURVE_ID = "meltyq_best_fit_solid_black"
TEMPERATURE_CURVE_ID = "temperature_profile"
COMPARISON_WAVELENGTH_MIN_MICRON = 0.75
COMPARISON_WAVELENGTH_MAX_MICRON = 8.0
DEEP_PROFILE_LAYERS = 64
RUN_METADATA_SCHEMA_VERSION = "meltyq.figure3.run-metadata@1"
ARCHIVED_METADATA_SCHEMA_VERSION = (
    "meltyq.figure3.documentation-forward-metadata@1"
)

COMPONENTS = {
    "molecule_CH4": {
        "model_column": "molecule_CH4_depth_ppm",
        "label": r"CH$_4$",
        "color": "#4daf4a",
        "linestyle": "-",
    },
    "molecule_H2O": {
        "model_column": "molecule_H2O_depth_ppm",
        "label": r"H$_2$O",
        "color": "#377eb8",
        "linestyle": "-",
    },
    "molecule_CO": {
        "model_column": "molecule_CO_depth_ppm",
        "label": "CO",
        "color": "#984ea3",
        "linestyle": "-",
    },
    "molecule_CO2": {
        "model_column": "molecule_CO2_depth_ppm",
        "label": r"CO$_2$",
        "color": "#e41a1c",
        "linestyle": "-",
    },
    "aerosols": {
        "model_column": "aerosols_depth_ppm",
        "label": "Aerosols",
        "color": "#ff7f00",
        "linestyle": "-",
    },
    "rayleigh_plus_cia": {
        "model_column": "rayleigh_plus_cia_depth_ppm",
        "label": "Rayleigh+CIA",
        "color": "#ff7f00",
        "linestyle": "--",
    },
}

TEXT = {
    "ja": {
        "font": "Noto Sans CJK JP",
        "suptitle": (
            "MELTYQ Figure 3 raster digitization と "
            "ExoExamples public CKD demo"
        ),
        "raw_title": "(a) Total: shiftなしの表示座標",
        "aligned_title": "(b) Total: 一つの共有定数shift後",
        "difference_title": "(c) shifted demo − digitized raster",
        "component_title": (
            "(d) Component: 同じ共有shift、個別alignmentなし"
        ),
        "tp_title": "(e) T–P profile: upper + deep atmosphere",
        "paper_total": "論文rasterのMELTYQ黒線",
        "demo_raw": "ExoExamples public CKD demo (shiftなし)",
        "demo_shifted": "ExoExamples public CKD demo (共有shift後)",
        "clean_samples": "shift決定用の非曖昧sample",
        "wavelength": "波長 (µm)",
        "depth": "Transit depth (ppm)",
        "difference": "差 (ppm)",
        "temperature": "温度 (K)",
        "pressure": "圧力 (Pa)",
        "paper_tp": "論文rasterのT–P中心線",
        "upper_tp": "public demo upper atmosphere",
        "deep_tp": "public demo deep atmosphere",
        "paper_style": "論文rasterのvisible pixel",
        "demo_style": "shift後のExoExamples scenario",
        "note": (
            "議論用のplot-coordinate比較。public CKD demoはFigure 9の丸めた"
            "posterior中央値を用い、未公開Figure 3 best fitではない。"
        ),
        "shift_text": "共有shift = {shift:+.3f} ppm",
        "metric_text": (
            "0.75–8 µmの非曖昧sample: N={count}\n"
            "表示座標RMS={rms:.3f} ppm, median|diff|={mad:.3f} ppm"
        ),
    },
    "en": {
        "font": "DejaVu Sans",
        "suptitle": (
            "MELTYQ Figure 3 raster digitization vs. "
            "ExoExamples public CKD demo"
        ),
        "raw_title": "(a) Total: displayed coordinates without a shift",
        "aligned_title": "(b) Total: after one shared constant shift",
        "difference_title": "(c) shifted demo − digitized raster",
        "component_title": (
            "(d) Components: same shared shift, no per-curve alignment"
        ),
        "tp_title": "(e) T–P profiles: upper + deep atmosphere",
        "paper_total": "MELTYQ black curve digitized from raster",
        "demo_raw": "ExoExamples public CKD demo (unshifted)",
        "demo_shifted": "ExoExamples public CKD demo (shared shift)",
        "clean_samples": "unambiguous samples used for the shift",
        "wavelength": "Wavelength (µm)",
        "depth": "Transit depth (ppm)",
        "difference": "Difference (ppm)",
        "temperature": "Temperature (K)",
        "pressure": "Pressure (Pa)",
        "paper_tp": "T–P centerline digitized from raster",
        "upper_tp": "public-demo upper atmosphere",
        "deep_tp": "public-demo deep atmosphere",
        "paper_style": "visible pixels in the published raster",
        "demo_style": "shifted ExoExamples scenario",
        "note": (
            "Discussion-only plot-coordinate comparison. The public CKD demo "
            "uses rounded Figure 9 posterior medians and is not the unpublished "
            "Figure 3 best fit."
        ),
        "shift_text": "shared shift = {shift:+.3f} ppm",
        "metric_text": (
            "Unambiguous 0.75–8 µm samples: N={count}\n"
            "display-coordinate RMS={rms:.3f} ppm, median |difference|="
            "{mad:.3f} ppm"
        ),
    },
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""

    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path.resolve())


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def normalize_repository_paths(value):
    """Replace repository-root prefixes in archived JSON values."""

    if isinstance(value, dict):
        return {
            key: normalize_repository_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_repository_paths(item) for item in value]
    if isinstance(value, str):
        prefix = str(REPOSITORY_ROOT.resolve()) + os.sep
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def archived_forward_metadata(metadata, model, model_path):
    """Build a machine-independent metadata snapshot linked to its model."""

    archived = normalize_repository_paths(metadata)
    archived["documentation_archive"] = {
        "schema_version": ARCHIVED_METADATA_SCHEMA_VERSION,
        "model_spectra": {
            "sha256": sha256_file(model_path),
            "spectral_points": int(model.size),
            "scenario_columns": list(model.dtype.names[1:]),
        },
    }
    return archived


def validate_archived_pair(metadata, model, model_path):
    """Require the archived metadata to identify the exact model CSV."""

    archive = metadata.get("documentation_archive", {})
    if archive.get("schema_version") != ARCHIVED_METADATA_SCHEMA_VERSION:
        raise ValueError("Archived forward metadata has no recognized schema.")
    model_record = archive.get("model_spectra", {})
    expected = {
        "sha256": sha256_file(model_path),
        "spectral_points": int(model.size),
        "scenario_columns": list(model.dtype.names[1:]),
    }
    if model_record != expected:
        raise ValueError("Archived forward metadata does not match the model CSV.")


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Expected a lowercase CSV boolean, got {value!r}.")


def load_raster_traces(path: Path):
    """Load the committed raster digitization grouped by curve identifier."""

    grouped: dict[str, list[dict[str, object]]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "reference_contract",
            "panel",
            "curve_id",
            "visibility",
            "pixel_x",
            "wavelength_micron",
            "transit_depth_ppm",
            "pressure_pa",
            "temperature_k",
            "estimated_uncertainty_ppm",
            "estimated_uncertainty_temperature_k",
            "ambiguous_algorithm",
            "possible_error_bar_overlap",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Raster reference CSV does not satisfy its schema.")
        for row in reader:
            if row["reference_contract"] != "published_raster_plot_digitization":
                raise ValueError("Unexpected raster reference contract.")
            parsed = {
                **row,
                "pixel_x": float(row["pixel_x"]),
                "wavelength_micron": (
                    float(row["wavelength_micron"])
                    if row["wavelength_micron"]
                    else np.nan
                ),
                "transit_depth_ppm": (
                    float(row["transit_depth_ppm"])
                    if row["transit_depth_ppm"]
                    else np.nan
                ),
                "pressure_pa": (
                    float(row["pressure_pa"]) if row["pressure_pa"] else np.nan
                ),
                "temperature_k": (
                    float(row["temperature_k"])
                    if row["temperature_k"]
                    else np.nan
                ),
                "estimated_uncertainty_ppm": (
                    float(row["estimated_uncertainty_ppm"])
                    if row["estimated_uncertainty_ppm"]
                    else np.nan
                ),
                "estimated_uncertainty_temperature_k": (
                    float(row["estimated_uncertainty_temperature_k"])
                    if row["estimated_uncertainty_temperature_k"]
                    else np.nan
                ),
                "ambiguous_algorithm": parse_bool(row["ambiguous_algorithm"]),
                "possible_error_bar_overlap": parse_bool(
                    row["possible_error_bar_overlap"]
                ),
            }
            grouped.setdefault(row["curve_id"], []).append(parsed)
    if BLACK_CURVE_ID not in grouped or TEMPERATURE_CURVE_ID not in grouped:
        raise ValueError("Raster reference is missing a required Figure 3 trace.")
    return grouped


def load_model(path: Path):
    """Load and validate the full-scenario public CKD model CSV."""

    model = np.genfromtxt(path, delimiter=",", names=True)
    required = {"wavelength_micron", "total_depth_ppm"} | {
        value["model_column"] for value in COMPONENTS.values()
    }
    if model.dtype.names is None or not required.issubset(model.dtype.names):
        raise ValueError("Model CSV does not contain every required scenario.")
    wavelength = np.asarray(model["wavelength_micron"], dtype=float)
    if (
        wavelength.ndim != 1
        or wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(np.diff(wavelength) <= 0.0)
    ):
        raise ValueError("Model wavelength must be finite and strictly increasing.")
    for name in required - {"wavelength_micron"}:
        if np.any(~np.isfinite(model[name])):
            raise ValueError(f"Model scenario {name} contains non-finite values.")
    return model


def trace_arrays(rows, x_name, y_name):
    x = np.asarray([row[x_name] for row in rows], dtype=float)
    y = np.asarray([row[y_name] for row in rows], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def difference_summary(difference):
    """Return descriptive plot-coordinate differences in ppm."""

    difference = np.asarray(difference, dtype=float)
    if difference.size == 0 or np.any(~np.isfinite(difference)):
        raise ValueError("Difference samples must be non-empty and finite.")
    return {
        "number_of_samples": int(difference.size),
        "mean_ppm": float(np.mean(difference)),
        "median_ppm": float(np.median(difference)),
        "rms_ppm": float(np.sqrt(np.mean(difference**2))),
        "median_absolute_ppm": float(np.median(np.abs(difference))),
        "maximum_absolute_ppm": float(np.max(np.abs(difference))),
    }


def temperature_difference_summary(difference):
    """Return descriptive temperature differences in kelvin."""

    difference = np.asarray(difference, dtype=float)
    if difference.size == 0 or np.any(~np.isfinite(difference)):
        raise ValueError("Temperature differences must be non-empty and finite.")
    return {
        "number_of_samples": int(difference.size),
        "mean_k": float(np.mean(difference)),
        "median_k": float(np.median(difference)),
        "rms_k": float(np.sqrt(np.mean(difference**2))),
        "median_absolute_k": float(np.median(np.abs(difference))),
        "maximum_absolute_k": float(np.max(np.abs(difference))),
    }


def pressure_centers_bar(top_bar, bottom_bar, nlayer):
    boundaries = np.logspace(np.log10(top_bar), np.log10(bottom_bar), nlayer + 1)
    return np.sqrt(boundaries[:-1] * boundaries[1:])


def public_demo_temperature_profiles(config):
    """Reconstruct the configured upper and deep T-P coordinates."""

    upper = config["upper"]
    magma = config["magma"]
    pressure_upper_bar = pressure_centers_bar(
        upper["pressure_top_bar"], upper["pressure_base_bar"], upper["nlayer"]
    )
    log10_pressure_pa = np.log10(pressure_upper_bar) + 5.0
    temperature_upper_k = np.interp(
        log10_pressure_pa,
        [-2.0, 2.0, 4.0, 6.0],
        [
            upper["temperature_1e_minus2_pa_k"],
            upper["temperature_100_pa_k"],
            upper["temperature_1e4_pa_k"],
            magma["temperature_base_k"],
        ],
    )
    window_size = int(
        upper["nlayer"] * upper["smoothing_window_percent"] / 100.0
    )
    if window_size % 2 == 0:
        window_size += 1
    if window_size > 1:
        smoothed = np.convolve(
            temperature_upper_k,
            np.ones(window_size) / window_size,
            mode="valid",
        )
        border = (temperature_upper_k.size - smoothed.size) // 2
        temperature_upper_k[border : temperature_upper_k.size - border] = smoothed

    pressure_upper_pa = pressure_upper_bar * 1.0e5
    pressure_upper_pa = np.concatenate(
        (
            [upper["pressure_top_bar"] * 1.0e5],
            pressure_upper_pa,
            [upper["pressure_base_bar"] * 1.0e5],
        )
    )
    temperature_upper_k = np.concatenate(
        (
            [upper["temperature_1e_minus2_pa_k"]],
            temperature_upper_k,
            [magma["temperature_base_k"]],
        )
    )

    pressure_deep_bar = pressure_centers_bar(
        upper["pressure_base_bar"],
        magma["pressure_melt_bar"],
        DEEP_PROFILE_LAYERS,
    )
    temperature_deep_k = np.interp(
        np.log10(pressure_deep_bar),
        [np.log10(upper["pressure_base_bar"]), np.log10(magma["pressure_melt_bar"])],
        [magma["temperature_base_k"], magma["temperature_melt_k"]],
    )
    pressure_deep_pa = np.concatenate(
        (
            [upper["pressure_base_bar"] * 1.0e5],
            pressure_deep_bar * 1.0e5,
            [magma["pressure_melt_bar"] * 1.0e5],
        )
    )
    temperature_deep_k = np.concatenate(
        (
            [magma["temperature_base_k"]],
            temperature_deep_k,
            [magma["temperature_melt_k"]],
        )
    )
    return {
        "upper_pressure_pa": pressure_upper_pa,
        "upper_temperature_k": temperature_upper_k,
        "deep_pressure_pa": pressure_deep_pa,
        "deep_temperature_k": temperature_deep_k,
        "smoothing_window_layers": window_size,
    }


def configure_wavelength_axis(axis, text):
    axis.set_xscale("log")
    axis.set_xlim(0.7, 12.0)
    axis.set_xticks([0.8, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel(text["wavelength"])
    axis.grid(alpha=0.25)


def plot_segmented(axis, rows, *, color, linestyle):
    """Plot only contiguous runs of visible raster pixels."""

    ordered = sorted(rows, key=lambda row: row["pixel_x"])
    starts = [0]
    pixel_x = np.asarray([row["pixel_x"] for row in ordered])
    starts.extend((np.flatnonzero(np.diff(pixel_x) > 1.5) + 1).tolist())
    starts.append(len(ordered))
    for start, stop in zip(starts[:-1], starts[1:]):
        segment = ordered[start:stop]
        if not segment:
            continue
        wavelength = np.asarray([row["wavelength_micron"] for row in segment])
        depth = np.asarray([row["transit_depth_ppm"] for row in segment])
        axis.plot(
            wavelength,
            depth,
            color=color,
            linestyle=linestyle,
            linewidth=3.0,
            alpha=0.38,
            solid_capstyle="round",
        )


def render_comparison(
    path,
    language,
    traces,
    model,
    selected_black,
    shared_shift_ppm,
    aligned_summary,
    temperature_profiles,
):
    """Render one localized comparison figure."""

    text = TEXT[language]
    model_wavelength = np.asarray(model["wavelength_micron"])
    black_wavelength, black_depth = trace_arrays(
        traces[BLACK_CURVE_ID], "wavelength_micron", "transit_depth_ppm"
    )
    selected_wavelength, selected_depth = trace_arrays(
        selected_black, "wavelength_micron", "transit_depth_ppm"
    )
    selected_model = np.interp(
        selected_wavelength, model_wavelength, model["total_depth_ppm"]
    )
    selected_difference = selected_model + shared_shift_ppm - selected_depth

    with plt.rc_context({"font.family": text["font"], "font.size": 10.0}):
        figure, axes = plt.subplot_mosaic(
            [
                ["raw", "aligned", "difference"],
                ["components", "components", "tp"],
            ],
            figsize=(18.0, 10.2),
            gridspec_kw={"height_ratios": [1.0, 1.1]},
        )

        raw = axes["raw"]
        raw.plot(
            black_wavelength,
            black_depth,
            color="black",
            linewidth=1.5,
            label=text["paper_total"],
        )
        raw.plot(
            model_wavelength,
            model["total_depth_ppm"],
            color="#0072b2",
            linewidth=1.2,
            label=text["demo_raw"],
        )
        raw.scatter(
            selected_wavelength,
            selected_depth,
            s=2.0,
            color="#d55e00",
            alpha=0.55,
            label=text["clean_samples"],
            zorder=3,
        )
        raw.set_title(text["raw_title"])
        raw.set_ylabel(text["depth"])
        configure_wavelength_axis(raw, text)
        raw.legend(fontsize=8)

        aligned = axes["aligned"]
        aligned.plot(
            black_wavelength,
            black_depth,
            color="black",
            linewidth=1.5,
            label=text["paper_total"],
        )
        aligned.plot(
            model_wavelength,
            model["total_depth_ppm"] + shared_shift_ppm,
            color="#d55e00",
            linewidth=1.2,
            label=text["demo_shifted"],
        )
        aligned.set_title(text["aligned_title"])
        aligned.set_ylabel(text["depth"])
        configure_wavelength_axis(aligned, text)
        aligned.text(
            0.03,
            0.04,
            text["shift_text"].format(shift=shared_shift_ppm),
            transform=aligned.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        aligned.legend(fontsize=8)

        difference = axes["difference"]
        difference.scatter(
            selected_wavelength,
            selected_difference,
            s=6.0,
            color="#d55e00",
            alpha=0.7,
        )
        difference.axhline(0.0, color="black", linewidth=0.8)
        difference.set_title(text["difference_title"])
        difference.set_ylabel(text["difference"])
        configure_wavelength_axis(difference, text)
        difference.text(
            0.03,
            0.04,
            text["metric_text"].format(
                count=aligned_summary["number_of_samples"],
                rms=aligned_summary["rms_ppm"],
                mad=aligned_summary["median_absolute_ppm"],
            ),
            transform=difference.transAxes,
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

        components = axes["components"]
        for curve_id, style in COMPONENTS.items():
            paper_rows = [
                row
                for row in traces[curve_id]
                if row["visibility"] == "visible_palette_pixels"
            ]
            plot_segmented(
                components,
                paper_rows,
                color=style["color"],
                linestyle=style["linestyle"],
            )
            components.plot(
                model_wavelength,
                model[style["model_column"]] + shared_shift_ppm,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.0,
                alpha=1.0,
            )
        components.set_ylim(2685.0, 3090.0)
        components.set_ylabel(text["depth"])
        components.set_title(text["component_title"])
        configure_wavelength_axis(components, text)
        component_handles = [
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=style["label"],
            )
            for style in COMPONENTS.values()
        ]
        source_handles = [
            Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=4.0,
                alpha=0.38,
                label=text["paper_style"],
            ),
            Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=1.0,
                label=text["demo_style"],
            ),
        ]
        component_legend = components.legend(
            handles=component_handles,
            loc="upper right",
            ncol=3,
            fontsize=8,
        )
        components.add_artist(component_legend)
        components.legend(handles=source_handles, loc="lower right", fontsize=8)

        tp = axes["tp"]
        tp_rows = traces[TEMPERATURE_CURVE_ID]
        paper_pressure, paper_temperature = trace_arrays(
            tp_rows, "pressure_pa", "temperature_k"
        )
        paper_uncertainty = np.asarray(
            [row["estimated_uncertainty_temperature_k"] for row in tp_rows]
        )
        order = np.argsort(np.asarray([row["pressure_pa"] for row in tp_rows]))
        paper_uncertainty = paper_uncertainty[order]
        tp.fill_betweenx(
            paper_pressure,
            paper_temperature - paper_uncertainty,
            paper_temperature + paper_uncertainty,
            color="black",
            alpha=0.12,
            linewidth=0.0,
        )
        tp.plot(
            paper_temperature,
            paper_pressure,
            color="black",
            linewidth=1.5,
            label=text["paper_tp"],
        )
        tp.plot(
            temperature_profiles["upper_temperature_k"],
            temperature_profiles["upper_pressure_pa"],
            color="#0072b2",
            linewidth=1.4,
            label=text["upper_tp"],
        )
        tp.plot(
            temperature_profiles["deep_temperature_k"],
            temperature_profiles["deep_pressure_pa"],
            color="#d55e00",
            linewidth=1.4,
            label=text["deep_tp"],
        )
        tp.set_yscale("log")
        tp.invert_yaxis()
        tp.set_xlabel(text["temperature"])
        tp.set_ylabel(text["pressure"])
        tp.set_title(text["tp_title"])
        tp.grid(alpha=0.25)
        tp.legend(fontsize=8)

        figure.suptitle(text["suptitle"], fontsize=15)
        figure.text(0.5, 0.012, text["note"], ha="center", fontsize=9)
        figure.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=180,
            metadata={"Software": "ExoExamples Figure 3 documentation builder"},
        )
        plt.close(figure)


def build(args):
    raster_path = args.raster_reference.resolve()
    config_path = args.public_config.resolve()
    archived_model_path = args.archived_model.resolve()
    archived_metadata_path = args.archived_metadata.resolve()
    model_input_path = args.model_output.resolve()
    metadata_input_path = args.metadata.resolve()
    update_model_archive = model_input_path != archived_model_path
    update_metadata_archive = metadata_input_path != archived_metadata_path
    if update_model_archive != update_metadata_archive:
        raise ValueError(
            "Update the archived model and metadata together by passing both "
            "--model-output and --metadata."
        )

    for required_path in (
        raster_path,
        model_input_path,
        metadata_input_path,
        config_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    traces = load_raster_traces(raster_path)
    model = load_model(model_input_path)
    metadata = load_json(metadata_input_path)
    config = load_json(config_path)
    if metadata["config"] != config:
        raise ValueError("Forward metadata config does not match the public config.")
    if metadata.get("schema_version") != RUN_METADATA_SCHEMA_VERSION:
        raise ValueError("Forward metadata has no recognized run schema.")
    if metadata["effective_opacity_mode"] != "ckd":
        raise ValueError("Raster documentation comparison requires the public CKD run.")
    if config["claim_status"] != "public_demo_not_unpublished_best_fit":
        raise ValueError("Public-demo claim status is not explicit.")
    if update_model_archive:
        source_model_record = metadata.get("outputs", {}).get(
            "model_spectra_csv",
            {},
        )
        if source_model_record.get("sha256") != sha256_file(model_input_path):
            raise ValueError(
                "Forward metadata does not identify the supplied model CSV."
            )
        archived_metadata = archived_forward_metadata(
            metadata,
            model,
            model_input_path,
        )
        archived_model_path.parent.mkdir(parents=True, exist_ok=True)
        archived_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        model_temporary = archived_model_path.with_name(
            archived_model_path.name + ".part"
        )
        metadata_temporary = archived_metadata_path.with_name(
            archived_metadata_path.name + ".part"
        )
        shutil.copyfile(model_input_path, model_temporary)
        metadata_temporary.write_text(
            json.dumps(
                archived_metadata,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(model_temporary, archived_model_path)
        os.replace(metadata_temporary, archived_metadata_path)
        model = load_model(archived_model_path)
        metadata = load_json(archived_metadata_path)

    model_source_path = archived_model_path
    metadata_path = archived_metadata_path
    validate_archived_pair(metadata, model, model_source_path)

    model_wavelength = np.asarray(model["wavelength_micron"])
    black_rows = traces[BLACK_CURVE_ID]
    selected_black = [
        row
        for row in black_rows
        if COMPARISON_WAVELENGTH_MIN_MICRON
        <= row["wavelength_micron"]
        <= COMPARISON_WAVELENGTH_MAX_MICRON
        and not row["ambiguous_algorithm"]
        and not row["possible_error_bar_overlap"]
    ]
    selected_wavelength, selected_depth = trace_arrays(
        selected_black, "wavelength_micron", "transit_depth_ppm"
    )
    selected_model = np.interp(
        selected_wavelength, model_wavelength, model["total_depth_ppm"]
    )
    raw_difference = selected_model - selected_depth
    shared_shift_ppm = -float(np.median(raw_difference))
    aligned_difference = raw_difference + shared_shift_ppm
    raw_summary = difference_summary(raw_difference)
    aligned_summary = difference_summary(aligned_difference)

    component_summaries = {}
    for curve_id, style in COMPONENTS.items():
        visible_rows = [
            row
            for row in traces[curve_id]
            if row["visibility"] == "visible_palette_pixels"
        ]
        wavelength, depth = trace_arrays(
            visible_rows, "wavelength_micron", "transit_depth_ppm"
        )
        covered = (wavelength >= model_wavelength[0]) & (
            wavelength <= model_wavelength[-1]
        )
        sampled_model = np.interp(
            wavelength[covered], model_wavelength, model[style["model_column"]]
        )
        difference = sampled_model + shared_shift_ppm - depth[covered]
        component_summaries[curve_id] = {
            "paper_visibility_contract": "visible_palette_pixels",
            "model_scenario_column": style["model_column"],
            "shared_shift_ppm": shared_shift_ppm,
            "individual_realignment_applied": False,
            "visible_samples_in_csv": len(visible_rows),
            "visible_samples_within_model_coverage": int(np.count_nonzero(covered)),
            "wavelength_min_micron": float(np.min(wavelength[covered])),
            "wavelength_max_micron": float(np.max(wavelength[covered])),
            "model_plus_shared_shift_minus_raster": difference_summary(difference),
        }

    temperature_profiles = public_demo_temperature_profiles(config)
    tp_rows = traces[TEMPERATURE_CURVE_ID]
    raster_pressure_pa = np.asarray([row["pressure_pa"] for row in tp_rows])
    raster_temperature_k = np.asarray(
        [row["temperature_k"] for row in tp_rows]
    )
    raster_unambiguous = np.asarray(
        [not row["ambiguous_algorithm"] for row in tp_rows]
    )
    tp_comparison = {}
    comparison_regions = (
        (
            "upper",
            1.0e-4,
            1.0e6,
            "upper_pressure_pa",
            "upper_temperature_k",
        ),
        (
            "deep",
            1.0e6,
            config["magma"]["pressure_melt_bar"] * 1.0e5,
            "deep_pressure_pa",
            "deep_temperature_k",
        ),
    )
    for (
        name,
        pressure_min_pa,
        pressure_max_pa,
        model_pressure_key,
        model_temperature_key,
    ) in comparison_regions:
        selected = (
            (raster_pressure_pa >= pressure_min_pa)
            & (raster_pressure_pa <= pressure_max_pa)
            & raster_unambiguous
        )
        sampled_public_demo = np.interp(
            np.log10(raster_pressure_pa[selected]),
            np.log10(temperature_profiles[model_pressure_key]),
            temperature_profiles[model_temperature_key],
        )
        temperature_difference = (
            sampled_public_demo - raster_temperature_k[selected]
        )
        tp_comparison[name] = {
            "pressure_min_pa": pressure_min_pa,
            "pressure_max_pa": pressure_max_pa,
            "ambiguous_algorithm": False,
            "interpolation_coordinate": "log10_pressure_pa",
            "public_demo_minus_digitized_raster": (
                temperature_difference_summary(temperature_difference)
            ),
        }
    ja_path = args.ja_figure.resolve()
    en_path = args.en_figure.resolve()
    for path, language in ((ja_path, "ja"), (en_path, "en")):
        render_comparison(
            path,
            language,
            traces,
            model,
            selected_black,
            shared_shift_ppm,
            aligned_summary,
            temperature_profiles,
        )

    source_outputs = {
        "model_spectra_csv": {
            "path": repository_path(model_source_path),
            "sha256": sha256_file(model_source_path),
        },
        "metadata_json": {
            "path": repository_path(metadata_path),
            "sha256": sha256_file(metadata_path),
            "created_at_utc": metadata["created_at_utc"],
        },
    }
    summary = {
        "schema_version": "meltyq.figure3.raster_comparison@1",
        "comparison_contract": (
            "provisional_plot_coordinate_comparison_not_goodness_of_fit"
        ),
        "paper": metadata["paper"],
        "claim_status": config["claim_status"],
        "inputs": {
            "raster_reference": {
                "path": repository_path(raster_path),
                "sha256": sha256_file(raster_path),
                "row_count": int(sum(len(rows) for rows in traces.values())),
            },
            "public_config": {
                "path": repository_path(config_path),
                "sha256": sha256_file(config_path),
            },
            "source_outputs": source_outputs,
            "archived_model": {
                "path": repository_path(archived_model_path),
                "sha256": sha256_file(archived_model_path),
                "spectral_points": int(model_wavelength.size),
                "scenario_columns": list(model.dtype.names[1:]),
            },
            "builder": {
                "path": repository_path(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
            "package_versions_from_forward_metadata": metadata["package_versions"],
        },
        "total_curve_comparison": {
            "difference_definition": "public_ckd_demo_minus_digitized_raster_ppm",
            "selection": {
                "curve_id": BLACK_CURVE_ID,
                "wavelength_min_micron": COMPARISON_WAVELENGTH_MIN_MICRON,
                "wavelength_max_micron": COMPARISON_WAVELENGTH_MAX_MICRON,
                "ambiguous_algorithm": False,
                "possible_error_bar_overlap": False,
                "selected_samples": len(selected_black),
                "all_digitized_black_samples": len(black_rows),
                "median_estimated_digitization_uncertainty_ppm": float(
                    np.median(
                        [row["estimated_uncertainty_ppm"] for row in selected_black]
                    )
                ),
            },
            "unshifted_descriptive_difference": raw_summary,
            "shared_vertical_shift": {
                "formula": "-median(public_ckd_demo_minus_digitized_raster)",
                "shift_added_to_every_public_demo_scenario_ppm": shared_shift_ppm,
                "individual_component_realignment_applied": False,
            },
            "after_shared_shift_descriptive_difference": aligned_summary,
        },
        "component_curve_comparison": component_summaries,
        "temperature_pressure_comparison": {
            "difference_definition": (
                "public_demo_temperature_minus_digitized_raster_temperature_k"
            ),
            "published_raster_curve_id": TEMPERATURE_CURVE_ID,
            "published_digitized_samples": len(tp_rows),
            "public_upper_layers": config["upper"]["nlayer"],
            "public_upper_smoothing_window_layers": temperature_profiles[
                "smoothing_window_layers"
            ],
            "public_deep_layers": DEEP_PROFILE_LAYERS,
            "deep_temperature_rule": (
                "linear in log10 pressure between the 10-bar base and melt interface"
            ),
            "regions": tp_comparison,
        },
        "outputs": {
            "ja_figure": {
                "path": repository_path(ja_path),
                "sha256": sha256_file(ja_path),
            },
            "en_figure": {
                "path": repository_path(en_path),
                "sha256": sha256_file(en_path),
            },
        },
        "interpretation_limits": [
            "The published curves are coordinates digitized from a raster image, not author numerical tables.",
            "The public CKD demo uses rounded Figure 9 posterior medians and is not the unpublished Figure 3 best fit.",
            "The shared vertical shift is a display-coordinate alignment, not a fitted physical parameter.",
            "Descriptive differences are not likelihoods or goodness-of-fit statistics.",
            "Published component summaries use only visible palette pixels and cannot recover curves clipped below the plot boundary.",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raster-reference", type=Path, default=DEFAULT_RASTER_REFERENCE)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_ARCHIVED_MODEL)
    parser.add_argument("--archived-model", type=Path, default=DEFAULT_ARCHIVED_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_ARCHIVED_METADATA)
    parser.add_argument(
        "--archived-metadata",
        type=Path,
        default=DEFAULT_ARCHIVED_METADATA,
    )
    parser.add_argument("--public-config", type=Path, default=DEFAULT_PUBLIC_CONFIG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ja-figure", type=Path, default=DEFAULT_JA_FIGURE)
    parser.add_argument("--en-figure", type=Path, default=DEFAULT_EN_FIGURE)
    return parser.parse_args()


def main():
    summary = build(parse_args())
    total = summary["total_curve_comparison"]
    print("Built MELTYQ Figure 3 raster comparison assets.")
    print(
        "  Shared shift: "
        f"{total['shared_vertical_shift']['shift_added_to_every_public_demo_scenario_ppm']:.6f} ppm"
    )
    print(
        "  Shifted total display-coordinate RMS: "
        f"{total['after_shared_shift_descriptive_difference']['rms_ppm']:.6f} ppm"
    )


if __name__ == "__main__":
    main()
