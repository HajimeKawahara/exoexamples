"""Shared analysis helpers for the bilingual MELTYQ Figure 8 documents."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


CARBON_BASIS_FACTOR = 28.0101 / 12.0107
NITROGEN_BASIS_FACTOR = 28.0134 / 14.0067
PAPER_COMPOSITION_HALF_LINEWIDTH_DEX = 0.03190
PAPER_RADIUS_HALF_LINEWIDTH_REARTH = 0.003190
FORTNEY_CORE_RADIUS_REARTH = 1.442319865

CALCULATION_PACKAGES = (
    "jax",
    "numpy",
    "exogibbs",
    "exoeos",
    "exojax",
    "exopie",
)
SOURCE_FINGERPRINT_PACKAGES = ("exogibbs", "exoeos", "exojax", "exopie")
FORCE_REGENERATION_VARIABLE = (
    "EXOEXAMPLES_FORCE_MELTYQ_DOCUMENT_RESULTS"
)
DOCUMENT_CACHE_SCHEMA_VERSION = "meltyq.figure8.document-cache@1"
DOCUMENT_CACHE_ARTIFACTS = ("results.csv", "figure8.png")

SPECIES = ("H2", "O2", "H2O", "CO", "CO2", "CH4", "N2", "NH3")
SPECIES_COLORS = {
    "H2": "#8a00ff",
    "O2": "#009e73",
    "H2O": "#56b4e9",
    "CO": "#e69f00",
    "CO2": "#d8c900",
    "CH4": "#0072b2",
    "N2": "#a52a2a",
    "NH3": "#f4a3a8",
}
SWEEP_ORDER = (
    "pressure_melt",
    "temperature_melt",
    "oxygen_fugacity",
    "co_melt",
    "n_melt",
    "temperature_base",
)
LOG_SWEEPS = {"pressure_melt", "oxygen_fugacity", "co_melt", "n_melt"}


def find_repo_root(start=None):
    """Find the repository root from either the notebook or repository cwd."""

    start = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (start, *start.parents):
        if (
            candidate / "examples" / "meltyq" / "meltyq_figure8.py"
        ).is_file():
            return candidate
    raise RuntimeError("ExoExamples repository root was not found.")


def runtime_versions():
    """Return versions of the modules used by the running interpreter."""

    versions = {"python": sys.version.split()[0]}
    for package in CALCULATION_PACKAGES:
        try:
            module_version = getattr(
                importlib.import_module(package),
                "__version__",
                None,
            )
        except ImportError:
            module_version = None
        if module_version is not None:
            versions[package] = str(module_version)
            continue
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def runtime_source_states():
    """Fingerprint provider sources and material ExoPie data assets."""

    states = {}
    for package in SOURCE_FINGERPRINT_PACKAGES:
        module = importlib.import_module(package)
        package_directory = Path(module.__file__).resolve().parent
        digest = hashlib.sha256()
        source_files = sorted(package_directory.rglob("*.py"))
        for source_file in source_files:
            digest.update(str(source_file.relative_to(package_directory)).encode())
            digest.update(b"\0")
            digest.update(source_file.read_bytes())
            digest.update(b"\0")
        states[package] = {
            "python_source_files": len(source_files),
            "python_source_inventory_sha256": digest.hexdigest(),
        }
        if package == "exopie":
            data_files = sorted((package_directory / "Data").glob("*.pkl"))
            data_digest = hashlib.sha256()
            for data_file in data_files:
                data_digest.update(
                    str(data_file.relative_to(package_directory)).encode()
                )
                data_digest.update(b"\0")
                data_digest.update(data_file.read_bytes())
                data_digest.update(b"\0")
            states[package].update(
                {
                    "data_files": len(data_files),
                    "data_inventory_sha256": data_digest.hexdigest(),
                }
            )
    return states


def _calculation_fingerprint(repo_root, *, basis_aligned):
    """Fingerprint calculation code and installed distribution versions."""

    paths = [
        repo_root / "examples" / "meltyq" / "meltyq_figure8.py",
        repo_root / "examples" / "meltyq" / "meltyq_clear_forward.py",
    ]
    if basis_aligned:
        paths.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    digest.update(f"basis_aligned={basis_aligned}\n".encode())
    for path in paths:
        digest.update(str(path.relative_to(repo_root)).encode())
        digest.update(path.read_bytes())
    for name, package_version in sorted(runtime_versions().items()):
        digest.update(f"{name}={package_version}\n".encode())
    for name, state in sorted(runtime_source_states().items()):
        digest.update(
            f"{name}_state={json.dumps(state, sort_keys=True)}\n".encode()
        )
    return digest.hexdigest()


def _artifact_records(output_directory):
    records = {}
    for filename in DOCUMENT_CACHE_ARTIFACTS:
        path = output_directory / filename
        records[filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    return records


def _cache_matches(metadata_path, fingerprint):
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        document_cache = metadata["document_cache"]
        return (
            document_cache.get("schema") == DOCUMENT_CACHE_SCHEMA_VERSION
            and document_cache.get("calculation_fingerprint") == fingerprint
            and document_cache.get("artifacts")
            == _artifact_records(metadata_path.parent)
        )
    except (KeyError, OSError, json.JSONDecodeError):
        return False


def _generation_provenance(fingerprint, output_directory):
    return {
        "schema": DOCUMENT_CACHE_SCHEMA_VERSION,
        "calculation_fingerprint": fingerprint,
        "artifacts": _artifact_records(output_directory),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_versions": runtime_versions(),
        "runtime_source_states": runtime_source_states(),
        "fingerprint_scope": (
            "ExoExamples calculation sources, imported module versions, and "
            "the Python source inventories of ExoGibbs, ExoEOS, ExoJAX, and "
            "the external ExoPie package, including ExoPie's packaged "
            "mass-radius data tables."
        ),
    }


def _record_generation_provenance(metadata_path, fingerprint):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["document_cache"] = _generation_provenance(
        fingerprint,
        metadata_path.parent,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _run_example(repo_root, output_directory):
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault(
        "MPLCONFIGDIR",
        str(repo_root / ".cache" / "matplotlib"),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "examples" / "meltyq" / "meltyq_figure8.py"),
            "--output-directory",
            str(output_directory),
        ],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip().splitlines()[-3:]


def ensure_raw_results(repo_root):
    """Generate the documented same-numeric-input run when it is absent."""

    output_directory = repo_root / "outputs" / "meltyq_figure8"
    metadata_path = output_directory / "metadata.json"
    fingerprint = _calculation_fingerprint(
        repo_root,
        basis_aligned=False,
    )
    required = (
        output_directory / "results.csv",
        metadata_path,
        output_directory / "figure8.png",
    )
    force = os.environ.get(FORCE_REGENERATION_VARIABLE) == "1"
    if force or not all(path.is_file() for path in required) or not (
        _cache_matches(metadata_path, fingerprint)
    ):
        messages = _run_example(repo_root, output_directory)
        _record_generation_provenance(metadata_path, fingerprint)
        return output_directory, True, messages
    return output_directory, False, []


def _import_model(repo_root):
    repository_directory = str(repo_root)
    if repository_directory not in sys.path:
        sys.path.insert(0, repository_directory)
    from examples.meltyq import meltyq_figure8 as model

    return model


def _paper_to_provider_parameters(parameters):
    return replace(
        parameters,
        co_melt_mole_ratio=(
            parameters.co_melt_mole_ratio * CARBON_BASIS_FACTOR
        ),
        n_melt_mole_ratio=(
            parameters.n_melt_mole_ratio * NITROGEN_BASIS_FACTOR
        ),
    )


def _run_basis_aligned_sweeps(model, context):
    paper_baseline = model.BASELINE
    provider_baseline = _paper_to_provider_parameters(paper_baseline)
    cache = {}

    def cached_solve(parameters, init):
        key = tuple(asdict(parameters).values())
        if key not in cache:
            cache[key] = model.solve_point(context, parameters, init=init)
        return cache[key]

    baseline_result, baseline_init = cached_solve(provider_baseline, None)
    if baseline_result.error or baseline_init is None:
        raise RuntimeError(baseline_result.error or "Mapped baseline failed.")

    runs = []
    for sweep in model.SWEEPS:
        paper_values = np.asarray(sweep.values)
        paper_baseline_value = getattr(paper_baseline, sweep.field)
        baseline_index = int(
            np.argmin(np.abs(paper_values - paper_baseline_value))
        )
        panel_results = [None] * len(sweep.values)
        panel_results[baseline_index] = model.SweepPoint(
            sweep=sweep,
            requested_value=sweep.values[baseline_index],
            result=baseline_result,
        )
        for indices in (
            range(baseline_index - 1, -1, -1),
            range(baseline_index + 1, len(sweep.values)),
        ):
            continuation_init = baseline_init
            for index in indices:
                requested_value = sweep.values[index]
                paper_parameters = replace(
                    paper_baseline,
                    **{sweep.field: requested_value},
                )
                provider_parameters = _paper_to_provider_parameters(
                    paper_parameters
                )
                result, candidate_init = cached_solve(
                    provider_parameters,
                    continuation_init,
                )
                if result.error:
                    raise RuntimeError(
                        f"{sweep.name}={requested_value}: {result.error}"
                    )
                panel_results[index] = model.SweepPoint(
                    sweep=sweep,
                    requested_value=requested_value,
                    result=result,
                )
                if candidate_init is not None:
                    continuation_init = candidate_init
        runs.append(panel_results)
    return runs, cache, baseline_result, provider_baseline


def ensure_basis_aligned_results(repo_root):
    """Generate the explicit paper-to-provider C/N basis-mapped sensitivity run."""

    output_directory = repo_root / "outputs" / "meltyq_figure8_basis_aligned"
    results_path = output_directory / "results.csv"
    metadata_path = output_directory / "metadata.json"
    figure_path = output_directory / "figure8.png"
    fingerprint = _calculation_fingerprint(
        repo_root,
        basis_aligned=True,
    )
    force = os.environ.get(FORCE_REGENERATION_VARIABLE) == "1"
    if not force and all(
        path.is_file() for path in (results_path, metadata_path, figure_path)
    ) and _cache_matches(metadata_path, fingerprint):
        return output_directory, False

    model = _import_model(repo_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    context = model.build_context()
    if context.thermochemistry_clipped:
        raise RuntimeError(
            "The basis-aligned run requires temperature-dependent "
            "thermochemistry."
        )
    runs, cache, baseline_result, provider_baseline = (
        _run_basis_aligned_sweeps(model, context)
    )
    model.write_results_csv(
        results_path,
        runs,
        context,
        baseline_result=baseline_result,
    )
    model.plot_results(figure_path, runs)
    payload = {
        "paper": model.PAPER_DOI,
        "purpose": (
            "Sensitivity run with the published CO and N2 melt mole-ratio "
            "axes explicitly mapped to ExoGibbs elemental-C and atomic-N "
            "dilute mole-ratio inputs."
        ),
        "paper_baseline": asdict(model.BASELINE),
        "provider_baseline": asdict(provider_baseline),
        "basis_factors": {
            "co_paper_to_elemental_c": CARBON_BASIS_FACTOR,
            "n2_paper_to_atomic_n": NITROGEN_BASIS_FACTOR,
        },
        "unique_points": len(cache),
        "rocky_radius_rearth": context.rocky_radius_rearth,
        "deep_nlayer": context.deep_nlayer,
        "max_outer_iterations": context.magma_options.max_iter,
        "document_cache": _generation_provenance(
            fingerprint,
            output_directory,
        ),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_directory, True


def load_document_data(repo_root=None):
    """Load raw, basis-aligned, metadata, and vector-reference data."""

    repo_root = find_repo_root() if repo_root is None else Path(repo_root)
    raw_directory, raw_generated, messages = ensure_raw_results(repo_root)
    aligned_directory, aligned_generated = ensure_basis_aligned_results(
        repo_root
    )
    reference_path = (
        repo_root
        / "docs"
        / "meltyq"
        / "data"
        / "meltyq_figure8_vector_reference.csv"
    )
    return {
        "repo_root": repo_root,
        "raw": pd.read_csv(raw_directory / "results.csv"),
        "aligned": pd.read_csv(aligned_directory / "results.csv"),
        "raw_metadata": json.loads(
            (raw_directory / "metadata.json").read_text(encoding="utf-8")
        ),
        "aligned_metadata": json.loads(
            (aligned_directory / "metadata.json").read_text(encoding="utf-8")
        ),
        "reference": pd.read_csv(reference_path),
        "raw_generated": raw_generated,
        "aligned_generated": aligned_generated,
        "generation_messages": messages,
    }


def _axis_value(sweep, requested_value):
    if sweep in LOG_SWEEPS:
        return np.log10(requested_value)
    return requested_value


def _reference_value(reference, sweep, species, level, x):
    curve = reference[
        (reference["sweep"] == sweep)
        & (reference["species"] == species)
        & (reference["level"] == level)
    ].sort_values("x")
    if len(curve) < 2 or x < curve["x"].min() or x > curve["x"].max():
        return np.nan
    return float(np.interp(x, curve["x"], curve["value"]))


def compare_with_reference(results, reference, label):
    """Compare a result table with visible published vector centerlines."""

    records = []
    for row in results.itertuples(index=False):
        x = _axis_value(row.sweep, row.sweep_value)
        for species in SPECIES:
            for endpoint, level, prefix in (
                ("surface", "surface", "x_melt_"),
                ("base", "base", "x_10bar_"),
            ):
                paper_value = _reference_value(
                    reference,
                    row.sweep,
                    species,
                    level,
                    x,
                )
                mole_fraction = getattr(row, f"{prefix}{species}")
                model_value = (
                    np.log10(mole_fraction)
                    if np.isfinite(mole_fraction) and mole_fraction > 0.0
                    else np.nan
                )
                visible = bool(
                    np.isfinite(paper_value)
                    and np.isfinite(model_value)
                    and paper_value > -9.95
                )
                records.append(
                    {
                        "label": label,
                        "sweep": row.sweep,
                        "sweep_value": row.sweep_value,
                        "x": x,
                        "species": species,
                        "endpoint": endpoint,
                        "paper": paper_value,
                        "model": model_value,
                        "delta": model_value - paper_value,
                        "visible": visible,
                    }
                )
        paper_radius = _reference_value(
            reference,
            row.sweep,
            "radius",
            "radius",
            x,
        )
        inverse_column = (
            1.0 / row.rocky_radius_rearth
            - 1.0 / row.radius_base_rearth
        )
        fortney_core_radius = 1.0 / (
            1.0 / FORTNEY_CORE_RADIUS_REARTH - inverse_column
        )
        records.append(
            {
                "label": label,
                "sweep": row.sweep,
                "sweep_value": row.sweep_value,
                "x": x,
                "species": "radius",
                "endpoint": "radius",
                "paper": paper_radius,
                "model": row.radius_base_rearth,
                "delta": row.radius_base_rearth - paper_radius,
                "counterfactual_model": fortney_core_radius,
                "counterfactual_delta": fortney_core_radius - paper_radius,
                "visible": np.isfinite(paper_radius),
            }
        )
    return pd.DataFrame.from_records(records)


def build_comparison(data):
    return pd.concat(
        [
            compare_with_reference(data["raw"], data["reference"], "raw"),
            compare_with_reference(
                data["aligned"],
                data["reference"],
                "basis-aligned",
            ),
        ],
        ignore_index=True,
    )


def _set_plot_language(language):
    if language == "ja":
        plt.rcParams["font.family"] = "Noto Sans CJK JP"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"


def plot_architecture(language="en"):
    """Plot the package ownership and forward data flow."""

    _set_plot_language(language)
    labels = {
        "ja": (
            "論文座標・sweep\nExoExamples",
            "magma–gas境界\nExoGibbs\n(fugacity: ExoEOS)",
            "log P grid・T(log P)\nExoJAX / ExoExamples",
            "平衡profile・厳密10 bar\nExoGibbs",
            "混合密度\nExoEOS",
            "静水圧半径\nExoJAX\n(Rrock: 外部ExoPie)",
            "CSV・診断・図\nExoExamples",
        ),
        "en": (
            "Paper coordinates & sweeps\nExoExamples",
            "Magma–gas boundary\nExoGibbs\n(fugacity: ExoEOS)",
            "log P grid & T(log P)\nExoJAX / ExoExamples",
            "Equilibrium profile & exact 10 bar\nExoGibbs",
            "Mixture density\nExoEOS",
            "Hydrostatic radius\nExoJAX\n(Rrock: external ExoPie)",
            "CSV, diagnostics & plots\nExoExamples",
        ),
    }[language]
    title = {
        "ja": "Figure 8 forward comparisonの責務とdata flow",
        "en": "Responsibilities and data flow in the Figure 8 forward comparison",
    }[language]
    colors = ("#e8f1fb", "#efe6fa", "#e8f1fb", "#efe6fa", "#e7f5eb", "#e8f1fb", "#e8f1fb")
    figure, axis = plt.subplots(figsize=(16, 3.3))
    axis.set_xlim(0.0, 16.0)
    axis.set_ylim(0.0, 3.0)
    axis.axis("off")
    width = 1.85
    x_positions = np.linspace(0.15, 14.0, len(labels))
    for index, (x_position, label, color) in enumerate(
        zip(x_positions, labels, colors)
    ):
        box = FancyBboxPatch(
            (x_position, 0.85),
            width,
            1.2,
            boxstyle="round,pad=0.04",
            facecolor=color,
            edgecolor="0.3",
            linewidth=1.2,
        )
        axis.add_patch(box)
        axis.text(
            x_position + width / 2.0,
            1.45,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )
        if index:
            axis.annotate(
                "",
                xy=(x_position, 1.45),
                xytext=(x_positions[index - 1] + width, 1.45),
                arrowprops={"arrowstyle": "->", "color": "0.25"},
            )
    axis.set_title(title, fontsize=14, pad=8)
    figure.tight_layout()
    return figure


def plot_forward_curves(results, language="en"):
    """Plot the current same-numeric-input results from the CSV table."""

    _set_plot_language(language)
    titles = {
        "ja": (
            r"(a) $P_{melt}$ 依存性",
            r"(b) $T_{melt}$ 依存性",
            r"(c) $f_{O_2}$ 依存性",
            "(d) provider elemental-C入力依存性",
            "(e) provider atomic-N入力依存性",
            r"(f) $T_b$ 依存性",
        ),
        "en": (
            r"(a) Dependence on $P_{melt}$",
            r"(b) Dependence on $T_{melt}$",
            r"(c) Dependence on $f_{O_2}$",
            "(d) Dependence on provider elemental-C input",
            "(e) Dependence on provider atomic-N input",
            r"(f) Dependence on $T_b$",
        ),
    }[language]
    xlabels = (
        r"Log($P_{melt}$ [bar])",
        r"$T_{melt}$ [K]",
        r"Log($f_{O_2}$ [bar])",
        r"Log($x_{C,provider}$)",
        r"Log($x_{N,provider}$)",
        r"$T_b$ [K]",
    )
    y_label = "Log(モル分率)" if language == "ja" else "Log(Mole fraction)"
    radius_label = (
        r"10 bar半径 [$R_\oplus$]"
        if language == "ja"
        else r"Radius at 10 bar [$R_\oplus$]"
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    for axis, sweep, title, xlabel in zip(
        axes.flat,
        SWEEP_ORDER,
        titles,
        xlabels,
    ):
        panel = results[results["sweep"] == sweep].sort_values("sweep_value")
        raw_x = panel["sweep_value"].to_numpy()
        x = np.log10(raw_x) if sweep in LOG_SWEEPS else raw_x
        for species in SPECIES:
            surface = np.log10(
                np.clip(panel[f"x_melt_{species}"].to_numpy(), 1.0e-10, 1.0)
            )
            base = np.log10(
                np.clip(panel[f"x_10bar_{species}"].to_numpy(), 1.0e-10, 1.0)
            )
            axis.plot(x, surface, color=SPECIES_COLORS[species], label=species)
            axis.plot(x, base, color=SPECIES_COLORS[species], linestyle=":")
        axis.set(title=title, xlabel=xlabel, ylabel=y_label, ylim=(-10.0, 0.0))
        radius_axis = axis.twinx()
        radius_axis.plot(
            x,
            panel["radius_base_rearth"],
            color="0.65",
            linewidth=2.0,
        )
        radius_axis.set(ylabel=radius_label, ylim=(1.4, 2.4))
        axis.legend(loc="lower left", ncol=2, fontsize=7)
    figure.suptitle(
        "ExoExamples workflow: same-numeric-input forward comparison"
        if language == "en"
        else "ExoExamples workflow：同数値入力forward comparison",
        fontsize=15,
    )
    figure.tight_layout()
    return figure


def plot_basis_mapping_effect(comparison, language="en"):
    """Plot the effect of making the published and provider bases explicit."""

    _set_plot_language(language)
    chemistry = comparison[
        (comparison["endpoint"] != "radius") & comparison["visible"]
    ].copy()
    chemistry["absolute_delta"] = chemistry["delta"].abs()
    chemistry_summary = (
        chemistry.groupby(["label", "species"])["absolute_delta"]
        .median()
        .unstack("label")
        .reindex(SPECIES)
    )
    radius = comparison[comparison["endpoint"] == "radius"].copy()
    radius["absolute_delta"] = radius["delta"].abs()
    radius_summary = radius.groupby("label")["absolute_delta"].median()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(SPECIES))
    width = 0.38
    axes[0].bar(
        positions - width / 2.0,
        chemistry_summary["raw"],
        width,
        label="Same numeric input" if language == "en" else "同数値入力",
        color="0.65",
    )
    axes[0].bar(
        positions + width / 2.0,
        chemistry_summary["basis-aligned"],
        width,
        label="Basis aligned" if language == "en" else "入力基底整合後",
        color="#0072b2",
    )
    axes[0].set(
        xticks=positions,
        xticklabels=SPECIES,
        yscale="log",
        ylabel=(
            r"Median $|\Delta\log_{10}x|$ [dex]"
            if language == "en"
            else r"中央値 $|\Delta\log_{10}x|$ [dex]"
        ),
        title=(
            "Visible composition traces"
            if language == "en"
            else "表示範囲内の組成curve"
        ),
    )
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        [0, 1],
        [radius_summary["raw"], radius_summary["basis-aligned"]],
        color=["0.65", "#0072b2"],
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=(
            ["Same numeric", "Basis aligned"]
            if language == "en"
            else ["同数値", "基底整合後"]
        ),
        ylabel=(
            r"Median $|\Delta R_{10\,bar}|$ [$R_\oplus$]"
            if language == "en"
            else r"中央値 $|\Delta R_{10\,bar}|$ [$R_\oplus$]"
        ),
        title=(
            "10-bar radius"
            if language == "en"
            else "10 bar半径"
        ),
    )
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Effect of the paper-to-provider C/N basis mapping"
        if language == "en"
        else "論文座標からprovider入力へのC/N基底変換の効果",
        fontsize=14,
    )
    figure.tight_layout()
    return figure


def plot_basis_aligned_overlay(comparison, reference, language="en"):
    """Overlay basis-aligned samples on the published vector centerlines."""

    _set_plot_language(language)
    aligned = comparison[comparison["label"] == "basis-aligned"]
    titles = {
        "ja": (
            r"(a) $P_{melt}$",
            r"(b) $T_{melt}$",
            r"(c) $f_{O_2}$",
            "(d) paper melt-CO座標",
            r"(e) paper melt-N$_2$座標",
            r"(f) $T_b$",
        ),
        "en": (
            r"(a) $P_{melt}$",
            r"(b) $T_{melt}$",
            r"(c) $f_{O_2}$",
            "(d) Paper melt-CO coordinate",
            r"(e) Paper melt-N$_2$ coordinate",
            r"(f) $T_b$",
        ),
    }[language]
    xlabels = (
        r"Log($P_{melt}$ [bar])",
        r"$T_{melt}$ [K]",
        r"Log($f_{O_2}$ [bar])",
        r"Log($x_{CO,paper}$)",
        r"Log($x_{N_2,paper}$)",
        r"$T_b$ [K]",
    )
    y_label = "Log(モル分率)" if language == "ja" else "Log(Mole fraction)"
    radius_label = (
        r"10 bar半径 [$R_\oplus$]"
        if language == "ja"
        else r"Radius at 10 bar [$R_\oplus$]"
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    for axis, sweep, title, xlabel in zip(
        axes.flat,
        SWEEP_ORDER,
        titles,
        xlabels,
    ):
        panel_model = aligned[aligned["sweep"] == sweep]
        panel_reference = reference[reference["sweep"] == sweep]
        for species in SPECIES:
            for endpoint, level, linestyle, marker in (
                ("surface", "surface", "-", "o"),
                ("base", "base", ":", "s"),
            ):
                paper_curve = panel_reference[
                    (panel_reference["species"] == species)
                    & (panel_reference["level"] == level)
                ].sort_values("x")
                model_curve = panel_model[
                    (panel_model["species"] == species)
                    & (panel_model["endpoint"] == endpoint)
                ].sort_values("x")
                axis.plot(
                    paper_curve["x"],
                    paper_curve["value"],
                    color=SPECIES_COLORS[species],
                    linestyle=linestyle,
                    linewidth=4.0,
                    alpha=0.24,
                )
                axis.plot(
                    model_curve["x"],
                    model_curve["model"],
                    color=SPECIES_COLORS[species],
                    linestyle=linestyle,
                    linewidth=1.0,
                    marker=marker,
                    markersize=2.8,
                )
        paper_radius = panel_reference[
            panel_reference["level"] == "radius"
        ].sort_values("x")
        model_radius = panel_model[
            panel_model["endpoint"] == "radius"
        ].sort_values("x")
        radius_axis = axis.twinx()
        radius_axis.plot(
            paper_radius["x"],
            paper_radius["value"],
            color="0.65",
            linewidth=4.0,
            alpha=0.35,
        )
        radius_axis.plot(
            model_radius["x"],
            model_radius["model"],
            color="0.25",
            marker="o",
            markersize=2.8,
            linewidth=1.0,
        )
        axis.set(
            title=title,
            xlabel=xlabel,
            ylabel=y_label,
            ylim=(-10.0, 0.0),
        )
        radius_axis.set(ylabel=radius_label, ylim=(1.4, 2.4))
        axis.grid(alpha=0.15)
    species_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[name], label=name)
        for name in SPECIES
    ]
    source_labels = (
        ("paper vector stroke", "ExoExamples workflow samples")
        if language == "en"
        else ("paper vector線", "ExoExamples workflow sample")
    )
    source_handles = [
        Line2D(
            [0],
            [0],
            color="0.45",
            linewidth=4.0,
            alpha=0.35,
            label=source_labels[0],
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            linewidth=1.0,
            label=source_labels[1],
        ),
    ]
    figure.legend(
        handles=species_handles + source_handles,
        loc="lower center",
        ncol=5,
        fontsize=8,
    )
    figure.suptitle(
        "Basis-aligned ExoExamples workflow samples over MELTYQ Figure 8 vectors"
        if language == "en"
        else "基底整合後ExoExamples workflow sampleとMELTYQ Fig. 8 vector線",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure


def plot_basis_aligned_differences(comparison, language="en"):
    """Plot basis-aligned differences from the published vector centerlines."""

    _set_plot_language(language)
    aligned = comparison[comparison["label"] == "basis-aligned"]
    titles = {
        "ja": (
            r"(a) $P_{melt}$",
            r"(b) $T_{melt}$",
            r"(c) $f_{O_2}$",
            "(d) paper melt-CO座標",
            r"(e) paper melt-N$_2$座標",
            r"(f) $T_b$",
        ),
        "en": (
            r"(a) $P_{melt}$",
            r"(b) $T_{melt}$",
            r"(c) $f_{O_2}$",
            "(d) Paper melt-CO coordinate",
            r"(e) Paper melt-N$_2$ coordinate",
            r"(f) $T_b$",
        ),
    }[language]
    xlabels = (
        r"Log($P_{melt}$ [bar])",
        r"$T_{melt}$ [K]",
        r"Log($f_{O_2}$ [bar])",
        r"Log($x_{CO,paper}$)",
        r"Log($x_{N_2,paper}$)",
        r"$T_b$ [K]",
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    for axis, sweep, title, xlabel in zip(
        axes.flat,
        SWEEP_ORDER,
        titles,
        xlabels,
    ):
        panel = aligned[aligned["sweep"] == sweep]
        axis.axhspan(
            -PAPER_COMPOSITION_HALF_LINEWIDTH_DEX,
            PAPER_COMPOSITION_HALF_LINEWIDTH_DEX,
            color="0.92",
            zorder=0,
        )
        axis.axhline(0.0, color="0.45", linewidth=0.8)
        for species in SPECIES:
            for endpoint, linestyle in (("surface", "-"), ("base", ":")):
                curve = panel[
                    (panel["species"] == species)
                    & (panel["endpoint"] == endpoint)
                    & panel["visible"]
                ].sort_values("x")
                if curve.empty:
                    continue
                axis.plot(
                    curve["x"],
                    curve["delta"],
                    color=SPECIES_COLORS[species],
                    linestyle=linestyle,
                    marker=".",
                    markersize=4,
                    label=species if endpoint == "surface" else None,
                )
        radius = panel[panel["endpoint"] == "radius"].sort_values("x")
        radius_axis = axis.twinx()
        radius_axis.plot(
            radius["x"],
            radius["delta"],
            color="0.45",
            linewidth=2.0,
        )
        radius_axis.plot(
            radius["x"],
            radius["counterfactual_delta"],
            color="0.45",
            linewidth=1.2,
            linestyle="--",
        )
        radius_axis.axhline(0.0, color="0.65", linewidth=0.6)
        radius_axis.set_ylim(-0.005, 0.015)
        radius_axis.set_ylabel(
            r"$\Delta R_{10\,bar}$ [$R_\oplus$]",
            color="0.35",
        )
        axis.set_ylim((-1.0, 6.2) if sweep == "pressure_melt" else (-0.7, 0.7))
        axis.set(
            title=title,
            xlabel=xlabel,
            ylabel=r"$\Delta\log_{10}x$ [dex]",
        )
        axis.grid(alpha=0.2)
    species_handles = [
        Line2D([0], [0], color=SPECIES_COLORS[name], label=name)
        for name in SPECIES
    ]
    style_labels = (
        (
            "$P_{melt}$ gas",
            "10 bar gas",
            "$R_{10 bar}$ (external ExoPie rocky boundary)",
            "$R_{10 bar}$ (Fortney rocky-boundary counterfactual)",
        )
        if language == "en"
        else (
            "$P_{melt}$ 気相",
            "10 bar気相",
            "$R_{10 bar}$ (外部ExoPie岩石境界)",
            "$R_{10 bar}$ (Fortney岩石境界の仮想計算)",
        )
    )
    style_handles = [
        Line2D([0], [0], color="0.2", linestyle="-", label=style_labels[0]),
        Line2D([0], [0], color="0.2", linestyle=":", label=style_labels[1]),
        Line2D([0], [0], color="0.45", linewidth=2, label=style_labels[2]),
        Line2D(
            [0],
            [0],
            color="0.45",
            linewidth=1.2,
            linestyle="--",
            label=style_labels[3],
        ),
    ]
    figure.legend(
        handles=species_handles + style_handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
    )
    figure.suptitle(
        "Basis-aligned ExoExamples workflow minus digitized MELTYQ Figure 8"
        if language == "en"
        else "基底整合後のExoExamples workflow − MELTYQ Fig. 8 digitization",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    return figure
