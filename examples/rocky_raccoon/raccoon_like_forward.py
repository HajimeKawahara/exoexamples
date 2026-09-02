"""Run the provisional Rocky Raccoon-like coupled forward column.

This executable is an integration benchmark for the current ExoGibbs and
ExoEOS adapters.  It is deliberately labelled as a raccoon-like calculation,
not as a reproduction of the paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping

import jax


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY = REPOSITORY_ROOT / ".cache/jax"
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / "outputs/rocky_raccoon_2026/raccoon_like_forward"
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / ".cache/matplotlib"),
)
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        str(DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY),
    )
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __package__:
    from .model import (
        PRESET_CASE_IDENTIFIERS,
        RaccoonLikeConfig,
        RaccoonLikeResult,
        build_context,
        preset_composition,
        run_column,
    )
else:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from examples.rocky_raccoon.model import (
        PRESET_CASE_IDENTIFIERS,
        RaccoonLikeConfig,
        RaccoonLikeResult,
        build_context,
        preset_composition,
        run_column,
    )


CLAIM_STATUS = "raccoon_like_not_paper_reproduction"
SCHEMA_VERSION = 1
VALIDITY_MODES = ("paper_extrapolated", "strict_validity")
PACKAGE_NAMES = ("jax", "numpy", "scipy", "matplotlib", "exogibbs", "exoeos")
SOURCE_PACKAGE_NAMES = ("exogibbs", "exoeos")
SOURCE_PROVENANCE_SCHEMA = "rocky_raccoon.source_provenance@1"
ACCEPTED_LAYER_SNAPSHOT_SCHEMA = "rocky_raccoon.accepted_layer_snapshot@1"
MANAGED_OUTPUT_FILENAMES = frozenset(
    {"run_status.json", "profiles.csv", "summary.json", "profile.png"}
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without running chemistry."""

    defaults = RaccoonLikeConfig()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_CASE_IDENTIFIERS),
        default=defaults.preset,
    )
    parser.add_argument(
        "--validity",
        "--validity-mode",
        dest="validity_mode",
        choices=VALIDITY_MODES,
        default=defaults.validity_mode,
    )
    parser.add_argument(
        "--pressure-base-bar",
        "--pbase",
        dest="pressure_base_bar",
        type=float,
        default=defaults.pressure_base_bar,
    )
    parser.add_argument(
        "--pressure-top-bar",
        type=float,
        default=defaults.pressure_top_bar,
    )
    parser.add_argument(
        "--transit-pressure-bar",
        type=float,
        default=defaults.transit_pressure_bar,
    )
    parser.add_argument(
        "--luminosity-w",
        "--luminosity",
        dest="luminosity_w",
        type=float,
        default=defaults.luminosity_w,
    )
    parser.add_argument(
        "--pressure-ratio",
        type=float,
        default=defaults.pressure_ratio,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.max_steps,
    )
    parser.add_argument(
        "--silicon-per-hydrogen",
        "--abundance",
        dest="silicon_per_hydrogen",
        type=float,
        default=None,
        help="Basal Si/H number ratio; any abundance flag enables overrides.",
    )
    parser.add_argument(
        "--magnesium-to-silicon",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--oxygen-to-silicon",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--carbon-per-hydrogen",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--transport",
        choices=("constant",),
        default="constant",
        help="Transport closure; only the current constant closure is available.",
    )
    parser.add_argument(
        "--rosseland-opacity-m2-kg",
        "--opacity",
        dest="rosseland_opacity_m2_kg",
        type=float,
        default=defaults.transport.rosseland_opacity_m2_kg,
    )
    parser.add_argument(
        "--thermal-conductivity-w-m-k",
        "--conductivity",
        dest="thermal_conductivity_w_m_k",
        type=float,
        default=defaults.transport.thermal_conductivity_w_m_k,
    )
    parser.add_argument(
        "--output-dir",
        "--output-directory",
        dest="output_directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--accepted-layer-snapshot",
        type=Path,
        default=None,
        help=(
            "Atomically overwrite a diagnostic NPZ after each accepted layer; "
            "this is not a restart file."
        ),
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Build and report the configured providers without solving a column.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> RaccoonLikeConfig:
    """Translate CLI values into the model's public configuration."""

    defaults = RaccoonLikeConfig()
    abundance_values = (
        args.silicon_per_hydrogen,
        args.magnesium_to_silicon,
        args.oxygen_to_silicon,
        args.carbon_per_hydrogen,
    )
    composition = None
    if any(value is not None for value in abundance_values):
        composition_defaults = preset_composition(args.preset)
        composition = replace(
            composition_defaults,
            silicon_per_hydrogen=(
                composition_defaults.silicon_per_hydrogen
                if args.silicon_per_hydrogen is None
                else args.silicon_per_hydrogen
            ),
            magnesium_to_silicon=(
                composition_defaults.magnesium_to_silicon
                if args.magnesium_to_silicon is None
                else args.magnesium_to_silicon
            ),
            oxygen_to_silicon=(
                composition_defaults.oxygen_to_silicon
                if args.oxygen_to_silicon is None
                else args.oxygen_to_silicon
            ),
            carbon_per_hydrogen=(
                composition_defaults.carbon_per_hydrogen
                if args.carbon_per_hydrogen is None
                else args.carbon_per_hydrogen
            ),
        )
    return replace(
        defaults,
        preset=args.preset,
        validity_mode=args.validity_mode,
        pressure_base_bar=args.pressure_base_bar,
        pressure_top_bar=args.pressure_top_bar,
        transit_pressure_bar=args.transit_pressure_bar,
        luminosity_w=args.luminosity_w,
        pressure_ratio=args.pressure_ratio,
        max_steps=args.max_steps,
        composition=composition,
        transport=replace(
            defaults.transport,
            rosseland_opacity_m2_kg=args.rosseland_opacity_m2_kg,
            thermal_conductivity_w_m_k=args.thermal_conductivity_w_m_k,
        ),
    )


def _validate_output_paths(args: argparse.Namespace) -> None:
    """Reject a snapshot destination owned by the normal output writer."""

    if args.accepted_layer_snapshot is None:
        return
    output_directory = Path(args.output_directory).resolve()
    snapshot_path = Path(args.accepted_layer_snapshot).resolve()
    managed_paths = {
        (output_directory / filename).resolve()
        for filename in MANAGED_OUTPUT_FILENAMES
    }
    if snapshot_path in managed_paths:
        raise ValueError(
            "--accepted-layer-snapshot must not resolve to a managed output "
            f"artifact: {snapshot_path}."
        )


def package_versions() -> dict[str, str | None]:
    """Return versions from the modules used by the running interpreter."""

    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in PACKAGE_NAMES:
        try:
            module_version = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            module_version = None
        if module_version is not None:
            versions[name] = str(module_version)
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _python_source_inventory_sha256(source_directory: Path) -> str | None:
    files = sorted(source_directory.rglob("*.py"))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(source_directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(directory: Path, *arguments: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(directory), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return completed.returncode == 0, completed.stdout.strip()


def _source_record(module_file: Path, source_directory: Path) -> dict[str, Any]:
    module_path = module_file.resolve()
    source_path = source_directory.resolve()
    root_ok, root_text = _git_output(source_path, "rev-parse", "--show-toplevel")
    revision = None
    dirty = None
    if root_ok:
        git_root = Path(root_text)
        revision_ok, revision_text = _git_output(git_root, "rev-parse", "HEAD")
        status_ok, status_text = _git_output(
            git_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            os.path.relpath(source_path, git_root),
        )
        revision = revision_text if revision_ok else None
        dirty = bool(status_text) if status_ok else None
    return {
        "module_file": str(module_path),
        "git_revision": revision,
        "git_source_dirty": dirty,
        "python_source_inventory_sha256": _python_source_inventory_sha256(
            source_path
        ),
    }


def source_provenance() -> dict[str, Any]:
    """Identify the exact local Python sources used by the forward model."""

    providers = {
        "exoexamples": _source_record(Path(__file__), Path(__file__).parent),
    }
    for name in SOURCE_PACKAGE_NAMES:
        module = importlib.import_module(name)
        module_file = Path(module.__file__)
        providers[name] = _source_record(module_file, module_file.parent)
    return {
        "schema_version": SOURCE_PROVENANCE_SCHEMA,
        "providers": providers,
    }


def _species_metadata(context) -> dict[str, list[str]]:
    setup = context.chemistry_setup
    return {
        "elements": list(setup.elements),
        "gas": list(setup.gas_species),
        "condensates": list(setup.condensate_species),
    }


def _named_bounds(names: tuple[str, ...], values) -> dict[str, float] | None:
    if values is None:
        return None
    bounds = tuple(float(value) for value in values)
    if len(bounds) != len(names):
        raise ValueError("Condensate validity bounds do not match the species list.")
    return dict(zip(names, bounds))


def _validity_metadata(context) -> dict[str, Any]:
    setup = context.chemistry_setup
    condensates = setup.condensate_setup
    metadata = dict(condensates.metadata or {})
    names = tuple(setup.condensate_species)
    original = metadata.get("rocky_raccoon_original_temperature_validity_upper")
    return {
        "mode": context.config.validity_mode,
        "scope": metadata.get("rocky_raccoon_validity_scope"),
        "active_condensate_upper_temperature_k": _named_bounds(
            names,
            condensates.temperature_validity_upper,
        ),
        "original_condensate_upper_temperature_k": _named_bounds(
            names,
            original,
        ),
    }


def _inventory_metadata(context) -> dict[str, Any]:
    names = tuple(context.chemistry_setup.elements)
    inventory = np.asarray(context.element_inventory, dtype=float)
    if inventory.shape != (len(names),):
        raise ValueError("The context inventory does not match its element names.")
    return {
        "names": list(names),
        "gauge": "normalized_positive_non_charge_element_sum",
        "physical_element_sum": float(
            sum(value for name, value in zip(names, inventory) if name != "e-")
        ),
        "charge_constraint_name": "e-",
        "charge_constraint_inventory": float(inventory[names.index("e-")]),
        "interpretation": (
            "Composition propagated after accepted rainout; not absolute "
            "retained element amount."
        ),
    }


def build_input_report(
    context,
    *,
    versions: Mapping[str, str | None] | None = None,
    source_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe provider inputs without evaluating chemical equilibrium."""

    return {
        "schema_version": SCHEMA_VERSION,
        "claim_status": CLAIM_STATUS,
        "ready": True,
        "check_scope": "providers_built_column_not_solved",
        "config": asdict(context.config),
        "effective_composition": asdict(context.composition),
        "case_identifier": context.case.identifier,
        "species": _species_metadata(context),
        "element_inventory": _inventory_metadata(context),
        "validity": _validity_metadata(context),
        "package_versions": dict(
            package_versions() if versions is None else versions
        ),
        "source_provenance": dict(
            source_provenance() if source_report is None else source_report
        ),
    }


def _accepted_transition(layer):
    transition = layer.chemistry.payload
    if transition is None or not hasattr(transition, "equilibrium"):
        raise TypeError("Every accepted layer must contain a rainout transition.")
    return transition, transition.equilibrium


def _vector(values, names: tuple[str, ...], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (len(names),):
        raise ValueError(f"{label} must have one value per named entry.")
    return result


def write_accepted_layer_snapshot(
    path: Path,
    layer_index: int,
    layer,
    context,
    *,
    source_report: Mapping[str, Any],
) -> None:
    """Atomically store one accepted layer for provider diagnostics."""

    transition, equilibrium = _accepted_transition(layer)
    setup = context.chemistry_setup
    element_names = tuple(setup.elements)
    gas_names = tuple(setup.gas_species)
    condensate_names = tuple(setup.condensate_species)
    inventory_in = _vector(
        transition.inventory_in,
        element_names,
        "inventory_in",
    )
    inventory_out = _vector(
        transition.inventory_out,
        element_names,
        "inventory_out",
    )
    gas_ln_n = _vector(equilibrium.gas_ln_n, gas_names, "gas_ln_n")
    gas_n = _vector(equilibrium.gas_n, gas_names, "gas_n")
    condensate_amounts = _vector(
        equilibrium.condensate_amounts,
        condensate_names,
        "condensate_amounts",
    )
    support = np.asarray(
        equilibrium.condensate_support_indices,
        dtype=np.int64,
    )
    if support.ndim != 1 or any(
        index < 0 or index >= len(condensate_names)
        for index in support.tolist()
    ):
        raise ValueError("condensate_support_indices are invalid.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                schema=np.asarray(ACCEPTED_LAYER_SNAPSHOT_SCHEMA),
                layer_index=np.asarray(layer_index, dtype=np.int64),
                pressure_bar=np.asarray(layer.pressure_bar, dtype=float),
                temperature_k=np.asarray(layer.temperature_k, dtype=float),
                radius_m=np.asarray(layer.radius_m, dtype=float),
                mass_density_kg_m3=np.asarray(
                    layer.mass_density_kg_m3,
                    dtype=float,
                ),
                transport=np.asarray(layer.transport),
                element_names=np.asarray(element_names),
                gas_species=np.asarray(gas_names),
                condensate_species=np.asarray(condensate_names),
                inventory_in=inventory_in,
                inventory_out=inventory_out,
                gas_ln_n=gas_ln_n,
                gas_n=gas_n,
                gas_ntot=np.asarray(equilibrium.gas_ntot, dtype=float),
                condensate_amounts=condensate_amounts,
                condensate_support_indices=support,
                source_provenance_json=np.asarray(
                    json.dumps(
                        dict(source_report),
                        sort_keys=True,
                        allow_nan=False,
                    )
                ),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def accepted_layer_snapshot_callback(
    path: Path,
    context,
    *,
    source_report: Mapping[str, Any],
) -> Callable[[int, Any], None]:
    """Return an observer that keeps only the latest committed layer."""

    def write_snapshot(layer_index: int, layer) -> None:
        write_accepted_layer_snapshot(
            path,
            layer_index,
            layer,
            context,
            source_report=source_report,
        )

    return write_snapshot


def profile_table(result: RaccoonLikeResult) -> tuple[list[str], list[dict[str, Any]]]:
    """Build the fully named accepted-layer table for CSV output."""

    context = result.context
    profile = result.profile
    setup = context.chemistry_setup
    gas_names = tuple(setup.gas_species)
    condensate_names = tuple(setup.condensate_species)
    element_names = tuple(setup.elements)
    scalar_fields = [
        "layer_index",
        "pressure_bar",
        "temperature_k",
        "radius_m",
        "mass_density_kg_m3",
        "mean_molar_mass_kg_mol",
        "transport",
        "convective_temperature_log_gradient",
        "convective_molar_mass_log_gradient",
        "convective_equation_1_gradient",
        "nonconvective_temperature_log_gradient",
        "nonconvective_molar_mass_log_gradient",
        "nonconvective_equation_1_gradient",
        "equilibrium_converged",
        "equilibrium_status",
        "acceptance_tier",
        "selected_route",
        "condensate_support_names",
        "shell_mass_kg",
        "hydrogen_mass_fraction",
        "shell_hydrogen_mass_kg",
    ]
    gas_fields = [f"gas_x[{name}]" for name in gas_names]
    condensate_amount_fields = [
        f"condensate_amount[{name}]" for name in condensate_names
    ]
    condensate_support_fields = [
        f"condensate_support[{name}]" for name in condensate_names
    ]
    inventory_in_fields = [
        f"normalized_inventory_in[{name}]" for name in element_names
    ]
    inventory_out_fields = [
        f"normalized_inventory_out[{name}]" for name in element_names
    ]
    fieldnames = (
        scalar_fields
        + gas_fields
        + condensate_amount_fields
        + condensate_support_fields
        + inventory_in_fields
        + inventory_out_fields
    )

    rows = []
    shell_masses = profile.shell_mass_kg
    for index, layer in enumerate(profile.layers):
        transition, equilibrium = _accepted_transition(layer)
        gas_x = _vector(equilibrium.gas_x, gas_names, "gas_x")
        condensate_amounts = _vector(
            equilibrium.condensate_amounts,
            condensate_names,
            "condensate_amounts",
        )
        inventory_in = _vector(
            transition.inventory_in,
            element_names,
            "inventory_in",
        )
        inventory_out = _vector(
            transition.inventory_out,
            element_names,
            "inventory_out",
        )
        support_indices = {
            int(value) for value in np.asarray(equilibrium.condensate_support_indices)
        }
        if any(
            value < 0 or value >= len(condensate_names)
            for value in support_indices
        ):
            raise ValueError("A condensate support index is out of range.")
        hydrogen_fraction = float(
            context.ideal_gas.hydrogen_mass_fraction(gas_x)
        )
        shell_mass = float(shell_masses[index]) if index < len(shell_masses) else None
        decision = profile.decisions[index - 1] if index else None
        row: dict[str, Any] = {
            "layer_index": index,
            "pressure_bar": float(layer.pressure_bar),
            "temperature_k": float(layer.temperature_k),
            "radius_m": float(layer.radius_m),
            "mass_density_kg_m3": float(layer.mass_density_kg_m3),
            "mean_molar_mass_kg_mol": float(
                layer.chemistry.mean_molar_mass_kg_mol
            ),
            "transport": layer.transport,
            "convective_temperature_log_gradient": (
                None
                if decision is None
                else decision.convective.temperature_log_gradient
            ),
            "convective_molar_mass_log_gradient": (
                None
                if decision is None
                else decision.convective.molar_mass_log_gradient
            ),
            "convective_equation_1_gradient": (
                None if decision is None else decision.convective.equation_1_gradient
            ),
            "nonconvective_temperature_log_gradient": (
                None
                if decision is None
                else decision.nonconvective.temperature_log_gradient
            ),
            "nonconvective_molar_mass_log_gradient": (
                None
                if decision is None
                else decision.nonconvective.molar_mass_log_gradient
            ),
            "nonconvective_equation_1_gradient": (
                None if decision is None else decision.nonconvective.equation_1_gradient
            ),
            "equilibrium_converged": bool(equilibrium.converged),
            "equilibrium_status": str(equilibrium.status),
            "acceptance_tier": str(equilibrium.acceptance_tier),
            "selected_route": str(equilibrium.selected_route),
            "condensate_support_names": "|".join(
                condensate_names[value] for value in sorted(support_indices)
            ),
            "shell_mass_kg": shell_mass,
            "hydrogen_mass_fraction": hydrogen_fraction,
            "shell_hydrogen_mass_kg": (
                None if shell_mass is None else shell_mass * hydrogen_fraction
            ),
        }
        row.update(dict(zip(gas_fields, gas_x)))
        row.update(dict(zip(condensate_amount_fields, condensate_amounts)))
        row.update(
            {
                field: int(species_index in support_indices)
                for species_index, field in enumerate(condensate_support_fields)
            }
        )
        row.update(dict(zip(inventory_in_fields, inventory_in)))
        row.update(dict(zip(inventory_out_fields, inventory_out)))
        rows.append(row)
    return fieldnames, rows


def write_profiles_csv(path: Path, result: RaccoonLikeResult) -> None:
    """Write the accepted-layer table."""

    fieldnames, rows = profile_table(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _solver_diagnostics(
    result: RaccoonLikeResult,
) -> dict[str, Any]:
    equilibria = [_accepted_transition(layer)[1] for layer in result.profile.layers]
    supports = [
        frozenset(
            int(value) for value in np.asarray(item.condensate_support_indices)
        )
        for item in equilibria
    ]
    diagnostics: dict[str, Any] = {
        "mode": "fixed_boundary",
        "effective_pressure_base_bar": result.profile.inputs.pressure_base_bar,
        "effective_luminosity_w": result.profile.inputs.luminosity_w,
        "accepted_layer_count": len(result.profile.layers),
        "candidate_pair_count": len(result.profile.decisions),
        "all_equilibria_converged": all(bool(item.converged) for item in equilibria),
        "unconverged_layer_indices": [
            index for index, item in enumerate(equilibria) if not bool(item.converged)
        ],
        "equilibrium_status_counts": dict(
            Counter(str(item.status) for item in equilibria)
        ),
        "acceptance_tier_counts": dict(
            Counter(str(item.acceptance_tier) for item in equilibria)
        ),
        "selected_route_counts": dict(
            Counter(str(item.selected_route) for item in equilibria)
        ),
        "transport_counts": dict(Counter(result.profile.transport)),
        "phase_support_change_count": sum(
            left != right for left, right in zip(supports[:-1], supports[1:])
        ),
        "layers_with_provider_diagnostics": sum(
            item.diagnostics is not None for item in equilibria
        ),
    }
    return diagnostics


def build_summary(
    result: RaccoonLikeResult,
    *,
    versions: Mapping[str, str | None] | None = None,
    source_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build claim-limited metadata for one completed column."""

    metrics = asdict(result.metrics)
    metrics["transit_radius_earth"] = result.metrics.transit_radius_earth
    metrics["outer_rcb_radius_earth"] = result.metrics.outer_rcb_radius_earth
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": CLAIM_STATUS,
        "config": asdict(result.context.config),
        "effective_composition": asdict(result.context.composition),
        "case_identifier": result.context.case.identifier,
        "species": _species_metadata(result.context),
        "element_inventory": _inventory_metadata(result.context),
        "validity": _validity_metadata(result.context),
        "package_versions": dict(
            package_versions() if versions is None else versions
        ),
        "source_provenance": dict(
            source_provenance() if source_report is None else source_report
        ),
        "metrics": metrics,
        "solver_diagnostics": _solver_diagnostics(result),
        "artifacts": {
            "profile_table": "profiles.csv",
            "profile_figure": "profile.png",
            "summary": "summary.json",
            "latest_run_status": "run_status.json",
        },
    }


def write_summary_json(path: Path, summary: Mapping[str, Any]) -> None:
    """Write JSON while rejecting non-finite numerical metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plot_profile(path: Path, result: RaccoonLikeResult) -> None:
    """Write a compact three-panel physical profile."""

    profile = result.profile
    pressure = profile.pressure_bar
    density = profile.mass_density_kg_m3
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 4.5), sharey=True)
    axes[0].plot(profile.temperature_k, pressure, color="tab:red")
    axes[0].set_xlabel("Temperature (K)")
    axes[1].plot(profile.radius_m, pressure, color="tab:blue")
    axes[1].set_xlabel("Radius (m)")
    axes[2].plot(density, pressure, color="tab:green")
    axes[2].set_xlabel("Density (kg m$^{-3}$)")
    if np.max(density) / np.min(density) >= 10.0:
        axes[2].set_xscale("log")
    axes[0].set(ylabel="Pressure (bar)", yscale="log")
    axes[0].invert_yaxis()
    figure.suptitle("Rocky Raccoon-like forward column (not a reproduction)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_outputs(
    output_directory: Path,
    result: RaccoonLikeResult,
    *,
    versions: Mapping[str, str | None] | None = None,
    source_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the CSV, JSON, and profile figure for one completed run."""

    output_directory.mkdir(parents=True, exist_ok=True)
    write_profiles_csv(output_directory / "profiles.csv", result)
    plot_profile(output_directory / "profile.png", result)
    summary = build_summary(
        result,
        versions=versions,
        source_report=source_report,
    )
    write_summary_json(output_directory / "summary.json", summary)
    return summary


def run(
    args: argparse.Namespace,
    *,
    versions: Mapping[str, str | None] | None = None,
    source_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build providers, solve the requested column, and write artifacts."""

    _validate_output_paths(args)
    context = build_context(config_from_args(args))
    snapshot_callback = None
    if args.accepted_layer_snapshot is not None:
        snapshot_callback = accepted_layer_snapshot_callback(
            Path(args.accepted_layer_snapshot),
            context,
            source_report=(
                source_provenance()
                if source_report is None
                else source_report
            ),
        )
    result = run_column(
        context,
        accepted_layer_callback=snapshot_callback,
    )
    return write_outputs(
        Path(args.output_directory),
        result,
        versions=versions,
        source_report=source_report,
    )


def _run_status(
    status: str,
    *,
    config: Mapping[str, Any],
    versions: Mapping[str, str | None],
    source_report: Mapping[str, Any],
    error: Exception | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": CLAIM_STATUS,
        "status": status,
        "config": dict(config),
        "package_versions": dict(versions),
        "source_provenance": dict(source_report),
    }
    if error is not None:
        report.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    return report


def _unvalidated_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return JSON-safe raw arguments when config construction fails."""

    return {
        "unvalidated_cli_arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        }
    }


def main(argv: list[str] | None = None) -> int:
    """Run the command-line application."""

    args = build_parser().parse_args(argv)
    if args.check_inputs:
        try:
            context = build_context(config_from_args(args))
            report = build_input_report(context)
        except Exception as error:
            report = {
                "schema_version": SCHEMA_VERSION,
                "claim_status": CLAIM_STATUS,
                "ready": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 0 if report["ready"] else 1

    output_directory = Path(args.output_directory)
    status_path = output_directory / "run_status.json"
    versions = package_versions()
    source_report = source_provenance()
    try:
        serialized_config = asdict(config_from_args(args))
        _validate_output_paths(args)
    except Exception as error:
        write_summary_json(
            status_path,
            _run_status(
                "failed",
                config=_unvalidated_cli_config(args),
                versions=versions,
                source_report=source_report,
                error=error,
            ),
        )
        print(
            "Rocky Raccoon-like forward column failed; "
            f"see {status_path}.",
            file=sys.stderr,
        )
        return 1
    write_summary_json(
        status_path,
        _run_status(
            "running",
            config=serialized_config,
            versions=versions,
            source_report=source_report,
        ),
    )
    try:
        summary = run(
            args,
            versions=versions,
            source_report=source_report,
        )
    except Exception as error:
        write_summary_json(
            status_path,
            _run_status(
                "failed",
                config=serialized_config,
                versions=versions,
                source_report=source_report,
                error=error,
            ),
        )
        print(
            "Rocky Raccoon-like forward column failed; "
            f"see {status_path}.",
            file=sys.stderr,
        )
        return 1
    write_summary_json(
        status_path,
        _run_status(
            "completed",
            config=serialized_config,
            versions=versions,
            source_report=source_report,
        ),
    )
    print("Rocky Raccoon-like forward column completed.")
    print(f"  Claim status: {summary['claim_status']}")
    print(f"  Output directory: {args.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
