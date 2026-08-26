"""Build one hash-pinned molecular Diffgrid archive for MELTYQ Figure 3.

This is an intentionally explicit, expensive preparation command.  It uses
the complete ExoMol transition files exposed by ``MdbExomol``; it does not use
super-line continua or silently fall back to another line list.  Build one
species per process so the line database, PreMODIT teacher, and Diffgrid table
can be released before starting the next molecule.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time


DEFAULT_NUMBA_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / ".cache" / "numba"
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(DEFAULT_NUMBA_CACHE_DIRECTORY),
)

import jax


jax.config.update("jax_enable_x64", True)

import numpy as np
from exojax.opacity.diffgrid.diagnostics import (
    compare_diffgrid_with_teacher,
    diffgrid_interval_midpoint_temperatures,
)
from exojax.utils.instfunc import (
    nx_even_from_resolution_eslog,
    resolution_eslog,
)

if __package__:
    from .meltyq_figure3 import (
        ABSORBING_SPECIES,
        DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION,
        DIFFGRID_MANIFEST_SCHEMA_VERSION,
        PAPER_ISOTOPOLOGUES,
        PAPER_LINE_LISTS,
        PRESSURE_BASE_BAR,
        PRESSURE_TOP_BAR,
        UPPER_NLAYER,
    )
else:
    from meltyq_figure3 import (
        ABSORBING_SPECIES,
        DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION,
        DIFFGRID_MANIFEST_SCHEMA_VERSION,
        PAPER_ISOTOPOLOGUES,
        PAPER_LINE_LISTS,
        PRESSURE_BASE_BAR,
        PRESSURE_TOP_BAR,
        UPPER_NLAYER,
    )


def preparation_estimate(args) -> dict[str, object]:
    """Return grid and persistent-table estimates without touching databases."""

    wavenumber_min = 1.0e4 / args.wavelength_max_micron
    wavenumber_max = 1.0e4 / args.wavelength_min_micron
    points = nx_even_from_resolution_eslog(
        wavenumber_min,
        wavenumber_max,
        args.minimum_resolving_power,
        definition="pointwise",
    )
    estimate_grid = np.logspace(
        np.log10(wavenumber_min),
        np.log10(wavenumber_max),
        points,
        dtype=np.float64,
    )
    measured_resolution = resolution_eslog(
        estimate_grid,
        definition="pointwise",
    )
    table_bytes = (
        2
        * UPPER_NLAYER
        * args.number_temperature_nodes
        * points
        * np.dtype(np.float64).itemsize
    )
    return {
        "species": args.species,
        "wavenumber_range_cm1": [wavenumber_min, wavenumber_max],
        "spectral_points": points,
        "measured_grid_resolving_power": measured_resolution,
        "temperature_nodes": args.number_temperature_nodes,
        "one_species_value_and_derivative_table_gib": table_bytes / 2.0**30,
        "line_database_warning": (
            "This estimate excludes the ExoMol transition database, PreMODIT "
            "teacher, JAX/XLA temporaries, and archive compression workspace."
        ),
        "yt34to10_warning": (
            "YT34to10 declares about 34 billion transitions; a full CH4 build "
            "requires exceptional disk, conversion time, host RAM, and compute."
            if args.species == "CH4"
            else None
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def broadening_provenance(molecular_database, args) -> dict[str, object]:
    """Identify requested broadener data and any definition-file fallback."""

    broad_files = getattr(molecular_database, "broad_files", {})
    requested_path = broad_files.get(args.background_broadener)
    requested_path = Path(requested_path) if requested_path is not None else None
    requested_file_present = bool(
        requested_path is not None and requested_path.is_file()
    )
    missing_policy = (
        "allow_definition_file_defaults"
        if args.allow_default_broadening_fallback
        else "reject"
    )
    if not requested_file_present and missing_policy == "reject":
        raise FileNotFoundError(
            f"No {args.background_broadener} broadening file is available for "
            f"{args.species}. Use --allow-default-broadening-fallback only "
            "for an explicitly acknowledged approximation."
        )

    alpha_ref = np.asarray(molecular_database.alpha_ref, dtype=float)
    n_temperature_exponent = np.asarray(
        molecular_database.n_Texp,
        dtype=float,
    )
    if (
        alpha_ref.ndim != 1
        or n_temperature_exponent.shape != alpha_ref.shape
        or alpha_ref.size == 0
        or np.any(~np.isfinite(alpha_ref))
        or np.any(alpha_ref <= 0.0)
        or np.any(~np.isfinite(n_temperature_exponent))
    ):
        raise ValueError("Invalid effective pressure-broadening parameters.")
    alpha_ref_default = float(molecular_database.alpha_ref_def)
    n_temperature_exponent_default = float(molecular_database.n_Texp_def)
    if (
        not np.isfinite(alpha_ref_default)
        or alpha_ref_default <= 0.0
        or not np.isfinite(n_temperature_exponent_default)
    ):
        raise ValueError("Invalid definition-file broadening defaults.")

    alpha_ref_range = [float(np.min(alpha_ref)), float(np.max(alpha_ref))]
    n_temperature_exponent_range = [
        float(np.min(n_temperature_exponent)),
        float(np.max(n_temperature_exponent)),
    ]
    if not requested_file_present and (
        not np.allclose(
            alpha_ref_range,
            alpha_ref_default,
            rtol=1.0e-12,
            atol=0.0,
        )
        or not np.allclose(
            n_temperature_exponent_range,
            n_temperature_exponent_default,
            rtol=1.0e-12,
            atol=0.0,
        )
    ):
        raise ValueError(
            "Requested broadening file is absent, but effective parameters "
            "do not match the definition-file defaults."
        )

    return {
        "schema_version": "meltyq.figure3.broadening-provenance@1",
        "requested_background_broadener": args.background_broadener,
        "missing_requested_broadener_policy": missing_policy,
        "effective_source": (
            "requested_file_with_definition_defaults_for_unmatched_lines"
            if requested_file_present
            else "definition_file_defaults_missing_requested_file"
        ),
        "requested_file": (
            str(requested_path.resolve())
            if requested_file_present
            else str(requested_path) if requested_path is not None else None
        ),
        "requested_file_present": requested_file_present,
        "requested_file_sha256": (
            _sha256_file(requested_path) if requested_file_present else None
        ),
        "definition_default_alpha_ref_cm1_bar": alpha_ref_default,
        "definition_default_temperature_exponent": (
            n_temperature_exponent_default
        ),
        "effective_alpha_ref_range_cm1_bar": alpha_ref_range,
        "effective_temperature_exponent_range": (
            n_temperature_exponent_range
        ),
        "selected_line_parameter_count": int(alpha_ref.size),
    }


def source_file_inventory(database_path: Path) -> tuple[list[dict[str, object]], str]:
    """Hash every regular file that defines or caches the selected line list."""

    database_path = database_path.resolve()
    files = sorted(path for path in database_path.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No line-database files found under {database_path}.")
    inventory = [
        {
            "relative_path": str(path.relative_to(database_path)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in files
    ]
    canonical_payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return inventory, hashlib.sha256(canonical_payload).hexdigest()


def _float64_little_endian_sha256(values) -> str:
    """Hash a numeric coordinate array in a portable canonical encoding."""

    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def runtime_package_version(package_name: str) -> str:
    """Return the version of the module imported by this interpreter."""

    module_version = getattr(
        importlib.import_module(package_name),
        "__version__",
        None,
    )
    if module_version is not None:
        return str(module_version)
    return importlib.metadata.version(package_name)


def python_package_source_state(package_name: str) -> dict[str, object]:
    """Fingerprint imported Python sources and their enclosing Git checkout."""

    module = importlib.import_module(package_name)
    package_directory = Path(module.__file__).resolve().parent
    source_files = sorted(package_directory.rglob("*.py"))
    inventory = [
        {
            "relative_path": str(path.relative_to(package_directory)),
            "sha256": _sha256_file(path),
        }
        for path in source_files
    ]
    inventory_payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "python_source_files": len(inventory),
        "python_source_inventory_sha256": hashlib.sha256(
            inventory_payload
        ).hexdigest(),
    }
    repository_root = next(
        (
            parent
            for parent in (package_directory, *package_directory.parents)
            if (parent / ".git").exists()
        ),
        None,
    )
    if repository_root is not None:
        relative_package = package_directory.relative_to(repository_root)
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                f":(glob){relative_package}/**/*.py",
                f":(glob){relative_package}/*.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result.update(
            {
                "git_revision": revision,
                "git_source_dirty": bool(status),
                "git_status_sha256": hashlib.sha256(
                    status.encode("utf-8")
                ).hexdigest(),
            }
        )
    return result


def build_contract(args, nu_grid, pressure_grid, temperature_nodes) -> dict[str, object]:
    """Describe all settings that must agree across species archives."""

    nu_grid = np.asarray(nu_grid, dtype=float)
    pressure_grid = np.asarray(pressure_grid, dtype=float)
    temperature_nodes = np.asarray(temperature_nodes, dtype=float)
    return {
        "schema_version": "meltyq.figure3.diffgrid-build-contract@1",
        "spectral_grid": {
            "kind": "ESLOG",
            "unit": "cm-1",
            "wavelength_order": "ascending",
            "points": int(nu_grid.size),
            "wavenumber_min_cm1": float(nu_grid[0]),
            "wavenumber_max_cm1": float(nu_grid[-1]),
            "minimum_pointwise_resolving_power": float(
                resolution_eslog(nu_grid, definition="pointwise")
            ),
            "float64_little_endian_sha256": _float64_little_endian_sha256(
                nu_grid
            ),
        },
        "pressure_grid": {
            "unit": "bar",
            "layers": int(pressure_grid.size),
            "top_boundary_bar": PRESSURE_TOP_BAR,
            "bottom_boundary_bar": PRESSURE_BASE_BAR,
            "float64_little_endian_sha256": _float64_little_endian_sha256(
                pressure_grid
            ),
        },
        "temperature_grid": {
            "unit": "K",
            "nodes": int(temperature_nodes.size),
            "minimum_k": float(np.min(temperature_nodes)),
            "maximum_k": float(np.max(temperature_nodes)),
            "uniform_in_inverse_temperature": True,
            "float64_little_endian_sha256": _float64_little_endian_sha256(
                temperature_nodes
            ),
        },
        "teacher": {
            "method": "premodit",
            "complete_transition_files": True,
            "line_strength_cutoff": args.line_strength_cutoff,
            "typical_temperature_k": args.typical_temperature_k,
            "requested_background_broadener": args.background_broadener,
            "missing_requested_broadener_policy": (
                "allow_definition_file_defaults"
                if args.allow_default_broadening_fallback
                else "reject"
            ),
            "diffmode": 1,
            "broadening_resolution": args.broadening_resolution,
            "nstitch": args.nstitch,
            "cutwing": args.cutwing,
            "cutwing_effective": args.nstitch > 1,
            "minimum_cross_section_cm2": args.minimum_cross_section_cm2,
            "validation_temperature_rule": (
                "every inverse-temperature interval midpoint"
            ),
            "validation_profile_count": args.number_temperature_nodes - 1,
            "maximum_p99_absolute_log_cross_section_error": (
                args.maximum_p99_log_cross_section_error
            ),
            "maximum_absolute_log_cross_section_error": (
                args.maximum_log_cross_section_error
            ),
        },
        "software": {
            "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
            "versions": {
                package: runtime_package_version(package)
                for package in ("exojax", "jax", "jaxlib", "radis")
            },
            "exojax_source": python_package_source_state("exojax"),
        },
    }


def _write_manifest(
    manifest_path: Path,
    descriptor: dict[str, object],
    minimum_resolving_power: float,
    common_build_contract: dict[str, object],
) -> None:
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != DIFFGRID_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Refusing to update an unknown manifest schema.")
        if not np.isclose(
            manifest.get("minimum_resolving_power"),
            minimum_resolving_power,
        ):
            raise ValueError("Manifest resolving power differs from this build.")
        if manifest.get("build_contract") != common_build_contract:
            raise ValueError(
                "Manifest spectral, pressure, temperature, or teacher contract "
                "differs from this build."
            )
    else:
        manifest = {
            "schema_version": DIFFGRID_MANIFEST_SCHEMA_VERSION,
            "minimum_resolving_power": minimum_resolving_power,
            "build_contract": common_build_contract,
            "notes": [
                "Generated one species at a time by meltyq_figure3_build_diffgrid.py.",
                "The manifest is runnable only after all five descriptors exist.",
            ],
            "files": {},
        }
    species = str(descriptor["molecule"])
    if species in manifest["files"]:
        raise FileExistsError(
            f"Manifest already contains {species}; use a new output directory."
        )
    manifest["files"][species] = descriptor
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(manifest_path)


def build_one(args) -> dict[str, object]:
    """Build and save one complete-line-list OpaDiffgrid archive."""

    from exojax.database.exomol.api import MdbExomol
    from exojax.opacity import OpaDiffgrid, OpaPremodit, saveopa
    from exojax.rt import ArtTransPure
    from exojax.utils.grids import wavenumber_grid

    estimate = preparation_estimate(args)
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    nu_grid, _, _ = wavenumber_grid(
        estimate["wavenumber_range_cm1"][0],
        estimate["wavenumber_range_cm1"][1],
        estimate["spectral_points"],
        unit="cm-1",
        xsmode="diffgrid",
        wavelength_order="ascending",
    )
    art = ArtTransPure.from_pressure_boundaries(
        PRESSURE_TOP_BAR,
        PRESSURE_BASE_BAR,
        UPPER_NLAYER,
        integration="simpson",
        nu_grid=nu_grid,
    )
    inverse_temperature = np.linspace(
        1.0 / args.temperature_max_k,
        1.0 / args.temperature_min_k,
        args.number_temperature_nodes,
    )
    temperature_nodes = 1.0 / inverse_temperature
    common_build_contract = build_contract(
        args,
        nu_grid,
        art.pressure,
        temperature_nodes,
    )
    archive_path = output_directory / f"{args.species}.npz"
    metadata_path = output_directory / f"{args.species}_metadata.json"
    manifest_path = output_directory / "manifest.json"
    for path in (archive_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}.")
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            existing_manifest = json.load(stream)
        if existing_manifest.get("schema_version") != DIFFGRID_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Refusing to update an unknown manifest schema.")
        if args.species in existing_manifest.get("files", {}):
            raise FileExistsError(
                f"Manifest already contains {args.species}; use a new output directory."
            )
        if not np.isclose(
            existing_manifest.get("minimum_resolving_power"),
            args.minimum_resolving_power,
        ):
            raise ValueError("Manifest resolving power differs from this build.")
        if existing_manifest.get("build_contract") != common_build_contract:
            raise ValueError(
                "Manifest spectral, pressure, temperature, or teacher contract "
                "differs from this build."
            )
    measured_resolution = float(
        resolution_eslog(nu_grid, definition="pointwise")
    )
    database_path = (
        args.database_root
        / args.species
        / PAPER_ISOTOPOLOGUES[args.species]
        / PAPER_LINE_LISTS[args.species]
    )

    timings = {}
    start = time.perf_counter()
    molecular_database = MdbExomol(
        str(database_path),
        nurange=nu_grid,
        crit=args.line_strength_cutoff,
        elower_max=None,
        Ttyp=args.typical_temperature_k,
        bkgdatm=args.background_broadener,
        broadf=True,
        broadf_download=args.allow_download,
        gpu_transfer=False,
        inherit_dataframe=False,
        optional_quantum_states=False,
        engine="vaex",
    )
    timings["molecular_database_seconds"] = time.perf_counter() - start
    effective_broadening = broadening_provenance(molecular_database, args)
    number_of_lines = int(len(molecular_database.nu_lines))
    if number_of_lines == 0:
        raise ValueError("The selected ExoMol database contains no lines on the grid.")
    selected_line_center_range_cm1 = [
        float(np.min(molecular_database.nu_lines)),
        float(np.max(molecular_database.nu_lines)),
    ]
    start = time.perf_counter()
    line_database_files, line_database_inventory_sha256 = source_file_inventory(
        database_path
    )
    timings["source_file_hash_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    teacher = OpaPremodit(
        mdb=molecular_database,
        nu_grid=nu_grid,
        diffmode=1,
        broadening_resolution={
            "mode": "manual",
            "value": args.broadening_resolution,
        },
        auto_trange=(args.temperature_min_k, args.temperature_max_k),
        allow_32bit=False,
        nstitch=args.nstitch,
        cutwing=args.cutwing,
        wavelength_order="ascending",
    )
    timings["premodit_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    diffgrid = OpaDiffgrid(
        teacher,
        temperature_grid=temperature_nodes,
        pressure_grid=np.asarray(art.pressure),
        min_cross_section=args.minimum_cross_section_cm2,
    )
    jax.block_until_ready(diffgrid.log_cross_section_grid)
    timings["diffgrid_seconds"] = time.perf_counter() - start

    validation_temperatures = diffgrid_interval_midpoint_temperatures(
        diffgrid
    )
    interpolation_validation = []
    start = time.perf_counter()
    for temperature in validation_temperatures:
        profile = np.full(UPPER_NLAYER, temperature)
        comparison = compare_diffgrid_with_teacher(
            diffgrid,
            teacher,
            profile,
            quantiles=(0.99,),
        )
        maximum_error = comparison.maximum_absolute_log_cross_section_error
        p99_error = (
            comparison.absolute_log_cross_section_error_quantiles[0]
        )
        interpolation_validation.append(
            {
                "isothermal_temperature_k": float(temperature),
                "maximum_absolute_log_cross_section_error": maximum_error,
                "p99_absolute_log_cross_section_error": p99_error,
            }
        )
        if p99_error > args.maximum_p99_log_cross_section_error:
            raise ValueError(
                "Diffgrid p99 log-cross-section error exceeds the configured "
                f"quality gate at {temperature:.6g} K: {p99_error:.6g}."
            )
        if maximum_error > args.maximum_log_cross_section_error:
            raise ValueError(
                "Diffgrid maximum log-cross-section error exceeds the configured "
                f"quality gate at {temperature:.6g} K: {maximum_error:.6g}."
            )
    timings["interpolation_validation_seconds"] = time.perf_counter() - start

    user_meta = {
        "schema_version": DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION,
        "build_contract": common_build_contract,
        "molecule": args.species,
        "isotopologue": PAPER_ISOTOPOLOGUES[args.species],
        "line_list": PAPER_LINE_LISTS[args.species],
        "source_grid_minimum_resolving_power": measured_resolution,
        "source_grid_points": int(nu_grid.size),
        "source_wavenumber_range_cm1": [
            float(nu_grid[0]),
            float(nu_grid[-1]),
        ],
        "line_database_path": str(database_path.resolve()),
        "line_database_source_url": (
            "https://www.exomol.com/db/"
            f"{args.species}/{PAPER_ISOTOPOLOGUES[args.species]}/"
            f"{PAPER_LINE_LISTS[args.species]}/"
        ),
        "line_database_content": "complete ExoMol .trans files; no super-lines",
        "line_database_files": line_database_files,
        "source_database_inventory_sha256": line_database_inventory_sha256,
        "number_of_selected_lines": number_of_lines,
        "selected_line_center_range_cm1": selected_line_center_range_cm1,
        "line_strength_cutoff": args.line_strength_cutoff,
        "line_strength_typical_temperature_k": args.typical_temperature_k,
        "broadening_provenance": effective_broadening,
        "temperature_range_k": [
            args.temperature_min_k,
            args.temperature_max_k,
        ],
        "number_temperature_nodes": args.number_temperature_nodes,
        "inverse_temperature_nodes": True,
        "premodit_diffmode": 1,
        "premodit_broadening_resolution": args.broadening_resolution,
        "premodit_nstitch": args.nstitch,
        "premodit_cutwing": args.cutwing,
        "premodit_cutwing_effective": args.nstitch > 1,
        "premodit_wing_policy": (
            "stitch-edge line-wing truncation"
            if args.nstitch > 1
            else "cutwing is an OpaPremodit no-op when nstitch=1"
        ),
        "minimum_cross_section_cm2": args.minimum_cross_section_cm2,
        "pressure_top_bar": PRESSURE_TOP_BAR,
        "pressure_base_bar": PRESSURE_BASE_BAR,
        "number_pressure_layers": UPPER_NLAYER,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "diffgrid_teacher_interpolation_validation": interpolation_validation,
        "diffgrid_teacher_interpolation_quality_gate": {
            "status": "passed",
            "metric": (
                "absolute natural-log cross-section error after applying the "
                "configured cross-section floor"
            ),
            "maximum_p99": args.maximum_p99_log_cross_section_error,
            "maximum": args.maximum_log_cross_section_error,
            "scope": (
                "every inverse-temperature interval midpoint as an isothermal "
                "profile on the full pressure grid"
            ),
            "observable_level_convergence": "not tested by this gate",
        },
        "package_versions": {
            package: runtime_package_version(package)
            for package in ("exojax", "jax", "jaxlib", "radis")
        },
    }
    start = time.perf_counter()
    saveopa(
        diffgrid,
        str(archive_path),
        format="npz",
        extra_meta=user_meta,
    )
    timings["save_seconds"] = time.perf_counter() - start

    descriptor = {
        "path": archive_path.name,
        "sha256": _sha256_file(archive_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "molecule": args.species,
        "isotopologue": PAPER_ISOTOPOLOGUES[args.species],
        "line_list": PAPER_LINE_LISTS[args.species],
        "teacher_method": "premodit",
        "source_grid_minimum_resolving_power": measured_resolution,
        "source_database_inventory_sha256": line_database_inventory_sha256,
        "broadening_provenance": effective_broadening,
        "interpolation_validation": {
            "status": "passed",
            "maximum_p99_log_error": (
                args.maximum_p99_log_cross_section_error
            ),
            "maximum_log_error": args.maximum_log_cross_section_error,
        },
    }
    _write_manifest(
        manifest_path,
        descriptor,
        args.minimum_resolving_power,
        common_build_contract,
    )
    return {
        **estimate,
        "archive": str(archive_path),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "number_of_selected_lines": number_of_lines,
        "measured_grid_resolving_power": measured_resolution,
        "timing_seconds": timings,
    }


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=ABSORBING_SPECIES, required=True)
    parser.add_argument(
        "--database-root",
        type=Path,
        default=repository_root / ".database",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root / ".database/meltyq_figure3_diffgrid",
    )
    parser.add_argument("--wavelength-min-micron", type=float, default=0.65)
    parser.add_argument("--wavelength-max-micron", type=float, default=12.0)
    parser.add_argument("--minimum-resolving-power", type=float, default=50000.0)
    parser.add_argument("--temperature-min-k", type=float, default=200.0)
    parser.add_argument("--temperature-max-k", type=float, default=1200.0)
    parser.add_argument("--number-temperature-nodes", type=int, default=21)
    parser.add_argument("--line-strength-cutoff", type=float, default=0.0)
    parser.add_argument("--typical-temperature-k", type=float, default=1000.0)
    parser.add_argument("--background-broadener", default="H2")
    parser.add_argument(
        "--allow-default-broadening-fallback",
        action="store_true",
        help=(
            "Allow an absent requested .broad file to use and record the "
            "ExoMol definition-file defaults; the strict default rejects it."
        ),
    )
    parser.add_argument("--broadening-resolution", type=float, default=0.2)
    parser.add_argument("--nstitch", type=int, default=1)
    parser.add_argument("--cutwing", type=float, default=1.0)
    parser.add_argument("--minimum-cross-section-cm2", type=float, default=1.0e-35)
    parser.add_argument(
        "--maximum-p99-log-cross-section-error",
        type=float,
        default=0.05,
        help="Reject an archive if an off-node validation p99 exceeds this value.",
    )
    parser.add_argument(
        "--maximum-log-cross-section-error",
        type=float,
        default=0.5,
        help="Reject an archive if an off-node validation maximum exceeds this value.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Acknowledge that MdbExomol may download extremely large files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid/table estimate without downloads or writes.",
    )
    return parser


def _validate_arguments(args) -> None:
    for label in (
        "wavelength_min_micron",
        "wavelength_max_micron",
        "temperature_min_k",
        "temperature_max_k",
        "typical_temperature_k",
        "broadening_resolution",
        "cutwing",
        "minimum_cross_section_cm2",
        "maximum_p99_log_cross_section_error",
        "maximum_log_cross_section_error",
    ):
        value = getattr(args, label)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive.")
    if args.wavelength_max_micron <= args.wavelength_min_micron:
        raise ValueError("Wavelength bounds are reversed.")
    if args.temperature_max_k <= args.temperature_min_k:
        raise ValueError("Temperature bounds are reversed.")
    if (
        not np.isfinite(args.minimum_resolving_power)
        or args.minimum_resolving_power < 50000.0
    ):
        raise ValueError(
            "minimum_resolving_power must be finite and at least 50000."
        )
    if args.number_temperature_nodes < 2:
        raise ValueError("number_temperature_nodes must be at least two.")
    if args.nstitch < 1:
        raise ValueError("nstitch must be positive.")
    spectral_points = preparation_estimate(args)["spectral_points"]
    if spectral_points % args.nstitch:
        raise ValueError(
            "The ESLOG spectral point count must be divisible by nstitch."
        )
    if not np.isfinite(args.line_strength_cutoff) or args.line_strength_cutoff < 0.0:
        raise ValueError("line_strength_cutoff must be finite and non-negative.")
    if (
        args.maximum_log_cross_section_error
        < args.maximum_p99_log_cross_section_error
    ):
        raise ValueError(
            "maximum_log_cross_section_error must be at least the p99 limit."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_arguments(args)
    if args.dry_run:
        print(json.dumps(preparation_estimate(args), indent=2, allow_nan=False))
        return
    if not args.allow_download:
        parser.error("A real build requires explicit --allow-download consent.")
    result = build_one(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
