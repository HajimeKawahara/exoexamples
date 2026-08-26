"""Build a MELTYQ Figure 3-style forward spectral comparison for K2-18 b.

The application joins the published magma/deep-atmosphere boundary to an
ExoJAX upper atmosphere, opacity calculation, and transmission solver.  It
does not claim to reproduce the unpublished MELTYQ best-fit vector or
posterior samples.  Those artifacts enter through explicit config and
reference-spectrum contracts and do not change the forward-model code.

Two opacity paths are supported:

``ckd``
    A fast, runnable ExoMolOP R=1000 comparison.  Molecular k coefficients
    are added at matching g ordinates, the perfect-correlation convention
    currently used by the ExoJAX broadband example.  This is not the R=50000
    cross-section treatment described by MELTYQ.

``diffgrid``
    Paper-line-list-aligned R=50000 cross sections stored as ExoJAX
    ``OpaDiffgrid`` archives.  A hash-pinned manifest and archive provenance
    make the external opacity products a replaceable input rather than
    repository code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from typing import Literal, Mapping

import jax


DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[2] / ".cache" / "jax"
)
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"),
)
if jax.config.jax_compilation_cache_dir is None:
    jax.config.update(
        "jax_compilation_cache_dir",
        str(DEFAULT_JAX_COMPILATION_CACHE_DIRECTORY),
    )
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from exogibbs.presets.magma_gas import MELTYQ_SPECIES
from exojax.atm.atmprof import hydrostatic_radius_profile_ideal_gas
from exojax.atm.idealgas import number_density
from exojax.database.molinfo import mean_molmass_manual
from exojax.postproc.binning import (
    apply_bin_operator,
    band_mean_bin_operator,
    piecewise_linear_bin_operator,
)
from exojax.rt.layeropacity import (
    layer_optical_depth_from_cross_section,
    layer_optical_depth_from_extinction,
    layer_optical_depth_from_log_cia,
)
from exojax.utils.constants import G, ME, RE
from exojax.utils.instfunc import resolution_eslog

if __package__:
    from .meltyq_figure8 import Figure8Parameters, build_context, solve_point
    from .meltyq_jwst_data import (
        PUBLIC_OSF_ASSETS,
        TransitSpectrum,
        fetch_figure3_public_assets,
        load_figure3_public_spectra,
        verify_sha256,
    )
    from .meltyq_spectral_components import (
        gray_cloud_layer_optical_depth,
        lee_extinction_coefficient_m_inv,
        TAUREX_LEE_PROFILE_CHANGE,
        TAUREX_LEE_SOURCE,
        TAUREX_RAYLEIGH_SOURCE,
        TAUREX_RAYLEIGH_SPECIES,
        TAUREX_CIA_SOURCE,
        TAUREX_PLANET_SOURCE,
        TAUREX_SIMPLEMODEL_SOURCE,
        TAUREX_TRANSMISSION_SOURCE,
        taurex_old_path_rectangle_transit_radius_squared,
        taurex_rayleigh_cross_section_cm2,
        transit_depth_ppm,
        upper_temperature_profile_k,
    )
else:
    from meltyq_figure8 import Figure8Parameters, build_context, solve_point
    from meltyq_jwst_data import (
        PUBLIC_OSF_ASSETS,
        TransitSpectrum,
        fetch_figure3_public_assets,
        load_figure3_public_spectra,
        verify_sha256,
    )
    from meltyq_spectral_components import (
        gray_cloud_layer_optical_depth,
        lee_extinction_coefficient_m_inv,
        TAUREX_LEE_PROFILE_CHANGE,
        TAUREX_LEE_SOURCE,
        TAUREX_RAYLEIGH_SOURCE,
        TAUREX_RAYLEIGH_SPECIES,
        TAUREX_CIA_SOURCE,
        TAUREX_PLANET_SOURCE,
        TAUREX_SIMPLEMODEL_SOURCE,
        TAUREX_TRANSMISSION_SOURCE,
        taurex_old_path_rectangle_transit_radius_squared,
        taurex_rayleigh_cross_section_cm2,
        transit_depth_ppm,
        upper_temperature_profile_k,
    )


PAPER_DOI = "https://doi.org/10.3847/1538-4357/ae6917"
CONFIG_SCHEMA_VERSION = "meltyq.figure3.config@1"
RUN_METADATA_SCHEMA_VERSION = "meltyq.figure3.run-metadata@1"
DIFFGRID_MANIFEST_SCHEMA_VERSION = "meltyq.figure3.diffgrid-manifest@1"
DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION = (
    "meltyq.figure3.diffgrid-archive-provenance@1"
)
RSUN_CM = 6.957e10
PRESSURE_BASE_BAR = 10.0
PRESSURE_TOP_BAR = 1.0e-10
UPPER_NLAYER = 100
OPAQUE_CLOUD_VERTICAL_DTAU = 1.0e20
TAUREX_GRAVITATIONAL_CONSTANT_CGS = 6.67430e-8
"""Astropy gravitational constant used by TauREx, in cm3 g-1 s-2."""

ABSORBING_SPECIES = ("H2O", "CO", "CO2", "CH4", "NH3")
RAYLEIGH_SPECIES = tuple(MELTYQ_SPECIES)
SCENARIO_NAMES = ("total", "aerosols", "rayleigh_plus_cia") + tuple(
    f"molecule_{species}" for species in ABSORBING_SPECIES
)

PAPER_LINE_LISTS = {
    "H2O": "POKAZATEL",
    "CO": "Li2015",
    "CO2": "UCL-4000",
    "CH4": "YT34to10",
    "NH3": "BYTe",
}

PAPER_ISOTOPOLOGUES = {
    "H2O": "1H2-16O",
    "CO": "12C-16O",
    "CO2": "12C-16O2",
    "CH4": "12C-1H4",
    "NH3": "14N-1H3",
}

# ExoMolOP has no BYTe R=1000 table.  CoYuTe is an explicit fast-path
# substitution; the diffgrid path requires the paper-cited BYTe source.
CKD_RELATIVE_PATHS = {
    "H2O": Path("H2O/1H2-16O/POKAZATEL"),
    "CO": Path("CO/12C-16O/Li2015"),
    "CO2": Path("CO2/12C-16O2/UCL-4000"),
    "CH4": Path("CH4/12C-1H4/YT34to10"),
    "NH3": Path("NH3/14N-1H3/CoYuTe"),
}

CIA_FILENAMES = {
    "H2-H2": "H2-H2_2011.cia",
    "H2-He": "H2-He_2011.cia",
}

PAPER_TO_PROVIDER_FACTORS = {
    "co_melt_mole_ratio": 28.0101 / 12.0107,
    "n_melt_mole_ratio": 28.0134 / 14.0067,
}


@dataclass(frozen=True)
class PlanetConfig:
    """Fixed bulk quantities used by the forward calculation."""

    mass_earth: float
    stellar_radius_solar: float
    rocky_core_mass_fraction: float = 0.33
    rocky_radius_rearth: float | None = None


@dataclass(frozen=True)
class MagmaConfig:
    """Magma and deep-atmosphere parameters."""

    pressure_melt_bar: float
    temperature_melt_k: float
    oxygen_fugacity_bar: float
    co_melt_mole_ratio: float
    n_melt_mole_ratio: float
    temperature_base_k: float
    input_basis: Literal[
        "exogibbs_elemental",
        "paper_labelled_operational_mapping",
    ]


@dataclass(frozen=True)
class UpperConfig:
    """Observable-atmosphere temperature and discretization parameters."""

    pressure_top_bar: float
    pressure_base_bar: float
    nlayer: int
    temperature_1e4_pa_k: float
    temperature_100_pa_k: float
    temperature_1e_minus2_pa_k: float
    smoothing_window_percent: float


@dataclass(frozen=True)
class CloudConfig:
    """Opaque gray deck configuration."""

    enabled: bool
    top_pressure_pa: float


@dataclass(frozen=True)
class HazeConfig:
    """Lee haze configuration."""

    enabled: bool
    particle_radius_micron: float
    q0: float
    particle_number_density_m3: float
    mid_pressure_pa: float
    log10_pressure_extent: float = 2.0
    vertical_profile: Literal["exp_decay", "constant"] = "exp_decay"


@dataclass(frozen=True)
class ObservationConfig:
    """Choice of public reductions and nuisance offset convention."""

    niriss_resolution: Literal["lowres", "native"]
    miri_reduction: Literal["jexores", "jexopipe"]
    offset_group_by_dataset: Mapping[str, str]
    offsets_ppm: Mapping[str, float]
    anchor_offset_group: str | None


@dataclass(frozen=True)
class SpectralConfig:
    """Spectral grid and opacity selection."""

    opacity_mode: Literal["ckd", "diffgrid"]
    wavelength_min_micron: float
    wavelength_max_micron: float
    radiative_transfer_scheme: Literal[
        "taurex_rectangle",
        "exojax_simpson",
    ]
    rayleigh_provider: Literal["taurex", "exojax"]


@dataclass(frozen=True)
class Figure3Config:
    """Validated top-level comparison configuration."""

    schema_version: str
    name: str
    claim_status: str
    source_notes: tuple[str, ...]
    planet: PlanetConfig
    magma: MagmaConfig
    upper: UpperConfig
    cloud: CloudConfig
    haze: HazeConfig
    observations: ObservationConfig
    spectral: SpectralConfig


@dataclass(frozen=True)
class OpacityBundle:
    """Loaded molecular opacities sharing one spectral coordinate."""

    mode: Literal["ckd", "diffgrid"]
    calculators: Mapping[str, object]
    wavenumber_cm1: np.ndarray
    weights: np.ndarray | None
    provenance: Mapping[str, object]
    band_edges_cm1: np.ndarray | None = None


@dataclass(frozen=True)
class DiffgridArchiveOpacity:
    """A validated archive loaded only while its cross section is evaluated."""

    archive_path: Path
    temperature_grid: np.ndarray
    pressure_grid: np.ndarray
    molmass: float
    persistent_table_bytes: int

    def xsmatrix(self, temperature_k):
        """Load, interpolate, synchronize, and release one species table."""

        from exojax.opacity import OpaDiffgrid

        calculator = OpaDiffgrid.from_saved_opa(
            str(self.archive_path),
            strict=False,
        )
        cross_section = calculator.xsmatrix(temperature_k)
        jax.block_until_ready(cross_section)
        del calculator
        return cross_section


@dataclass(frozen=True)
class BinnedSpectrum:
    """One observation and all model scenarios on its exact bins."""

    observation: TransitSpectrum
    offset_group: str
    offset_ppm: float
    scenario_depth_ppm: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PreparedSpectralOpacity:
    """Temperature-evaluated opacity arrays passed dynamically to the RT JIT."""

    molecular_cross_sections: object
    log_cia_coefficients: object
    cia_coverage_masks: object
    rayleigh_cross_sections: object
    preparation_seconds_including_any_compilation: float


def _require_finite_positive(label: str, value: float) -> None:
    try:
        valid = bool(np.isfinite(value)) and value > 0.0
    except TypeError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite and positive.")


def load_config(path: str | Path) -> Figure3Config:
    """Load and validate a Figure 3 comparison JSON file."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version {CONFIG_SCHEMA_VERSION!r}; "
            f"got {payload.get('schema_version')!r}."
        )
    try:
        config = Figure3Config(
            schema_version=payload["schema_version"],
            name=payload["name"],
            claim_status=payload["claim_status"],
            source_notes=tuple(payload.get("source_notes", ())),
            planet=PlanetConfig(**payload["planet"]),
            magma=MagmaConfig(**payload["magma"]),
            upper=UpperConfig(**payload["upper"]),
            cloud=CloudConfig(**payload["cloud"]),
            haze=HazeConfig(**payload["haze"]),
            observations=ObservationConfig(**payload["observations"]),
            spectral=SpectralConfig(**payload["spectral"]),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid Figure 3 config structure: {error}") from error
    validate_config(config)
    return config


def validate_config(config: Figure3Config) -> None:
    """Reject invalid units, grids, or silently ambiguous conventions."""

    _require_finite_positive("planet.mass_earth", config.planet.mass_earth)
    _require_finite_positive(
        "planet.stellar_radius_solar", config.planet.stellar_radius_solar
    )
    if config.planet.rocky_radius_rearth is not None:
        _require_finite_positive(
            "planet.rocky_radius_rearth",
            config.planet.rocky_radius_rearth,
        )
    if not 0.0 <= config.planet.rocky_core_mass_fraction <= 1.0:
        raise ValueError("planet.rocky_core_mass_fraction must lie in [0, 1].")
    for field_name in (
        "pressure_melt_bar",
        "temperature_melt_k",
        "oxygen_fugacity_bar",
        "co_melt_mole_ratio",
        "n_melt_mole_ratio",
        "temperature_base_k",
    ):
        _require_finite_positive(
            f"magma.{field_name}", getattr(config.magma, field_name)
        )
    if not PRESSURE_BASE_BAR <= config.magma.pressure_melt_bar <= 1.0e5:
        raise ValueError("magma.pressure_melt_bar must lie between 10 bar and 10 GPa.")
    if config.magma.input_basis not in (
        "exogibbs_elemental",
        "paper_labelled_operational_mapping",
    ):
        raise ValueError("Unknown magma.input_basis.")
    if config.upper.nlayer != UPPER_NLAYER:
        raise ValueError("The published MELTYQ upper atmosphere uses 100 layers.")
    if not np.isclose(config.upper.pressure_top_bar, PRESSURE_TOP_BAR):
        raise ValueError("The published MELTYQ upper boundary is 1e-10 bar.")
    if not np.isclose(config.upper.pressure_base_bar, PRESSURE_BASE_BAR):
        raise ValueError("MELTYQ and the deep solver join at exactly 10 bar.")
    if config.upper.pressure_top_bar >= config.upper.pressure_base_bar:
        raise ValueError("Upper pressure boundaries are reversed.")
    for field_name in (
        "temperature_1e4_pa_k",
        "temperature_100_pa_k",
        "temperature_1e_minus2_pa_k",
    ):
        _require_finite_positive(
            f"upper.{field_name}", getattr(config.upper, field_name)
        )
    try:
        smoothing_window_size = int(
            config.upper.nlayer
            * (config.upper.smoothing_window_percent / 100.0)
        )
        if smoothing_window_size % 2 == 0:
            smoothing_window_size += 1
        smoothing_window_valid = (
            np.isfinite(config.upper.smoothing_window_percent)
            and config.upper.smoothing_window_percent >= 0.0
            and smoothing_window_size <= config.upper.nlayer
        )
    except TypeError:
        smoothing_window_valid = False
    if not smoothing_window_valid:
        raise ValueError(
            "upper.smoothing_window_percent produces an invalid smoothing window."
        )
    if config.cloud.enabled:
        _require_finite_positive(
            "cloud.top_pressure_pa", config.cloud.top_pressure_pa
        )
        if not (
            config.upper.pressure_top_bar * 1.0e5
            <= config.cloud.top_pressure_pa
            <= config.upper.pressure_base_bar * 1.0e5
        ):
            raise ValueError("cloud.top_pressure_pa lies outside the upper atmosphere.")
    if config.haze.enabled:
        for field_name in (
            "particle_radius_micron",
            "q0",
            "particle_number_density_m3",
            "mid_pressure_pa",
            "log10_pressure_extent",
        ):
            _require_finite_positive(
                f"haze.{field_name}", getattr(config.haze, field_name)
            )
        with np.errstate(over="ignore", invalid="ignore"):
            haze_half_width_factor = np.power(
                10.0,
                0.5 * config.haze.log10_pressure_extent,
            )
        haze_top_pressure_pa = (
            config.haze.mid_pressure_pa / haze_half_width_factor
        )
        haze_bottom_pressure_pa = (
            config.haze.mid_pressure_pa * haze_half_width_factor
        )
        atmosphere_top_pressure_pa = config.upper.pressure_top_bar * 1.0e5
        atmosphere_bottom_pressure_pa = config.upper.pressure_base_bar * 1.0e5
        if not np.isfinite(haze_half_width_factor) or not (
            atmosphere_top_pressure_pa <= haze_top_pressure_pa
            and haze_bottom_pressure_pa <= atmosphere_bottom_pressure_pa
        ):
            raise ValueError(
                "The complete Lee haze pressure interval must lie inside "
                "the upper atmosphere."
            )
    if config.haze.vertical_profile not in ("exp_decay", "constant"):
        raise ValueError("Unknown haze.vertical_profile.")
    if config.spectral.opacity_mode not in ("ckd", "diffgrid"):
        raise ValueError("spectral.opacity_mode must be 'ckd' or 'diffgrid'.")
    if config.spectral.radiative_transfer_scheme not in (
        "taurex_rectangle",
        "exojax_simpson",
    ):
        raise ValueError("Unknown spectral.radiative_transfer_scheme.")
    if config.spectral.rayleigh_provider not in ("taurex", "exojax"):
        raise ValueError("Unknown spectral.rayleigh_provider.")
    _require_finite_positive(
        "spectral.wavelength_min_micron",
        config.spectral.wavelength_min_micron,
    )
    _require_finite_positive(
        "spectral.wavelength_max_micron",
        config.spectral.wavelength_max_micron,
    )
    if (
        config.spectral.wavelength_max_micron
        <= config.spectral.wavelength_min_micron
    ):
        raise ValueError("The spectral wavelength range is reversed.")
    if config.observations.niriss_resolution not in ("lowres", "native"):
        raise ValueError("Unknown observations.niriss_resolution.")
    if config.observations.miri_reduction not in ("jexores", "jexopipe"):
        raise ValueError("Unknown observations.miri_reduction.")
    groups = set(config.observations.offset_group_by_dataset.values())
    expected_datasets = {
        "niriss_soss",
        "nirspec_g235h_nrs1",
        "nirspec_g235h_nrs2",
        "nirspec_g395h_nrs1",
        "nirspec_g395h_nrs2",
        "miri_lrs",
    }
    assigned_datasets = set(config.observations.offset_group_by_dataset)
    if assigned_datasets != expected_datasets:
        missing_datasets = expected_datasets.difference(assigned_datasets)
        extra_datasets = assigned_datasets.difference(expected_datasets)
        details = []
        if missing_datasets:
            details.append("missing " + ", ".join(sorted(missing_datasets)))
        if extra_datasets:
            details.append("unexpected " + ", ".join(sorted(extra_datasets)))
        raise ValueError(
            "Offset group assignments must match the six datasets exactly: "
            + "; ".join(details)
        )
    if (
        config.observations.anchor_offset_group is not None
        and config.observations.anchor_offset_group not in groups
    ):
        raise ValueError("anchor_offset_group is not assigned to any dataset.")
    missing_offsets = groups.difference(config.observations.offsets_ppm)
    extra_offsets = set(config.observations.offsets_ppm).difference(groups)
    if missing_offsets or extra_offsets:
        details = []
        if missing_offsets:
            details.append("missing " + ", ".join(sorted(missing_offsets)))
        if extra_offsets:
            details.append("unused " + ", ".join(sorted(extra_offsets)))
        raise ValueError(
            "offsets_ppm must match assigned groups exactly: "
            + "; ".join(details)
        )
    for group, offset_ppm in config.observations.offsets_ppm.items():
        try:
            finite = bool(np.isfinite(offset_ppm))
        except TypeError:
            finite = False
        if not finite:
            raise ValueError(f"observations.offsets_ppm[{group!r}] must be finite.")


def provider_magma_parameters(config: MagmaConfig) -> Figure8Parameters:
    """Map an explicit melt-input convention to ExoGibbs public inputs."""

    co_melt = config.co_melt_mole_ratio
    n_melt = config.n_melt_mole_ratio
    if config.input_basis == "paper_labelled_operational_mapping":
        co_melt *= PAPER_TO_PROVIDER_FACTORS["co_melt_mole_ratio"]
        n_melt *= PAPER_TO_PROVIDER_FACTORS["n_melt_mole_ratio"]
    return Figure8Parameters(
        pressure_melt_bar=config.pressure_melt_bar,
        temperature_melt_k=config.temperature_melt_k,
        oxygen_fugacity_bar=config.oxygen_fugacity_bar,
        co_melt_mole_ratio=co_melt,
        n_melt_mole_ratio=n_melt,
        temperature_base_k=config.temperature_base_k,
    )


def _single_nonempty_h5(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    files = [path for path in sorted(directory.glob("*.h5")) if path.stat().st_size]
    if len(files) > 1:
        raise ValueError(f"Multiple CKD h5 files found in {directory}.")
    return files[0] if files else None


def _validate_ckd_table_identity(table_path: str | Path, species: str):
    """Verify an ExoMolOP filename and lightweight molecular metadata."""

    import h5py
    from exojax.provider.url import petitRADTRANS_ktable_filenames

    table_path = Path(table_path)
    try:
        molecule, isotopologue, line_list = CKD_RELATIVE_PATHS[species].parts
    except (KeyError, ValueError) as error:
        raise ValueError(f"No CKD identity contract for {species}.") from error
    if molecule != species:
        raise ValueError(f"Invalid CKD path contract for {species}.")
    expected_filenames = set(
        petitRADTRANS_ktable_filenames(isotopologue, line_list)
    )
    if table_path.name not in expected_filenames:
        raise ValueError(
            f"Unexpected CKD filename for {species}: {table_path.name}."
        )

    with h5py.File(table_path, "r") as archive:
        if "mol_mass" not in archive:
            raise ValueError(f"CKD table has no mol_mass for {species}.")
        molecular_mass_values = np.asarray(archive["mol_mass"][()]).reshape(-1)
        if molecular_mass_values.size != 1:
            raise ValueError(f"CKD mol_mass must be scalar for {species}.")
        molecular_mass = float(molecular_mass_values[0])
        archive_molecule = None
        if "mol_name" in archive:
            molecule_values = np.asarray(archive["mol_name"][()]).reshape(-1)
            if molecule_values.size != 1:
                raise ValueError(f"CKD mol_name must be scalar for {species}.")
            archive_molecule = molecule_values[0]
            if isinstance(archive_molecule, bytes):
                archive_molecule = archive_molecule.decode("utf-8")
            else:
                archive_molecule = str(archive_molecule)
            if archive_molecule.strip() != species:
                raise ValueError(
                    f"CKD mol_name mismatch for {species}: "
                    f"{archive_molecule!r}."
                )

    expected_molecular_mass = float(mean_molmass_manual(species))
    if (
        not np.isfinite(molecular_mass)
        or molecular_mass <= 0.0
        or not np.isclose(
            molecular_mass,
            expected_molecular_mass,
            # Public ExoMolOP tables store some molecular masses as integers.
            rtol=3.0e-3,
            atol=0.0,
        )
    ):
        raise ValueError(
            f"CKD molecular mass mismatch for {species}: "
            f"{molecular_mass!r}."
        )
    return {
        "molecule": species,
        "isotopologue": isotopologue,
        "line_list": line_list,
        "basename": table_path.name,
        "archive_mol_name": archive_molecule,
        "molecular_mass": molecular_mass,
    }


def load_ckd_opacities(
    opacity_root: str | Path,
    wavenumber_range_cm1,
    *,
    allow_download: bool,
) -> OpacityBundle:
    """Load the fast ExoMolOP opacity path with explicit substitutions."""

    from exojax.opacity import OpaCKD
    from exojax.provider.exomolop import download_exomolop_h5

    opacity_root = Path(opacity_root)
    calculators = {}
    source_files = {}
    for species, relative_path in CKD_RELATIVE_PATHS.items():
        directory = opacity_root / relative_path
        existing = _single_nonempty_h5(directory)
        if existing is None and not allow_download:
            raise FileNotFoundError(
                f"Missing CKD table for {species}: {directory}. "
                "Run with --allow-opacity-download once."
            )
        if existing is None:
            table_path = Path(download_exomolop_h5(directory))
        else:
            table_path = existing
        identity = _validate_ckd_table_identity(table_path, species)
        calculator = OpaCKD.from_external(
            "exomolop",
            str(table_path),
            nurange=wavenumber_range_cm1,
        )
        calculators[species] = calculator
        source_files[species] = {
            "path": str(table_path),
            "sha256": _sha256_file(table_path),
            "identity": identity,
        }

    reference_species = ABSORBING_SPECIES[0]
    reference = calculators[reference_species]
    reference_arrays = {
        "wavenumber": np.asarray(reference.nu_bands),
        "g_grid": np.asarray(reference.ckd_info.ggrid),
        "weights": np.asarray(reference.ckd_info.weights),
        "band_edges": np.asarray(reference.band_edges),
    }
    for species, calculator in calculators.items():
        candidate_arrays = {
            "wavenumber": np.asarray(calculator.nu_bands),
            "g_grid": np.asarray(calculator.ckd_info.ggrid),
            "weights": np.asarray(calculator.ckd_info.weights),
            "band_edges": np.asarray(calculator.band_edges),
        }
        for label, expected in reference_arrays.items():
            candidate = candidate_arrays[label]
            if candidate.shape != expected.shape or not np.array_equal(
                candidate,
                expected,
            ):
                raise ValueError(
                    f"CKD {label} mismatch for {species} and {reference_species}."
                )
    return OpacityBundle(
        mode="ckd",
        calculators=calculators,
        wavenumber_cm1=reference_arrays["wavenumber"],
        weights=reference_arrays["weights"],
        provenance={
            "provider": "ExoMolOP",
            "nominal_resolving_power": 1000,
            "mixture_convention": "matching-g perfect correlation",
            "files": source_files,
            "line_lists": {
                **PAPER_LINE_LISTS,
                "NH3": "CoYuTe (fast-path substitution for paper BYTe)",
            },
            "bounds_policy": (
                "Temperature must be inside every table; pressure outside a "
                "table is endpoint-clamped by OpaCKD and reported in metadata."
            ),
        },
        band_edges_cm1=reference_arrays["band_edges"],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float64_little_endian_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _source_database_inventory_sha256(inventory) -> str:
    """Validate and hash the builder's canonical line-database inventory."""

    if not isinstance(inventory, list) or not inventory:
        raise ValueError("Line-database file inventory must be a nonempty list.")
    relative_paths = []
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Invalid line-database inventory entry.")
        relative_path = item.get("relative_path")
        size_bytes = item.get("size_bytes")
        sha256 = item.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in sha256.lower()
            )
        ):
            raise ValueError("Invalid line-database inventory entry.")
        relative_paths.append(relative_path)
    if relative_paths != sorted(set(relative_paths)):
        raise ValueError("Line-database inventory paths must be unique and sorted.")
    payload = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_diffgrid_build_contract(manifest, declared_resolution):
    """Require the builder's common grid and teacher contract."""

    contract = manifest.get("build_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != (
        "meltyq.figure3.diffgrid-build-contract@1"
    ):
        raise ValueError("Diffgrid manifest has no recognized build_contract.")
    for section in (
        "spectral_grid",
        "pressure_grid",
        "temperature_grid",
        "teacher",
        "software",
    ):
        if not isinstance(contract.get(section), dict):
            raise ValueError(f"Diffgrid build_contract is missing {section}.")
    contract_resolution = float(
        contract["spectral_grid"].get(
            "minimum_pointwise_resolving_power",
            0.0,
        )
    )
    if (
        not np.isfinite(contract_resolution)
        or contract_resolution < declared_resolution
    ):
        raise ValueError("Diffgrid build_contract resolving power is too low.")
    return contract


def _validate_diffgrid_coordinate_contract(
    build_contract,
    wavenumber,
    pressure,
    temperature,
    species,
):
    """Match archive coordinate arrays to the common canonical hashes."""

    checks = (
        ("spectral_grid", wavenumber),
        ("pressure_grid", pressure),
        ("temperature_grid", temperature),
    )
    for section, values in checks:
        expected = build_contract[section].get(
            "float64_little_endian_sha256"
        )
        actual = _float64_little_endian_sha256(values)
        if expected != actual:
            raise ValueError(
                f"Diffgrid {section} contract mismatch for {species}."
            )


def _diffgrid_interpolation_contract(descriptor, species):
    """Validate and return a manifest's required interpolation quality gate."""

    contract = descriptor.get("interpolation_validation", {})
    if contract.get("status") != "passed":
        raise ValueError(
            f"Diffgrid interpolation validation did not pass for {species}."
        )
    try:
        maximum_p99 = float(contract["maximum_p99_log_error"])
        maximum = float(contract["maximum_log_error"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid Diffgrid interpolation limits for {species}."
        ) from error
    if (
        not np.isfinite(maximum_p99)
        or not np.isfinite(maximum)
        or maximum_p99 < 0.0
        or maximum < maximum_p99
    ):
        raise ValueError(
            f"Invalid Diffgrid interpolation limits for {species}."
        )
    return {
        "status": "passed",
        "maximum_p99_log_error": maximum_p99,
        "maximum_log_error": maximum,
    }


def _diffgrid_broadening_contract(descriptor, build_contract, species):
    """Validate the requested and effective pressure-broadening provenance."""

    provenance = descriptor.get("broadening_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema_version")
        != "meltyq.figure3.broadening-provenance@1"
    ):
        raise ValueError(f"Missing broadening provenance for {species}.")
    teacher = build_contract["teacher"]
    requested = teacher.get("requested_background_broadener")
    missing_policy = teacher.get("missing_requested_broadener_policy")
    if (
        not isinstance(requested, str)
        or not requested
        or missing_policy not in ("reject", "allow_definition_file_defaults")
        or provenance.get("requested_background_broadener") != requested
        or provenance.get("missing_requested_broadener_policy")
        != missing_policy
    ):
        raise ValueError(f"Broadening build contract mismatch for {species}.")

    requested_file_present = provenance.get("requested_file_present")
    effective_source = provenance.get("effective_source")
    if requested_file_present is True:
        if effective_source != (
            "requested_file_with_definition_defaults_for_unmatched_lines"
        ):
            raise ValueError(f"Invalid broadening source for {species}.")
        requested_file_sha256 = provenance.get("requested_file_sha256")
        if (
            not isinstance(requested_file_sha256, str)
            or len(requested_file_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in requested_file_sha256.lower()
            )
        ):
            raise ValueError(f"Invalid broadening-file hash for {species}.")
    elif requested_file_present is False:
        if (
            missing_policy != "allow_definition_file_defaults"
            or effective_source
            != "definition_file_defaults_missing_requested_file"
            or provenance.get("requested_file_sha256") is not None
        ):
            raise ValueError(f"Unacknowledged broadening fallback for {species}.")
    else:
        raise ValueError(f"Invalid broadening-file state for {species}.")

    try:
        alpha_default = float(
            provenance["definition_default_alpha_ref_cm1_bar"]
        )
        exponent_default = float(
            provenance["definition_default_temperature_exponent"]
        )
        alpha_range = np.asarray(
            provenance["effective_alpha_ref_range_cm1_bar"],
            dtype=float,
        )
        exponent_range = np.asarray(
            provenance["effective_temperature_exponent_range"],
            dtype=float,
        )
        parameter_count = int(provenance["selected_line_parameter_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid effective broadening parameters for {species}."
        ) from error
    if (
        not np.isfinite(alpha_default)
        or alpha_default <= 0.0
        or not np.isfinite(exponent_default)
        or alpha_range.shape != (2,)
        or exponent_range.shape != (2,)
        or np.any(~np.isfinite(alpha_range))
        or np.any(alpha_range <= 0.0)
        or np.any(~np.isfinite(exponent_range))
        or alpha_range[1] < alpha_range[0]
        or exponent_range[1] < exponent_range[0]
        or parameter_count <= 0
    ):
        raise ValueError(
            f"Invalid effective broadening parameters for {species}."
        )
    if requested_file_present is False and (
        not np.allclose(
            alpha_range,
            alpha_default,
            rtol=1.0e-12,
            atol=0.0,
        )
        or not np.allclose(
            exponent_range,
            exponent_default,
            rtol=1.0e-12,
            atol=0.0,
        )
    ):
        raise ValueError(f"Invalid definition-file fallback for {species}.")
    return provenance


def _validate_archive_interpolation_gate(
    user_meta,
    contract,
    build_contract,
    species,
):
    """Validate every recorded teacher/Diffgrid interpolation measurement."""

    archive_gate = user_meta.get(
        "diffgrid_teacher_interpolation_quality_gate",
        {},
    )
    if archive_gate.get("status") != "passed":
        raise ValueError(
            f"Archive interpolation validation did not pass for {species}."
        )
    if not np.isclose(
        archive_gate.get("maximum_p99", np.nan),
        contract["maximum_p99_log_error"],
        rtol=1.0e-12,
        atol=0.0,
    ) or not np.isclose(
        archive_gate.get("maximum", np.nan),
        contract["maximum_log_error"],
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            f"Archive interpolation limits mismatch for {species}."
        )

    try:
        temperature_contract = build_contract["temperature_grid"]
        teacher_contract = build_contract["teacher"]
        number_temperature_nodes = int(temperature_contract["nodes"])
        minimum_temperature = float(temperature_contract["minimum_k"])
        maximum_temperature = float(temperature_contract["maximum_k"])
        validation_profile_count = int(
            teacher_contract["validation_profile_count"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid interpolation temperature contract for {species}."
        ) from error
    if (
        number_temperature_nodes < 2
        or validation_profile_count != number_temperature_nodes - 1
        or not np.isfinite(minimum_temperature)
        or not np.isfinite(maximum_temperature)
        or minimum_temperature <= 0.0
        or maximum_temperature <= minimum_temperature
        or temperature_contract.get("uniform_in_inverse_temperature") is not True
        or teacher_contract.get("validation_temperature_rule")
        != "every inverse-temperature interval midpoint"
    ):
        raise ValueError(
            f"Invalid interpolation temperature contract for {species}."
        )

    measurements = user_meta.get(
        "diffgrid_teacher_interpolation_validation"
    )
    if (
        not isinstance(measurements, list)
        or len(measurements) != validation_profile_count
    ):
        raise ValueError(
            "Archive must record all interpolation validation profiles for "
            f"{species}."
        )
    inverse_temperature_nodes = np.linspace(
        1.0 / maximum_temperature,
        1.0 / minimum_temperature,
        number_temperature_nodes,
    )
    expected_temperatures = 1.0 / (
        0.5
        * (
            inverse_temperature_nodes[:-1]
            + inverse_temperature_nodes[1:]
        )
    )
    for index, (measurement, expected_temperature) in enumerate(
        zip(measurements, expected_temperatures)
    ):
        if not isinstance(measurement, dict):
            raise ValueError(
                f"Invalid interpolation measurement {index} for {species}."
            )
        try:
            measured_temperature = float(
                measurement["isothermal_temperature_k"]
            )
            p99_error = float(
                measurement[
                    "p99_absolute_log_cross_section_error"
                ]
            )
            maximum_error = float(
                measurement[
                    "maximum_absolute_log_cross_section_error"
                ]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid interpolation measurement {index} for {species}."
            ) from error
        if (
            not np.isfinite(measured_temperature)
            or not np.isclose(
                measured_temperature,
                expected_temperature,
                rtol=1.0e-12,
                atol=0.0,
            )
            or not np.isfinite(p99_error)
            or not np.isfinite(maximum_error)
            or p99_error < 0.0
            or maximum_error < p99_error
            or p99_error > contract["maximum_p99_log_error"]
            or maximum_error > contract["maximum_log_error"]
        ):
            raise ValueError(
                f"Interpolation measurement {index} failed validation for "
                f"{species}."
            )


def _validate_wavenumber_coverage(wavenumber_cm1, required_range_cm1):
    """Require a monotonic grid to cover both requested endpoints."""

    if required_range_cm1 is None:
        return
    wavenumber = np.asarray(wavenumber_cm1, dtype=float)
    required = np.asarray(required_range_cm1, dtype=float)
    if required.shape != (2,) or np.any(~np.isfinite(required)):
        raise ValueError("Required wavenumber range must contain two finite values.")
    required_minimum, required_maximum = np.sort(required)
    tolerance = 64.0 * np.finfo(float).eps * required_maximum
    if (
        wavenumber[0] > required_minimum + tolerance
        or wavenumber[-1] < required_maximum - tolerance
    ):
        raise ValueError(
            "Diffgrid wavenumber grid does not cover the configured spectral range."
        )


def validate_diffgrid_manifest_lightweight(
    manifest_path: str | Path,
    expected_pressure_bar,
    required_wavenumber_range_cm1=None,
) -> Mapping[str, object]:
    """Validate a Diffgrid contract without loading cross-section tensors."""

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != DIFFGRID_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unknown Diffgrid manifest schema_version.")
    declared_resolution = float(manifest.get("minimum_resolving_power", 0.0))
    if not np.isfinite(declared_resolution) or declared_resolution < 50000.0:
        raise ValueError("Diffgrid manifest must certify resolving power >= 50000.")
    build_contract = _validated_diffgrid_build_contract(
        manifest,
        declared_resolution,
    )

    expected_pressure_bar = np.asarray(expected_pressure_bar)
    reference_wavenumber = None
    files = {}
    archive_inputs = {}
    persistent_table_bytes = 0
    for species in ABSORBING_SPECIES:
        try:
            descriptor = manifest["files"][species]
        except KeyError as error:
            raise ValueError(f"Diffgrid manifest is missing {species}.") from error
        expected_descriptor = {
            "molecule": species,
            "isotopologue": PAPER_ISOTOPOLOGUES[species],
            "line_list": PAPER_LINE_LISTS[species],
        }
        for key, expected in expected_descriptor.items():
            if descriptor.get(key) != expected:
                raise ValueError(
                    f"Diffgrid descriptor {key} mismatch for {species}."
                )
        teacher_method = descriptor.get("teacher_method")
        if not isinstance(teacher_method, str) or not teacher_method:
            raise ValueError(f"Diffgrid descriptor teacher_method missing for {species}.")
        source_resolution = float(
            descriptor.get("source_grid_minimum_resolving_power", 0.0)
        )
        if (
            not np.isfinite(source_resolution)
            or source_resolution < declared_resolution
        ):
            raise ValueError(
                f"Diffgrid source grid resolution is too low for {species}."
            )
        interpolation_contract = _diffgrid_interpolation_contract(
            descriptor,
            species,
        )
        broadening_contract = _diffgrid_broadening_contract(
            descriptor,
            build_contract,
            species,
        )
        teacher_contract = build_contract["teacher"]
        if not np.isclose(
            interpolation_contract["maximum_p99_log_error"],
            teacher_contract.get(
                "maximum_p99_absolute_log_cross_section_error",
                np.nan,
            ),
            rtol=1.0e-12,
            atol=0.0,
        ) or not np.isclose(
            interpolation_contract["maximum_log_error"],
            teacher_contract.get(
                "maximum_absolute_log_cross_section_error",
                np.nan,
            ),
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"Diffgrid interpolation contract mismatch for {species}."
            )
        if not np.isclose(
            source_resolution,
            build_contract["spectral_grid"][
                "minimum_pointwise_resolving_power"
            ],
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"Diffgrid source resolution contract mismatch for {species}."
            )
        source_inventory_sha256 = descriptor.get(
            "source_database_inventory_sha256"
        )
        if (
            not isinstance(source_inventory_sha256, str)
            or len(source_inventory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_inventory_sha256.lower()
            )
        ):
            raise ValueError(
                f"Invalid source-database inventory SHA-256 for {species}."
            )

        archive_path = Path(descriptor["path"])
        if not archive_path.is_absolute():
            archive_path = manifest_path.parent / archive_path
        if archive_path.suffix != ".npz":
            raise ValueError("Figure 3 Diffgrid manifests require NPZ archives.")
        metadata_path = archive_path.with_name(
            archive_path.stem + "_metadata.json"
        )
        archive_sha256 = _sha256_file(archive_path)
        metadata_sha256 = _sha256_file(metadata_path)
        if archive_sha256 != descriptor["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {archive_path}.")
        if metadata_sha256 != descriptor["metadata_sha256"]:
            raise ValueError(f"SHA-256 mismatch for {metadata_path}.")

        with metadata_path.open(encoding="utf-8") as stream:
            archive_metadata = json.load(stream)
        if archive_metadata.get("opa_type") != "OpaDiffgrid":
            raise ValueError(f"Archive type mismatch for {species}.")
        if archive_metadata.get("opa_state", {}).get(
            "teacher_method"
        ) != teacher_method:
            raise ValueError(f"Archive teacher method mismatch for {species}.")
        user_meta = archive_metadata.get("user_meta", {})
        if user_meta.get("build_contract") != build_contract:
            raise ValueError(f"Archive build_contract mismatch for {species}.")
        expected_user_meta = {
            "schema_version": DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION,
            **expected_descriptor,
            "source_grid_minimum_resolving_power": source_resolution,
            "source_database_inventory_sha256": source_inventory_sha256,
            "broadening_provenance": broadening_contract,
        }
        for key, expected in expected_user_meta.items():
            actual = user_meta.get(key)
            if isinstance(expected, float):
                try:
                    matches = np.isclose(
                        actual,
                        expected,
                        rtol=1.0e-12,
                        atol=0.0,
                    )
                except TypeError:
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise ValueError(
                    f"Archive user_meta[{key!r}] mismatch for {species}."
                )
        if _source_database_inventory_sha256(
            user_meta.get("line_database_files")
        ) != source_inventory_sha256:
            raise ValueError(
                f"Archive source-database inventory mismatch for {species}."
            )
        _validate_archive_interpolation_gate(
            user_meta,
            interpolation_contract,
            build_contract,
            species,
        )

        shapes = archive_metadata.get("array_shapes", {})
        dtypes = archive_metadata.get("array_dtypes", {})
        table_names = (
            "log_cross_section_grid",
            "log_cross_section_derivative_grid",
        )
        species_table_bytes = 0
        for name in table_names:
            shape = tuple(shapes.get(name, ()))
            if len(shape) != 3:
                raise ValueError(f"Invalid {name} shape metadata for {species}.")
            array_bytes = int(np.prod(shape)) * np.dtype(dtypes[name]).itemsize
            species_table_bytes += array_bytes
            persistent_table_bytes += array_bytes

        with np.load(archive_path, allow_pickle=False) as archive:
            wavenumber = np.asarray(archive["nu_grid"])
            pressure = np.asarray(archive["pressure_grid"])
            temperature_grid = np.asarray(archive["temperature_grid"])
        if pressure.shape != expected_pressure_bar.shape or not np.allclose(
            pressure,
            expected_pressure_bar,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(f"Diffgrid pressure mismatch for {species}.")
        if reference_wavenumber is None:
            reference_wavenumber = wavenumber
        elif wavenumber.shape != reference_wavenumber.shape or not np.allclose(
            wavenumber,
            reference_wavenumber,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(f"Diffgrid wavenumber mismatch for {species}.")
        expected_table_shape = (
            pressure.size,
            int(shapes["temperature_grid"][0]),
            wavenumber.size,
        )
        if (
            temperature_grid.shape != tuple(shapes["temperature_grid"])
            or temperature_grid.ndim != 1
            or np.any(~np.isfinite(temperature_grid))
            or np.any(temperature_grid <= 0.0)
            or np.any(np.diff(temperature_grid) >= 0.0)
        ):
            raise ValueError(f"Invalid temperature grid for {species}.")
        _validate_diffgrid_coordinate_contract(
            build_contract,
            wavenumber,
            pressure,
            temperature_grid,
            species,
        )
        for name in table_names:
            if tuple(shapes[name]) != expected_table_shape:
                raise ValueError(f"{name} dimensions mismatch for {species}.")
        files[species] = {
            **descriptor,
            "resolved_path": str(archive_path),
            "resolved_metadata_path": str(metadata_path),
        }
        molmass = archive_metadata.get("opa_state", {}).get(
            "optional_attributes",
            {},
        ).get("molmass")
        if molmass is None:
            molmass = mean_molmass_manual(species)
        if not np.isfinite(molmass) or molmass <= 0.0:
            raise ValueError(f"Invalid molecular mass for {species}.")
        archive_inputs[species] = {
            "archive_path": archive_path,
            "temperature_grid": temperature_grid,
            "pressure_grid": pressure,
            "molmass": float(molmass),
            "persistent_table_bytes": species_table_bytes,
        }

    spacing = np.diff(reference_wavenumber)
    if np.any(~np.isfinite(reference_wavenumber)) or np.any(spacing <= 0.0):
        raise ValueError("Diffgrid wavenumber must be finite and increasing.")
    resolving_power = reference_wavenumber[:-1] / spacing
    measured_minimum = float(
        resolution_eslog(reference_wavenumber, definition="pointwise")
    )
    resolution_tolerance = (
        64.0 * np.finfo(float).eps * declared_resolution
    )
    if measured_minimum + resolution_tolerance < declared_resolution:
        raise ValueError("Diffgrid grid is below its declared minimum resolution.")
    _validate_wavenumber_coverage(
        reference_wavenumber,
        required_wavenumber_range_cm1,
    )
    return {
        "files": files,
        "archive_inputs": archive_inputs,
        "build_contract": build_contract,
        "declared_minimum_resolving_power": declared_resolution,
        "wavenumber_cm1": reference_wavenumber,
        "measured_minimum_grid_resolving_power": measured_minimum,
        "measured_median_grid_resolving_power": float(
            np.median(resolving_power)
        ),
        "persistent_cross_section_table_gib": persistent_table_bytes / 2.0**30,
    }


def load_diffgrid_opacities(
    manifest_path: str | Path,
    expected_pressure_bar,
    required_wavenumber_range_cm1=None,
    *,
    lazy: bool = True,
) -> OpacityBundle:
    """Load hash-pinned R=50000 ExoJAX Diffgrid NPZ archives."""

    validation = validate_diffgrid_manifest_lightweight(
        manifest_path,
        expected_pressure_bar,
        required_wavenumber_range_cm1,
    )
    if lazy:
        calculators = {
            species: DiffgridArchiveOpacity(**validation["archive_inputs"][species])
            for species in ABSORBING_SPECIES
        }
        per_species_table_gib = {
            species: (
                calculators[species].persistent_table_bytes / 2.0**30
            )
            for species in ABSORBING_SPECIES
        }
        return OpacityBundle(
            mode="diffgrid",
            calculators=calculators,
            wavenumber_cm1=validation["wavenumber_cm1"],
            weights=None,
            provenance={
                "provider": "ExoJAX OpaDiffgrid",
                "nominal_resolving_power": validation[
                    "declared_minimum_resolving_power"
                ],
                "measured_minimum_grid_resolving_power": validation[
                    "measured_minimum_grid_resolving_power"
                ],
                "measured_median_grid_resolving_power": validation[
                    "measured_median_grid_resolving_power"
                ],
                "mixture_convention": (
                    "cross sections summed wavelength by wavelength"
                ),
                "loading_strategy": (
                    "one archive at a time: load, temperature-interpolate, "
                    "synchronize, and release"
                ),
                "persistent_table_total_gib": validation[
                    "persistent_cross_section_table_gib"
                ],
                "persistent_table_peak_one_species_gib": max(
                    per_species_table_gib.values()
                ),
                "persistent_table_gib_by_species": per_species_table_gib,
                "build_contract": validation["build_contract"],
                "files": validation["files"],
            },
        )

    from exojax.opacity import OpaDiffgrid

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != DIFFGRID_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unknown Diffgrid manifest schema_version.")
    declared_resolution = float(manifest.get("minimum_resolving_power", 0.0))
    if not np.isfinite(declared_resolution) or declared_resolution < 50000.0:
        raise ValueError("Diffgrid manifest must certify resolving power >= 50000.")

    calculators = {}
    provenance_files = {}
    for species in ABSORBING_SPECIES:
        try:
            descriptor = manifest["files"][species]
        except KeyError as error:
            raise ValueError(f"Diffgrid manifest is missing {species}.") from error
        if descriptor.get("molecule") != species:
            raise ValueError(f"Diffgrid descriptor molecule mismatch for {species}.")
        if descriptor.get("isotopologue") != PAPER_ISOTOPOLOGUES[species]:
            raise ValueError(
                f"{species} must use isotopologue {PAPER_ISOTOPOLOGUES[species]}."
            )
        if descriptor.get("line_list") != PAPER_LINE_LISTS[species]:
            raise ValueError(
                f"{species} must use paper line list {PAPER_LINE_LISTS[species]}."
            )
        archive_path = Path(descriptor["path"])
        if not archive_path.is_absolute():
            archive_path = manifest_path.parent / archive_path
        if archive_path.suffix != ".npz":
            raise ValueError("Figure 3 Diffgrid manifests require NPZ archives.")
        metadata_path = archive_path.with_name(
            archive_path.stem + "_metadata.json"
        )
        teacher_method = descriptor.get("teacher_method")
        if not isinstance(teacher_method, str) or not teacher_method:
            raise ValueError(f"Diffgrid descriptor teacher_method missing for {species}.")
        source_resolution = float(
            descriptor.get("source_grid_minimum_resolving_power", 0.0)
        )
        if (
            not np.isfinite(source_resolution)
            or source_resolution < declared_resolution
        ):
            raise ValueError(
                f"Diffgrid source grid resolution is too low for {species}."
            )
        interpolation_contract = _diffgrid_interpolation_contract(
            descriptor,
            species,
        )
        source_inventory_sha256 = descriptor.get(
            "source_database_inventory_sha256"
        )
        if (
            not isinstance(source_inventory_sha256, str)
            or len(source_inventory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_inventory_sha256.lower()
            )
        ):
            raise ValueError(
                f"Invalid source-database inventory SHA-256 for {species}."
            )
        expected_sha256 = descriptor["sha256"]
        actual_sha256 = _sha256_file(archive_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {archive_path}: expected "
                f"{expected_sha256}, got {actual_sha256}."
            )
        expected_metadata_sha256 = descriptor["metadata_sha256"]
        actual_metadata_sha256 = _sha256_file(metadata_path)
        if actual_metadata_sha256 != expected_metadata_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {metadata_path}: expected "
                f"{expected_metadata_sha256}, got {actual_metadata_sha256}."
            )
        calculator = OpaDiffgrid.from_saved_opa(str(archive_path), strict=False)
        calculator.check_pressure_grid(expected_pressure_bar)
        if calculator.teacher_method != teacher_method:
            raise ValueError(
                f"Diffgrid teacher method mismatch for {species}: expected "
                f"{teacher_method!r}, got {calculator.teacher_method!r}."
            )
        expected_user_meta = {
            "schema_version": DIFFGRID_ARCHIVE_PROVENANCE_SCHEMA_VERSION,
            "molecule": species,
            "isotopologue": PAPER_ISOTOPOLOGUES[species],
            "line_list": PAPER_LINE_LISTS[species],
            "source_database_inventory_sha256": source_inventory_sha256,
        }
        for key, expected in expected_user_meta.items():
            actual = calculator.user_meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"Diffgrid archive user_meta[{key!r}] mismatch for "
                    f"{species}: expected {expected!r}, got {actual!r}."
                )
        _validate_archive_interpolation_gate(
            calculator.user_meta,
            interpolation_contract,
            validation["build_contract"],
            species,
        )
        archive_source_resolution = float(
            calculator.user_meta.get(
                "source_grid_minimum_resolving_power",
                0.0,
            )
        )
        if not np.isclose(
            archive_source_resolution,
            source_resolution,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"Diffgrid archive source-grid provenance mismatch for {species}."
            )
        calculators[species] = calculator
        provenance_files[species] = {
            **descriptor,
            "resolved_path": str(archive_path),
            "resolved_metadata_path": str(metadata_path),
            "sha256_verified": actual_sha256,
            "metadata_sha256_verified": actual_metadata_sha256,
            "archive_user_meta_verified": True,
        }

    reference_species = ABSORBING_SPECIES[0]
    reference_wavenumber = np.asarray(calculators[reference_species].nu_grid)
    for species, calculator in calculators.items():
        candidate = np.asarray(calculator.nu_grid)
        if candidate.shape != reference_wavenumber.shape or not np.allclose(
            candidate,
            reference_wavenumber,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                f"Diffgrid wavenumber mismatch for {species} and "
                f"{reference_species}."
            )
    spacing = np.diff(reference_wavenumber)
    if np.any(spacing <= 0.0):
        raise ValueError("Diffgrid wavenumber must be strictly increasing.")
    resolving_power = reference_wavenumber[:-1] / spacing
    measured_minimum_resolution = float(
        resolution_eslog(reference_wavenumber, definition="pointwise")
    )
    resolution_tolerance = (
        64.0 * np.finfo(float).eps * declared_resolution
    )
    if measured_minimum_resolution + resolution_tolerance < declared_resolution:
        raise ValueError(
            "Diffgrid wavenumber sampling is inconsistent with its declared "
            "minimum resolving power."
        )
    _validate_wavenumber_coverage(
        reference_wavenumber,
        required_wavenumber_range_cm1,
    )
    return OpacityBundle(
        mode="diffgrid",
        calculators=calculators,
        wavenumber_cm1=reference_wavenumber,
        weights=None,
        provenance={
            "provider": "ExoJAX OpaDiffgrid",
            "nominal_resolving_power": declared_resolution,
            "measured_minimum_grid_resolving_power": measured_minimum_resolution,
            "measured_median_grid_resolving_power": float(
                np.median(resolving_power)
            ),
            "mixture_convention": "cross sections summed wavelength by wavelength",
            "files": provenance_files,
        },
    )


def build_art(config: UpperConfig, wavenumber_cm1):
    """Construct ExoJAX transmission layers from exact pressure boundaries."""

    from exojax.rt import ArtTransPure

    return ArtTransPure.from_pressure_boundaries(
        pressure_top_boundary=config.pressure_top_bar,
        pressure_btm_boundary=config.pressure_base_bar,
        nlayer=config.nlayer,
        integration="simpson",
        nu_grid=wavenumber_cm1,
    )


def _validate_cia_interpolation_grid(database, pair: str) -> None:
    """Require the rectangular, unique grid used by the pinned CIA files."""

    native_wavenumber = np.asarray(database.nucia, dtype=float)
    native_temperature = np.asarray(database.tcia, dtype=float)
    native_log_coefficient = np.asarray(database.logac, dtype=float)
    if (
        native_wavenumber.ndim != 1
        or native_wavenumber.size < 2
        or np.any(~np.isfinite(native_wavenumber))
        or np.any(np.diff(native_wavenumber) <= 0.0)
    ):
        raise ValueError(
            f"{pair} CIA wavenumbers must be finite, unique, and "
            "strictly increasing for TauREx interpolation."
        )
    if (
        native_temperature.ndim != 1
        or native_temperature.size < 2
        or np.any(~np.isfinite(native_temperature))
        or np.any(np.diff(native_temperature) <= 0.0)
    ):
        raise ValueError(
            f"{pair} CIA temperatures must be finite, unique, and "
            "strictly increasing for TauREx interpolation."
        )
    if native_log_coefficient.shape != (
        native_temperature.size,
        native_wavenumber.size,
    ) or np.any(~np.isfinite(native_log_coefficient)):
        raise ValueError(f"{pair} CIA coefficients form an invalid grid.")


def _validate_cia_file_identity(path: str | Path, expected_pair: str) -> None:
    """Require the HITRAN CIA header to identify the configured pair."""

    path = Path(path)
    with path.open(encoding="ascii") as stream:
        header = stream.readline().split()
    if not header or header[0] != expected_pair:
        actual_pair = header[0] if header else None
        raise ValueError(
            f"CIA pair mismatch for {path}: expected {expected_pair!r}, "
            f"got {actual_pair!r}."
        )


def load_cia_opacities(
    opacity_root: str | Path,
    wavenumber_cm1,
    *,
    allow_download: bool,
):
    """Load H2-H2 and H2-He CIA and record their interpolation coverage."""

    from exojax.database import CdbCIA
    from exojax.opacity import OpaCIA

    opacity_root = Path(opacity_root)
    result = {}
    provenance = {}
    for pair, filename in CIA_FILENAMES.items():
        path = opacity_root / filename
        if not path.exists() and not allow_download:
            raise FileNotFoundError(
                f"Missing CIA table {path}. Run with --allow-opacity-download once."
            )
        if path.exists():
            _validate_cia_file_identity(path, pair)
        database = CdbCIA(str(path), nurange=wavenumber_cm1)
        _validate_cia_file_identity(path, pair)
        _validate_cia_interpolation_grid(database, pair)
        calculator = OpaCIA(
            database,
            nu_grid=wavenumber_cm1,
            wavenumber_interpolation="interp",
        )
        coverage = (
            np.asarray(wavenumber_cm1) >= float(jnp.min(database.nucia))
        ) & (np.asarray(wavenumber_cm1) <= float(jnp.max(database.nucia)))
        result[pair] = (calculator, jnp.asarray(coverage))
        provenance[pair] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "wavenumber_min_cm1": float(jnp.min(database.nucia)),
            "wavenumber_max_cm1": float(jnp.max(database.nucia)),
            "temperature_min_k": float(jnp.min(database.tcia)),
            "temperature_max_k": float(jnp.max(database.tcia)),
            "implementation": "exojax.opacity.OpaCIA",
            "wavenumber_interpolation": "interp",
            "coverage_by_scheme": {
                "taurex_rectangle": (
                    "use constant edge values outside native coverage"
                ),
                "exojax_simpson": (
                    "set coefficients to zero outside native coverage"
                ),
            },
        }
    return result, provenance


def build_rayleigh_cross_sections(wavenumber_cm1, provider):
    """Precompute Rayleigh cross sections for all nine gases."""

    if provider == "taurex":
        return {
            species: jnp.asarray(
                taurex_rayleigh_cross_section_cm2(species, wavenumber_cm1)
            )
            for species in RAYLEIGH_SPECIES
            if species in TAUREX_RAYLEIGH_SPECIES
        }
    if provider == "exojax":
        from exojax.opacity import OpaRayleigh

        return {
            species: OpaRayleigh(wavenumber_cm1, species).xsvector()
            for species in RAYLEIGH_SPECIES
        }
    raise ValueError("Rayleigh provider must be 'taurex' or 'exojax'.")


def taurex_rectangle_transit_radius_squared(
    normalized_layer_height_top_to_bottom,
    dtau,
):
    """Apply TauREx's legacy chord path and rectangle area quadrature.

    The returned value is normalized by the square of the lower-boundary
    radius, as in ``ArtTransPure.run``.
    """

    return taurex_old_path_rectangle_transit_radius_squared(
        normalized_layer_height_top_to_bottom,
        dtau,
    )


def make_spectral_kernel(
    art,
    opacity_bundle: OpacityBundle,
    cia_opacities,
    rayleigh_cross_sections,
    *,
    cloud_enabled: bool,
    haze_enabled: bool,
    haze_vertical_profile: str,
    radiative_transfer_scheme: str = "exojax_simpson",
):
    """Create one JIT kernel for total and all component spectra.

    All scenarios are stacked before a single ``vmap`` radiative-transfer call,
    so requesting component curves does not trigger one compilation per curve.
    """

    if radiative_transfer_scheme not in (
        "taurex_rectangle",
        "exojax_simpson",
    ):
        raise ValueError("Unknown radiative-transfer scheme.")

    species_index = {name: index for index, name in enumerate(MELTYQ_SPECIES)}
    molar_masses = jnp.asarray(
        [mean_molmass_manual(species) for species in MELTYQ_SPECIES]
    )
    wavenumber = jnp.asarray(opacity_bundle.wavenumber_cm1)
    cia_pairs = tuple(cia_opacities)
    rayleigh_species = tuple(rayleigh_cross_sections)
    molecular_mass_values = []
    for species in ABSORBING_SPECIES:
        molecular_mass = getattr(
            opacity_bundle.calculators[species],
            "molmass",
            None,
        )
        if molecular_mass is None:
            molecular_mass = molar_masses[species_index[species]]
        molecular_mass_values.append(molecular_mass)
    molecular_masses = jnp.asarray(molecular_mass_values)
    pressure_bar = jnp.asarray(art.pressure)
    pressure_boundary_bar = jnp.asarray(art.pressure_boundary)
    pressure_pa = pressure_bar * 1.0e5
    is_ckd = opacity_bundle.mode == "ckd"
    if is_ckd:
        weights = jnp.asarray(opacity_bundle.weights)
        spectral_shape = (
            art.nlayer,
            weights.size,
            wavenumber.size,
        )
    else:
        weights = None
        spectral_shape = (art.nlayer, wavenumber.size)

    def kernel(
        temperature_k,
        base_mole_fractions,
        molecular_cross_sections,
        log_cia_coefficients,
        cia_coverage_masks,
        rayleigh_cross_section_matrix,
        radius_base_cm,
        gravity_base_cgs,
        cloud_top_pressure_pa,
        haze_particle_radius_micron,
        haze_q0,
        haze_particle_number_density_m3,
        haze_mid_pressure_pa,
        haze_log10_pressure_extent,
    ):
        mean_molecular_weight = jnp.sum(base_mole_fractions * molar_masses)
        mmw_profile = jnp.full_like(temperature_k, mean_molecular_weight)
        if radiative_transfer_scheme == "taurex_rectangle":
            radius_boundaries_cm, _ = hydrostatic_radius_profile_ideal_gas(
                pressure_boundary_bar,
                temperature_k,
                mmw_profile,
                radius_base_cm,
                gravity_base_cgs,
                hydrostatic_scheme="layer_constant_gravity",
            )
            taurex_layer_height_cm = (
                radius_boundaries_cm[:-1] - radius_boundaries_cm[1:]
            )
            taurex_number_density_cm3 = number_density(
                pressure_bar,
                temperature_k,
            )
            normalized_taurex_layer_height = (
                taurex_layer_height_cm / radius_base_cm
            )
            gravity_profile = None
        else:
            gravity_profile = art.gravity_profile(
                temperature_k,
                mmw_profile,
                radius_base_cm,
                gravity_base_cgs,
            )
        rayleigh_plus_cia = jnp.zeros(
            spectral_shape,
            dtype=temperature_k.dtype,
        )
        aerosols = jnp.zeros(spectral_shape, dtype=temperature_k.dtype)

        h2_vmr = jnp.full_like(
            temperature_k, base_mole_fractions[species_index["H2"]]
        )
        he_vmr = jnp.full_like(
            temperature_k, base_mole_fractions[species_index["He"]]
        )
        for cia_index, pair in enumerate(cia_pairs):
            log_cia_coefficient = log_cia_coefficients[cia_index]
            coverage = cia_coverage_masks[cia_index]
            if pair == "H2-H2":
                vmr_a, vmr_b = h2_vmr, h2_vmr
            else:
                vmr_a, vmr_b = h2_vmr, he_vmr
            if radiative_transfer_scheme == "taurex_rectangle":
                dtau_cia = layer_optical_depth_from_log_cia(
                    log_cia_coefficient,
                    vmr_a * taurex_number_density_cm3,
                    vmr_b * taurex_number_density_cm3,
                    taurex_layer_height_cm,
                )
            else:
                dtau_cia = art.opacity_profile_cia(
                    log_cia_coefficient,
                    temperature_k,
                    vmr_a,
                    vmr_b,
                    mmw_profile,
                    gravity_profile,
                )
            dtau_cia = dtau_cia * coverage[None, :]
            rayleigh_plus_cia = rayleigh_plus_cia + (
                dtau_cia[:, None, :] if is_ckd else dtau_cia
            )

        for rayleigh_index, species in enumerate(rayleigh_species):
            cross_section = rayleigh_cross_section_matrix[rayleigh_index]
            vmr_profile = jnp.full_like(
                temperature_k,
                base_mole_fractions[species_index[species]],
            )
            if radiative_transfer_scheme == "taurex_rectangle":
                absorber_number_column = (
                    vmr_profile
                    * taurex_number_density_cm3
                    * taurex_layer_height_cm
                )
                dtau_rayleigh = layer_optical_depth_from_cross_section(
                    jnp.abs(cross_section),
                    absorber_number_column,
                )
            else:
                dtau_rayleigh = art.opacity_profile_xs(
                    cross_section,
                    vmr_profile,
                    mmw_profile,
                    gravity_profile,
                )
            rayleigh_plus_cia = rayleigh_plus_cia + (
                dtau_rayleigh[:, None, :] if is_ckd else dtau_rayleigh
            )

        if cloud_enabled:
            dtau_cloud = gray_cloud_layer_optical_depth(
                pressure_pa,
                wavenumber,
                cloud_top_pressure_pa,
            )
            # TauREx expresses an opaque deck as infinity.  ExoJAX's chord
            # matrix contains structural zeros, for which 0 * inf is NaN.
            # A finite value this large is already completely opaque while
            # preserving those exact zeros.
            dtau_cloud = jnp.nan_to_num(
                dtau_cloud,
                posinf=OPAQUE_CLOUD_VERTICAL_DTAU,
            )
            aerosols = aerosols + (
                dtau_cloud[:, None, :] if is_ckd else dtau_cloud
            )

        if haze_enabled:
            extinction_m_inv = lee_extinction_coefficient_m_inv(
                pressure_pa,
                wavenumber,
                haze_particle_radius_micron,
                haze_q0,
                haze_particle_number_density_m3,
                haze_mid_pressure_pa,
                haze_log10_pressure_extent,
                haze_vertical_profile,
            )
            if radiative_transfer_scheme == "taurex_rectangle":
                haze_layer_height_cm = taurex_layer_height_cm
            else:
                normalized_height, _ = art.atmosphere_height(
                    temperature_k,
                    mmw_profile,
                    radius_base_cm,
                    gravity_base_cgs,
                )
                haze_layer_height_cm = normalized_height * radius_base_cm
            dtau_haze = layer_optical_depth_from_extinction(
                extinction_m_inv * 1.0e-2,
                haze_layer_height_cm,
            )
            aerosols = aerosols + (
                dtau_haze[:, None, :] if is_ckd else dtau_haze
            )

        molecular_depths = []
        for molecule_index, species in enumerate(ABSORBING_SPECIES):
            cross_section = molecular_cross_sections[molecule_index]
            molecular_mass = molecular_masses[molecule_index]
            vmr = base_mole_fractions[species_index[species]]
            if radiative_transfer_scheme == "taurex_rectangle":
                vmr_profile = jnp.full_like(temperature_k, vmr)
                absorber_number_column = (
                    vmr_profile
                    * taurex_number_density_cm3
                    * taurex_layer_height_cm
                )
                dtau_molecule = layer_optical_depth_from_cross_section(
                    jnp.abs(cross_section),
                    absorber_number_column,
                )
            else:
                mmr_profile = jnp.full_like(
                    temperature_k,
                    vmr * molecular_mass / mean_molecular_weight,
                )
                opacity_profile = (
                    art.opacity_profile_xs_ckd
                    if is_ckd
                    else art.opacity_profile_xs
                )
                dtau_molecule = opacity_profile(
                    cross_section,
                    mmr_profile,
                    molecular_mass,
                    gravity_profile,
                )
            molecular_depths.append(dtau_molecule)

        total = aerosols + rayleigh_plus_cia
        for molecular_depth in molecular_depths:
            total = total + molecular_depth
        scenarios = jnp.stack(
            [total, aerosols, rayleigh_plus_cia] + molecular_depths
        )

        if radiative_transfer_scheme == "taurex_rectangle":
            if is_ckd:
                def run_one(dtau):
                    flattened = dtau.reshape(
                        (art.nlayer, weights.size * wavenumber.size)
                    )
                    radius_squared = taurex_rectangle_transit_radius_squared(
                        normalized_taurex_layer_height,
                        flattened,
                    ).reshape((weights.size, wavenumber.size))
                    return jnp.einsum("g,gn->n", weights, radius_squared)
            else:
                run_one = lambda dtau: taurex_rectangle_transit_radius_squared(
                    normalized_taurex_layer_height,
                    dtau,
                )
        elif is_ckd:
            run_one = lambda dtau: art.run_ckd(
                dtau,
                temperature_k,
                mmw_profile,
                radius_base_cm,
                gravity_base_cgs,
                weights,
            )
        else:
            run_one = lambda dtau: art.run(
                dtau,
                temperature_k,
                mmw_profile,
                radius_base_cm,
                gravity_base_cgs,
            )
        normalized_radius_squared = jax.vmap(run_one)(scenarios)
        return normalized_radius_squared, mean_molecular_weight

    return jax.jit(kernel)


def prepare_spectral_opacity(
    opacity_bundle: OpacityBundle,
    cia_opacities,
    rayleigh_cross_sections,
    temperature_k,
    pressure_bar,
    *,
    radiative_transfer_scheme="exojax_simpson",
) -> PreparedSpectralOpacity:
    """Evaluate opacity tables before tracing the radiative-transfer kernel.

    In particular, the large Diffgrid value and derivative arrays remain
    dynamic arguments of ExoJAX's interpolation JIT instead of becoming
    closed-over constants in the radiative-transfer executable.
    """

    start = time.perf_counter()
    molecular = []
    for species in ABSORBING_SPECIES:
        calculator = opacity_bundle.calculators[species]
        if opacity_bundle.mode == "ckd":
            cross_section = calculator.xstensor_ckd(
                temperature_k,
                pressure_bar,
            )
        else:
            cross_section = calculator.xsmatrix(temperature_k)
        molecular.append(cross_section)
    molecular_cross_sections = jnp.stack(molecular)

    log_cia = [
        calculator.logacia_matrix(temperature_k)
        for calculator, _ in cia_opacities.values()
    ]
    if radiative_transfer_scheme == "taurex_rectangle":
        cia_coverage = [
            jnp.ones_like(coverage, dtype=bool)
            for _, coverage in cia_opacities.values()
        ]
    elif radiative_transfer_scheme == "exojax_simpson":
        cia_coverage = [coverage for _, coverage in cia_opacities.values()]
    else:
        raise ValueError("Unknown radiative-transfer scheme.")
    spectral_points = int(np.asarray(opacity_bundle.wavenumber_cm1).size)
    if log_cia:
        log_cia_coefficients = jnp.stack(log_cia)
        cia_coverage_masks = jnp.stack(cia_coverage)
    else:
        log_cia_coefficients = jnp.empty(
            (0, temperature_k.size, spectral_points),
            dtype=temperature_k.dtype,
        )
        cia_coverage_masks = jnp.empty(
            (0, spectral_points),
            dtype=bool,
        )

    if rayleigh_cross_sections:
        rayleigh_cross_section_matrix = jnp.stack(
            list(rayleigh_cross_sections.values())
        )
    else:
        rayleigh_cross_section_matrix = jnp.empty(
            (0, spectral_points),
            dtype=temperature_k.dtype,
        )
    arrays = (
        molecular_cross_sections,
        log_cia_coefficients,
        cia_coverage_masks,
        rayleigh_cross_section_matrix,
    )
    jax.block_until_ready(arrays)
    return PreparedSpectralOpacity(
        molecular_cross_sections=molecular_cross_sections,
        log_cia_coefficients=log_cia_coefficients,
        cia_coverage_masks=cia_coverage_masks,
        rayleigh_cross_sections=rayleigh_cross_section_matrix,
        preparation_seconds_including_any_compilation=(
            time.perf_counter() - start
        ),
    )


def run_spectral_kernel(
    kernel,
    config: Figure3Config,
    art,
    opacity_bundle,
    cia_opacities,
    rayleigh_cross_sections,
    base_mole_fractions,
    radius_base_rearth,
    *,
    benchmark_repeats: int,
):
    """Evaluate and time one compiled spectral kernel shape."""

    radius_base_cm = float(radius_base_rearth) * RE
    gravitational_constant_cgs = (
        TAUREX_GRAVITATIONAL_CONSTANT_CGS
        if config.spectral.radiative_transfer_scheme == "taurex_rectangle"
        else G
    )
    gravity_base_cgs = (
        gravitational_constant_cgs
        * config.planet.mass_earth
        * ME
        / radius_base_cm**2
    )
    temperature_k = upper_temperature_profile_k(
        art.pressure,
        config.magma.temperature_base_k,
        config.upper.temperature_1e4_pa_k,
        config.upper.temperature_100_pa_k,
        config.upper.temperature_1e_minus2_pa_k,
        config.upper.smoothing_window_percent,
    )
    prepared_opacity = prepare_spectral_opacity(
        opacity_bundle,
        cia_opacities,
        rayleigh_cross_sections,
        temperature_k,
        art.pressure,
        radiative_transfer_scheme=config.spectral.radiative_transfer_scheme,
    )
    arguments = (
        temperature_k,
        jnp.asarray(base_mole_fractions),
        prepared_opacity.molecular_cross_sections,
        prepared_opacity.log_cia_coefficients,
        prepared_opacity.cia_coverage_masks,
        prepared_opacity.rayleigh_cross_sections,
        jnp.asarray(radius_base_cm),
        jnp.asarray(gravity_base_cgs),
        jnp.asarray(
            config.cloud.top_pressure_pa
            if config.cloud.enabled
            else config.upper.pressure_base_bar * 1.0e5
        ),
        jnp.asarray(
            config.haze.particle_radius_micron
            if config.haze.enabled
            else 1.0
        ),
        jnp.asarray(config.haze.q0 if config.haze.enabled else 1.0),
        jnp.asarray(
            config.haze.particle_number_density_m3
            if config.haze.enabled
            else 1.0
        ),
        jnp.asarray(
            config.haze.mid_pressure_pa
            if config.haze.enabled
            else 1.0
        ),
        jnp.asarray(
            config.haze.log10_pressure_extent
            if config.haze.enabled
            else 1.0
        ),
    )
    start = time.perf_counter()
    normalized_radius_squared, mean_molecular_weight = kernel(*arguments)
    jax.block_until_ready(normalized_radius_squared)
    first_call_seconds = time.perf_counter() - start

    repeat_seconds = []
    for _ in range(max(0, benchmark_repeats)):
        start = time.perf_counter()
        repeated, _ = kernel(*arguments)
        jax.block_until_ready(repeated)
        repeat_seconds.append(time.perf_counter() - start)

    stellar_radius_cm = config.planet.stellar_radius_solar * RSUN_CM
    depth_ppm = transit_depth_ppm(
        normalized_radius_squared,
        radius_base_cm,
        stellar_radius_cm,
    )
    timing = {
        "first_call_seconds_including_any_compilation": first_call_seconds,
        "repeat_call_seconds": repeat_seconds,
        "repeat_median_seconds": (
            float(np.median(repeat_seconds)) if repeat_seconds else None
        ),
        "compiled_scenario_count": len(SCENARIO_NAMES),
        "compilation_strategy": "one stacked vmap kernel for all scenarios",
        "opacity_preparation_seconds_including_any_compilation": (
            prepared_opacity.preparation_seconds_including_any_compilation
        ),
        "opacity_table_capture_policy": (
            "temperature-evaluated cross sections are dynamic RT arguments; "
            "large opacity tables are not RT-kernel closure constants"
        ),
    }
    return (
        np.asarray(temperature_k),
        float(mean_molecular_weight),
        np.asarray(depth_ppm),
        timing,
    )


def validate_opacity_state(opacity_bundle: OpacityBundle, pressure_bar, temperature_k):
    """Validate temperature bounds and report any deliberate pressure clamp."""

    pressure_bar = np.asarray(pressure_bar)
    temperature_k = np.asarray(temperature_k)
    report = {}
    for species, calculator in opacity_bundle.calculators.items():
        if opacity_bundle.mode == "ckd":
            temperature_grid = np.asarray(calculator.ckd_info.T_grid)
            pressure_grid = np.asarray(calculator.ckd_info.P_grid)
            pressure_below = int(np.count_nonzero(pressure_bar < pressure_grid.min()))
            pressure_above = int(np.count_nonzero(pressure_bar > pressure_grid.max()))
        else:
            temperature_grid = np.asarray(calculator.temperature_grid)
            pressure_grid = np.asarray(calculator.pressure_grid)
            pressure_below = 0
            pressure_above = 0
        temperature_min = float(np.min(temperature_grid))
        temperature_max = float(np.max(temperature_grid))
        if np.any(temperature_k < temperature_min) or np.any(
            temperature_k > temperature_max
        ):
            raise ValueError(
                f"Upper temperature lies outside the {species} opacity range "
                f"[{temperature_min}, {temperature_max}] K."
            )
        report[species] = {
            "temperature_range_k": [temperature_min, temperature_max],
            "pressure_range_bar": [
                float(np.min(pressure_grid)),
                float(np.max(pressure_grid)),
            ],
            "layers_pressure_endpoint_clamped_below": pressure_below,
            "layers_pressure_endpoint_clamped_above": pressure_above,
        }
    return report


def _asset_paths_without_download(
    cache_directory: Path,
    observation_config: ObservationConfig,
) -> Mapping[str, Path]:
    keys = (
        f"niriss_soss_{observation_config.niriss_resolution}",
        "nirspec_hu2025_archive",
        f"miri_lrs_{observation_config.miri_reduction}",
    )
    paths = {}
    for key in keys:
        asset = PUBLIC_OSF_ASSETS[key]
        path = cache_directory / asset.filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing public observation {path}. Run with "
                "--fetch-public-data once."
            )
        verify_sha256(path.read_bytes(), asset.sha256, str(path))
        paths[key] = path
    return paths


def load_public_observations(
    config: ObservationConfig,
    cache_directory: str | Path,
    *,
    fetch: bool,
) -> tuple[TransitSpectrum, ...]:
    """Load hash-verified public observations without implicit network I/O."""

    cache_directory = Path(cache_directory)
    if fetch:
        paths = fetch_figure3_public_assets(
            cache_directory,
            niriss_resolution=config.niriss_resolution,
            miri_reduction=config.miri_reduction,
        )
    else:
        paths = _asset_paths_without_download(cache_directory, config)
    return load_figure3_public_spectra(
        paths,
        niriss_resolution=config.niriss_resolution,
        miri_reduction=config.miri_reduction,
    )


def bin_model_scenarios(
    wavelength_micron,
    scenario_depth_ppm,
    observations,
    observation_config: ObservationConfig,
    *,
    profile_offsets: bool,
    native_band_edges_micron=None,
) -> tuple[tuple[BinnedSpectrum, ...], Mapping[str, float]]:
    """Integrate spectra over observed bins and apply model-side offsets.

    Diffgrid samples are integrated as a piecewise-linear spectrum.  When
    finite native band edges are supplied, CKD values are instead treated as
    piecewise-constant means over those bands.
    """

    observations = tuple(observations)
    native_band_edges = (
        None
        if native_band_edges_micron is None
        else np.asarray(native_band_edges_micron)
    )
    raw_by_dataset = {}
    if observations:
        bin_counts = tuple(observation.size for observation in observations)
        target_bin_edges = np.concatenate(
            [
                np.column_stack(
                    (
                        observation.wavelength_lower_um,
                        observation.wavelength_upper_um,
                    )
                )
                for observation in observations
            ],
            axis=0,
        )
        if native_band_edges is None:
            operator = piecewise_linear_bin_operator(
                wavelength_micron,
                target_bin_edges,
            )
        else:
            operator = band_mean_bin_operator(
                native_band_edges,
                target_bin_edges,
            )
        binned_scenarios = np.asarray(
            apply_bin_operator(operator, scenario_depth_ppm)
        )
        bin_start = 0
        for observation, bin_count in zip(observations, bin_counts):
            bin_stop = bin_start + bin_count
            raw_by_dataset[observation.dataset_id] = {
                name: values[bin_start:bin_stop]
                for name, values in zip(SCENARIO_NAMES, binned_scenarios)
            }
            bin_start = bin_stop

    offsets_ppm = dict(observation_config.offsets_ppm)
    if profile_offsets:
        grouped_residuals = {}
        for observation in observations:
            group = observation_config.offset_group_by_dataset[
                observation.dataset_id
            ]
            if (
                observation_config.anchor_offset_group is not None
                and group == observation_config.anchor_offset_group
            ):
                continue
            residual_ppm = (
                observation.transit_depth_ppm
                - raw_by_dataset[observation.dataset_id]["total"]
            )
            uncertainty_ppm = observation.transit_depth_uncertainty * 1.0e6
            grouped_residuals.setdefault(group, []).append(
                (residual_ppm, uncertainty_ppm)
            )
        for group, blocks in grouped_residuals.items():
            residual = np.concatenate([block[0] for block in blocks])
            uncertainty = np.concatenate([block[1] for block in blocks])
            weight = uncertainty**-2
            offsets_ppm[group] = float(np.sum(weight * residual) / np.sum(weight))

    binned = []
    for observation in observations:
        group = observation_config.offset_group_by_dataset[
            observation.dataset_id
        ]
        offset_ppm = float(offsets_ppm[group])
        shifted = {
            name: values + offset_ppm
            for name, values in raw_by_dataset[observation.dataset_id].items()
        }
        binned.append(
            BinnedSpectrum(
                observation=observation,
                offset_group=group,
                offset_ppm=offset_ppm,
                scenario_depth_ppm=shifted,
            )
        )
    return tuple(binned), offsets_ppm


def gaussian_summary(binned_spectra) -> Mapping[str, float]:
    """Return diagonal-Gaussian residual diagnostics, without a retrieval."""

    residual_blocks = []
    sigma_blocks = []
    for item in binned_spectra:
        residual_blocks.append(
            item.observation.transit_depth_ppm
            - item.scenario_depth_ppm["total"]
        )
        sigma_blocks.append(item.observation.transit_depth_uncertainty * 1.0e6)
    residual = np.concatenate(residual_blocks)
    sigma = np.concatenate(sigma_blocks)
    chi2 = float(np.sum((residual / sigma) ** 2))
    log_likelihood = float(
        -0.5 * np.sum((residual / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))
    )
    return {
        "number_of_points": int(residual.size),
        "chi_squared_diagonal_only": chi2,
        "reduced_chi_squared_no_parameter_correction": chi2 / residual.size,
        "gaussian_log_likelihood_ppm_units": log_likelihood,
        "residual_rms_ppm": float(np.sqrt(np.mean(residual**2))),
        "residual_median_absolute_ppm": float(np.median(np.abs(residual))),
    }


def load_reference_spectrum(path: str | Path):
    """Load an intrinsic, unoffset author curve with an explicit CSV contract."""

    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "spectrum_contract",
            "wavelength_micron",
            "transit_depth_ppm",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Reference CSV requires spectrum_contract, wavelength_micron, "
                "and transit_depth_ppm."
            )
        rows = list(reader)
    contracts = {row["spectrum_contract"] for row in rows}
    if contracts != {"intrinsic_unoffset_model"}:
        raise ValueError(
            "Reference spectrum_contract must be intrinsic_unoffset_model."
        )
    wavelength = np.asarray([float(row["wavelength_micron"]) for row in rows])
    depth_ppm = np.asarray([float(row["transit_depth_ppm"]) for row in rows])
    if (
        wavelength.size < 2
        or np.any(~np.isfinite(wavelength))
        or np.any(np.diff(wavelength) <= 0.0)
        or np.any(~np.isfinite(depth_ppm))
    ):
        raise ValueError("Reference spectrum must be finite and wavelength ordered.")
    return wavelength, depth_ppm


def compare_reference(
    model_wavelength_micron,
    model_depth_ppm,
    reference_wavelength_micron,
    reference_depth_ppm,
):
    """Compare a reference curve over its covered wavelength samples."""

    covered = (
        (reference_wavelength_micron >= model_wavelength_micron[0])
        & (reference_wavelength_micron <= model_wavelength_micron[-1])
    )
    if np.count_nonzero(covered) < 2:
        raise ValueError("Reference and model wavelength ranges do not overlap.")
    sampled_model = np.interp(
        reference_wavelength_micron[covered],
        model_wavelength_micron,
        model_depth_ppm,
    )
    residual = sampled_model - reference_depth_ppm[covered]
    return {
        "number_of_samples": int(residual.size),
        "model_minus_reference_rms_ppm": float(np.sqrt(np.mean(residual**2))),
        "model_minus_reference_median_absolute_ppm": float(
            np.median(np.abs(residual))
        ),
        "model_minus_reference_maximum_absolute_ppm": float(
            np.max(np.abs(residual))
        ),
    }


def write_spectral_csv(path, wavelength_micron, scenario_depth_ppm) -> None:
    """Write the native model grid and every component scenario."""

    matrix = np.column_stack([wavelength_micron, scenario_depth_ppm.T])
    header = ",".join(
        ["wavelength_micron"]
        + [f"{name}_depth_ppm" for name in SCENARIO_NAMES]
    )
    np.savetxt(path, matrix, delimiter=",", header=header, comments="")


def write_binned_csv(path, binned_spectra) -> None:
    """Write observations, exact bin edges, offsets, and model scenarios."""

    fieldnames = [
        "dataset_id",
        "instrument",
        "mode",
        "offset_group",
        "offset_ppm",
        "wavelength_lower_micron",
        "wavelength_micron",
        "wavelength_upper_micron",
        "observed_depth_ppm",
        "observed_uncertainty_ppm",
        *[f"model_{name}_depth_ppm" for name in SCENARIO_NAMES],
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in binned_spectra:
            observation = item.observation
            for index in range(observation.size):
                row = {
                    "dataset_id": observation.dataset_id,
                    "instrument": observation.instrument,
                    "mode": observation.mode,
                    "offset_group": item.offset_group,
                    "offset_ppm": item.offset_ppm,
                    "wavelength_lower_micron": observation.wavelength_lower_um[index],
                    "wavelength_micron": observation.wavelength_um[index],
                    "wavelength_upper_micron": observation.wavelength_upper_um[index],
                    "observed_depth_ppm": observation.transit_depth_ppm[index],
                    "observed_uncertainty_ppm": (
                        observation.transit_depth_uncertainty[index] * 1.0e6
                    ),
                }
                row.update(
                    {
                        f"model_{name}_depth_ppm": values[index]
                        for name, values in item.scenario_depth_ppm.items()
                    }
                )
                writer.writerow(row)


def plot_comparison(
    path,
    wavelength_micron,
    scenario_depth_ppm,
    pressure_bar,
    temperature_k,
    binned_spectra,
    reference=None,
) -> None:
    """Write a Figure 3-style spectrum and temperature-profile diagnostic."""

    import matplotlib.pyplot as plt

    figure, (spectrum_axis, temperature_axis) = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [3.2, 1.0]},
    )
    scenario = dict(zip(SCENARIO_NAMES, scenario_depth_ppm))
    spectrum_axis.plot(
        wavelength_micron,
        scenario["total"],
        color="black",
        linewidth=1.4,
        label="Intrinsic total (before dataset offsets)",
    )
    component_styles = {
        "molecule_CH4": ("tab:blue", "CH4 only"),
        "molecule_H2O": ("tab:cyan", "H2O only"),
        "molecule_CO": ("tab:orange", "CO only"),
        "molecule_CO2": ("tab:green", "CO2 only"),
        "aerosols": ("0.45", "Aerosols only"),
        "rayleigh_plus_cia": ("tab:red", "Rayleigh + CIA only"),
    }
    for name, (color, label) in component_styles.items():
        spectrum_axis.plot(
            wavelength_micron,
            scenario[name],
            color=color,
            linewidth=0.9,
            alpha=0.9,
            label=label,
        )
    if reference is not None:
        spectrum_axis.plot(
            reference[0],
            reference[1],
            color="tab:red",
            linewidth=1.0,
            linestyle="--",
            label="Supplied MELTYQ reference",
        )
    colors = plt.get_cmap("tab10")
    for index, item in enumerate(binned_spectra):
        observation = item.observation
        spectrum_axis.errorbar(
            observation.wavelength_um,
            observation.transit_depth_ppm,
            xerr=(
                observation.wavelength_um - observation.wavelength_lower_um,
                observation.wavelength_upper_um - observation.wavelength_um,
            ),
            yerr=observation.transit_depth_uncertainty * 1.0e6,
            fmt=".",
            markersize=2.5,
            linewidth=0.5,
            color=colors(index % 10),
            alpha=0.75,
            label=observation.dataset_id,
        )
        spectrum_axis.plot(
            observation.wavelength_um,
            item.scenario_depth_ppm["total"],
            color=colors(index % 10),
            linewidth=0.9,
            alpha=0.9,
        )
    spectrum_axis.set(
        xlabel="Wavelength (micron)",
        ylabel="Transit depth (ppm)",
        xlim=(wavelength_micron[0], wavelength_micron[-1]),
        xscale="log",
    )
    spectrum_axis.legend(fontsize=7, ncol=2)

    temperature_axis.plot(temperature_k, pressure_bar * 1.0e5, color="black")
    temperature_axis.set(
        xlabel="Temperature (K)",
        ylabel="Pressure (Pa)",
        yscale="log",
    )
    temperature_axis.invert_yaxis()
    figure.suptitle("MELTYQ Figure 3-style forward comparison")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _package_versions() -> Mapping[str, str | None]:
    """Return versions of the modules used by the running interpreter.

    Editable installs can leave an older distribution metadata record on
    ``sys.path``.  Prefer the imported module's version so provenance follows
    the code that actually executed, then fall back to distribution metadata
    for packages such as ExoPie that do not expose ``__version__``.
    """

    result = {}
    for package in ("exojax", "exogibbs", "exoeos", "exopie", "jax"):
        try:
            module_version = getattr(
                importlib.import_module(package),
                "__version__",
                None,
            )
        except ImportError:
            module_version = None
        if module_version is not None:
            result[package] = str(module_version)
            continue
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _package_source_states() -> Mapping[str, Mapping[str, object]]:
    """Fingerprint provider sources and material ExoPie data assets."""

    states = {}
    for package in ("exojax", "exogibbs", "exoeos", "exopie"):
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


def spectral_memory_lower_bound(opacity_bundle: OpacityBundle, nlayer: int):
    """Estimate persistent tables and major float64 RT work arrays.

    The work-array value is deliberately a lower bound: XLA compiler
    temporaries and backend allocator overhead are not knowable here.
    """

    spectral_points = int(np.asarray(opacity_bundle.wavenumber_cm1).size)
    g_points = int(np.asarray(opacity_bundle.weights).size) if (
        opacity_bundle.weights is not None
    ) else 1
    values_per_layer_grid = nlayer * g_points * spectral_points
    major_work_array_count = len(SCENARIO_NAMES) + len(ABSORBING_SPECIES) + 4
    work_bytes = values_per_layer_grid * major_work_array_count * 8
    evaluated_molecular_cross_section_bytes = (
        values_per_layer_grid * len(ABSORBING_SPECIES) * 8
    )

    table_bytes = 0
    peak_table_bytes = 0
    if opacity_bundle.mode == "diffgrid":
        field_names = (
            "pressure_grid",
            "temperature_grid",
            "inverse_temperature_grid",
            "log_cross_section_grid",
            "log_cross_section_derivative_grid",
            "log_cross_section_floor",
        )
        for calculator in opacity_bundle.calculators.values():
            if hasattr(calculator, "persistent_table_bytes"):
                calculator_table_bytes = calculator.persistent_table_bytes
            else:
                calculator_table_bytes = 0
                info = calculator.diffgrid_info
                for field_name in field_names:
                    value = getattr(info, field_name)
                    calculator_table_bytes += int(np.prod(value.shape)) * np.dtype(
                        value.dtype
                    ).itemsize
            table_bytes += calculator_table_bytes
            peak_table_bytes = max(peak_table_bytes, calculator_table_bytes)
    opacity_preparation_stage_bytes = (
        peak_table_bytes + evaluated_molecular_cross_section_bytes
    )
    radiative_transfer_stage_bytes = (
        evaluated_molecular_cross_section_bytes + work_bytes
    )
    return {
        "spectral_points": spectral_points,
        "g_points": g_points,
        "archive_opacity_table_total_gib": table_bytes / 2.0**30,
        "peak_resident_opacity_table_gib": peak_table_bytes / 2.0**30,
        "retained_evaluated_molecular_cross_sections_gib": (
            evaluated_molecular_cross_section_bytes / 2.0**30
        ),
        "major_rt_work_arrays_lower_bound_gib": work_bytes / 2.0**30,
        "stage_peak_lower_bound_gib": max(
            opacity_preparation_stage_bytes,
            radiative_transfer_stage_bytes,
        )
        / 2.0**30,
        "warning": (
            "Stage peak is max(one streamed table plus retained cross sections, "
            "retained cross sections plus major RT work arrays). It excludes "
            "CIA/Rayleigh arrays, XLA temporaries, and allocator overhead."
        ),
    }


def input_path_report(config: Figure3Config, args) -> Mapping[str, object]:
    """Validate every external input used by the requested run."""

    opacity_mode = args.opacity_mode or config.spectral.opacity_mode
    missing = []
    observations = {}
    observation_keys = (
        f"niriss_soss_{config.observations.niriss_resolution}",
        "nirspec_hu2025_archive",
        f"miri_lrs_{config.observations.miri_reduction}",
    )
    for key in observation_keys:
        asset = PUBLIC_OSF_ASSETS[key]
        path = Path(args.observation_cache) / asset.filename
        present = path.is_file() and path.stat().st_size > 0
        valid = False
        error = None
        if present:
            try:
                verify_sha256(path.read_bytes(), asset.sha256, str(path))
                valid = True
            except ValueError as exception:
                error = str(exception)
        observations[key] = {
            "path": str(path),
            "present": present,
            "sha256_valid": valid,
            "error": error,
        }
        if not valid:
            missing.append(str(path))
    parsed_observations = {"valid": False, "spectra": []}
    if all(observations[key]["sha256_valid"] for key in observation_keys):
        try:
            spectra = load_figure3_public_spectra(
                {
                    key: Path(args.observation_cache)
                    / PUBLIC_OSF_ASSETS[key].filename
                    for key in observation_keys
                },
                niriss_resolution=config.observations.niriss_resolution,
                miri_reduction=config.observations.miri_reduction,
            )
            parsed_observations = {
                "valid": True,
                "spectra": [
                    {"dataset_id": spectrum.dataset_id, "points": spectrum.size}
                    for spectrum in spectra
                ],
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            parsed_observations["error"] = str(error)
            missing.append("parseable public observation products")

    molecular = {}
    if opacity_mode == "ckd":
        for species, relative_path in CKD_RELATIVE_PATHS.items():
            directory = Path(args.opacity_root) / relative_path
            table = _single_nonempty_h5(directory)
            present = table is not None
            molecular[species] = {
                "path": str(table or directory),
                "present": present,
                "identity_valid": False,
            }
            if not present:
                missing.append(str(directory))
            elif table is not None:
                molecular[species]["sha256"] = _sha256_file(table)
                try:
                    molecular[species]["identity"] = (
                        _validate_ckd_table_identity(table, species)
                    )
                    molecular[species]["identity_valid"] = True
                except (OSError, TypeError, ValueError) as error:
                    molecular[species]["identity_error"] = str(error)
                    missing.append(f"valid CKD identity for {species}")
    else:
        manifest_path = Path(args.diffgrid_manifest)
        if not manifest_path.is_file():
            missing.append(str(manifest_path))
        molecular["manifest"] = {
            "path": str(manifest_path),
            "present": manifest_path.is_file(),
            "validated": False,
        }
        if manifest_path.is_file():
            try:
                from exojax.atm.atmprof import (
                    pressure_layer_logspace_from_boundaries,
                )

                wavelength_range_cm1 = np.asarray(
                    [
                        1.0e4 / config.spectral.wavelength_max_micron,
                        1.0e4 / config.spectral.wavelength_min_micron,
                    ]
                )
                pressure_bar, _, _, _ = (
                    pressure_layer_logspace_from_boundaries(
                        np.log10(config.upper.pressure_top_bar),
                        np.log10(config.upper.pressure_base_bar),
                        config.upper.nlayer,
                        numpy=True,
                    )
                )
                validation = validate_diffgrid_manifest_lightweight(
                    manifest_path,
                    pressure_bar,
                    wavelength_range_cm1,
                )
                molecular["manifest"]["validated"] = True
                for species, descriptor in validation["files"].items():
                    molecular[species] = {
                        "path": descriptor["resolved_path"],
                        "present": True,
                        "sha256_valid": True,
                        "archive_provenance_valid": True,
                    }
                table_bytes_by_species = [
                    value["persistent_table_bytes"]
                    for value in validation["archive_inputs"].values()
                ]
                retained_cross_section_bytes = (
                    config.upper.nlayer
                    * validation["wavenumber_cm1"].size
                    * len(ABSORBING_SPECIES)
                    * 8
                )
                rt_work_bytes = (
                    config.upper.nlayer
                    * validation["wavenumber_cm1"].size
                    * (len(SCENARIO_NAMES) + len(ABSORBING_SPECIES) + 4)
                    * 8
                )
                stage_peak_bytes = max(
                    max(table_bytes_by_species) + retained_cross_section_bytes,
                    retained_cross_section_bytes + rt_work_bytes,
                )
                molecular["grid"] = {
                    "minimum_resolving_power": validation[
                        "measured_minimum_grid_resolving_power"
                    ],
                    "median_resolving_power": validation[
                        "measured_median_grid_resolving_power"
                    ],
                    "archive_cross_section_table_total_gib": validation[
                        "persistent_cross_section_table_gib"
                    ],
                    "peak_resident_one_species_table_gib": (
                        max(table_bytes_by_species) / 2.0**30
                    ),
                    "retained_evaluated_molecular_cross_sections_gib": (
                        retained_cross_section_bytes / 2.0**30
                    ),
                    "major_rt_work_arrays_lower_bound_gib": (
                        rt_work_bytes / 2.0**30
                    ),
                    "stage_peak_lower_bound_gib": (
                        stage_peak_bytes / 2.0**30
                    ),
                    "memory_warning": (
                        "Stage peak excludes CIA/Rayleigh arrays, XLA temporaries, "
                        "and allocator overhead."
                    ),
                }
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                molecular["manifest_error"] = str(error)
                missing.append("fully validated Diffgrid manifest and archives")

    cia = {}
    for pair, filename in CIA_FILENAMES.items():
        path = Path(args.opacity_root) / filename
        present = path.is_file() and path.stat().st_size > 0
        cia[pair] = {"path": str(path), "present": present}
        if not present:
            missing.append(str(path))
        else:
            cia[pair]["sha256"] = _sha256_file(path)
            try:
                _validate_cia_file_identity(path, pair)
                cia[pair]["identity_valid"] = True
            except (OSError, UnicodeError, ValueError) as error:
                cia[pair]["identity_valid"] = False
                cia[pair]["identity_error"] = str(error)
                missing.append(f"valid CIA identity for {pair}")
    reference_spectrum = {"requested": args.reference_spectrum is not None}
    if args.reference_spectrum is not None:
        reference_path = Path(args.reference_spectrum)
        reference_spectrum.update(
            {
                "path": str(reference_path),
                "present": reference_path.is_file(),
                "valid": False,
            }
        )
        try:
            if opacity_mode != "diffgrid":
                raise ValueError(
                    "Reference-curve comparison requires Diffgrid point samples; "
                    "CKD band means cannot be compared as band-center samples."
                )
            wavelength, _ = load_reference_spectrum(reference_path)
            actual_sha256 = _sha256_file(reference_path)
            expected_sha256 = args.reference_spectrum_sha256
            if (
                expected_sha256 is not None
                and actual_sha256 != expected_sha256.lower()
            ):
                raise ValueError("Reference-spectrum SHA-256 mismatch.")
            reference_spectrum.update(
                {
                    "valid": True,
                    "sha256": actual_sha256,
                    "sha256_expected": expected_sha256,
                    "samples": int(wavelength.size),
                    "wavelength_range_micron": [
                        float(wavelength[0]),
                        float(wavelength[-1]),
                    ],
                }
            )
        except (OSError, TypeError, ValueError) as error:
            reference_spectrum["error"] = str(error)
            missing.append("valid reference spectrum")
    return {
        "ready": not missing,
        "effective_opacity_mode": opacity_mode,
        "observations": observations,
        "parsed_observations": parsed_observations,
        "molecular_opacities": molecular,
        "cia": cia,
        "reference_spectrum": reference_spectrum,
        "missing": missing,
    }


def run(args) -> Mapping[str, object]:
    """Run the complete deep-to-observed-bin forward comparison."""

    if args.max_outer_iterations <= 0:
        raise ValueError("max_outer_iterations must be positive.")
    if args.benchmark_repeats < 0:
        raise ValueError("benchmark_repeats must be non-negative.")

    config = load_config(args.config)
    opacity_mode = args.opacity_mode or config.spectral.opacity_mode
    if args.reference_spectrum is not None and opacity_mode != "diffgrid":
        raise ValueError(
            "Reference-curve comparison requires opacity_mode=diffgrid."
        )
    observations = load_public_observations(
        config.observations,
        args.observation_cache,
        fetch=args.fetch_public_data,
    )

    wavenumber_range_cm1 = (
        1.0e4 / config.spectral.wavelength_max_micron,
        1.0e4 / config.spectral.wavelength_min_micron,
    )
    if opacity_mode == "ckd":
        opacity_bundle = load_ckd_opacities(
            args.opacity_root,
            wavenumber_range_cm1,
            allow_download=args.allow_opacity_download,
        )
        art = build_art(config.upper, opacity_bundle.wavenumber_cm1)
    else:
        provisional_art = build_art(config.upper, np.asarray(wavenumber_range_cm1))
        opacity_bundle = load_diffgrid_opacities(
            args.diffgrid_manifest,
            provisional_art.pressure,
            wavenumber_range_cm1,
        )
        art = build_art(config.upper, opacity_bundle.wavenumber_cm1)

    cia_opacities, cia_provenance = load_cia_opacities(
        args.opacity_root,
        opacity_bundle.wavenumber_cm1,
        allow_download=args.allow_opacity_download,
    )
    rayleigh_cross_sections = build_rayleigh_cross_sections(
        opacity_bundle.wavenumber_cm1,
        config.spectral.rayleigh_provider,
    )
    upper_temperature_probe = upper_temperature_profile_k(
        art.pressure,
        config.magma.temperature_base_k,
        config.upper.temperature_1e4_pa_k,
        config.upper.temperature_100_pa_k,
        config.upper.temperature_1e_minus2_pa_k,
        config.upper.smoothing_window_percent,
    )
    opacity_bounds = validate_opacity_state(
        opacity_bundle,
        art.pressure,
        upper_temperature_probe,
    )
    upper_temperature_host = np.asarray(upper_temperature_probe)
    for pair, bounds in cia_provenance.items():
        if np.any(upper_temperature_host < bounds["temperature_min_k"]) or np.any(
            upper_temperature_host > bounds["temperature_max_k"]
        ):
            raise ValueError(
                f"Upper temperature lies outside the {pair} CIA range "
                f"[{bounds['temperature_min_k']}, "
                f"{bounds['temperature_max_k']}] K."
            )

    context_start = time.perf_counter()
    context = build_context(
        max_outer_iterations=args.max_outer_iterations,
        planet_mass_earth=config.planet.mass_earth,
        rocky_core_mass_fraction=config.planet.rocky_core_mass_fraction,
        rocky_radius_rearth=config.planet.rocky_radius_rearth,
    )
    context_seconds = time.perf_counter() - context_start
    deep_start = time.perf_counter()
    deep_result, _ = solve_point(context, provider_magma_parameters(config.magma))
    deep_seconds = time.perf_counter() - deep_start
    if deep_result.error or not (
        deep_result.boundary_converged
        and deep_result.profile_converged
        and deep_result.base_converged
    ):
        raise RuntimeError(
            "Magma/deep-atmosphere forward model failed: " + deep_result.error
        )

    kernel = make_spectral_kernel(
        art,
        opacity_bundle,
        cia_opacities,
        rayleigh_cross_sections,
        cloud_enabled=config.cloud.enabled,
        haze_enabled=config.haze.enabled,
        haze_vertical_profile=config.haze.vertical_profile,
        radiative_transfer_scheme=(
            config.spectral.radiative_transfer_scheme
        ),
    )
    temperature_k, mean_molecular_weight, scenario_depth_ppm, timing = (
        run_spectral_kernel(
            kernel,
            config,
            art,
            opacity_bundle,
            cia_opacities,
            rayleigh_cross_sections,
            deep_result.base_mole_fractions,
            deep_result.radius_base_rearth,
            benchmark_repeats=args.benchmark_repeats,
        )
    )

    wavelength_micron = 1.0e4 / opacity_bundle.wavenumber_cm1
    order = np.argsort(wavelength_micron)
    wavelength_micron = wavelength_micron[order]
    scenario_depth_ppm = scenario_depth_ppm[:, order]
    native_band_edges_micron = None
    if opacity_bundle.band_edges_cm1 is not None:
        band_edges_cm1 = np.asarray(opacity_bundle.band_edges_cm1)
        native_band_edges_micron = np.column_stack(
            (1.0e4 / band_edges_cm1[:, 1], 1.0e4 / band_edges_cm1[:, 0])
        )[order]
    binned_spectra, offsets_ppm = bin_model_scenarios(
        wavelength_micron,
        scenario_depth_ppm,
        observations,
        config.observations,
        profile_offsets=args.profile_offsets,
        native_band_edges_micron=native_band_edges_micron,
    )
    residual_summary = gaussian_summary(binned_spectra)

    reference = None
    reference_summary = None
    reference_provenance = None
    if args.reference_spectrum is not None:
        if args.reference_spectrum_sha256 is not None:
            actual_reference_sha256 = _sha256_file(args.reference_spectrum)
            if actual_reference_sha256 != args.reference_spectrum_sha256.lower():
                raise ValueError("Reference-spectrum SHA-256 mismatch.")
        reference = load_reference_spectrum(args.reference_spectrum)
        reference_path = Path(args.reference_spectrum)
        reference_provenance = {
            "path": str(reference_path),
            "sha256": _sha256_file(reference_path),
        }
        reference_summary = compare_reference(
            wavelength_micron,
            scenario_depth_ppm[SCENARIO_NAMES.index("total")],
            reference[0],
            reference[1],
        )

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_spectral_csv(
        output_directory / "model_spectra.csv",
        wavelength_micron,
        scenario_depth_ppm,
    )
    write_binned_csv(output_directory / "binned_comparison.csv", binned_spectra)
    plot_comparison(
        output_directory / "figure3_forward_comparison.png",
        wavelength_micron,
        scenario_depth_ppm,
        np.asarray(art.pressure),
        temperature_k,
        binned_spectra,
        reference=reference,
    )

    model_spectra_path = output_directory / "model_spectra.csv"
    binned_comparison_path = output_directory / "binned_comparison.csv"
    forward_figure_path = output_directory / "figure3_forward_comparison.png"
    metadata = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_DOI,
        "config": asdict(config),
        "effective_opacity_mode": opacity_mode,
        "package_versions": _package_versions(),
        "package_source_states": _package_source_states(),
        "deep_model": {
            "provider_parameters": asdict(provider_magma_parameters(config.magma)),
            "rocky_radius_rearth": context.rocky_radius_rearth,
            "radius_at_10_bar_rearth": deep_result.radius_base_rearth,
            "base_mole_fractions": {
                species: float(value)
                for species, value in zip(
                    MELTYQ_SPECIES, deep_result.base_mole_fractions
                )
            },
            "boundary_iterations": deep_result.boundary_iterations,
            "profile_max_iterations": deep_result.profile_max_iterations,
            "base_iterations": deep_result.base_iterations,
        },
        "upper_model": {
            "mean_molecular_weight": mean_molecular_weight,
            "temperature_min_k": float(np.min(temperature_k)),
            "temperature_max_k": float(np.max(temperature_k)),
            "radiative_transfer": {
                "scheme": config.spectral.radiative_transfer_scheme,
                "gravity_constant_cgs": (
                    TAUREX_GRAVITATIONAL_CONSTANT_CGS
                    if config.spectral.radiative_transfer_scheme
                    == "taurex_rectangle"
                    else G
                ),
                "taurex_sources": (
                    {
                        "hydrostatic_equations": TAUREX_PLANET_SOURCE,
                        "ideal_gas_density_convention": (
                            TAUREX_SIMPLEMODEL_SOURCE
                        ),
                        "cia_interpolation_convention": TAUREX_CIA_SOURCE,
                        "legacy_chord_and_rectangle_area": (
                            TAUREX_TRANSMISSION_SOURCE
                        ),
                    }
                    if config.spectral.radiative_transfer_scheme
                    == "taurex_rectangle"
                    else None
                ),
                "exojax_implementations": (
                    {
                        "ideal_gas_geometry": (
                            "exojax.atm.atmprof."
                            "hydrostatic_radius_profile_ideal_gas"
                        ),
                        "ideal_gas_number_density": (
                            "exojax.atm.idealgas.number_density"
                        ),
                        "geometric_layer_optical_depth": (
                            "exojax.rt.layeropacity"
                        ),
                        "cia_interpolation": (
                            "exojax.opacity.OpaCIA[interp]"
                        ),
                    }
                    if config.spectral.radiative_transfer_scheme
                    == "taurex_rectangle"
                    else {
                        "pressure_coordinate_radiative_transfer": (
                            "exojax.rt.ArtTransPure"
                        ),
                        "geometric_haze_layer_optical_depth": (
                            "exojax.rt.layeropacity."
                            "layer_optical_depth_from_extinction"
                            if config.haze.enabled
                            else None
                        ),
                        "cia_interpolation": (
                            "exojax.opacity.OpaCIA[interp]"
                        ),
                    }
                ),
                "taurex_discretization": (
                    "ExoJAX layer-constant-gravity geometry, center-density "
                    "column extinction, TauREx legacy chords, and rectangle "
                    "annulus area"
                    if config.spectral.radiative_transfer_scheme
                    == "taurex_rectangle"
                    else None
                ),
            },
            "rayleigh": {
                "provider": config.spectral.rayleigh_provider,
                "taurex_source": (
                    TAUREX_RAYLEIGH_SOURCE
                    if config.spectral.rayleigh_provider == "taurex"
                    else None
                ),
                "input_cross_section_unit": "cm2 molecule-1",
            },
            "haze": {
                "enabled": config.haze.enabled,
                "vertical_profile": config.haze.vertical_profile,
                "public_formula_source": (
                    TAUREX_LEE_SOURCE if config.haze.enabled else None
                ),
                "historical_formula_change": (
                    TAUREX_LEE_PROFILE_CHANGE if config.haze.enabled else None
                ),
                "meltyq_runtime_revision_confirmed": False,
            },
            "opaque_cloud_vertical_dtau_used_by_rt": (
                OPAQUE_CLOUD_VERTICAL_DTAU if config.cloud.enabled else None
            ),
        },
        "opacities": opacity_bundle.provenance,
        "opacity_interpolation_bounds": opacity_bounds,
        "cia": cia_provenance,
        "observations": [
            {
                "dataset_id": spectrum.dataset_id,
                "points": spectrum.size,
                "provenance": spectrum.provenance,
            }
            for spectrum in observations
        ],
        "offsets_ppm": offsets_ppm,
        "offsets_profiled_analytically": args.profile_offsets,
        "observation_binning": (
            "exojax.postproc.binning.band_mean_bin_operator: wavelength-overlap "
            "of piecewise-constant CKD band means"
            if opacity_mode == "ckd"
            else "exojax.postproc.binning.piecewise_linear_bin_operator: exact "
            "wavelength top-hat of piecewise-linear Diffgrid samples"
        ),
        "residual_summary": residual_summary,
        "reference_summary": reference_summary,
        "reference_spectrum": reference_provenance,
        "memory_estimate": spectral_memory_lower_bound(
            opacity_bundle,
            config.upper.nlayer,
        ),
        "timing": {
            "context_build_seconds": context_seconds,
            "deep_first_solve_seconds_including_any_compilation": deep_seconds,
            **timing,
        },
        "outputs": {
            "model_spectra_csv": {
                "filename": model_spectra_path.name,
                "sha256": _sha256_file(model_spectra_path),
            },
            "binned_comparison_csv": {
                "filename": binned_comparison_path.name,
                "sha256": _sha256_file(binned_comparison_path),
            },
            "forward_figure_png": {
                "filename": forward_figure_path.name,
                "sha256": _sha256_file(forward_figure_path),
            },
        },
        "interpretation_limits": [
            "The checked-in public demo is not the unpublished MELTYQ best fit.",
            "Diagonal residual metrics do not include channel covariance.",
            "Lee vertical-profile selection remains author-confirmation dependent.",
            (
                "At high pressure ExoGibbs uses the primary-source square-root "
                "pressure N-solubility form; the typeset MELTYQ Appendix shows "
                "linear pressure."
            ),
            (
                "CKD mode is a fast R=1000 perfect-correlation approximation, "
                "not MELTYQ's R=50000 cross-section calculation."
                if opacity_mode == "ckd"
                else "Diffgrid fidelity is limited by its external table provenance."
            ),
        ],
    }
    with (output_directory / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface without parsing at import time."""

    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="MELTYQ Figure 3-style K2-18 b forward comparison.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root
        / "examples/meltyq/configs/meltyq_figure3_public_demo.json",
    )
    parser.add_argument(
        "--opacity-mode",
        choices=("ckd", "diffgrid"),
        help="Override the opacity mode stored in the config.",
    )
    parser.add_argument(
        "--opacity-root",
        type=Path,
        default=repository_root / ".database",
    )
    parser.add_argument(
        "--diffgrid-manifest",
        type=Path,
        default=repository_root
        / "examples/meltyq/configs/meltyq_figure3_diffgrid_manifest.template.json",
    )
    parser.add_argument(
        "--observation-cache",
        type=Path,
        default=repository_root / ".database/meltyq_figure3_observations",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=repository_root / "outputs/meltyq_figure3",
    )
    parser.add_argument("--fetch-public-data", action="store_true")
    parser.add_argument("--allow-opacity-download", action="store_true")
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help=(
            "Verify paths, hashes, provenance, and small coordinate arrays "
            "without loading cross-section tensors or solving."
        ),
    )
    parser.add_argument("--profile-offsets", action="store_true")
    parser.add_argument("--reference-spectrum", type=Path)
    parser.add_argument(
        "--reference-spectrum-sha256",
        help="Optional required SHA-256 digest for the reference curve.",
    )
    parser.add_argument("--max-outer-iterations", type=int, default=100)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    return parser


def main() -> None:
    """Run the CLI and print the claim-limited numerical summary."""

    parser = build_parser()
    args = parser.parse_args()
    if (
        args.reference_spectrum_sha256 is not None
        and args.reference_spectrum is None
    ):
        parser.error("--reference-spectrum-sha256 requires --reference-spectrum.")
    if args.reference_spectrum_sha256 is not None and (
        len(args.reference_spectrum_sha256) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in args.reference_spectrum_sha256
        )
    ):
        parser.error("--reference-spectrum-sha256 must be 64 hexadecimal digits.")
    if args.max_outer_iterations <= 0:
        parser.error("--max-outer-iterations must be positive.")
    if args.benchmark_repeats < 0:
        parser.error("--benchmark-repeats must be non-negative.")
    if args.check_inputs:
        config = load_config(args.config)
        if args.fetch_public_data:
            load_public_observations(
                config.observations,
                args.observation_cache,
                fetch=True,
            )
        report = input_path_report(config, args)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["ready"]:
            raise SystemExit(1)
        return
    metadata = run(args)
    print("MELTYQ Figure 3-style forward comparison")
    print(f"  Claim status: {metadata['config']['claim_status']}")
    print(
        "  Radius at 10 bar: "
        f"{metadata['deep_model']['radius_at_10_bar_rearth']:.6f} R_earth"
    )
    print(
        "  Residual RMS: "
        f"{metadata['residual_summary']['residual_rms_ppm']:.3f} ppm"
    )
    print(
        "  Spectral repeat runtime: "
        f"{metadata['timing']['repeat_median_seconds']} s"
    )


if __name__ == "__main__":
    main()
