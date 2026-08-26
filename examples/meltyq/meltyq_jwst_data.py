"""Public JWST data used for a MELTYQ Figure 3-style comparison.

The numerical data remain external to ExoExamples.  This module pins the
public OSF revisions and SHA-256 digests, downloads them into an explicit
cache, and converts the heterogeneous text formats to one small data model.
No dataset offset is applied while loading: offsets are nuisance parameters
of the spectral comparison, not corrections to the archived observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class RemoteAsset:
    """A hash-pinned public file."""

    key: str
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class ArchiveSpectrum:
    """One spectrum stored inside the Hu et al. OSF archive."""

    dataset_id: str
    member_path: str
    sha256: str
    instrument: str
    mode: str
    reduction: str
    offset_group: str


@dataclass(frozen=True)
class TransitSpectrum:
    """Canonical representation of one independently offset spectrum.

    Wavelengths are in micron.  Transit depths and their one-standard-
    deviation uncertainties are dimensionless fractions, not ppm.
    """

    dataset_id: str
    instrument: str
    mode: str
    reduction: str
    offset_group: str
    wavelength_um: np.ndarray
    wavelength_lower_um: np.ndarray
    wavelength_upper_um: np.ndarray
    transit_depth: np.ndarray
    transit_depth_uncertainty: np.ndarray
    provenance: str

    def __post_init__(self) -> None:
        array_names = (
            "wavelength_um",
            "wavelength_lower_um",
            "wavelength_upper_um",
            "transit_depth",
            "transit_depth_uncertainty",
        )
        arrays = []
        for name in array_names:
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)

        lengths = {value.size for value in arrays}
        if len(lengths) != 1 or not arrays[0].size:
            raise ValueError("All spectrum arrays must have one nonzero length.")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Spectrum arrays must contain only finite values.")
        if not np.all(np.diff(self.wavelength_um) > 0.0):
            raise ValueError("Wavelength centers must be strictly increasing.")
        if not np.all(
            (self.wavelength_lower_um < self.wavelength_um)
            & (self.wavelength_um < self.wavelength_upper_um)
        ):
            raise ValueError("Each wavelength center must lie inside its bin.")
        if not np.all(self.transit_depth_uncertainty > 0.0):
            raise ValueError("Transit-depth uncertainties must be positive.")
        if not np.all(
            (self.transit_depth > 0.0) & (self.transit_depth < 1.0)
        ):
            raise ValueError("Transit depths must be dimensionless fractions.")

    @property
    def size(self) -> int:
        """Number of spectral bins."""

        return self.wavelength_um.size

    @property
    def transit_depth_ppm(self) -> np.ndarray:
        """Read-only convenience view of transit depth in ppm."""

        value = self.transit_depth * 1.0e6
        value.setflags(write=False)
        return value


PUBLIC_OSF_ASSETS: Mapping[str, RemoteAsset] = {
    "niriss_soss_lowres": RemoteAsset(
        key="niriss_soss_lowres",
        filename="K2-18b_niriss_soss_lowres.txt",
        url="https://osf.io/download/82d6b/?revision=1",
        sha256=(
            "b4499b16456d19e2b09c35f910a60fbf96d8c14a006fe9f532f17edca051458b"
        ),
    ),
    "niriss_soss_native": RemoteAsset(
        key="niriss_soss_native",
        filename="K2-18b_niriss_soss_native.txt",
        url="https://osf.io/download/t5gz7/?revision=1",
        sha256=(
            "779a3511a1f72496429989cf228e6a034d97a80ae506863e3b32a8c6915d64a1"
        ),
    ),
    "nirspec_hu2025_archive": RemoteAsset(
        key="nirspec_hu2025_archive",
        filename="spectra-k2-18b.zip",
        url="https://osf.io/download/j6p7r/?revision=1",
        sha256=(
            "4ee5cb6ad42015bd8fb10f64e54329d250137ab1fa129c89a14830946adc8f18"
        ),
    ),
    "miri_lrs_jexores": RemoteAsset(
        key="miri_lrs_jexores",
        filename="K2-18b_miri_lrs_jexores.txt",
        url="https://osf.io/download/w8qtb/?revision=1",
        sha256=(
            "fcace11f382706fd474cd27f45b881bfbcf0e4d3042459d1225f846a873b1df1"
        ),
    ),
    "miri_lrs_jexopipe": RemoteAsset(
        key="miri_lrs_jexopipe",
        filename="K2-18b_miri_lrs_jexopipe.txt",
        url="https://osf.io/download/6pev4/?revision=1",
        sha256=(
            "42db622a0d742e783c632190b0f01baba62be9da1237bb74f453768073d66a87"
        ),
    ),
}


NIRSPEC_SHIFTED_AVERAGE: tuple[ArchiveSpectrum, ...] = (
    ArchiveSpectrum(
        dataset_id="nirspec_g235h_nrs1",
        member_path=(
            "spectra/nirspec_combined/"
            "K2_G235H_NRS1_ave2_shifted.dat"
        ),
        sha256=(
            "9cdebd5fa35881c42547f02e44a82eb05be26200a161fad66e24053c49241729"
        ),
        instrument="JWST/NIRSpec",
        mode="G235H/NRS1",
        reduction="Hu et al. (2025) shifted average",
        offset_group="nirspec_g235h_nrs1",
    ),
    ArchiveSpectrum(
        dataset_id="nirspec_g235h_nrs2",
        member_path=(
            "spectra/nirspec_combined/"
            "K2_G235H_NRS2_ave2_shifted.dat"
        ),
        sha256=(
            "4e8f0c73570661c7a3d285aa6e3fa0bf0b52d2e62b89849187b5043b456a7323"
        ),
        instrument="JWST/NIRSpec",
        mode="G235H/NRS2",
        reduction="Hu et al. (2025) shifted average",
        offset_group="nirspec_g235h_nrs2",
    ),
    ArchiveSpectrum(
        dataset_id="nirspec_g395h_nrs1",
        member_path=(
            "spectra/nirspec_combined/"
            "K2_G395H_NRS1_ave3_shifted.dat"
        ),
        sha256=(
            "0837913712ef6aee5df328a7632b6ae2e918972a11a88c8a211c04890a7dfae8"
        ),
        instrument="JWST/NIRSpec",
        mode="G395H/NRS1",
        reduction="Hu et al. (2025) shifted average",
        offset_group="nirspec_g395h_nrs1",
    ),
    ArchiveSpectrum(
        dataset_id="nirspec_g395h_nrs2",
        member_path=(
            "spectra/nirspec_combined/"
            "K2_G395H_NRS2_ave3_shifted.dat"
        ),
        sha256=(
            "dae57f569166ab12caa81ddcbf571bd17a313f1fdde6dc96e2c7d34ab3e837e5"
        ),
        instrument="JWST/NIRSpec",
        mode="G395H/NRS2",
        reduction="Hu et al. (2025) shifted average",
        offset_group="nirspec_g395h_nrs2",
    ),
)


def sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of ``payload``."""

    return hashlib.sha256(payload).hexdigest()


def verify_sha256(payload: bytes, expected_sha256: str, label: str) -> None:
    """Raise ``ValueError`` when a downloaded or archived file changed."""

    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )


def fetch_asset(
    asset: RemoteAsset,
    cache_directory: str | Path,
    opener: Callable[[str], object] | None = None,
) -> Path:
    """Return a verified cached asset, downloading it only when absent.

    ``opener`` is injectable so download behavior can be tested without
    network access.  A corrupt existing cache entry is rejected rather than
    silently overwritten.
    """

    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / asset.filename

    if destination.exists():
        payload = destination.read_bytes()
        verify_sha256(payload, asset.sha256, str(destination))
        return destination

    if opener is None:
        opener = urlopen
    response = opener(asset.url)
    try:
        payload = response.read()
    finally:
        close = getattr(response, "close", None)
        if close is not None:
            close()
    verify_sha256(payload, asset.sha256, asset.url)

    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return destination


def _read_text_matrix(
    payload: bytes,
    *,
    skip_rows: int,
    expected_columns: int,
    label: str,
) -> np.ndarray:
    try:
        matrix = np.loadtxt(BytesIO(payload), skiprows=skip_rows, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Could not parse {label}: {error}") from error
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"{label} must contain {expected_columns} numeric columns; "
            f"found shape {matrix.shape}."
        )
    return matrix


def parse_center_half_width_spectrum(
    payload: bytes,
    *,
    dataset_id: str,
    instrument: str,
    mode: str,
    reduction: str,
    offset_group: str,
    provenance: str,
) -> TransitSpectrum:
    """Parse the four-column Madhusudhan NIRISS/MIRI text format."""

    matrix = _read_text_matrix(
        payload,
        skip_rows=1,
        expected_columns=4,
        label=provenance,
    )
    wavelength_um, half_width_um, depth, uncertainty = matrix.T
    if not np.all(half_width_um > 0.0):
        raise ValueError(f"{provenance} has a non-positive bin half-width.")
    return TransitSpectrum(
        dataset_id=dataset_id,
        instrument=instrument,
        mode=mode,
        reduction=reduction,
        offset_group=offset_group,
        wavelength_um=wavelength_um,
        wavelength_lower_um=wavelength_um - half_width_um,
        wavelength_upper_um=wavelength_um + half_width_um,
        transit_depth=depth,
        transit_depth_uncertainty=uncertainty,
        provenance=provenance,
    )


def parse_edge_center_spectrum(
    payload: bytes,
    *,
    dataset_id: str,
    instrument: str,
    mode: str,
    reduction: str,
    offset_group: str,
    provenance: str,
) -> TransitSpectrum:
    """Parse a five-column Hu et al. combined NIRSpec spectrum."""

    matrix = _read_text_matrix(
        payload,
        skip_rows=0,
        expected_columns=5,
        label=provenance,
    )
    lower_um, upper_um, wavelength_um, depth, uncertainty = matrix.T
    return TransitSpectrum(
        dataset_id=dataset_id,
        instrument=instrument,
        mode=mode,
        reduction=reduction,
        offset_group=offset_group,
        wavelength_um=wavelength_um,
        wavelength_lower_um=lower_um,
        wavelength_upper_um=upper_um,
        transit_depth=depth,
        transit_depth_uncertainty=uncertainty,
        provenance=provenance,
    )


def load_niriss_soss(
    path: str | Path,
    resolution: Literal["lowres", "native"] = "lowres",
) -> TransitSpectrum:
    """Load the Madhusudhan et al. (2023) NIRISS/SOSS spectrum."""

    if resolution not in ("lowres", "native"):
        raise ValueError("resolution must be 'lowres' or 'native'.")
    path = Path(path)
    return parse_center_half_width_spectrum(
        path.read_bytes(),
        dataset_id="niriss_soss",
        instrument="JWST/NIRISS",
        mode="SOSS",
        reduction=f"Madhusudhan et al. (2023) {resolution}",
        offset_group="niriss_soss",
        provenance=str(path),
    )


def load_miri_lrs(
    path: str | Path,
    reduction: Literal["jexores", "jexopipe"] = "jexores",
) -> TransitSpectrum:
    """Load one Madhusudhan et al. (2025) MIRI/LRS reduction."""

    if reduction not in ("jexores", "jexopipe"):
        raise ValueError("reduction must be 'jexores' or 'jexopipe'.")
    path = Path(path)
    display_reduction = "JExoRES" if reduction == "jexores" else "JexoPipe"
    return parse_center_half_width_spectrum(
        path.read_bytes(),
        dataset_id="miri_lrs",
        instrument="JWST/MIRI",
        mode="LRS",
        reduction=f"Madhusudhan et al. (2025) {display_reduction}",
        offset_group="miri_lrs",
        provenance=str(path),
    )


def load_nirspec_shifted_average(
    archive_path: str | Path,
    members: Iterable[ArchiveSpectrum] = NIRSPEC_SHIFTED_AVERAGE,
) -> tuple[TransitSpectrum, ...]:
    """Load the four detector-separated Hu et al. shifted averages."""

    archive_path = Path(archive_path)
    spectra = []
    with ZipFile(archive_path) as archive:
        for member in members:
            try:
                payload = archive.read(member.member_path)
            except KeyError as error:
                raise ValueError(
                    f"Missing archive member {member.member_path}."
                ) from error
            provenance = f"{archive_path}:{member.member_path}"
            verify_sha256(payload, member.sha256, provenance)
            spectra.append(
                parse_edge_center_spectrum(
                    payload,
                    dataset_id=member.dataset_id,
                    instrument=member.instrument,
                    mode=member.mode,
                    reduction=member.reduction,
                    offset_group=member.offset_group,
                    provenance=provenance,
                )
            )
    return tuple(spectra)


def fetch_figure3_public_assets(
    cache_directory: str | Path,
    *,
    niriss_resolution: Literal["lowres", "native"] = "lowres",
    miri_reduction: Literal["jexores", "jexopipe"] = "jexores",
    opener: Callable[[str], object] | None = None,
) -> Mapping[str, Path]:
    """Fetch the three public source products needed for a Fig. 3 plot."""

    niriss_key = f"niriss_soss_{niriss_resolution}"
    miri_key = f"miri_lrs_{miri_reduction}"
    keys = (niriss_key, "nirspec_hu2025_archive", miri_key)
    return {
        key: fetch_asset(PUBLIC_OSF_ASSETS[key], cache_directory, opener)
        for key in keys
    }


def load_figure3_public_spectra(
    asset_paths: Mapping[str, str | Path],
    *,
    niriss_resolution: Literal["lowres", "native"] = "lowres",
    miri_reduction: Literal["jexores", "jexopipe"] = "jexores",
) -> tuple[TransitSpectrum, ...]:
    """Load public Fig. 3 inputs previously obtained with the manifest.

    The tuple contains NIRISS, four detector-separated NIRSpec spectra, and
    MIRI.  The old G395H product in OSF project 36djh and the alternative
    NIRISS product inside the Hu archive are intentionally not combined,
    avoiding duplicate observations.
    """

    niriss_key = f"niriss_soss_{niriss_resolution}"
    miri_key = f"miri_lrs_{miri_reduction}"
    niriss = load_niriss_soss(
        asset_paths[niriss_key],
        resolution=niriss_resolution,
    )
    nirspec = load_nirspec_shifted_average(
        asset_paths["nirspec_hu2025_archive"]
    )
    miri = load_miri_lrs(
        asset_paths[miri_key],
        reduction=miri_reduction,
    )
    return (niriss, *nirspec, miri)


def model_with_dataset_offset(
    model_transit_depth: np.ndarray,
    offset_ppm: float,
) -> np.ndarray:
    """Add a wavelength-independent nuisance offset to a model spectrum.

    Positive ``offset_ppm`` increases the model transit depth.  Archived
    observations are never mutated.  One offset group must be anchored (or
    otherwise constrained) by the calling fit to remove the common-mode
    radius/offset degeneracy.
    """

    return np.asarray(model_transit_depth, dtype=float) + offset_ppm * 1.0e-6
