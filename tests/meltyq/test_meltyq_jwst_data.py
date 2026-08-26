from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from examples.meltyq.meltyq_jwst_data import (
    ArchiveSpectrum,
    PUBLIC_OSF_ASSETS,
    RemoteAsset,
    fetch_asset,
    load_nirspec_shifted_average,
    model_with_dataset_offset,
    parse_center_half_width_spectrum,
    parse_edge_center_spectrum,
    sha256_bytes,
)


CENTER_HALF_WIDTH_FIXTURE = b"""wavelength, half width, depth, uncertainty
1.00 0.05 0.00300 0.00010
1.20 0.10 0.00320 0.00020
"""

EDGE_CENTER_FIXTURE = b"""1.00 1.10 1.05 0.00300 0.00010
1.10 1.20 1.15 0.00320 0.00020
"""


def _parse_center_fixture():
    return parse_center_half_width_spectrum(
        CENTER_HALF_WIDTH_FIXTURE,
        dataset_id="niriss_soss",
        instrument="JWST/NIRISS",
        mode="SOSS",
        reduction="fixture",
        offset_group="niriss_soss",
        provenance="fixture.txt",
    )


def test_center_half_width_parser_preserves_fractional_depths_and_edges():
    spectrum = _parse_center_fixture()

    np.testing.assert_allclose(spectrum.wavelength_lower_um, [0.95, 1.10])
    np.testing.assert_allclose(spectrum.wavelength_upper_um, [1.05, 1.30])
    np.testing.assert_allclose(spectrum.transit_depth, [0.0030, 0.0032])
    np.testing.assert_allclose(spectrum.transit_depth_ppm, [3000.0, 3200.0])
    assert spectrum.size == 2
    assert not spectrum.transit_depth.flags.writeable


def test_edge_center_parser_uses_explicit_bin_edges():
    spectrum = parse_edge_center_spectrum(
        EDGE_CENTER_FIXTURE,
        dataset_id="nirspec_g235h_nrs1",
        instrument="JWST/NIRSpec",
        mode="G235H/NRS1",
        reduction="fixture",
        offset_group="nirspec_g235h_nrs1",
        provenance="fixture.dat",
    )

    np.testing.assert_allclose(spectrum.wavelength_um, [1.05, 1.15])
    np.testing.assert_allclose(spectrum.wavelength_lower_um, [1.0, 1.1])
    np.testing.assert_allclose(spectrum.wavelength_upper_um, [1.1, 1.2])


def test_invalid_units_are_rejected_as_nonfractional_depths():
    payload = b"header\n1.0 0.1 3000.0 30.0\n"
    with pytest.raises(ValueError, match="dimensionless fractions"):
        parse_center_half_width_spectrum(
            payload,
            dataset_id="bad",
            instrument="test",
            mode="test",
            reduction="test",
            offset_group="bad",
            provenance="bad.txt",
        )


def test_model_offset_is_in_ppm_and_does_not_mutate_input():
    original = np.array([0.0030, 0.0031])
    shifted = model_with_dataset_offset(original, offset_ppm=125.0)

    np.testing.assert_allclose(shifted, [0.003125, 0.003225])
    np.testing.assert_allclose(original, [0.0030, 0.0031])


def test_fetch_asset_uses_hash_and_never_needs_network(tmp_path):
    payload = b"fixture payload"
    asset = RemoteAsset(
        key="fixture",
        filename="fixture.dat",
        url="https://invalid.example/fixture.dat",
        sha256=sha256_bytes(payload),
    )
    calls = []

    def opener(url):
        calls.append(url)
        return BytesIO(payload)

    path = fetch_asset(asset, tmp_path, opener=opener)
    cached_path = fetch_asset(
        asset,
        tmp_path,
        opener=lambda _url: pytest.fail("cache should avoid a second fetch"),
    )

    assert path == cached_path == tmp_path / "fixture.dat"
    assert path.read_bytes() == payload
    assert calls == [asset.url]


def test_fetch_asset_rejects_a_corrupt_existing_cache(tmp_path):
    payload = b"correct"
    asset = RemoteAsset(
        key="fixture",
        filename="fixture.dat",
        url="https://invalid.example/fixture.dat",
        sha256=sha256_bytes(payload),
    )
    (tmp_path / asset.filename).write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_asset(asset, tmp_path, opener=lambda _url: BytesIO(payload))


def test_nirspec_archive_member_is_hash_checked(tmp_path):
    member_path = "spectra/nirspec_combined/fixture.dat"
    archive_path = tmp_path / "fixture.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(member_path, EDGE_CENTER_FIXTURE)

    descriptor = ArchiveSpectrum(
        dataset_id="nirspec_fixture",
        member_path=member_path,
        sha256=sha256_bytes(EDGE_CENTER_FIXTURE),
        instrument="JWST/NIRSpec",
        mode="G235H/NRS1",
        reduction="fixture",
        offset_group="nirspec_fixture",
    )
    spectra = load_nirspec_shifted_average(
        archive_path,
        members=(descriptor,),
    )

    assert len(spectra) == 1
    assert spectra[0].size == 2
    assert spectra[0].offset_group == "nirspec_fixture"

    bad_descriptor = ArchiveSpectrum(
        dataset_id=descriptor.dataset_id,
        member_path=descriptor.member_path,
        sha256="0" * 64,
        instrument=descriptor.instrument,
        mode=descriptor.mode,
        reduction=descriptor.reduction,
        offset_group=descriptor.offset_group,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_nirspec_shifted_average(
            archive_path,
            members=(bad_descriptor,),
        )


def test_manifest_pins_revisioned_osf_urls_and_known_archive_hash():
    assert all(
        asset.url.startswith("https://osf.io/download/")
        and asset.url.endswith("?revision=1")
        for asset in PUBLIC_OSF_ASSETS.values()
    )
    assert (
        PUBLIC_OSF_ASSETS["nirspec_hu2025_archive"].sha256
        == "4ee5cb6ad42015bd8fb10f64e54329d250137ab1fa129c89a14830946adc8f18"
    )
