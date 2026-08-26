from argparse import Namespace
import json

import numpy as np
import pytest

from examples.meltyq.meltyq_figure3_build_diffgrid import (
    _write_manifest,
    _validate_arguments,
    broadening_provenance,
    build_contract,
    build_parser,
    preparation_estimate,
)


def _arguments(**updates):
    arguments = vars(
        build_parser().parse_args(["--species", "CO", "--dry-run"])
    )
    arguments.update(updates)
    return Namespace(**arguments)


def test_dry_run_estimate_is_one_species_and_requires_no_database():
    estimate = preparation_estimate(_arguments())

    assert estimate["species"] == "CO"
    assert estimate["spectral_points"] == 145788
    assert estimate["measured_grid_resolving_power"] >= 50000.0
    assert estimate["one_species_value_and_derivative_table_gib"] > 4.0


def test_manifest_refuses_a_different_common_build_contract(tmp_path):
    arguments = _arguments()
    nu_grid = np.geomspace(1000.0, 2000.0, 8)
    pressure_grid = np.geomspace(1.0e-10, 10.0, 4)
    temperature_grid = np.array([1200.0, 500.0, 200.0])
    contract = build_contract(
        arguments,
        nu_grid,
        pressure_grid,
        temperature_grid,
    )
    descriptor = {
        "molecule": "CO",
        "path": "CO.npz",
    }
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, descriptor, 50000.0, contract)

    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["build_contract"] == contract

    changed_contract = build_contract(
        arguments,
        np.geomspace(1000.0, 2100.0, 8),
        pressure_grid,
        temperature_grid,
    )
    with pytest.raises(ValueError, match="contract differs"):
        _write_manifest(
            manifest_path,
            {"molecule": "H2O", "path": "H2O.npz"},
            50000.0,
            changed_contract,
        )


def test_builder_rejects_incompatible_stitch_and_error_limits_before_loading_data():
    with pytest.raises(ValueError, match="at least 50000"):
        _validate_arguments(_arguments(minimum_resolving_power=1000.0))

    with pytest.raises(ValueError, match="divisible by nstitch"):
        _validate_arguments(_arguments(nstitch=5))

    with pytest.raises(ValueError, match="at least the p99"):
        _validate_arguments(
            _arguments(
                maximum_p99_log_cross_section_error=0.1,
                maximum_log_cross_section_error=0.05,
            )
        )


def test_broadening_fallback_requires_explicit_acknowledgement(tmp_path):
    class MolecularDatabase:
        broad_files = {"H2": tmp_path / "12C-16O__H2.broad"}
        alpha_ref = np.full(4, 0.07)
        n_Texp = np.full(4, 0.5)
        alpha_ref_def = 0.07
        n_Texp_def = 0.5

    arguments = _arguments(allow_default_broadening_fallback=False)
    with pytest.raises(FileNotFoundError, match="broadening file"):
        broadening_provenance(MolecularDatabase(), arguments)

    arguments.allow_default_broadening_fallback = True
    provenance = broadening_provenance(MolecularDatabase(), arguments)
    assert provenance["requested_file_present"] is False
    assert provenance["effective_source"] == (
        "definition_file_defaults_missing_requested_file"
    )
    assert provenance["effective_alpha_ref_range_cm1_bar"] == [0.07, 0.07]
