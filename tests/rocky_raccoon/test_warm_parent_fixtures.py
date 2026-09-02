"""Contract checks for the positional Rocky Raccoon warm-parent fixtures."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from examples.rocky_raccoon.paper import (
    FASTCHEM_CHARGE_ELEMENT,
    PAPER_GAS_SPECIES,
    PAPER_PHYSICAL_ELEMENTS,
)


DATA_DIRECTORY = Path(__file__).with_name("data")
FIXTURE_GAS_ORDER_SHA256 = (
    "014c19083a9a9ba5737c687adea998f856c51d178f46c7dfa2dca2147ce06290"
)
REQUIRED_KEYS = {
    "parent_condensate_amounts",
    "parent_gas_ln_n",
    "parent_gas_n",
    "parent_gas_ntot",
    "parent_inventory",
    "parent_pressure",
    "parent_support",
    "parent_temperature",
    "target_inventory",
    "target_pressure",
    "target_temperature",
}
FIXTURE_CASES = (
    (
        "default_column_step_1075_warm_parent.npz",
        "d747a72d147f1254766eb4a9c5567768695f4afeb4c5035ee8412c12096cfe3f",
        0,
        21,
        15,
    ),
    (
        "default_column_step_1084_warm_parent.npz",
        "a5311bacf99cc5b1abb48bbc8058f5ad11f304f64cde7dfbcf692a348c1d4b03",
        19,
        2,
        19,
    ),
)
NAMED_FIXTURE_CASES = (
    (
        "default_column_step_1076_warm_parent.npz",
        "5b971c4f35c02b80390466f8a5b95e79d48011b64a74ad262bc490327adc79f6",
        "889e80d5b2d0a47e795d1e0b29be2225e1af4fc3a69dbfa682f7e0cc6a38a79d",
        1075,
        1076,
        "nonconvective",
        163.3828868151611,
        4.063162466408924,
        160.63170064473724,
        4.0225308417448344,
        22,
        18,
    ),
    (
        "default_column_step_1077_convective_warm_parent.npz",
        "dc026260c895a33a3f64801f1d38c96f54b0dee3153f9e317c26fd5cec207163",
        "cde0a592c5468393b357809ab71c2a7ebfdb1b5438915c83a1b802167905e692",
        1076,
        1077,
        "convective",
        162.91607856710917,
        4.022530841744834,
        162.45060405690884,
        3.982305533327386,
        20,
        17,
    ),
    (
        "default_column_step_1186_convective_warm_parent.npz",
        "24c017328b99882ac3b7edb96fbc8e3ea17476780033a2d3c8109f7de754a39c",
        "0b2f9439610d8743abf0e40ca49d9561e6be31518e829e649bafcef082e064f6",
        1185,
        1186,
        "convective",
        119.26654880286326,
        1.3450412196053743,
        118.9257872348488,
        1.3315908074093206,
        32,
        28,
    ),
    (
        "default_column_step_1342_nonconvective_warm_parent.npz",
        "09206f20e4a651fec9dd97c5b2f4838e87da3a204e70c2a5a96558715f52088b",
        "8135fc9ca65a5a8028ab05d6fabce01baf5dc4f3563dafed2c8456673e3314d1",
        1341,
        1342,
        "nonconvective",
        76.32561973848323,
        0.2804309550102501,
        74.96492038416324,
        0.2776266454601476,
        54,
        50,
    ),
    (
        "default_column_step_1372_convective_warm_parent.npz",
        "2c3f4420c4731a75b91fedeb9064151666bb34d50a0a14d55255bb6e5b6fcb91",
        "046420b5bd8e83465a7e27b7da71ba0cfb7c46565feb62a76e18d3a10dced64c",
        1371,
        1372,
        "convective",
        70.04736736757862,
        0.207434882130714,
        69.84723203223899,
        0.20536053330940687,
        54,
        53,
    ),
    (
        "default_column_step_1383_convective_warm_parent.npz",
        "0b3f6c190e60427733955f0032373a24044605bb4dc71a03fcb505cb011ce835",
        "dce9771690ed7f0585447703d953b1979fccb75b7b19bb958f13aded9270c239",
        1382,
        1383,
        "convective",
        67.8770604814665,
        0.18572438523927615,
        67.68312602294445,
        0.18386714138688337,
        54,
        54,
    ),
)
NAMED_FIXTURE_REQUIRED_KEYS = {
    "element_names",
    "gas_species",
    "parent_gas_ln_n",
    "parent_inventory",
    "parent_layer_index",
    "parent_pressure",
    "parent_temperature",
    "schema",
    "source_provenance_json",
    "source_snapshot_sha256",
    "target_candidate",
    "target_inventory",
    "target_pressure",
    "target_step_index",
    "target_temperature",
}


def _ordered_name_digest(names) -> str:
    return sha256("\0".join(names).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    (
        "filename",
        "expected_sha256",
        "absent_count",
        "subfloor_count",
        "linear_zero_count",
    ),
    FIXTURE_CASES,
)
def test_warm_parent_fixture_contract(
    filename,
    expected_sha256,
    absent_count,
    subfloor_count,
    linear_zero_count,
):
    path = DATA_DIRECTORY / filename
    assert sha256(path.read_bytes()).hexdigest() == expected_sha256
    assert _ordered_name_digest(PAPER_GAS_SPECIES) == FIXTURE_GAS_ORDER_SHA256
    assert PAPER_PHYSICAL_ELEMENTS + (FASTCHEM_CHARGE_ELEMENT,) == (
        "H",
        "Mg",
        "Si",
        "O",
        "C",
        "e-",
    )

    with np.load(path) as stored:
        assert set(stored.files) == REQUIRED_KEYS
        gas_logs = np.asarray(stored["parent_gas_ln_n"])
        gas_amounts = np.asarray(stored["parent_gas_n"])
        assert gas_logs.shape == gas_amounts.shape == (len(PAPER_GAS_SPECIES),)
        assert np.asarray(stored["parent_inventory"]).shape == (6,)
        assert np.asarray(stored["target_inventory"]).shape == (6,)
        assert np.all(np.isfinite(gas_logs) | np.isneginf(gas_logs))
        assert np.all(np.isfinite(gas_amounts))
        assert np.all(gas_amounts >= 0.0)

        finite = np.isfinite(gas_logs)
        relative_floor = (
            np.logaddexp.reduce(gas_logs[finite]) + np.log(1.0e-300)
        )
        assert np.count_nonzero(~finite) == absent_count
        assert np.count_nonzero(gas_logs[finite] < relative_floor) == (
            subfloor_count
        )
        assert np.count_nonzero(gas_amounts == 0.0) == linear_zero_count


@pytest.mark.parametrize(
    (
        "filename",
        "expected_sha256",
        "source_snapshot_sha256",
        "parent_layer_index",
        "target_step_index",
        "target_candidate",
        "parent_temperature",
        "parent_pressure",
        "target_temperature",
        "target_pressure",
        "subfloor_count",
        "linear_zero_count",
    ),
    NAMED_FIXTURE_CASES,
)
def test_warm_parent_fixture_is_named_and_self_describing(
    filename,
    expected_sha256,
    source_snapshot_sha256,
    parent_layer_index,
    target_step_index,
    target_candidate,
    parent_temperature,
    parent_pressure,
    target_temperature,
    target_pressure,
    subfloor_count,
    linear_zero_count,
):
    path = DATA_DIRECTORY / filename
    assert sha256(path.read_bytes()).hexdigest() == expected_sha256

    with np.load(path) as stored:
        assert set(stored.files) == NAMED_FIXTURE_REQUIRED_KEYS
        assert stored["schema"].item() == "rocky_raccoon.warm_parent_case@1"
        assert stored["parent_layer_index"].item() == parent_layer_index
        assert stored["target_step_index"].item() == target_step_index
        assert stored["target_candidate"].item() == target_candidate
        assert tuple(stored["element_names"].tolist()) == (
            PAPER_PHYSICAL_ELEMENTS + (FASTCHEM_CHARGE_ELEMENT,)
        )
        assert tuple(stored["gas_species"].tolist()) == PAPER_GAS_SPECIES
        assert stored["source_snapshot_sha256"].item() == (
            source_snapshot_sha256
        )
        assert json.loads(stored["source_provenance_json"].item())[
            "schema_version"
        ] == "rocky_raccoon.source_provenance@1"

        gas_logs = np.asarray(stored["parent_gas_ln_n"])
        assert gas_logs.shape == (len(PAPER_GAS_SPECIES),)
        assert np.all(np.isfinite(gas_logs))
        relative_floor = np.logaddexp.reduce(gas_logs) + np.log(1.0e-300)
        assert np.count_nonzero(gas_logs < relative_floor) == subfloor_count
        assert np.count_nonzero(np.exp(gas_logs) == 0.0) == linear_zero_count
        assert np.asarray(stored["parent_inventory"]).shape == (6,)
        assert np.asarray(stored["target_inventory"]).shape == (6,)
        assert stored["parent_temperature"].item() == parent_temperature
        assert stored["parent_pressure"].item() == parent_pressure
        assert stored["target_temperature"].item() == target_temperature
        assert stored["target_pressure"].item() == target_pressure
