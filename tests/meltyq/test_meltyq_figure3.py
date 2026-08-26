import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import examples.meltyq.meltyq_figure3 as figure3
from exogibbs.presets.magma_gas import MELTYQ_SPECIES
from exojax.opacity import OpaCIA
from exojax.rt import ArtTransPure

from examples.meltyq.meltyq_figure3 import (
    ABSORBING_SPECIES,
    OpacityBundle,
    PAPER_TO_PROVIDER_FACTORS,
    SCENARIO_NAMES,
    _diffgrid_broadening_contract,
    _validate_cia_interpolation_grid,
    _validate_cia_file_identity,
    _source_database_inventory_sha256,
    _validate_archive_interpolation_gate,
    _validate_ckd_table_identity,
    _validate_wavenumber_coverage,
    bin_model_scenarios,
    build_art,
    compare_reference,
    load_config,
    load_reference_spectrum,
    make_spectral_kernel,
    prepare_spectral_opacity,
    provider_magma_parameters,
    validate_config,
    validate_opacity_state,
)
from examples.meltyq.meltyq_jwst_data import TransitSpectrum


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_cia_grid_contract_rejects_duplicate_wavenumbers():
    class Database:
        nucia = np.array([1000.0, 1000.0])
        tcia = np.array([200.0, 400.0])
        logac = np.zeros((2, 2))

    with pytest.raises(ValueError, match="unique"):
        _validate_cia_interpolation_grid(Database(), "H2-H2")


def test_cia_grid_contract_accepts_rectangular_unique_grid():
    class Database:
        nucia = np.array([1000.0, 2000.0])
        tcia = np.array([200.0, 400.0])
        logac = np.full((2, 2), -45.0)

    _validate_cia_interpolation_grid(Database(), "H2-H2")


def test_cia_file_identity_rejects_a_renamed_pair(tmp_path):
    cia_path = tmp_path / "H2-H2_2011.cia"
    cia_path.write_text(
        "H2-He 20.0 21.0 2 200.0 1.0e-45 0.0 1\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="CIA pair mismatch"):
        _validate_cia_file_identity(cia_path, "H2-H2")

    _validate_cia_file_identity(cia_path, "H2-He")


@pytest.mark.parametrize(
    ("radiative_transfer_scheme", "expected_coverage"),
    (
        ("taurex_rectangle", [True, True, True]),
        ("exojax_simpson", [False, True, False]),
    ),
)
def test_preparation_uses_exojax_cia_interpolation_and_edge_values(
    radiative_transfer_scheme,
    expected_coverage,
):
    class CiaDatabase:
        nucia = jnp.array([1000.0, 2000.0])
        tcia = jnp.array([200.0, 400.0])
        logac = jnp.log10(
            jnp.array([[1.0e-46, 5.0e-46], [9.0e-46, 13.0e-46]])
        )

    class MolecularCalculator:
        molmass = 20.0

        def xsmatrix(self, temperature_k):
            return jnp.full((temperature_k.size, 3), 1.0e-30)

    bundle = OpacityBundle(
        mode="diffgrid",
        calculators={
            species: MolecularCalculator() for species in ABSORBING_SPECIES
        },
        wavenumber_cm1=np.array([500.0, 1500.0, 2500.0]),
        weights=None,
        provenance={},
    )
    cia_calculator = OpaCIA(
        CiaDatabase(),
        nu_grid=bundle.wavenumber_cm1,
        wavenumber_interpolation="interp",
    )
    prepared = prepare_spectral_opacity(
        bundle,
        {"H2-H2": (cia_calculator, jnp.array([False, True, False]))},
        {},
        jnp.array([300.0]),
        jnp.array([1.0]),
        radiative_transfer_scheme=radiative_transfer_scheme,
    )

    np.testing.assert_allclose(
        10.0 ** np.asarray(prepared.log_cia_coefficients),
        [[[5.0e-46, 7.0e-46, 9.0e-46]]],
        rtol=3.0e-6,
    )
    np.testing.assert_array_equal(
        prepared.cia_coverage_masks,
        [expected_coverage],
    )


PUBLIC_DEMO_CONFIG = (
    REPOSITORY_ROOT
    / "examples/meltyq/configs/meltyq_figure3_public_demo.json"
)
AUTHOR_TEMPLATE_CONFIG = (
    REPOSITORY_ROOT
    / "examples/meltyq/configs/meltyq_figure3_author.template.json"
)


class ConstantDiffgridOpacity:
    def __init__(self, value, molmass):
        self.value = value
        self.molmass = molmass

    def xsmatrix(self, temperature_k):
        return jnp.full((temperature_k.size, 4), self.value)


class ConstantCkdOpacity:
    def __init__(self, value, molmass):
        self.value = value
        self.molmass = molmass

    def xstensor_ckd(self, temperature_k, pressure_bar):
        return jnp.full((temperature_k.size, 2, 4), self.value)


def _observation(dataset_id, wavelength, depth_ppm, sigma_ppm):
    wavelength = np.asarray(wavelength)
    return TransitSpectrum(
        dataset_id=dataset_id,
        instrument="test",
        mode="test",
        reduction="test",
        offset_group=dataset_id,
        wavelength_um=wavelength,
        wavelength_lower_um=wavelength - 0.05,
        wavelength_upper_um=wavelength + 0.05,
        transit_depth=np.asarray(depth_ppm) * 1.0e-6,
        transit_depth_uncertainty=np.asarray(sigma_ppm) * 1.0e-6,
        provenance="fixture",
    )


def test_public_demo_config_uses_exojax_native_rt_and_is_not_author_best_fit():
    config = load_config(PUBLIC_DEMO_CONFIG)

    assert config.claim_status == "public_demo_not_unpublished_best_fit"
    assert config.upper.nlayer == 100
    assert config.spectral.opacity_mode == "ckd"
    assert config.spectral.radiative_transfer_scheme == "exojax_simpson"
    assert config.spectral.rayleigh_provider == "taurex"
    art = build_art(config.upper, np.linspace(1000.0, 2000.0, 4))
    assert art.integration == "simpson"


def test_config_rejects_a_truncated_haze_interval():
    config = load_config(PUBLIC_DEMO_CONFIG)
    truncated_haze = replace(
        config.haze,
        mid_pressure_pa=config.upper.pressure_top_bar * 1.0e5,
    )

    with pytest.raises(ValueError, match="complete Lee haze"):
        validate_config(replace(config, haze=truncated_haze))


def test_config_rejects_unused_offset_keys():
    config = load_config(PUBLIC_DEMO_CONFIG)
    invalid_observations = replace(
        config.observations,
        offsets_ppm={**config.observations.offsets_ppm, "typo": 0.0},
    )

    with pytest.raises(ValueError, match="unused typo"):
        validate_config(replace(config, observations=invalid_observations))


def test_author_template_cannot_be_mistaken_for_a_runnable_config():
    payload = json.loads(AUTHOR_TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    assert payload["spectral"]["radiative_transfer_scheme"] == "taurex_rectangle"
    with pytest.raises(ValueError, match="must be finite and positive"):
        load_config(AUTHOR_TEMPLATE_CONFIG)


def test_paper_labelled_melt_mapping_is_explicit_and_reversible_by_choice():
    config = load_config(PUBLIC_DEMO_CONFIG)
    mapped = provider_magma_parameters(config.magma)

    assert mapped.co_melt_mole_ratio == pytest.approx(
        config.magma.co_melt_mole_ratio
        * PAPER_TO_PROVIDER_FACTORS["co_melt_mole_ratio"]
    )
    assert mapped.n_melt_mole_ratio == pytest.approx(
        config.magma.n_melt_mole_ratio
        * PAPER_TO_PROVIDER_FACTORS["n_melt_mole_ratio"]
    )

    provider_config = replace(config.magma, input_basis="exogibbs_elemental")
    direct = provider_magma_parameters(provider_config)
    assert direct.co_melt_mole_ratio == config.magma.co_melt_mole_ratio
    assert direct.n_melt_mole_ratio == config.magma.n_melt_mole_ratio


def test_profiled_offsets_use_one_group_and_leave_anchor_fixed():
    config = load_config(PUBLIC_DEMO_CONFIG).observations
    config = replace(
        config,
        offset_group_by_dataset={"anchor": "anchor", "free": "free"},
        offsets_ppm={"anchor": 0.0, "free": 0.0},
        anchor_offset_group="anchor",
    )
    wavelength = np.linspace(1.0, 2.0, 21)
    scenario = np.vstack(
        [np.full_like(wavelength, 3000.0 + index) for index in range(len(SCENARIO_NAMES))]
    )
    observations = (
        _observation("anchor", [1.25], [3000.0], [10.0]),
        _observation("free", [1.75], [3125.0], [10.0]),
    )

    binned, offsets = bin_model_scenarios(
        wavelength,
        scenario,
        observations,
        config,
        profile_offsets=True,
    )

    assert offsets["anchor"] == 0.0
    assert offsets["free"] == pytest.approx(125.0)
    assert binned[1].scenario_depth_ppm["total"][0] == pytest.approx(3125.0)


def test_diffgrid_observation_binning_uses_piecewise_linear_model():
    dataset_id = "diffgrid"
    config = replace(
        load_config(PUBLIC_DEMO_CONFIG).observations,
        offset_group_by_dataset={dataset_id: dataset_id},
        offsets_ppm={dataset_id: 0.0},
        anchor_offset_group=None,
    )
    wavelength = np.array([1.0, 2.0, 4.0])
    lower = np.array([1.25, 2.5])
    upper = np.array([1.75, 3.5])
    observation = TransitSpectrum(
        dataset_id=dataset_id,
        instrument="test",
        mode="test",
        reduction="test",
        offset_group=dataset_id,
        wavelength_um=0.5 * (lower + upper),
        wavelength_lower_um=lower,
        wavelength_upper_um=upper,
        transit_depth=np.full(lower.size, 3.0e-3),
        transit_depth_uncertainty=np.full(lower.size, 1.0e-4),
        provenance="fixture",
    )
    scenarios = np.stack(
        [
            2.0 * wavelength + 3.0 + scenario_index
            for scenario_index in range(len(SCENARIO_NAMES))
        ]
    )

    binned, _ = bin_model_scenarios(
        wavelength,
        scenarios,
        (observation,),
        config,
        profile_offsets=False,
    )

    for scenario_index, name in enumerate(SCENARIO_NAMES):
        np.testing.assert_allclose(
            binned[0].scenario_depth_ppm[name],
            [6.0 + scenario_index, 9.0 + scenario_index],
        )


def test_ckd_observation_binning_uses_finite_band_means():
    dataset_id = "ckd"
    config = replace(
        load_config(PUBLIC_DEMO_CONFIG).observations,
        offset_group_by_dataset={dataset_id: dataset_id},
        offsets_ppm={dataset_id: 0.0},
        anchor_offset_group=None,
    )
    native_edges = np.array([[1.0, 1.5], [1.5, 2.0], [2.0, 2.5]])
    lower = np.array([1.25, 1.75])
    upper = np.array([1.75, 2.25])
    observation = TransitSpectrum(
        dataset_id=dataset_id,
        instrument="test",
        mode="test",
        reduction="test",
        offset_group=dataset_id,
        wavelength_um=0.5 * (lower + upper),
        wavelength_lower_um=lower,
        wavelength_upper_um=upper,
        transit_depth=np.full(lower.size, 3.0e-3),
        transit_depth_uncertainty=np.full(lower.size, 1.0e-4),
        provenance="fixture",
    )
    band_means = np.array([10.0, 20.0, 40.0])
    scenarios = np.stack(
        [
            band_means + scenario_index
            for scenario_index in range(len(SCENARIO_NAMES))
        ]
    )

    binned, _ = bin_model_scenarios(
        0.5 * np.sum(native_edges, axis=1),
        scenarios,
        (observation,),
        config,
        profile_offsets=False,
        native_band_edges_micron=native_edges,
    )

    for scenario_index, name in enumerate(SCENARIO_NAMES):
        np.testing.assert_allclose(
            binned[0].scenario_depth_ppm[name],
            [15.0 + scenario_index, 30.0 + scenario_index],
        )


def test_reference_comparison_reports_known_ppm_residual():
    summary = compare_reference(
        np.array([1.0, 2.0, 3.0]),
        np.array([100.0, 200.0, 300.0]),
        np.array([1.5, 2.5]),
        np.array([140.0, 240.0]),
    )

    assert summary["model_minus_reference_rms_ppm"] == pytest.approx(10.0)
    assert summary["model_minus_reference_maximum_absolute_ppm"] == pytest.approx(10.0)


def test_reference_csv_requires_intrinsic_unoffset_contract(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "spectrum_contract,wavelength_micron,transit_depth_ppm\n"
        "intrinsic_unoffset_model,1.0,3000.0\n"
        "intrinsic_unoffset_model,2.0,3100.0\n",
        encoding="utf-8",
    )
    wavelength, depth = load_reference_spectrum(reference)
    np.testing.assert_allclose(wavelength, [1.0, 2.0])
    np.testing.assert_allclose(depth, [3000.0, 3100.0])

    reference.write_text(
        "spectrum_contract,wavelength_micron,transit_depth_ppm\n"
        "dataset_offset_applied,1.0,3000.0\n"
        "dataset_offset_applied,2.0,3100.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intrinsic_unoffset_model"):
        load_reference_spectrum(reference)


def test_archive_interpolation_gate_requires_every_inverse_temperature_midpoint():
    build_contract = {
        "temperature_grid": {
            "nodes": 3,
            "minimum_k": 200.0,
            "maximum_k": 1200.0,
            "uniform_in_inverse_temperature": True,
        },
        "teacher": {
            "validation_temperature_rule": (
                "every inverse-temperature interval midpoint"
            ),
            "validation_profile_count": 2,
        },
    }
    contract = {
        "maximum_p99_log_error": 0.05,
        "maximum_log_error": 0.5,
    }
    inverse_temperature = np.linspace(1.0 / 1200.0, 1.0 / 200.0, 3)
    expected_temperatures = 1.0 / (
        0.5 * (inverse_temperature[:-1] + inverse_temperature[1:])
    )
    measurements = [
        {
            "isothermal_temperature_k": float(temperature),
            "p99_absolute_log_cross_section_error": 0.01,
            "maximum_absolute_log_cross_section_error": 0.1,
        }
        for temperature in expected_temperatures
    ]
    user_meta = {
        "diffgrid_teacher_interpolation_quality_gate": {
            "status": "passed",
            "maximum_p99": 0.05,
            "maximum": 0.5,
        },
        "diffgrid_teacher_interpolation_validation": measurements,
    }

    _validate_archive_interpolation_gate(
        user_meta,
        contract,
        build_contract,
        "CO",
    )

    user_meta["diffgrid_teacher_interpolation_validation"] = measurements[:1]
    with pytest.raises(ValueError, match="all interpolation validation profiles"):
        _validate_archive_interpolation_gate(
            user_meta,
            contract,
            build_contract,
            "CO",
        )

    user_meta["diffgrid_teacher_interpolation_validation"] = measurements
    measurements[1]["maximum_absolute_log_cross_section_error"] = 0.6
    with pytest.raises(ValueError, match="failed validation"):
        _validate_archive_interpolation_gate(
            user_meta,
            contract,
            build_contract,
            "CO",
        )


def test_ckd_identity_rejects_a_renamed_table_for_another_molecule(tmp_path):
    h5py = pytest.importorskip("h5py")
    table = (
        tmp_path
        / "1H2-16O__POKAZATEL__R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    with h5py.File(table, "w") as archive:
        archive.create_dataset("mol_mass", data=np.array([44.0095]))

    with pytest.raises(ValueError, match="molecular mass mismatch"):
        _validate_ckd_table_identity(table, "H2O")

    with h5py.File(table, "w") as archive:
        archive.create_dataset("mol_mass", data=np.array([18.01528]))
        archive.create_dataset("mol_name", data=np.bytes_("CO2"))

    with pytest.raises(ValueError, match="mol_name mismatch"):
        _validate_ckd_table_identity(table, "H2O")


def test_ckd_identity_accepts_exomolop_integer_molecular_mass(tmp_path):
    h5py = pytest.importorskip("h5py")
    table = (
        tmp_path
        / "12C-1H4__YT34to10.R1000_0.3-50mu.ktable.petitRADTRANS.h5"
    )
    with h5py.File(table, "w") as archive:
        archive.create_dataset("mol_mass", data=np.array([16]))
        archive.create_dataset("mol_name", data=np.bytes_("CH4"))

    identity = _validate_ckd_table_identity(table, "CH4")

    assert identity["molecular_mass"] == 16.0


def test_spectral_runner_uses_the_supplied_base_radius(monkeypatch):
    config = SimpleNamespace(
        spectral=SimpleNamespace(radiative_transfer_scheme="exojax_simpson"),
        planet=SimpleNamespace(mass_earth=1.0, stellar_radius_solar=1.0),
        magma=SimpleNamespace(temperature_base_k=500.0),
        upper=SimpleNamespace(
            pressure_base_bar=10.0,
            temperature_1e4_pa_k=500.0,
            temperature_100_pa_k=500.0,
            temperature_1e_minus2_pa_k=500.0,
            smoothing_window_percent=0.0,
        ),
        cloud=SimpleNamespace(enabled=False, top_pressure_pa=1.0e3),
        haze=SimpleNamespace(
            enabled=False,
            particle_radius_micron=0.1,
            q0=1.0,
            particle_number_density_m3=1.0,
            mid_pressure_pa=1.0,
            log10_pressure_extent=1.0,
        ),
    )
    prepared = figure3.PreparedSpectralOpacity(
        molecular_cross_sections=jnp.empty((0, 1, 3)),
        log_cia_coefficients=jnp.empty((0, 1, 3)),
        cia_coverage_masks=jnp.empty((0, 3), dtype=bool),
        rayleigh_cross_sections=jnp.empty((0, 3)),
        preparation_seconds_including_any_compilation=0.0,
    )
    monkeypatch.setattr(
        figure3,
        "upper_temperature_profile_k",
        lambda *args: jnp.asarray([500.0]),
    )
    monkeypatch.setattr(
        figure3,
        "prepare_spectral_opacity",
        lambda *args, **kwargs: prepared,
    )

    def kernel(*arguments):
        assert float(arguments[6]) == pytest.approx(2.0 * figure3.RE)
        return jnp.ones((len(SCENARIO_NAMES), 3)), jnp.asarray(2.3)

    _, _, depth_ppm, _ = figure3.run_spectral_kernel(
        kernel,
        config,
        SimpleNamespace(pressure=jnp.asarray([1.0])),
        SimpleNamespace(),
        {},
        {},
        np.ones(len(MELTYQ_SPECIES)) / len(MELTYQ_SPECIES),
        2.0,
        benchmark_repeats=0,
    )

    assert depth_ppm.shape == (len(SCENARIO_NAMES), 3)
    assert np.all(np.isfinite(depth_ppm))


def test_diffgrid_broadening_contract_rejects_unacknowledged_fallback():
    build_contract = {
        "teacher": {
            "requested_background_broadener": "H2",
            "missing_requested_broadener_policy": "allow_definition_file_defaults",
        }
    }
    provenance = {
        "schema_version": "meltyq.figure3.broadening-provenance@1",
        "requested_background_broadener": "H2",
        "missing_requested_broadener_policy": "allow_definition_file_defaults",
        "effective_source": "definition_file_defaults_missing_requested_file",
        "requested_file": "12C-16O__H2.broad",
        "requested_file_present": False,
        "requested_file_sha256": None,
        "definition_default_alpha_ref_cm1_bar": 0.07,
        "definition_default_temperature_exponent": 0.5,
        "effective_alpha_ref_range_cm1_bar": [0.07, 0.07],
        "effective_temperature_exponent_range": [0.5, 0.5],
        "selected_line_parameter_count": 4,
    }
    descriptor = {"broadening_provenance": provenance}

    assert (
        _diffgrid_broadening_contract(descriptor, build_contract, "CO")
        is provenance
    )

    provenance["missing_requested_broadener_policy"] = "reject"
    with pytest.raises(ValueError, match="build contract mismatch"):
        _diffgrid_broadening_contract(descriptor, build_contract, "CO")


def test_one_diffgrid_kernel_returns_all_components_in_one_shape():
    wavenumber = np.array([1000.0, 1500.0, 2000.0, 2500.0])
    art = ArtTransPure.from_pressure_boundaries(
        1.0e-6,
        1.0,
        4,
        integration="trapezoid",
        nu_grid=wavenumber,
    )
    calculators = {
        species: ConstantDiffgridOpacity(1.0e-30 * (index + 1), 20.0 + index)
        for index, species in enumerate(ABSORBING_SPECIES)
    }
    bundle = OpacityBundle(
        mode="diffgrid",
        calculators=calculators,
        wavenumber_cm1=wavenumber,
        weights=None,
        provenance={},
    )
    kernel = make_spectral_kernel(
        art,
        bundle,
        cia_opacities={},
        rayleigh_cross_sections={},
        cloud_enabled=True,
        haze_enabled=False,
        haze_vertical_profile="exp_decay",
        radiative_transfer_scheme="taurex_rectangle",
    )
    temperature = jnp.full(4, 500.0)
    prepared = prepare_spectral_opacity(
        bundle,
        {},
        {},
        temperature,
        art.pressure,
    )
    vmr = jnp.asarray([0.8, 0.19, 0.0, 0.003, 0.001, 0.001, 0.003, 0.001, 0.001])
    result, mean_molecular_weight = kernel(
        temperature,
        vmr,
        prepared.molecular_cross_sections,
        prepared.log_cia_coefficients,
        prepared.cia_coverage_masks,
        prepared.rayleigh_cross_sections,
        jnp.asarray(2.0e9),
        jnp.asarray(1.0e3),
        jnp.asarray(1.0e4),
        jnp.asarray(0.1),
        jnp.asarray(50.0),
        jnp.asarray(1.0e8),
        jnp.asarray(100.0),
        jnp.asarray(2.0),
    )
    jax.block_until_ready(result)

    assert result.shape == (len(SCENARIO_NAMES), wavenumber.size)
    assert np.all(np.isfinite(np.asarray(result)))
    assert float(mean_molecular_weight) > 0.0
    assert np.all(np.asarray(result[0]) >= np.asarray(result[1]))
    assert tuple(MELTYQ_SPECIES) == (
        "H2",
        "He",
        "O2",
        "H2O",
        "CO",
        "CO2",
        "CH4",
        "N2",
        "NH3",
    )


def test_diffgrid_temperature_bounds_are_checked_before_jit():
    calculators = {}
    for species in ABSORBING_SPECIES:
        calculator = ConstantDiffgridOpacity(1.0e-30, 20.0)
        calculator.temperature_grid = np.array([1000.0, 100.0])
        calculator.pressure_grid = np.array([1.0e-5, 1.0])
        calculators[species] = calculator
    bundle = OpacityBundle(
        mode="diffgrid",
        calculators=calculators,
        wavenumber_cm1=np.array([1000.0, 2000.0]),
        weights=None,
        provenance={},
    )

    with pytest.raises(ValueError, match="outside"):
        validate_opacity_state(
            bundle,
            pressure_bar=np.array([1.0e-5, 1.0]),
            temperature_k=np.array([50.0, 500.0]),
        )


def test_diffgrid_grid_must_cover_both_configured_endpoints():
    _validate_wavenumber_coverage(
        np.array([800.0, 16000.0]),
        (1.0e4 / 12.0, 1.0e4 / 0.65),
    )

    with pytest.raises(ValueError, match="does not cover"):
        _validate_wavenumber_coverage(
            np.array([1000.0, 10000.0]),
            (1.0e4 / 12.0, 1.0e4 / 0.65),
        )


def test_source_database_inventory_hash_requires_sorted_unique_entries():
    inventory = [
        {"relative_path": "a.def", "size_bytes": 2, "sha256": "0" * 64},
        {"relative_path": "b.trans", "size_bytes": 3, "sha256": "1" * 64},
    ]
    assert len(_source_database_inventory_sha256(inventory)) == 64

    with pytest.raises(ValueError, match="unique and sorted"):
        _source_database_inventory_sha256(inventory[::-1])


def test_ckd_kernel_broadcasts_finite_cloud_and_haze_over_g_grid():
    wavenumber = np.array([1000.0, 1500.0, 2000.0, 2500.0])
    art = ArtTransPure.from_pressure_boundaries(
        1.0e-6,
        1.0,
        4,
        integration="simpson",
        nu_grid=wavenumber,
    )
    bundle = OpacityBundle(
        mode="ckd",
        calculators={
            species: ConstantCkdOpacity(1.0e-30, 20.0)
            for species in ABSORBING_SPECIES
        },
        wavenumber_cm1=wavenumber,
        weights=np.array([0.5, 0.5]),
        provenance={},
    )
    kernel = make_spectral_kernel(
        art,
        bundle,
        cia_opacities={},
        rayleigh_cross_sections={},
        cloud_enabled=True,
        haze_enabled=True,
        haze_vertical_profile="constant",
    )
    temperature = jnp.full(4, 500.0)
    prepared = prepare_spectral_opacity(
        bundle,
        {},
        {},
        temperature,
        art.pressure,
    )
    result, _ = kernel(
        temperature,
        jnp.asarray([0.8, 0.19, 0.0, 0.003, 0.001, 0.001, 0.003, 0.001, 0.001]),
        prepared.molecular_cross_sections,
        prepared.log_cia_coefficients,
        prepared.cia_coverage_masks,
        prepared.rayleigh_cross_sections,
        jnp.asarray(2.0e9),
        jnp.asarray(1.0e3),
        jnp.asarray(1.0e4),
        jnp.asarray(0.1),
        jnp.asarray(50.0),
        jnp.asarray(1.0e8),
        jnp.asarray(100.0),
        jnp.asarray(2.0),
    )
    jax.block_until_ready(result)

    assert result.shape == (len(SCENARIO_NAMES), wavenumber.size)
    assert np.all(np.isfinite(np.asarray(result)))
