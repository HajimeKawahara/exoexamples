import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from exojax.atm.atmprof import hydrostatic_radius_profile_ideal_gas
from exojax.atm.idealgas import number_density
from exojax.opacity import OpaCIA
from exojax.rt.layeropacity import (
    layer_optical_depth_from_cross_section,
    layer_optical_depth_from_extinction,
    layer_optical_depth_from_log_cia,
)

from examples.meltyq.meltyq_spectral_components import (
    gray_cloud_layer_optical_depth,
    gray_cloud_pressure_mask,
    lee_extinction_coefficient_m_inv,
    lee_extinction_efficiency,
    lee_particle_cross_section_m2,
    lee_particle_number_density_m3,
    lee_pressure_bounds_pa,
    taurex_old_path_rectangle_transit_radius_squared,
    taurex_rayleigh_cross_section_cm2,
    transit_depth_ppm,
    upper_temperature_profile_k,
)


def test_lee_efficiency_and_cross_section_follow_definition():
    wavenumber_cm1 = np.array([1000.0, 5000.0])
    radius_micron = 0.1
    q0 = 40.0
    size_parameter = (
        2.0 * np.pi * radius_micron * wavenumber_cm1 / 1.0e4
    )
    expected_efficiency = 5.0 / (
        q0 * size_parameter ** (-4.0) + size_parameter**0.2
    )

    efficiency = np.asarray(
        lee_extinction_efficiency(wavenumber_cm1, radius_micron, q0)
    )
    cross_section_m2 = np.asarray(
        lee_particle_cross_section_m2(wavenumber_cm1, radius_micron, q0)
    )

    np.testing.assert_allclose(efficiency, expected_efficiency, rtol=1e-6)
    np.testing.assert_allclose(
        cross_section_m2,
        np.pi * (radius_micron * 1.0e-6) ** 2 * expected_efficiency,
        rtol=1e-6,
    )


def test_meltyq_pressure_extent_is_centered_in_log10_pressure():
    top_pressure_pa, bottom_pressure_pa = lee_pressure_bounds_pa(
        1.0e5,
        log10_pressure_extent=2.0,
    )

    assert top_pressure_pa == pytest.approx(1.0e4)
    assert bottom_pressure_pa == pytest.approx(1.0e6)


def test_default_profile_matches_pymiescatt_pressure_power_law():
    pressure_pa = np.array([1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])
    x_lee_m3 = 2.0e8
    density_m3 = np.asarray(
        lee_particle_number_density_m3(
            pressure_pa,
            x_lee_m3,
            mid_pressure_pa=1.0e5,
        )
    )

    expected_m3 = np.array(
        [
            0.0,
            x_lee_m3 * 1.0e-10,
            x_lee_m3 * 1.0e-5,
            x_lee_m3,
            0.0,
        ]
    )
    np.testing.assert_allclose(density_m3, expected_m3, rtol=1e-6)


def test_constant_profile_is_inclusive_at_both_pressure_bounds():
    pressure_pa = np.array([1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7])
    density_m3 = np.asarray(
        lee_particle_number_density_m3(
            pressure_pa,
            particle_number_density_m3=3.0,
            mid_pressure_pa=1.0e5,
            vertical_profile="constant",
        )
    )

    np.testing.assert_array_equal(density_m3, [0.0, 3.0, 3.0, 3.0, 0.0])


def test_unknown_vertical_profile_is_rejected():
    with pytest.raises(ValueError, match="vertical_profile"):
        lee_particle_number_density_m3(
            np.array([1.0e5]),
            particle_number_density_m3=1.0,
            mid_pressure_pa=1.0e5,
            vertical_profile="unknown",
        )


def test_extinction_coefficient_has_layer_by_wavenumber_shape():
    pressure_pa = np.array([1.0e3, 1.0e5, 1.0e7])
    wavenumber_cm1 = np.array([1000.0, 2000.0])
    coefficient_m_inv = np.asarray(
        lee_extinction_coefficient_m_inv(
            pressure_pa,
            wavenumber_cm1,
            particle_radius_micron=0.1,
            q0=40.0,
            particle_number_density_m3=1.0e8,
            mid_pressure_pa=1.0e5,
            vertical_profile="constant",
        )
    )

    assert coefficient_m_inv.shape == (3, 2)
    np.testing.assert_array_equal(coefficient_m_inv[[0, 2]], 0.0)
    assert np.all(coefficient_m_inv[1] > 0.0)


def test_gray_cloud_is_opaque_at_and_below_cloud_top():
    pressure_pa = np.array([1.0e2, 1.0e3, 1.0e4])
    mask = np.asarray(gray_cloud_pressure_mask(pressure_pa, 1.0e3))
    layer_optical_depth = np.asarray(
        gray_cloud_layer_optical_depth(
            pressure_pa,
            spectral_grid=np.array([1000.0, 2000.0]),
            cloud_top_pressure_pa=1.0e3,
        )
    )

    np.testing.assert_array_equal(mask, [False, True, True])
    assert layer_optical_depth.shape == (3, 2)
    np.testing.assert_array_equal(layer_optical_depth[0], 0.0)
    assert np.all(np.isinf(layer_optical_depth[1:]))


def test_upper_temperature_profile_interpolates_in_log_pressure():
    pressure_bar = np.array([1.0e-10, 1.0e-7, 1.0e-5, 1.0e-3, 0.1, 10.0])
    temperature_k = np.asarray(
        upper_temperature_profile_k(
            pressure_bar,
            temperature_base_k=1000.0,
            temperature_1e4_pa_k=800.0,
            temperature_100_pa_k=400.0,
            temperature_1e_minus2_pa_k=200.0,
        )
    )

    np.testing.assert_allclose(
        temperature_k,
        [200.0, 200.0, 300.0, 400.0, 800.0, 1000.0],
        atol=5.0e-5,
    )


def test_temperature_smoothing_matches_taurex_npoint_window_rule():
    pressure_bar = np.geomspace(1.0e-10, 10.0, 100)
    unsmoothed = np.asarray(
        upper_temperature_profile_k(
            pressure_bar,
            1000.0,
            800.0,
            400.0,
            200.0,
            smoothing_window_percent=0.0,
        )
    )
    smoothed = np.asarray(
        upper_temperature_profile_k(
            pressure_bar,
            1000.0,
            800.0,
            400.0,
            200.0,
            smoothing_window_percent=10.0,
        )
    )
    expected = unsmoothed.copy()
    expected[5:-5] = np.convolve(unsmoothed, np.ones(11) / 11.0, mode="valid")

    np.testing.assert_allclose(smoothed, expected, rtol=1.0e-6)


def test_taurex_h2_rayleigh_formula_is_converted_to_cm2():
    wavenumber_cm1 = np.array([1000.0, 5000.0, 10000.0])
    wavelength_angstrom = 1.0e8 / wavenumber_cm1
    expected_m2 = (
        8.14e-13
        * wavelength_angstrom ** (-4.0)
        * (
            1.0
            + 1.572e6 * wavelength_angstrom ** (-2.0)
            + 1.981e12 * wavelength_angstrom ** (-4.0)
        )
        * 1.0e-4
    )

    cross_section_cm2 = taurex_rayleigh_cross_section_cm2(
        "H2", wavenumber_cm1
    )

    np.testing.assert_allclose(cross_section_cm2, expected_m2 * 1.0e4)


def test_taurex_rayleigh_supports_every_meltyq_gas():
    wavenumber_cm1 = np.array([1000.0, 3000.0, 10000.0])
    for species in ("H2", "He", "O2", "H2O", "CO", "CO2", "CH4", "N2", "NH3"):
        cross_section = taurex_rayleigh_cross_section_cm2(
            species,
            wavenumber_cm1,
        )
        assert cross_section.shape == wavenumber_cm1.shape
        assert np.all(np.isfinite(cross_section))
        assert np.all(cross_section > 0.0)


def test_taurex_old_path_and_rectangle_area_match_literal_numpy_rule():
    height_top_to_bottom = np.array([0.006, 0.011, 0.017])
    vertical_depth_top_to_bottom = np.array(
        [
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.08],
        ]
    )
    dz = height_top_to_bottom[::-1]
    vertical_depth = vertical_depth_top_to_bottom[::-1]
    altitude = np.concatenate(([0.0], np.cumsum(dz[:-1])))
    path_lengths = []
    for layer in range(dz.size):
        tangent_squared = (1.0 + dz[0] / 2.0 + altitude[layer]) ** 2
        path = np.empty(dz.size - layer)
        path[0] = np.sqrt(
            (
                1.0
                + dz[0] / 2.0
                + altitude[layer]
                + dz[layer] / 2.0
            )
            ** 2
            - tangent_squared
        )
        if path.size > 1:
            path[1:] = np.sqrt(
                (
                    1.0
                    + dz[0] / 2.0
                    + altitude[layer + 1 :]
                    + dz[layer + 1 :] / 2.0
                )
                ** 2
                - tangent_squared
            )
            path[1:] -= np.sqrt(
                (
                    1.0
                    + dz[0] / 2.0
                    + altitude[layer:-1]
                    + dz[layer:-1] / 2.0
                )
                ** 2
                - tangent_squared
            )
        path_lengths.append(2.0 * path)
    chord_depth = np.stack(
        [
            path @ (vertical_depth[layer:] / dz[layer:, None])
            for layer, path in enumerate(path_lengths)
        ]
    )
    expected = 1.0 + np.sum(
        2.0
        * (1.0 + altitude)[:, None]
        * (1.0 - np.exp(-chord_depth))
        * dz[:, None],
        axis=0,
    )

    actual = np.asarray(
        taurex_old_path_rectangle_transit_radius_squared(
            height_top_to_bottom,
            vertical_depth_top_to_bottom,
        )
    )

    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-7)


def test_exojax_hydrostatic_profile_matches_bottom_up_euler_recurrence():
    pressure_boundary_pa = np.geomspace(1.0, 1.0e5, 5)
    pressure_center_pa = np.sqrt(
        pressure_boundary_pa[:-1] * pressure_boundary_pa[1:]
    )
    temperature_k = np.array([240.0, 300.0, 500.0, 800.0])
    mean_molecular_weight_amu = np.full(4, 2.3)
    radius_base_m = 1.5e7
    gravity_base_m_s2 = 12.0

    pressure_level_bottom = pressure_boundary_pa[::-1]
    temperature_bottom = temperature_k[::-1]
    altitude_lower_m = 0.0
    expected_height_bottom = []
    expected_gravity_bottom = []
    for index, layer_temperature in enumerate(temperature_bottom):
        gravity = gravity_base_m_s2 * (
            radius_base_m / (radius_base_m + altitude_lower_m)
        ) ** 2
        scale_height = (
            1.380649e-23
            * layer_temperature
            / (2.3 * 1.66053906660e-27 * gravity)
        )
        layer_height = scale_height * np.log(
            pressure_level_bottom[index]
            / pressure_level_bottom[index + 1]
        )
        expected_height_bottom.append(layer_height)
        expected_gravity_bottom.append(gravity)
        altitude_lower_m += layer_height

    radius_boundary_cm, gravity_boundary_cgs = (
        hydrostatic_radius_profile_ideal_gas(
            pressure_boundary_pa * 1.0e-5,
            temperature_k,
            mean_molecular_weight_amu,
            radius_base_m * 1.0e2,
            gravity_base_m_s2 * 1.0e2,
            hydrostatic_scheme="layer_constant_gravity",
        )
    )
    layer_height_m = (
        radius_boundary_cm[:-1] - radius_boundary_cm[1:]
    ) * 1.0e-2
    gravity_m_s2 = gravity_boundary_cgs[1:] * 1.0e-2
    number_density_m3 = (
        number_density(pressure_center_pa * 1.0e-5, temperature_k) * 1.0e6
    )

    np.testing.assert_allclose(
        layer_height_m,
        np.asarray(expected_height_bottom)[::-1],
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        gravity_m_s2,
        np.asarray(expected_gravity_bottom)[::-1],
        rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        number_density_m3,
        pressure_center_pa / (1.380649e-23 * temperature_k),
        rtol=2.0e-6,
    )


def test_exojax_layer_depth_unit_conversions_match_si_path_integrals():
    cross_section_cm2 = np.array([[2.0e-24, 3.0e-24], [4.0e-24, 5.0e-24]])
    number_density_m3 = np.array([1.0e25, 2.0e25])
    layer_height_m = np.array([100.0, 200.0])
    vmr = np.array([0.1, 0.2])
    molecular_depth = layer_optical_depth_from_cross_section(
        cross_section_cm2,
        vmr * number_density_m3 * 1.0e-6 * layer_height_m * 1.0e2,
    )
    expected_molecular = (
        cross_section_cm2
        * 1.0e-4
        * (vmr * number_density_m3 * layer_height_m)[:, None]
    )
    np.testing.assert_allclose(molecular_depth, expected_molecular, rtol=1.0e-6)

    log10_cia_cm5 = np.log10(np.array([[2.0e-46], [3.0e-46]]))
    cia_depth = layer_optical_depth_from_log_cia(
        log10_cia_cm5,
        vmr * number_density_m3 * 1.0e-6,
        (vmr / 2.0) * number_density_m3 * 1.0e-6,
        layer_height_m * 1.0e2,
    )
    expected_cia = (
        10.0**log10_cia_cm5
        * 1.0e-10
        * (vmr * (vmr / 2.0) * number_density_m3**2 * layer_height_m)[:, None]
    )
    np.testing.assert_allclose(cia_depth, expected_cia, rtol=2.0e-6)


def test_exojax_cia_interpolates_linear_coefficients_on_both_axes():
    native_temperature_k = np.array([200.0, 400.0])
    native_wavenumber_cm1 = np.array([1000.0, 2000.0])
    native_coefficient_cm5 = np.array(
        [
            [1.0e-46, 5.0e-46],
            [9.0e-46, 13.0e-46],
        ]
    )

    class CiaDatabase:
        nucia = native_wavenumber_cm1
        tcia = native_temperature_k
        logac = np.log10(native_coefficient_cm5)

    calculator = OpaCIA(
        CiaDatabase(),
        nu_grid=np.array([500.0, 1500.0, 2500.0]),
        wavenumber_interpolation="interp",
    )
    log_coefficient = calculator.logacia_matrix(np.array([300.0]))
    coefficient = 10.0**log_coefficient

    # At 300 K the native coefficients are [5, 9]e-46.  TauREx then
    # interpolates in coefficient space and holds the edge values constant.
    np.testing.assert_allclose(
        coefficient,
        [[5.0e-46, 7.0e-46, 9.0e-46]],
        rtol=3.0e-6,
    )

    direct_depth = layer_optical_depth_from_log_cia(
        log_coefficient,
        number_density_1=np.array([0.5 * 2.0e19]),
        number_density_2=np.array([0.25 * 2.0e19]),
        path_length=np.array([100.0 * 1.0e2]),
    )
    expected_depth = (
        coefficient * 1.0e-10 * 0.5 * 0.25 * (2.0e25) ** 2 * 100.0
    )
    np.testing.assert_allclose(direct_depth, expected_depth, rtol=3.0e-6)


def test_extinction_to_layer_depth_uses_geometric_height():
    coefficient_m_inv = np.array([[2.0, 3.0], [4.0, 5.0]])
    optical_depth = np.asarray(
        layer_optical_depth_from_extinction(
            coefficient_m_inv * 1.0e-2,
            path_length=np.array([1.0, 2.0]) * 1.0e2,
        )
    )

    np.testing.assert_allclose(optical_depth, [[2.0, 3.0], [8.0, 10.0]])


def test_transit_depth_ppm_uses_squared_radius_ratio():
    depth_ppm = np.asarray(
        transit_depth_ppm(
            np.array([1.0, 1.21]),
            radius_base_cm=1.0,
            stellar_radius_cm=10.0,
        )
    )

    np.testing.assert_allclose(depth_ppm, [10000.0, 12100.0])
