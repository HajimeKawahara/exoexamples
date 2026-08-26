"""MELTYQ upper-atmosphere and spectral comparison components.

This module retains only the comparison-specific formulas translated from
public TauREx sources: Rayleigh scattering, Lee haze, the legacy chord path,
and rectangle-rule transit-area integration.  The exact source revisions used
for the published MELTYQ run remain an external provenance input.  General
atmospheric geometry, layer optical depths, and spectral binning are provided
by ExoJAX.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np


LEE_LOG10_PRESSURE_EXTENT = 2.0
"""Full width of the MELTYQ Lee layer in log10 pressure."""

LEE_PRESSURE_POWER = 5.0
"""Pressure exponent in the current TauREx-PyMieScatt default profile."""


TAUREX_LEE_SOURCE = (
    "https://github.com/groningen-exoatmospheres/taurex-pymiescatt/blob/"
    "2973acec3985c2222281062be16a07428c43d621/"
    "taurex_pymiescatt/contributions/pyleemie.py"
)
"""Pinned public source of the current Lee-haze pressure profile."""


TAUREX_LEE_PROFILE_CHANGE = (
    "https://github.com/groningen-exoatmospheres/taurex-pymiescatt/commit/"
    "2973acec3985c2222281062be16a07428c43d621"
)
"""Public commit that replaced the preceding Lee-haze pressure relation."""


TAUREX3_REFERENCE_COMMIT = "7b6e82a86d4675f140e9e59f3d1410a863251c03"
"""Pinned TauREx 3.2.0 source revision used as a translation reference."""


TAUREX3_SOURCE_ROOT = (
    "https://github.com/ucl-exoplanets/taurex3/blob/"
    f"{TAUREX3_REFERENCE_COMMIT}/src/taurex"
)


TAUREX_RAYLEIGH_SOURCE = (
    f"{TAUREX3_SOURCE_ROOT}/util/scattering.py"
)
"""Source of the Rayleigh formulas copied into this comparison."""


TAUREX_TRANSMISSION_SOURCE = (
    f"{TAUREX3_SOURCE_ROOT}/model/transmission.py"
)
"""Source of the legacy path and rectangle-area formulas reproduced here."""


TAUREX_PLANET_SOURCE = (
    f"{TAUREX3_SOURCE_ROOT}/data/planet.py"
)
"""Source of the bottom-up hydrostatic recurrence used for compatibility."""


TAUREX_SIMPLEMODEL_SOURCE = (
    f"{TAUREX3_SOURCE_ROOT}/model/simplemodel.py"
)
"""Source of the ideal-gas density convention used for compatibility."""


TAUREX_CIA_SOURCE = (
    f"{TAUREX3_SOURCE_ROOT}/cia/hitrancia.py"
)
"""Source of the HITRAN CIA interpolation convention used for compatibility."""


TAUREX_RAYLEIGH_SPECIES = (
    "H2",
    "He",
    "N2",
    "O2",
    "CO2",
    "CH4",
    "CO",
    "NH3",
    "H2O",
)
"""Molecules for which TauREx defines a Rayleigh cross section."""


def _taurex_general_rayleigh_cross_section_m2(
    wavenumber_cm1,
    refractive_index,
    king_factor=1.0,
):
    """Evaluate the general TauREx Rayleigh expression in square metres."""

    wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=float)
    wavelength_m = (1.0e4 / wavenumber_cm1) * 1.0e-6
    number_density_air_m3 = 2.6867805e25
    refractive_factor = (refractive_index**2 - 1.0) / (
        number_density_air_m3 * (refractive_index**2 + 2.0)
    )
    return (
        24.0
        * np.pi**3
        * king_factor
        * refractive_factor**2
        / wavelength_m**4
    )


def taurex_rayleigh_cross_section_cm2(species, wavenumber_cm1):
    """Return the TauREx Rayleigh cross section in cm2 per molecule.

    The equations are a direct unit-preserving transcription of TauREx 3's
    Rayleigh contribution.  TauREx returns square metres, whereas ExoJAX's
    ``opacity_profile_xs`` consumes square centimetres.
    """

    wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=float)
    if (
        wavenumber_cm1.ndim != 1
        or wavenumber_cm1.size == 0
        or np.any(~np.isfinite(wavenumber_cm1))
        or np.any(wavenumber_cm1 <= 0.0)
    ):
        raise ValueError("wavenumber_cm1 must be a finite, positive 1D grid.")

    wavelength_angstrom = 1.0e8 / wavenumber_cm1
    wavelength_micron = 1.0e4 / wavenumber_cm1
    if species == "H2":
        cross_section_m2 = (
            8.14e-13
            * wavelength_angstrom ** (-4.0)
            * (
                1.0
                + 1.572e6 * wavelength_angstrom ** (-2.0)
                + 1.981e12 * wavelength_angstrom ** (-4.0)
            )
            * 1.0e-4
        )
    elif species == "He":
        cross_section_m2 = (
            5.484e-14
            * wavelength_angstrom ** (-4.0)
            * (1.0 + 2.44e5 * wavelength_angstrom ** (-2.0))
            * 1.0e-4
        )
    else:
        if species == "N2":
            refractive_index = 1.0 + (
                6498.2
                + 307.43305e12 / (14.4e9 - wavenumber_cm1**2)
            ) * 1.0e-8
            king_factor = 1.034 + 3.17e-12 * wavenumber_cm1**2
        elif species == "O2":
            refractive_index = (
                1.0
                + 1.181494e-4
                + 9.708931e-3
                / (75.4 - wavelength_micron ** (-2.0))
            )
            king_factor = 1.096
        elif species == "CO2":
            inverse_wavelength_squared = wavelength_micron ** (-2.0)
            refractive_index = (
                1.0
                + 6.991e-2 / (166.175 - inverse_wavelength_squared)
                + 1.44720e-3 / (79.609 - inverse_wavelength_squared)
                + 6.42941e-5 / (56.3064 - inverse_wavelength_squared)
                + 5.21306e-5 / (46.0196 - inverse_wavelength_squared)
                + 1.46847e-6 / (0.0584738 - inverse_wavelength_squared)
            )
            king_factor = 1.1364
        elif species == "CH4":
            wavelength_cm = wavelength_micron * 1.0e-4
            refractive_index = 1.0 + 1.0e-8 * (
                46662.0 + 4.02e-6 / wavelength_cm**2
            )
            king_factor = 1.0
        elif species == "CO":
            refractive_index = 1.0 + 32.7e-5 * (
                1.0 + 8.1e-3 / wavelength_micron**2
            )
            king_factor = 1.016
        elif species == "NH3":
            refractive_index = 1.0 + 37.0e-5 * (
                1.0 + 12.0e-3 / wavelength_micron**2
            )
            king_factor = 1.0
        elif species == "H2O":
            inverse_wavelength_squared = wavelength_micron ** (-2.0)
            refractive_index_air = 1.0 + (
                0.05792105 / (238.0185 - inverse_wavelength_squared)
                + 0.00167917 / (57.362 - inverse_wavelength_squared)
            )
            refractive_index = 0.85 * (refractive_index_air - 1.0) + 1.0
            depolarization = 0.17
            king_factor = (6.0 + 3.0 * depolarization) / (
                6.0 - 7.0 * depolarization
            )
        else:
            raise ValueError(f"TauREx has no Rayleigh formula for {species!r}.")
        cross_section_m2 = _taurex_general_rayleigh_cross_section_m2(
            wavenumber_cm1,
            refractive_index,
            king_factor,
        )
    return cross_section_m2 * 1.0e4


def taurex_old_path_rectangle_transit_radius_squared(
    normalized_height_top_to_bottom,
    vertical_optical_depth_top_to_bottom,
):
    """Apply TauREx's legacy chord path and rectangle annulus quadrature.

    Heights are normalized by the radius at the lower boundary.  The first
    axis of both inputs follows ExoJAX's top-to-bottom layer order.  The
    result is the effective squared transit radius divided by that same
    lower-boundary radius squared.

    This reproduces ``TransmissionModel(new_path_method=False)``: TauREx
    constructs chords through layer-centred shells and then integrates
    ``2 (R + z) [1 - exp(-tau)] dz`` with a rectangle rule.
    """

    height_top = jnp.asarray(normalized_height_top_to_bottom)
    vertical_depth_top = jnp.asarray(vertical_optical_depth_top_to_bottom)
    if height_top.ndim != 1:
        raise ValueError("normalized layer heights must be one-dimensional.")
    if vertical_depth_top.ndim != 2:
        raise ValueError("vertical optical depth must have shape (layer, grid).")
    if vertical_depth_top.shape[0] != height_top.size:
        raise ValueError("height and optical-depth layer counts differ.")

    dz = height_top[::-1]
    vertical_depth = vertical_depth_top[::-1]
    altitude = jnp.concatenate(
        (jnp.zeros(1, dtype=dz.dtype), jnp.cumsum(dz[:-1]))
    )
    radius_offset = 1.0 + 0.5 * dz[0]
    tangent_radius = radius_offset + altitude
    layer_midpoint_radius = radius_offset + altitude + 0.5 * dz

    row = jnp.arange(dz.size)[:, None]
    column = jnp.arange(dz.size)[None, :]
    left_argument = (
        layer_midpoint_radius[None, :] ** 2
        - tangent_radius[:, None] ** 2
    )
    previous_midpoint_radius = jnp.concatenate(
        (layer_midpoint_radius[:1], layer_midpoint_radius[:-1])
    )
    right_argument = (
        previous_midpoint_radius[None, :] ** 2
        - tangent_radius[:, None] ** 2
    )
    left = jnp.sqrt(jnp.maximum(left_argument, 0.0))
    right = jnp.sqrt(jnp.maximum(right_argument, 0.0))
    path_length = 2.0 * jnp.where(
        column >= row,
        left - jnp.where(column > row, right, 0.0),
        0.0,
    )
    path_to_vertical_ratio = path_length / dz[None, :]
    chord_optical_depth = path_to_vertical_ratio @ vertical_depth
    atmospheric_area = jnp.sum(
        2.0
        * (1.0 + altitude)[:, None]
        * (1.0 - jnp.exp(-chord_optical_depth))
        * dz[:, None],
        axis=0,
    )
    return 1.0 + atmospheric_area


def upper_temperature_profile_k(
    pressure_bar,
    temperature_base_k,
    temperature_1e4_pa_k,
    temperature_100_pa_k,
    temperature_1e_minus2_pa_k,
    smoothing_window_percent=0.0,
):
    """Return the MELTYQ upper-atmosphere temperature profile.

    The four retrieved anchors are at 10 bar, 10^4 Pa, 100 Pa, and
    10^-2 Pa.  Temperature is linear in log10 pressure between anchors and
    isothermal above 10^-2 Pa.
    """

    pressure_bar = jnp.asarray(pressure_bar)
    log10_pressure_pa = jnp.log10(pressure_bar) + 5.0
    anchor_log10_pressure_pa = jnp.asarray([-2.0, 2.0, 4.0, 6.0])
    anchor_temperature_k = jnp.asarray(
        [
            temperature_1e_minus2_pa_k,
            temperature_100_pa_k,
            temperature_1e4_pa_k,
            temperature_base_k,
        ]
    )
    profile = jnp.interp(
        log10_pressure_pa,
        anchor_log10_pressure_pa,
        anchor_temperature_k,
    )
    window_size = int(pressure_bar.size * (smoothing_window_percent / 100.0))
    if window_size % 2 == 0:
        window_size += 1
    if window_size <= 1:
        return profile
    if window_size > pressure_bar.size:
        raise ValueError("Temperature smoothing window exceeds the layer count.")
    smoothed = jnp.convolve(
        profile,
        jnp.ones(window_size, dtype=profile.dtype) / window_size,
        mode="valid",
    )
    border = (profile.size - smoothed.size) // 2
    return profile.at[border : profile.size - border].set(smoothed)


def transit_depth_ppm(
    normalized_transit_radius_squared,
    radius_base_cm,
    stellar_radius_cm,
):
    """Convert ExoJAX normalized squared radius to transit depth in ppm."""

    return (
        1.0e6
        * jnp.asarray(normalized_transit_radius_squared)
        * (radius_base_cm / stellar_radius_cm) ** 2
    )


def lee_extinction_efficiency(
    wavenumber_cm1,
    particle_radius_micron,
    q0,
):
    """Return the dimensionless Lee et al. (2013) extinction efficiency.

    The size parameter is ``x = 2 pi a / wavelength`` with both ``a`` and
    wavelength expressed in micron.  ``q0`` is the dimensionless empirical
    coefficient called ``Q_LEE`` by MELTYQ.
    """

    wavenumber_cm1 = jnp.asarray(wavenumber_cm1)
    size_parameter = (
        2.0 * jnp.pi * particle_radius_micron * wavenumber_cm1 / 1.0e4
    )
    return 5.0 / (
        q0 * size_parameter ** (-4.0) + size_parameter**0.2
    )


def lee_particle_cross_section_m2(
    wavenumber_cm1,
    particle_radius_micron,
    q0,
):
    """Return the extinction cross section of one Lee particle in m2."""

    radius_m = particle_radius_micron * 1.0e-6
    return (
        jnp.pi
        * radius_m**2
        * lee_extinction_efficiency(
            wavenumber_cm1,
            particle_radius_micron,
            q0,
        )
    )


def lee_pressure_bounds_pa(
    mid_pressure_pa,
    log10_pressure_extent=LEE_LOG10_PRESSURE_EXTENT,
):
    """Return inclusive ``(top, bottom)`` pressure bounds in Pa.

    ``mid_pressure_pa`` is MELTYQ's ``P_LEE``.  The extent is the full width
    around it in log10 pressure.  Hence the MELTYQ value 2 gives
    ``P_top = P_LEE / 10`` and ``P_bottom = 10 P_LEE``.
    """

    half_width_factor = 10.0 ** (0.5 * log10_pressure_extent)
    return (
        mid_pressure_pa / half_width_factor,
        mid_pressure_pa * half_width_factor,
    )


def lee_particle_number_density_m3(
    pressure_pa,
    particle_number_density_m3,
    mid_pressure_pa,
    log10_pressure_extent=LEE_LOG10_PRESSURE_EXTENT,
    vertical_profile: Literal["exp_decay", "constant"] = "exp_decay",
):
    """Return the Lee-particle number-density profile in particles m-3.

    ``exp_decay`` reproduces the public TauREx-PyMieScatt profile pinned by
    :data:`TAUREX_LEE_SOURCE`:

    ``n(P) = X_LEE (P / P_bottom)**5``

    inside the Lee layer and zero outside.  Despite its historical name, the
    implemented relation is a pressure power law.  Therefore ``X_LEE`` is the
    number density at the bottom of the haze layer.  ``constant`` reproduces
    the plugin's alternative vertically uniform profile.
    """

    pressure_pa = jnp.asarray(pressure_pa)
    top_pressure_pa, bottom_pressure_pa = lee_pressure_bounds_pa(
        mid_pressure_pa,
        log10_pressure_extent,
    )
    in_layer = (pressure_pa >= top_pressure_pa) & (
        pressure_pa <= bottom_pressure_pa
    )

    if vertical_profile == "exp_decay":
        density_m3 = particle_number_density_m3 * (
            pressure_pa / bottom_pressure_pa
        ) ** LEE_PRESSURE_POWER
    elif vertical_profile == "constant":
        density_m3 = jnp.broadcast_to(
            jnp.asarray(particle_number_density_m3),
            pressure_pa.shape,
        )
    else:
        raise ValueError(
            "vertical_profile must be 'exp_decay' or 'constant'."
        )

    return jnp.where(in_layer, density_m3, 0.0)


def lee_extinction_coefficient_m_inv(
    pressure_pa,
    wavenumber_cm1,
    particle_radius_micron,
    q0,
    particle_number_density_m3,
    mid_pressure_pa,
    log10_pressure_extent=LEE_LOG10_PRESSURE_EXTENT,
    vertical_profile: Literal["exp_decay", "constant"] = "exp_decay",
):
    """Return the layer-by-wavenumber Lee extinction coefficient in m-1.

    The result is ``alpha_ext(P, nu) = n_particle(P) sigma_ext(nu)``.  Its
    path integral is the dimensionless slant optical depth.
    """

    number_density_m3 = jnp.atleast_1d(
        lee_particle_number_density_m3(
            pressure_pa,
            particle_number_density_m3,
            mid_pressure_pa,
            log10_pressure_extent,
            vertical_profile,
        )
    )
    cross_section_m2 = jnp.atleast_1d(
        lee_particle_cross_section_m2(
            wavenumber_cm1,
            particle_radius_micron,
            q0,
        )
    )
    return number_density_m3[:, None] * cross_section_m2[None, :]


def gray_cloud_pressure_mask(pressure_pa, cloud_top_pressure_pa):
    """Return the TauREx gray-cloud mask, true at ``P >= P_CLOUDS``."""

    return jnp.asarray(pressure_pa) >= cloud_top_pressure_pa


def gray_cloud_layer_optical_depth(
    pressure_pa,
    spectral_grid,
    cloud_top_pressure_pa,
):
    """Return gray layer optical depths with exact TauREx cloud semantics.

    All wavelengths are assigned infinite optical depth at and below the
    cloud top.  Layers at lower pressure remain transparent.  ``spectral_grid``
    is used only to define the spectral dimension.
    """

    pressure_mask = jnp.atleast_1d(
        gray_cloud_pressure_mask(pressure_pa, cloud_top_pressure_pa)
    )
    spectral_grid = jnp.atleast_1d(jnp.asarray(spectral_grid))
    return jnp.where(
        pressure_mask[:, None],
        jnp.inf,
        jnp.zeros((pressure_mask.size, spectral_grid.size)),
    )
