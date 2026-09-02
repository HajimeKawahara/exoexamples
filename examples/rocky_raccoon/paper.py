"""Static policy for the Misener et al. (2026) paper benchmark.

This module records only values stated by the paper.  In particular, the
absolute elemental abundances and the numerical opacity and conductivity
closures are intentionally left unset.

Appendix A reports 70 gas species.  ExoGibbs minimizes this network directly;
it does not require the neutral atoms or free electron to be added as gas
reference species.  The ``e-`` formula-matrix row remains a charge constraint
with zero inventory.  The explicit network also excludes FastChem4 catalog
species, including ``Si2`` and ``Si3``, that are absent from the Appendix.
"""

from __future__ import annotations

from dataclasses import dataclass


PAPER_REFERENCE = "Misener et al. (2026), arXiv:2608.24873v1"

FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG = "FastChem4/logK/logK.dat"
FASTCHEM4_CONDENSATE_CATALOG = "FastChem4/logK/logK_condensates.dat"

PAPER_PHYSICAL_ELEMENTS = ("H", "Mg", "Si", "O", "C")
FIGURE_1_PHYSICAL_ELEMENTS = ("H", "Si", "O")
FASTCHEM_CHARGE_ELEMENT = "e-"

# Appendix A, Table 1, mapped to the labels in ExoGibbs' ion-inclusive
# FastChem4 catalog.  An element-only catalog filter must not replace this
# explicit tuple.
PAPER_GAS_SPECIES = (
    "C1H1",
    "C1H1O1",
    "C1H2",
    "C1H2O1",
    "C1H3",
    "C1H4",
    "C1H4O2",
    "C1O1",
    "C1O2",
    "C1Si1",
    "C1Si2",
    "C2",
    "C2H1",
    "C2H2",
    "C2H2O2",
    "C2H2O4",
    "C2H4",
    "C2H4O1",
    "C2H4O3",
    "C2H6O2",
    "C2Si1",
    "C2Si2",
    "C2O1",
    "C3",
    "C3H1",
    "C3O2",
    "C4",
    "C4H6O4",
    "C5",
    "C1+",
    "C1-",
    "C1H1+",
    "C1H1-",
    "C1H1O1+",
    "C1O2-",
    "C2-",
    "H1Mg1",
    "H1Mg1O1",
    "H1O1",
    "H1O2",
    "H1Si1",
    "H2",
    "H2Mg1O2",
    "H2O1",
    "H2O2",
    "H2Si1",
    "H3Si1",
    "H4Si1",
    "Mg1O1",
    "Mg2",
    "O1Si1",
    "O2",
    "O2Si1",
    "O3",
    "H1+",
    "H1-",
    "H1Mg1O1+",
    "H1O1+",
    "H1O1-",
    "H1Si1+",
    "H2+",
    "H2-",
    "H3O1+",
    "Mg1+",
    "O1+",
    "O1-",
    "O2+",
    "O2-",
    "Si1+",
    "Si1-",
)

PAPER_REPORTED_GAS_SPECIES = PAPER_GAS_SPECIES
PAPER_SOLVER_GAS_SPECIES = PAPER_REPORTED_GAS_SPECIES

SIO_CONDENSATE = "SiO(s)"

# Appendix A, Table 2, excluding SiO(s), as done in the canonical paper runs.
PAPER_CANONICAL_CONDENSATES = (
    "C(s)",
    "SiO2(s,l)",
    "Si(s,l)",
    "SiC(s)",
    "MgH2(s)",
    "Mg(OH)2(s)",
    "Mg(s,l)",
    "MgO(s,l)",
    "MgSiO3(s,l)",
    "Mg2SiO4(s,l)",
    "Mg2Si(s,l)",
    "MgCO3(s)",
    "MgC2(s)",
    "Mg2C3(s)",
)

# Preserve the Appendix order when inserting the Figure 5 sensitivity species.
PAPER_CONDENSATES_WITH_SIO = (
    PAPER_CANONICAL_CONDENSATES[:3]
    + (SIO_CONDENSATE,)
    + PAPER_CANONICAL_CONDENSATES[3:]
)

FIGURE_1_CONDENSATES = (
    "SiO2(s,l)",
    "Si(s,l)",
)

PAPER_CORE_MASS_EARTH = 4.0
PAPER_BASE_TEMPERATURE_K = 4000.0
PAPER_EQUILIBRIUM_TEMPERATURE_K = 1000.0
PAPER_HYDROGEN_MASS_FRACTION = 0.03
PAPER_CORE_MASS_RADIUS_EXPONENT = 0.25
PAPER_PRESSURE_FRACTIONAL_DECREMENT = 0.01
PAPER_PRESSURE_RATIO = 0.99
PAPER_TOP_PRESSURE_BAR = 1.0e-3
PAPER_TRANSIT_PRESSURE_BAR = 2.0e-2
PAPER_SHOOTING_FRACTIONAL_TOLERANCE = 1.0e-8
PAPER_VALIDITY_MODE = "paper_extrapolated"

PAPER_DEFAULT_GAMMA = 7.0 / 5.0
PAPER_GAMMA_OVERRIDES = (
    ("H4Si1", 1.3),
    ("H2O1", 4.0 / 3.0),
)


def paper_gamma(species: str) -> float:
    """Return the paper's prescribed ideal-gas heat-capacity ratio."""

    return dict(PAPER_GAMMA_OVERRIDES).get(species, PAPER_DEFAULT_GAMMA)


@dataclass(frozen=True)
class PaperCommonInputs:
    """Physical values shared by Figures 1, 2, and 5."""

    core_mass_earth: float = PAPER_CORE_MASS_EARTH
    base_temperature_k: float = PAPER_BASE_TEMPERATURE_K
    equilibrium_temperature_k: float = PAPER_EQUILIBRIUM_TEMPERATURE_K
    hydrogen_mass_fraction: float = PAPER_HYDROGEN_MASS_FRACTION
    core_mass_radius_exponent: float = PAPER_CORE_MASS_RADIUS_EXPONENT
    pressure_fractional_decrement: float = PAPER_PRESSURE_FRACTIONAL_DECREMENT
    top_pressure_bar: float = PAPER_TOP_PRESSURE_BAR
    transit_pressure_bar: float = PAPER_TRANSIT_PRESSURE_BAR
    shooting_fractional_tolerance: float = PAPER_SHOOTING_FRACTIONAL_TOLERANCE
    absolute_element_abundances: tuple[tuple[str, float], ...] | None = None
    opacity_parameters: tuple[tuple[str, float], ...] | None = None
    conductivity_parameters: tuple[tuple[str, float], ...] | None = None


@dataclass(frozen=True)
class PaperRadiusTargets:
    """Published outer-RCB and 20 mbar radii, in Earth radii."""

    outer_rcb_rearth: float
    transit_rearth: float
    transit_text_rearth: float | None = None


@dataclass(frozen=True)
class PaperFigureCase:
    """One published Figure 1, 2, or 5 chemistry-structure case."""

    identifier: str
    figure: int
    physical_elements: tuple[str, ...]
    prescribed_number_ratio: tuple[tuple[str, int], ...]
    condensate_species: tuple[str, ...]
    include_sio_s: bool
    radius_targets: PaperRadiusTargets
    validity_mode: str = PAPER_VALIDITY_MODE
    absolute_element_abundances: tuple[tuple[str, float], ...] | None = None


PAPER_COMMON_INPUTS = PaperCommonInputs()

FIGURE_1_CASES = (
    PaperFigureCase(
        identifier="figure1_si_o_1_4",
        figure=1,
        physical_elements=FIGURE_1_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Si", 1), ("O", 4)),
        condensate_species=FIGURE_1_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(1.55, 1.89),
    ),
    PaperFigureCase(
        identifier="figure1_si_o_1_2",
        figure=1,
        physical_elements=FIGURE_1_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Si", 1), ("O", 2)),
        condensate_species=FIGURE_1_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(1.62, 2.49),
    ),
    PaperFigureCase(
        identifier="figure1_si_o_1_1",
        figure=1,
        physical_elements=FIGURE_1_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Si", 1), ("O", 1)),
        condensate_species=FIGURE_1_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(
            1.56,
            2.33,
            transit_text_rearth=2.34,
        ),
    ),
)

FIGURE_2_CASES = (
    PaperFigureCase(
        identifier="figure2_mg_si_o_1_1_3",
        figure=2,
        physical_elements=PAPER_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Mg", 1), ("Si", 1), ("O", 3)),
        condensate_species=PAPER_CANONICAL_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(1.63, 2.51),
    ),
    PaperFigureCase(
        identifier="figure2_mg_si_o_1_1_4",
        figure=2,
        physical_elements=PAPER_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Mg", 1), ("Si", 1), ("O", 4)),
        condensate_species=PAPER_CANONICAL_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(1.59, 2.01),
    ),
)

FIGURE_5_CASES = (
    PaperFigureCase(
        identifier="figure5_mg_si_o_1_1_3_sio_off",
        figure=5,
        physical_elements=PAPER_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Mg", 1), ("Si", 1), ("O", 3)),
        condensate_species=PAPER_CANONICAL_CONDENSATES,
        include_sio_s=False,
        radius_targets=PaperRadiusTargets(1.63, 2.51),
    ),
    PaperFigureCase(
        identifier="figure5_mg_si_o_1_1_3_sio_on",
        figure=5,
        physical_elements=PAPER_PHYSICAL_ELEMENTS,
        prescribed_number_ratio=(("Mg", 1), ("Si", 1), ("O", 3)),
        condensate_species=PAPER_CONDENSATES_WITH_SIO,
        include_sio_s=True,
        radius_targets=PaperRadiusTargets(1.63, 2.28),
    ),
)

PAPER_FIGURE_CASES = FIGURE_1_CASES + FIGURE_2_CASES + FIGURE_5_CASES


def paper_case_by_identifier(identifier: str) -> PaperFigureCase:
    """Return one paper case while rejecting unknown or duplicate identifiers."""

    matches = tuple(
        case for case in PAPER_FIGURE_CASES if case.identifier == identifier
    )
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate paper-policy case {identifier!r}.")
    return matches[0]
