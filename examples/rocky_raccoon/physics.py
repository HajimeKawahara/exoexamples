"""Replaceable thermodynamic and transport physics for a raccoon-like model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from exoeos import IdealGas
from scipy.constants import G, R, Stefan_Boltzmann


BAR_TO_PA = 1.0e5
EARTH_MASS_KG = 5.9722e24
EARTH_RADIUS_M = 6.371e6

ATOMIC_MOLAR_MASS_KG_MOL = {
    "H": 1.00794e-3,
    "C": 12.0107e-3,
    "O": 15.9994e-3,
    "Mg": 24.3050e-3,
    "Si": 28.0855e-3,
    # The electron row is a charge constraint, not a material component.
    "e-": 0.0,
}


@dataclass(frozen=True)
class RaccoonLikeComposition:
    """Explicit basal number ratios for the non-reproduction model."""

    silicon_per_hydrogen: float = 1.0e-2
    magnesium_to_silicon: float = 1.0
    oxygen_to_silicon: float = 3.0
    carbon_per_hydrogen: float = 2.69e-4


@dataclass(frozen=True)
class ConstantTransport:
    """Constant opacity and conductivity closure for Equations (4)--(6)."""

    rosseland_opacity_m2_kg: float = 1.0e-2
    thermal_conductivity_w_m_k: float = 1.0e3

    def __post_init__(self) -> None:
        _positive(
            "rosseland_opacity_m2_kg",
            self.rosseland_opacity_m2_kg,
        )
        _positive(
            "thermal_conductivity_w_m_k",
            self.thermal_conductivity_w_m_k,
        )

    def conductive_opacity_m2_kg(
        self,
        temperature_k: float,
        mass_density_kg_m3: float,
    ) -> float:
        """Return the conductive opacity from paper Equation (6)."""

        temperature = _positive("temperature_k", temperature_k)
        density = _positive("mass_density_kg_m3", mass_density_kg_m3)
        return (
            16.0
            * Stefan_Boltzmann
            * temperature**3
            / (3.0 * density * self.thermal_conductivity_w_m_k)
        )

    def effective_opacity_m2_kg(
        self,
        temperature_k: float,
        mass_density_kg_m3: float,
    ) -> float:
        """Combine radiative and conductive transport harmonically."""

        conductive = self.conductive_opacity_m2_kg(
            temperature_k,
            mass_density_kg_m3,
        )
        return 1.0 / (
            1.0 / self.rosseland_opacity_m2_kg + 1.0 / conductive
        )

    def nonconvective_log_temperature_gradient(
        self,
        *,
        pressure_bar: float,
        temperature_k: float,
        mass_density_kg_m3: float,
        luminosity_w: float,
        planet_mass_kg: float,
    ) -> float:
        """Return the radiative-conductive gradient from paper Equation (4)."""

        pressure_pa = _positive("pressure_bar", pressure_bar) * BAR_TO_PA
        temperature = _positive("temperature_k", temperature_k)
        luminosity = _positive("luminosity_w", luminosity_w)
        mass = _positive("planet_mass_kg", planet_mass_kg)
        opacity = self.effective_opacity_m2_kg(
            temperature,
            mass_density_kg_m3,
        )
        return (
            3.0
            * opacity
            * pressure_pa
            * luminosity
            / (
                64.0
                * np.pi
                * G
                * mass
                * Stefan_Boltzmann
                * temperature**4
            )
        )


@dataclass(frozen=True)
class IdealGasMixture:
    """ExoEOS ideal-gas state with chemistry-setup species ordering."""

    species: tuple[str, ...]
    formula_matrix: np.ndarray
    elements: tuple[str, ...]
    molar_masses_kg_mol: np.ndarray
    hydrogen_atoms: np.ndarray
    eos: IdealGas

    @classmethod
    def from_chemical_setup(
        cls,
        setup,
        gamma: Callable[[str], float],
    ) -> "IdealGasMixture":
        """Build one ideal-gas provider from an ExoGibbs setup."""

        elements = tuple(setup.elements)
        species = tuple(setup.gas_species)
        missing_masses = tuple(
            name for name in elements if name not in ATOMIC_MOLAR_MASS_KG_MOL
        )
        if missing_masses:
            raise ValueError(
                "No atomic molar masses are configured for elements "
                f"{missing_masses!r}."
            )
        formula_matrix = np.asarray(setup.formula_matrix, dtype=float)
        if formula_matrix.shape != (len(elements), len(species)):
            raise ValueError("The gas formula matrix has an invalid shape.")

        element_masses = np.asarray(
            [ATOMIC_MOLAR_MASS_KG_MOL[name] for name in elements]
        )
        molar_masses = element_masses @ formula_matrix
        if np.any(~np.isfinite(molar_masses)) or np.any(molar_masses <= 0.0):
            raise ValueError("Every gas species must have positive molar mass.")

        gammas = np.asarray([float(gamma(name)) for name in species])
        if np.any(~np.isfinite(gammas)) or np.any(gammas <= 1.0):
            raise ValueError("Every heat-capacity ratio must exceed one.")
        molar_heat_capacities = R * gammas / (gammas - 1.0)
        hydrogen_atoms = (
            formula_matrix[elements.index("H")]
            if "H" in elements
            else np.zeros(len(species))
        )
        return cls(
            species=species,
            formula_matrix=formula_matrix,
            elements=elements,
            molar_masses_kg_mol=molar_masses,
            hydrogen_atoms=np.asarray(hydrogen_atoms, dtype=float),
            eos=IdealGas(
                molar_masses=molar_masses,
                molar_heat_capacities=molar_heat_capacities,
            ),
        )

    def validated_mole_fractions(self, mole_fractions) -> np.ndarray:
        """Return a finite, normalized composition in setup species order."""

        fractions = np.asarray(mole_fractions, dtype=float)
        if fractions.shape != (len(self.species),):
            raise ValueError("mole_fractions have an invalid shape.")
        if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
            raise ValueError("mole_fractions must be finite and non-negative.")
        total = float(np.sum(fractions))
        if not np.isclose(total, 1.0, rtol=1.0e-7, atol=1.0e-10):
            raise ValueError("mole_fractions must sum to one.")
        return fractions

    def state(self, temperature_k: float, pressure_bar: float, mole_fractions):
        """Return an ExoEOS state in SI units."""

        fractions = self.validated_mole_fractions(mole_fractions)
        return self.eos.state(
            _positive("temperature_k", temperature_k),
            _positive("pressure_bar", pressure_bar) * BAR_TO_PA,
            fractions,
        )

    def mean_molar_mass_kg_mol(self, mole_fractions) -> float:
        fractions = self.validated_mole_fractions(mole_fractions)
        return float(fractions @ self.molar_masses_kg_mol)

    def hydrogen_mass_fraction(self, mole_fractions) -> float:
        """Return the fraction of gas mass carried by hydrogen nuclei."""

        fractions = self.validated_mole_fractions(mole_fractions)
        mean_mass = float(fractions @ self.molar_masses_kg_mol)
        if "H" not in self.elements:
            return 0.0
        hydrogen_mass = float(
            fractions
            @ (
                self.hydrogen_atoms
                * ATOMIC_MOLAR_MASS_KG_MOL["H"]
            )
        )
        return hydrogen_mass / mean_mass


def elemental_inventory(
    element_names: Sequence[str],
    composition: RaccoonLikeComposition,
) -> np.ndarray:
    """Return a normalized elemental inventory in the requested order."""

    silicon = _nonnegative(
        "silicon_per_hydrogen",
        composition.silicon_per_hydrogen,
    )
    mapping: Mapping[str, float] = {
        "H": 1.0,
        "Si": silicon,
        "Mg": silicon
        * _nonnegative("magnesium_to_silicon", composition.magnesium_to_silicon),
        "O": silicon
        * _nonnegative("oxygen_to_silicon", composition.oxygen_to_silicon),
        "C": _nonnegative(
            "carbon_per_hydrogen",
            composition.carbon_per_hydrogen,
        ),
        "e-": 0.0,
    }
    unknown = tuple(name for name in element_names if name not in mapping)
    if unknown:
        raise ValueError(f"Unsupported inventory elements: {unknown!r}.")
    inventory = np.asarray([mapping[name] for name in element_names], dtype=float)
    physical_total = float(
        np.sum(
            [value for name, value in zip(element_names, inventory) if name != "e-"]
        )
    )
    if physical_total <= 0.0:
        raise ValueError("The physical elemental inventory must be positive.")
    return inventory / physical_total


def rocky_core_radius_m(
    planet_mass_kg: float,
    exponent: float = 0.25,
) -> float:
    """Return the paper-inspired rocky lower-boundary radius."""

    mass = _positive("planet_mass_kg", planet_mass_kg)
    exponent_value = float(exponent)
    if not np.isfinite(exponent_value):
        raise ValueError("exponent must be finite.")
    return EARTH_RADIUS_M * (mass / EARTH_MASS_KG) ** exponent_value


def _positive(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


__all__ = (
    "ATOMIC_MOLAR_MASS_KG_MOL",
    "BAR_TO_PA",
    "ConstantTransport",
    "EARTH_MASS_KG",
    "EARTH_RADIUS_M",
    "IdealGasMixture",
    "RaccoonLikeComposition",
    "elemental_inventory",
    "rocky_core_radius_m",
)
