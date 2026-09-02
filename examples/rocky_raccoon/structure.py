"""Chemistry-independent structure integration for Rocky Raccoon benchmarks.

The integrator advances from the envelope base toward lower pressure.  Chemical
equilibrium, density, and the two temperature gradients are injected so this
module does not depend on a particular chemistry or EOS package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np


BAR_PA = 1.0e5
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
PRESSURE_RATIO = 0.99

Transport = Literal["base", "convective", "nonconvective"]
CandidateTransport = Literal["convective", "nonconvective"]


@dataclass(frozen=True)
class ChemistryState:
    """Chemical result required by the structure calculation.

    ``mean_molar_mass_kg_mol`` is the gas mean molar mass in kg mol-1.
    ``remaining_inventory`` is the elemental inventory after condensates at
    this level have rained out.  ``payload`` may retain a provider-owned result,
    such as gas and condensate abundances, without exposing it to this module.
    """

    mean_molar_mass_kg_mol: float
    remaining_inventory: np.ndarray
    payload: Any = None


@dataclass(frozen=True)
class StructureInputs:
    """Fixed boundary conditions for one base-to-top integration.

    Pressure is in bar; temperature is in K; radius is in m; planet mass is in
    kg; and luminosity is in W.  Atmospheric self-gravity is not included.
    """

    pressure_base_bar: float
    pressure_top_bar: float
    temperature_base_k: float
    radius_base_m: float
    planet_mass_kg: float
    luminosity_w: float
    element_inventory: np.ndarray
    pressure_ratio: float = PRESSURE_RATIO
    max_steps: int = 100_000


@dataclass(frozen=True)
class LayerState:
    """Accepted state at one pressure level, using bar and SI units."""

    pressure_bar: float
    temperature_k: float
    radius_m: float
    mass_density_kg_m3: float
    chemistry: ChemistryState
    transport: Transport


@dataclass(frozen=True)
class CandidateState:
    """One trial state used in the Equation (1) stability comparison."""

    transport: CandidateTransport
    temperature_k: float
    temperature_log_gradient: float
    molar_mass_log_gradient: float
    equation_1_gradient: float
    chemistry: ChemistryState


@dataclass(frozen=True)
class StepDecision:
    """Both trial branches and the branch accepted at one pressure step."""

    pressure_bar: float
    convective: CandidateState
    nonconvective: CandidateState
    selected_transport: CandidateTransport

    @property
    def selected(self) -> CandidateState:
        """Return the accepted candidate."""

        if self.selected_transport == "convective":
            return self.convective
        return self.nonconvective


@dataclass(frozen=True)
class StructureProfile:
    """Accepted profile and the decisions made between adjacent levels."""

    inputs: StructureInputs
    layers: tuple[LayerState, ...]
    decisions: tuple[StepDecision, ...]

    @property
    def pressure_bar(self) -> np.ndarray:
        return np.asarray([layer.pressure_bar for layer in self.layers])

    @property
    def temperature_k(self) -> np.ndarray:
        return np.asarray([layer.temperature_k for layer in self.layers])

    @property
    def radius_m(self) -> np.ndarray:
        """Layer radii in m."""

        return np.asarray([layer.radius_m for layer in self.layers])

    @property
    def mass_density_kg_m3(self) -> np.ndarray:
        """Layer mass densities in kg m-3."""

        return np.asarray([layer.mass_density_kg_m3 for layer in self.layers])

    @property
    def mean_molar_mass_kg_mol(self) -> np.ndarray:
        """Layer gas mean molar masses in kg mol-1."""

        return np.asarray(
            [layer.chemistry.mean_molar_mass_kg_mol for layer in self.layers]
        )

    @property
    def shell_mass_kg(self) -> np.ndarray:
        """Shell masses in kg, using each shell's lower-state density."""

        radius = self.radius_m
        density = self.mass_density_kg_m3[:-1]
        return (4.0 * np.pi / 3.0) * density * (
            radius[1:] ** 3 - radius[:-1] ** 3
        )

    @property
    def remaining_inventory(self) -> np.ndarray:
        return np.stack(
            [layer.chemistry.remaining_inventory for layer in self.layers]
        )

    @property
    def transport(self) -> tuple[Transport, ...]:
        return tuple(layer.transport for layer in self.layers)


ChemistryCallback = Callable[
    [float, float, np.ndarray, Optional[ChemistryState]],
    ChemistryState,
]
MassDensityCallback = Callable[[float, float, ChemistryState], float]
TemperatureGradientCallback = Callable[[LayerState, float], float]
AcceptedLayerCallback = Callable[[int, LayerState], None]


@dataclass(frozen=True)
class StructureCallbacks:
    """Provider callbacks used during one integration.

    Chemistry and density callbacks receive temperature in K and pressure in
    bar.  Chemistry also receives the accepted previous state as a numerical
    initialization hint; the base call receives ``None``.  Density must be
    returned in kg m-3.  Gradient callbacks receive luminosity in W and return
    d(ln T)/d(ln P).
    """

    equilibrium: ChemistryCallback
    convective_log_temperature_gradient: TemperatureGradientCallback
    nonconvective_log_temperature_gradient: TemperatureGradientCallback
    mass_density: MassDensityCallback


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _validated_inputs(inputs: StructureInputs) -> np.ndarray:
    pressure_base = _finite_positive("pressure_base_bar", inputs.pressure_base_bar)
    pressure_top = _finite_positive("pressure_top_bar", inputs.pressure_top_bar)
    if pressure_top >= pressure_base:
        raise ValueError("pressure_top_bar must be below pressure_base_bar.")
    _finite_positive("temperature_base_k", inputs.temperature_base_k)
    _finite_positive("radius_base_m", inputs.radius_base_m)
    _finite_positive("planet_mass_kg", inputs.planet_mass_kg)
    _finite_positive("luminosity_w", inputs.luminosity_w)
    pressure_ratio = float(inputs.pressure_ratio)
    if not np.isfinite(pressure_ratio) or not 0.0 < pressure_ratio < 1.0:
        raise ValueError("pressure_ratio must lie strictly between zero and one.")
    if (
        not isinstance(inputs.max_steps, (int, np.integer))
        or isinstance(inputs.max_steps, (bool, np.bool_))
        or inputs.max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer.")
    inventory = np.asarray(inputs.element_inventory, dtype=float)
    if (
        inventory.ndim != 1
        or inventory.size == 0
        or np.any(~np.isfinite(inventory))
        or np.any(inventory < 0.0)
    ):
        raise ValueError(
            "element_inventory must be a finite, non-negative 1D array."
        )
    return np.array(inventory, copy=True)


def _validated_chemistry(
    state: ChemistryState,
    inventory_shape: tuple[int, ...],
) -> ChemistryState:
    if not isinstance(state, ChemistryState):
        raise TypeError("The equilibrium callback must return ChemistryState.")
    mean_molar_mass_kg_mol = _finite_positive(
        "mean_molar_mass_kg_mol", state.mean_molar_mass_kg_mol
    )
    inventory = np.asarray(state.remaining_inventory, dtype=float)
    if inventory.shape != inventory_shape:
        raise ValueError("The returned elemental inventory has an invalid shape.")
    if np.any(~np.isfinite(inventory)) or np.any(inventory < 0.0):
        raise ValueError(
            "The returned elemental inventory must be finite and non-negative."
        )
    inventory = np.array(inventory, copy=True)
    inventory.setflags(write=False)
    return ChemistryState(mean_molar_mass_kg_mol, inventory, state.payload)


def _equilibrium(
    callback: ChemistryCallback,
    temperature_k: float,
    pressure_bar: float,
    incoming_inventory: np.ndarray,
    previous_chemistry: ChemistryState | None,
) -> ChemistryState:
    # Each call receives a private, writable copy.  A provider may use an
    # in-place work array without mutating the competing branch or caller state.
    result = callback(
        temperature_k,
        pressure_bar,
        np.array(incoming_inventory, copy=True),
        previous_chemistry,
    )
    return _validated_chemistry(result, incoming_inventory.shape)


def _mass_density(
    callback: MassDensityCallback,
    pressure_bar: float,
    temperature_k: float,
    chemistry: ChemistryState,
) -> float:
    return _finite_positive(
        "mass density", callback(pressure_bar, temperature_k, chemistry)
    )


def _temperature_gradient(
    callback: TemperatureGradientCallback,
    layer: LayerState,
    luminosity_w: float,
) -> float:
    gradient = float(callback(layer, luminosity_w))
    if not np.isfinite(gradient):
        raise ValueError("Temperature gradients must be finite.")
    return gradient


def _candidate(
    transport: CandidateTransport,
    temperature_log_gradient: float,
    current: LayerState,
    pressure_next_bar: float,
    incoming_inventory: np.ndarray,
    previous_chemistry: ChemistryState,
    equilibrium: ChemistryCallback,
) -> CandidateState:
    # The paper states Delta P = P/100 while stepping upward.  Here Delta P in
    # T_new = T + Delta P (dT/dP) is interpreted as the signed difference
    # P_new - P, and dT/dP is evaluated at the accepted lower state.
    delta_pressure_bar = pressure_next_bar - current.pressure_bar
    temperature_next_k = current.temperature_k + delta_pressure_bar * (
        temperature_log_gradient
        * current.temperature_k
        / current.pressure_bar
    )
    _finite_positive("candidate temperature", temperature_next_k)
    chemistry = _equilibrium(
        equilibrium,
        temperature_next_k,
        pressure_next_bar,
        incoming_inventory,
        previous_chemistry,
    )
    delta_log_pressure = np.log(pressure_next_bar / current.pressure_bar)
    molar_mass_log_gradient = np.log(
        chemistry.mean_molar_mass_kg_mol
        / current.chemistry.mean_molar_mass_kg_mol
    ) / delta_log_pressure
    equation_1_gradient = (
        temperature_log_gradient - molar_mass_log_gradient
    )
    if not np.isfinite(equation_1_gradient):
        raise ValueError("Equation (1) gradients must be finite.")
    return CandidateState(
        transport=transport,
        temperature_k=temperature_next_k,
        temperature_log_gradient=temperature_log_gradient,
        molar_mass_log_gradient=molar_mass_log_gradient,
        equation_1_gradient=equation_1_gradient,
        chemistry=chemistry,
    )


def _radius_step(
    current: LayerState,
    pressure_next_bar: float,
    planet_mass_kg: float,
) -> float:
    # Explicit lower-state Euler step for dP/dr = -rho GM/r^2.  The original
    # paper does not state a centering rule for Equation (8), so this choice is
    # deliberately exposed by a numerical regression test.
    gravity_si = (
        GRAVITATIONAL_CONSTANT_SI * planet_mass_kg / current.radius_m**2
    )
    pressure_drop_pa = (
        current.pressure_bar - pressure_next_bar
    ) * BAR_PA
    radius_next_m = current.radius_m + pressure_drop_pa / (
        current.mass_density_kg_m3 * gravity_si
    )
    return _finite_positive("radius", radius_next_m)


def integrate_structure(
    inputs: StructureInputs,
    callbacks: StructureCallbacks,
    *,
    accepted_layer_callback: AcceptedLayerCallback | None = None,
) -> StructureProfile:
    """Integrate one fixed-base-pressure, fixed-luminosity envelope profile.

    Every full step uses ``P_new = pressure_ratio P``; its default is the
    paper's value of 0.99.  The top pressure is a stopping threshold: the
    returned profile includes the first level at or below it, rather than
    shortening the final step and changing the selected discretization.
    Equation (1) selects convection only for a strict ``<`` comparison;
    equality therefore remains stable and selects the non-convective branch.
    An optional observer is called only after each layer is fully validated
    and committed, including the base layer at index zero.  The observer must
    treat the supplied state as read-only; its exceptions propagate.
    """

    initial_inventory = _validated_inputs(inputs)
    base_chemistry = _equilibrium(
        callbacks.equilibrium,
        inputs.temperature_base_k,
        inputs.pressure_base_bar,
        initial_inventory,
        None,
    )
    base_density = _mass_density(
        callbacks.mass_density,
        inputs.pressure_base_bar,
        inputs.temperature_base_k,
        base_chemistry,
    )
    current = LayerState(
        pressure_bar=float(inputs.pressure_base_bar),
        temperature_k=float(inputs.temperature_base_k),
        radius_m=float(inputs.radius_base_m),
        mass_density_kg_m3=base_density,
        chemistry=base_chemistry,
        transport="base",
    )
    layers = [current]
    decisions = []
    if accepted_layer_callback is not None:
        accepted_layer_callback(0, current)

    while current.pressure_bar > inputs.pressure_top_bar:
        if len(decisions) >= inputs.max_steps:
            raise RuntimeError("Structure integration exceeded max_steps.")
        pressure_next_bar = current.pressure_bar * inputs.pressure_ratio
        incoming_inventory = np.array(
            current.chemistry.remaining_inventory,
            copy=True,
        )
        convective_gradient = _temperature_gradient(
            callbacks.convective_log_temperature_gradient,
            current,
            inputs.luminosity_w,
        )
        nonconvective_gradient = _temperature_gradient(
            callbacks.nonconvective_log_temperature_gradient,
            current,
            inputs.luminosity_w,
        )
        convective = _candidate(
            "convective",
            convective_gradient,
            current,
            pressure_next_bar,
            incoming_inventory,
            current.chemistry,
            callbacks.equilibrium,
        )
        nonconvective = _candidate(
            "nonconvective",
            nonconvective_gradient,
            current,
            pressure_next_bar,
            incoming_inventory,
            current.chemistry,
            callbacks.equilibrium,
        )

        # Equation (1) is a strict inequality.  Do not replace this with <= or
        # an isclose test: exact equality is stable against convection.
        if convective.equation_1_gradient < nonconvective.equation_1_gradient:
            selected_transport: CandidateTransport = "convective"
            selected = convective
        else:
            selected_transport = "nonconvective"
            selected = nonconvective
        radius_next_m = _radius_step(
            current,
            pressure_next_bar,
            inputs.planet_mass_kg,
        )
        density_next = _mass_density(
            callbacks.mass_density,
            pressure_next_bar,
            selected.temperature_k,
            selected.chemistry,
        )
        next_layer = LayerState(
            pressure_bar=pressure_next_bar,
            temperature_k=selected.temperature_k,
            radius_m=radius_next_m,
            mass_density_kg_m3=density_next,
            chemistry=selected.chemistry,
            transport=selected_transport,
        )
        decisions.append(
            StepDecision(
                pressure_bar=pressure_next_bar,
                convective=convective,
                nonconvective=nonconvective,
                selected_transport=selected_transport,
            )
        )
        current = next_layer
        layers.append(current)
        if accepted_layer_callback is not None:
            accepted_layer_callback(len(layers) - 1, current)

    return StructureProfile(inputs, tuple(layers), tuple(decisions))
