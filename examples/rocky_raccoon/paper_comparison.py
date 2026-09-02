"""Compare completed Rocky Raccoon-like runs with published profile targets.

This postprocessor does not rerun the structure model and does not claim a
paper reproduction.  It renders the saved gas, condensate, and temperature
profiles in the layout of Misener et al. (2026), compares temperature with
the published vector trace, and compares the stated outer-RCB and transit
radii.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(REPOSITORY_ROOT / ".cache/matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


if __package__:
    from .model import RaccoonLikeConfig, build_context
    from .paper import (
        PAPER_COMMON_INPUTS,
        PAPER_REFERENCE,
        PaperFigureCase,
        paper_case_by_identifier,
    )
    from .physics import EARTH_MASS_KG
    from .raccoon_like_forward import CLAIM_STATUS
else:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    from examples.rocky_raccoon.model import RaccoonLikeConfig, build_context
    from examples.rocky_raccoon.paper import (
        PAPER_COMMON_INPUTS,
        PAPER_REFERENCE,
        PaperFigureCase,
        paper_case_by_identifier,
    )
    from examples.rocky_raccoon.physics import EARTH_MASS_KG
    from examples.rocky_raccoon.raccoon_like_forward import CLAIM_STATUS


COMPARISON_SCHEMA = "rocky_raccoon.paper_comparison@1"
COMPARISON_CLAIM_STATUS = (
    "fixed_boundary_vs_published_vector_tp_and_scalar_targets_not_reproduction"
)
BOLTZMANN_CONSTANT_J_K = 1.380649e-23
# The saved amount gauge is reconstructed from one elemental row, so this
# postprocessing audit allows accumulated double-precision solver residuals.
CLOSURE_RELATIVE_TOLERANCE = 1.0e-9
CLOSURE_ABSOLUTE_TOLERANCE = 1.0e-12
OUTPUT_FIGURE_NAME = "paper_comparison.png"
OUTPUT_REPORT_NAME = "paper_comparison.json"
DEFAULT_TEMPERATURE_REFERENCE = (
    REPOSITORY_ROOT
    / "docs/rocky_raccoon/data/rocky_raccoon_temperature_vector_reference.csv"
)
TEMPERATURE_COMPARISON_GRID_SIZE = 512

# Species shown in the corresponding published panels.  Neutral atomic H is
# absent from the explicit Appendix-A network used by this example and is
# consequently not included here.
FIGURE_1_GASES = (
    ("H2", r"H$_2$"),
    ("O2", r"O$_2$"),
    ("H1O1", "OH"),
    ("H2Si1", r"SiH$_2$"),
    ("O1Si1", "SiO"),
    ("H2O1", r"H$_2$O"),
    ("H3Si1", r"SiH$_3$"),
    ("H4Si1", r"SiH$_4$"),
)
FIGURE_2_AND_5_GASES = (
    ("C1H3", r"CH$_3$"),
    ("C2H2", r"C$_2$H$_2$"),
    ("H2", r"H$_2$"),
    ("H3Si1", r"SiH$_3$"),
    ("C1H4", r"CH$_4$"),
    ("C2H4", r"C$_2$H$_4$"),
    ("H2Mg1O2", r"Mg(OH)$_2$"),
    ("H4Si1", r"SiH$_4$"),
    ("C1O1", "CO"),
    ("H2O1", r"H$_2$O"),
    ("O1Si1", "SiO"),
    ("C1O2", r"CO$_2$"),
)
FIGURE_1_CONDENSATES = (
    ("Si(s,l)", "Si(s,l)"),
    ("SiO2(s,l)", r"SiO$_2$(s,l)"),
)
FIGURE_2_AND_5_CONDENSATES = (
    ("C(s)", "C(s)"),
    ("Mg2SiO4(s,l)", r"Mg$_2$SiO$_4$(s,l)"),
    ("MgO(s,l)", "MgO(s,l)"),
    ("MgSiO3(s,l)", r"MgSiO$_3$(s,l)"),
    ("SiC(s)", "SiC(s)"),
    ("SiO2(s,l)", r"SiO$_2$(s,l)"),
    ("SiO(s)", "SiO(s)"),
)


@dataclass(frozen=True)
class CondensateDensityConversion:
    """Physical condensate density and the amount-gauge closure audit."""

    number_density_cm3: np.ndarray
    gas_amount_total: np.ndarray
    maximum_absolute_closure_residual: float
    maximum_tolerance_ratio: float


@dataclass(frozen=True)
class CompletedRun:
    """Validated saved data for one completed fixed-boundary column."""

    directory: Path
    summary: Mapping[str, Any]
    case: PaperFigureCase
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    transport: tuple[str, ...]
    gas_species: tuple[str, ...]
    gas_mixing_ratio: np.ndarray
    condensate_species: tuple[str, ...]
    condensate_number_density_cm3: np.ndarray
    density_conversion: CondensateDensityConversion


@dataclass(frozen=True)
class PublishedTemperatureReference:
    """One temperature curve measured from the published vector artwork."""

    case_identifier: str
    reference_contract: str
    pressure_bar: np.ndarray
    temperature_k: np.ndarray
    segment_index: np.ndarray
    transport: tuple[str, ...]


def load_temperature_references(
    path: Path = DEFAULT_TEMPERATURE_REFERENCE,
) -> dict[str, PublishedTemperatureReference]:
    """Load validated, case-keyed published temperature vector traces."""

    source = Path(path)
    try:
        with source.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise ValueError(f"Could not read temperature reference {source}.") from error
    if not rows:
        raise ValueError(f"Temperature reference {source} contains no rows.")
    required = {
        "reference_contract",
        "case_id",
        "quantity",
        "segment_index",
        "transport_regime",
        "pressure_bar",
        "temperature_k",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"Temperature reference is missing columns: {sorted(missing)!r}."
        )
    case_ids = tuple(dict.fromkeys(row["case_id"] for row in rows))
    references: dict[str, PublishedTemperatureReference] = {}
    for case_id in case_ids:
        case_rows = [row for row in rows if row["case_id"] == case_id]
        paper_case_by_identifier(case_id)
        contracts = {row["reference_contract"] for row in case_rows}
        quantities = {row["quantity"] for row in case_rows}
        if contracts != {"published_vector_plot_digitization"}:
            raise ValueError(
                f"Case {case_id!r} has an unsupported reference contract."
            )
        if quantities != {"temperature"}:
            raise ValueError(f"Case {case_id!r} is not a temperature reference.")
        try:
            pressure = np.asarray(
                [float(row["pressure_bar"]) for row in case_rows], dtype=float
            )
            temperature = np.asarray(
                [float(row["temperature_k"]) for row in case_rows], dtype=float
            )
            segments = np.asarray(
                [int(row["segment_index"]) for row in case_rows], dtype=int
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Case {case_id!r} contains invalid numerical reference data."
            ) from error
        transport = tuple(row["transport_regime"] for row in case_rows)
        if (
            np.any(~np.isfinite(pressure))
            or np.any(~np.isfinite(temperature))
            or np.any(pressure <= 0.0)
            or np.any(temperature <= 0.0)
        ):
            raise ValueError(
                f"Case {case_id!r} has non-finite or non-positive coordinates."
            )
        if not set(transport) <= {"convective", "non_convective"}:
            raise ValueError(f"Case {case_id!r} has an unknown transport regime.")
        for segment in np.unique(segments):
            selected = pressure[segments == segment]
            if selected.size < 2 or not (
                np.all(np.diff(selected) > 0.0)
                or np.all(np.diff(selected) < 0.0)
            ):
                raise ValueError(
                    f"Case {case_id!r}, segment {segment} is not monotone."
                )
        references[case_id] = PublishedTemperatureReference(
            case_identifier=case_id,
            reference_contract=contracts.pop(),
            pressure_bar=pressure,
            temperature_k=temperature,
            segment_index=segments,
            transport=transport,
        )
    return references


def _as_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _named_matrix(
    rows: Sequence[Mapping[str, str]],
    names: Sequence[str],
    field_template: str,
) -> np.ndarray:
    fields = tuple(field_template.format(name=name) for name in names)
    missing = tuple(field for field in fields if field not in rows[0])
    if missing:
        raise ValueError(f"profiles.csv is missing columns: {missing!r}.")
    try:
        values = np.asarray(
            [[float(row[field]) for field in fields] for row in rows],
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"profiles.csv contains a non-numeric {field_template!r} value."
        ) from error
    if np.any(~np.isfinite(values)):
        raise ValueError(
            f"profiles.csv contains a non-finite {field_template!r} value."
        )
    return values


def recover_condensate_number_density_cm3(
    *,
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
    gas_mixing_ratio: np.ndarray,
    condensate_amounts: np.ndarray,
    incoming_inventory: np.ndarray,
    gas_formula_matrix: np.ndarray,
    condensate_formula_matrix: np.ndarray,
    element_names: Sequence[str],
    relative_tolerance: float = CLOSURE_RELATIVE_TOLERANCE,
    absolute_tolerance: float = CLOSURE_ABSOLUTE_TOLERANCE,
) -> CondensateDensityConversion:
    """Recover physical condensate densities from ExoGibbs' amount gauge.

    The H conservation row fixes the total gas amount ``n_g`` in the saved
    normalized gauge.  Every elemental row is then audited against

    ``b = n_g A_g x + A_c m``.

    Only after this audit passes is ``m / n_g`` converted using the ideal-gas
    number density ``P / (k_B T)``.
    """

    pressure = np.asarray(pressure_bar, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    gas_x = np.asarray(gas_mixing_ratio, dtype=float)
    condensates = np.asarray(condensate_amounts, dtype=float)
    inventory = np.asarray(incoming_inventory, dtype=float)
    gas_formula = np.asarray(gas_formula_matrix, dtype=float)
    condensate_formula = np.asarray(condensate_formula_matrix, dtype=float)
    elements = tuple(str(name) for name in element_names)
    layer_count = pressure.size

    if pressure.shape != (layer_count,) or temperature.shape != (layer_count,):
        raise ValueError("Pressure and temperature must be one-dimensional.")
    if gas_x.shape != (layer_count, gas_formula.shape[1]):
        raise ValueError("Gas mixing ratios do not match the gas formula matrix.")
    if condensates.shape != (layer_count, condensate_formula.shape[1]):
        raise ValueError(
            "Condensate amounts do not match the condensate formula matrix."
        )
    if gas_formula.shape[0] != len(elements):
        raise ValueError("Gas formula rows do not match the element names.")
    if condensate_formula.shape[0] != len(elements):
        raise ValueError("Condensate formula rows do not match the element names.")
    if inventory.shape != (layer_count, len(elements)):
        raise ValueError("Incoming inventories do not match the element names.")
    if "H" not in elements:
        raise ValueError("The H conservation row is required for amount recovery.")
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be finite and positive.")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("absolute_tolerance must be finite and positive.")
    for label, values in (
        ("pressure", pressure),
        ("temperature", temperature),
        ("gas mixing ratio", gas_x),
        ("condensate amount", condensates),
        ("incoming inventory", inventory),
        ("gas formula matrix", gas_formula),
        ("condensate formula matrix", condensate_formula),
    ):
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{label} contains a non-finite value.")
    if np.any(pressure <= 0.0) or np.any(temperature <= 0.0):
        raise ValueError("Pressure and temperature must be positive.")
    if np.any(gas_x < 0.0) or np.any(condensates < 0.0):
        raise ValueError(
            "Gas mixing ratios and condensate amounts must be non-negative."
        )

    gas_inventory_per_total = gas_x @ gas_formula.T
    condensate_inventory = condensates @ condensate_formula.T
    hydrogen_index = elements.index("H")
    hydrogen_per_total = gas_inventory_per_total[:, hydrogen_index]
    hydrogen_available = (
        inventory[:, hydrogen_index]
        - condensate_inventory[:, hydrogen_index]
    )
    if np.any(hydrogen_per_total <= 0.0):
        layer = int(np.flatnonzero(hydrogen_per_total <= 0.0)[0])
        raise ValueError(
            "Cannot recover gas amount from a non-positive H coefficient at "
            f"layer {layer}."
        )
    gas_amount_total = hydrogen_available / hydrogen_per_total
    if np.any(~np.isfinite(gas_amount_total)) or np.any(gas_amount_total <= 0.0):
        layer = int(
            np.flatnonzero(
                (~np.isfinite(gas_amount_total)) | (gas_amount_total <= 0.0)
            )[0]
        )
        raise ValueError(
            "Recovered gas amount is not finite and positive at "
            f"layer {layer}."
        )

    reconstructed = (
        gas_amount_total[:, None] * gas_inventory_per_total
        + condensate_inventory
    )
    residual = reconstructed - inventory
    tolerance = absolute_tolerance + relative_tolerance * np.abs(inventory)
    tolerance_ratio = np.abs(residual) / tolerance
    if np.any(tolerance_ratio > 1.0):
        layer, element_index = np.unravel_index(
            int(np.argmax(tolerance_ratio)), tolerance_ratio.shape
        )
        raise ValueError(
            "Amount-gauge closure failed at layer "
            f"{layer}, element {elements[element_index]!r}: residual="
            f"{residual[layer, element_index]:.6e}, tolerance="
            f"{tolerance[layer, element_index]:.6e}."
        )

    physical_gas_density_cm3 = (
        pressure * 1.0e5 / (BOLTZMANN_CONSTANT_J_K * temperature) / 1.0e6
    )
    number_density = (
        condensates
        / gas_amount_total[:, None]
        * physical_gas_density_cm3[:, None]
    )
    if np.any(~np.isfinite(number_density)) or np.any(number_density < 0.0):
        raise ValueError("Recovered condensate number density is invalid.")
    return CondensateDensityConversion(
        number_density_cm3=number_density,
        gas_amount_total=gas_amount_total,
        maximum_absolute_closure_residual=float(np.max(np.abs(residual))),
        maximum_tolerance_ratio=float(np.max(tolerance_ratio)),
    )


def _setup_for_summary(summary: Mapping[str, Any]):
    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("summary.json does not contain a config object.")
    preset = config.get("preset")
    validity_mode = config.get("validity_mode")
    if not isinstance(preset, str) or not isinstance(validity_mode, str):
        raise ValueError("summary.json does not identify preset and validity mode.")
    context = build_context(
        RaccoonLikeConfig(preset=preset, validity_mode=validity_mode)
    )
    if context.case.identifier != summary.get("case_identifier"):
        raise ValueError(
            "The saved case identifier does not match its configured preset."
        )
    return context.chemistry_setup


def _validate_paper_comparison_inputs(
    summary: Mapping[str, Any],
    case: PaperFigureCase,
) -> None:
    """Require the saved inputs that are explicitly fixed by the paper."""

    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("summary.json does not contain a config object.")
    required = {
        "core_mass_earth": PAPER_COMMON_INPUTS.core_mass_earth,
        "temperature_base_k": PAPER_COMMON_INPUTS.base_temperature_k,
        "pressure_top_bar": PAPER_COMMON_INPUTS.top_pressure_bar,
        "transit_pressure_bar": PAPER_COMMON_INPUTS.transit_pressure_bar,
        "pressure_ratio": 1.0
        - PAPER_COMMON_INPUTS.pressure_fractional_decrement,
    }
    for name, paper_value in required.items():
        try:
            model_value = float(config[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"summary.json does not contain a valid {name}."
            ) from error
        if not math.isfinite(model_value) or not np.isclose(
            model_value,
            paper_value,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(
                f"Saved {name}={model_value!r} does not match the paper "
                f"comparison value {paper_value!r}."
            )

    composition = summary.get("effective_composition")
    if not isinstance(composition, Mapping):
        raise ValueError(
            "summary.json does not contain effective composition metadata."
        )
    ratio = dict(case.prescribed_number_ratio)
    silicon = float(ratio["Si"])
    expected_ratios = {
        "oxygen_to_silicon": float(ratio["O"]) / silicon,
    }
    if "Mg" in ratio:
        expected_ratios["magnesium_to_silicon"] = (
            float(ratio["Mg"]) / silicon
        )
    for name, paper_value in expected_ratios.items():
        try:
            model_value = float(composition[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"summary.json does not contain a valid effective {name}."
            ) from error
        if not math.isfinite(model_value) or not np.isclose(
            model_value,
            paper_value,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise ValueError(
                f"Effective {name}={model_value!r} does not match case "
                f"{case.identifier!r} ({paper_value!r})."
            )


def load_completed_run(
    run_directory: Path,
    *,
    chemistry_setup=None,
) -> CompletedRun:
    """Load and validate one completed forward-run directory."""

    directory = Path(run_directory).resolve()
    status = _as_json_object(directory / "run_status.json")
    summary = _as_json_object(directory / "summary.json")
    if status.get("status") != "completed":
        raise ValueError(
            f"{directory} is not a completed run (status={status.get('status')!r})."
        )
    if status.get("claim_status") != CLAIM_STATUS:
        raise ValueError(f"{directory} has an unsupported run claim status.")
    if summary.get("claim_status") != CLAIM_STATUS:
        raise ValueError(f"{directory} has an unsupported summary claim status.")
    if status.get("config") != summary.get("config"):
        raise ValueError("run_status.json and summary.json configs differ.")
    diagnostics = summary.get("solver_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("summary.json does not contain solver diagnostics.")
    if diagnostics.get("mode") != "fixed_boundary":
        raise ValueError("Only a fixed-boundary run can enter this comparison.")
    if diagnostics.get("all_equilibria_converged") is not True:
        raise ValueError("The saved run does not report converged equilibria.")

    case_identifier = summary.get("case_identifier")
    if not isinstance(case_identifier, str):
        raise ValueError("summary.json does not contain a case identifier.")
    case = paper_case_by_identifier(case_identifier)
    _validate_paper_comparison_inputs(summary, case)
    species = summary.get("species")
    if not isinstance(species, Mapping):
        raise ValueError("summary.json does not contain species metadata.")
    gas_species = tuple(species.get("gas", ()))
    condensate_species = tuple(species.get("condensates", ()))
    element_names = tuple(species.get("elements", ()))
    setup = (
        chemistry_setup
        if chemistry_setup is not None
        else _setup_for_summary(summary)
    )
    if gas_species != tuple(setup.gas_species):
        raise ValueError("Saved gas species do not match the chemical setup.")
    if condensate_species != tuple(setup.condensate_species):
        raise ValueError("Saved condensates do not match the chemical setup.")
    if element_names != tuple(setup.elements):
        raise ValueError("Saved elements do not match the chemical setup.")

    profile_path = directory / "profiles.csv"
    try:
        with profile_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as error:
        raise ValueError(f"Could not read {profile_path}.") from error
    if not rows:
        raise ValueError("profiles.csv contains no accepted layers.")
    accepted_count = diagnostics.get("accepted_layer_count")
    if accepted_count != len(rows):
        raise ValueError(
            "profiles.csv row count does not match the accepted-layer count."
        )
    try:
        pressure = np.asarray([float(row["pressure_bar"]) for row in rows])
        temperature = np.asarray([float(row["temperature_k"]) for row in rows])
        transport = tuple(row["transport"] for row in rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "profiles.csv has invalid profile coordinate columns."
        ) from error
    if (
        np.any(~np.isfinite(pressure))
        or np.any(~np.isfinite(temperature))
        or np.any(pressure <= 0.0)
        or np.any(temperature <= 0.0)
    ):
        raise ValueError(
            "Profile pressure and temperature must be finite and positive."
        )
    if np.any(np.diff(pressure) >= 0.0):
        raise ValueError("Profile pressure must decrease strictly upward.")
    if not set(transport) <= {"base", "convective", "nonconvective"}:
        raise ValueError("profiles.csv contains an unknown transport branch.")

    gas_x = _named_matrix(rows, gas_species, "gas_x[{name}]")
    if np.any(gas_x < 0.0) or not np.allclose(
        np.sum(gas_x, axis=1), 1.0, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("Saved gas mixing ratios are not normalized.")
    condensate_amounts = _named_matrix(
        rows, condensate_species, "condensate_amount[{name}]"
    )
    incoming_inventory = _named_matrix(
        rows, element_names, "normalized_inventory_in[{name}]"
    )
    conversion = recover_condensate_number_density_cm3(
        pressure_bar=pressure,
        temperature_k=temperature,
        gas_mixing_ratio=gas_x,
        condensate_amounts=condensate_amounts,
        incoming_inventory=incoming_inventory,
        gas_formula_matrix=np.asarray(setup.formula_matrix, dtype=float),
        condensate_formula_matrix=np.asarray(
            setup.formula_matrix_cond, dtype=float
        ),
        element_names=element_names,
    )
    return CompletedRun(
        directory=directory,
        summary=summary,
        case=case,
        pressure_bar=pressure,
        temperature_k=temperature,
        transport=transport,
        gas_species=gas_species,
        gas_mixing_ratio=gas_x,
        condensate_species=condensate_species,
        condensate_number_density_cm3=conversion.number_density_cm3,
        density_conversion=conversion,
    )


def _finite_metric(summary: Mapping[str, Any], name: str) -> float:
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("summary.json does not contain metrics.")
    value = metrics.get(name)
    if isinstance(value, bool):
        raise ValueError(f"Metric {name!r} is not numerical.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Metric {name!r} is not numerical.") from error
    if not math.isfinite(result):
        raise ValueError(f"Metric {name!r} is not finite.")
    return result


def _hydrogen_mass_fraction_comparison(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the paper f_H definition with its saved-run approximation."""

    config = summary.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("summary.json does not contain a config object.")
    try:
        core_mass_kg = float(config["core_mass_earth"]) * EARTH_MASS_KG
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("summary.json does not contain a valid core mass.") from error
    envelope_mass_kg = _finite_metric(summary, "envelope_mass_kg")
    hydrogen_mass_kg = _finite_metric(summary, "hydrogen_mass_kg")
    total_mass_kg = core_mass_kg + envelope_mass_kg
    if core_mass_kg <= 0.0 or envelope_mass_kg <= 0.0 or not (
        0.0 <= hydrogen_mass_kg <= envelope_mass_kg
    ):
        raise ValueError("Saved masses cannot define a hydrogen mass fraction.")
    model_fraction = hydrogen_mass_kg / total_mass_kg
    paper_fraction = PAPER_COMMON_INPUTS.hydrogen_mass_fraction
    return {
        "paper_hydrogen_mass_over_total_planet_mass": paper_fraction,
        "model_hydrogen_mass_over_core_plus_integrated_envelope_mass": (
            model_fraction
        ),
        "model_minus_paper": model_fraction - paper_fraction,
        "fractional_difference": model_fraction / paper_fraction - 1.0,
        "model_denominator_kg": total_mass_kg,
        "definition_note": (
            "The model value is the closest saved-run approximation to the "
            "paper definition: integrated atmospheric hydrogen mass divided "
            "by fixed core mass plus integrated envelope mass. The structure "
            "integration itself neglects envelope self-gravity."
        ),
    }


def _radius_comparison(model_value: float, paper_value: float) -> dict[str, float]:
    difference = model_value - paper_value
    return {
        "paper_rearth": paper_value,
        "model_rearth": model_value,
        "model_minus_paper_rearth": difference,
        "fractional_difference": difference / paper_value,
    }


def _outer_rcb_comparison(run: CompletedRun) -> dict[str, Any]:
    """Compare only a model boundary connected to the non-convective top."""

    metrics = run.summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("summary.json does not contain metrics.")
    saved_value = metrics.get("outer_rcb_radius_earth")
    saved_diagnostic = None
    if saved_value is not None:
        if isinstance(saved_value, bool):
            raise ValueError("Saved outer-RCB diagnostic is not numerical.")
        try:
            saved_diagnostic = float(saved_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Saved outer-RCB diagnostic is not numerical."
            ) from error
        if not math.isfinite(saved_diagnostic):
            raise ValueError("Saved outer-RCB diagnostic is not finite.")

    paper_value = run.case.radius_targets.outer_rcb_rearth
    top_transport = run.transport[-1]
    if top_transport != "nonconvective":
        return {
            "availability": "unavailable_without_top_connected_nonconvective_region",
            "paper_rearth": paper_value,
            "model_rearth": None,
            "model_minus_paper_rearth": None,
            "fractional_difference": None,
            "top_transport": top_transport,
            "legacy_saved_transition_rearth": saved_diagnostic,
            "definition_note": (
                "The paper outer RCB bounds a top-connected non-convective "
                "region. This model profile remains convective at the top, "
                "so a detached convective-to-nonconvective transition is not "
                "compared as the same physical boundary."
            ),
        }
    if saved_diagnostic is None:
        raise ValueError(
            "A top-connected non-convective profile has no saved outer RCB."
        )
    return {
        "availability": "available_top_connected_nonconvective_region",
        "top_transport": top_transport,
        **_radius_comparison(saved_diagnostic, paper_value),
    }


def _deduplicate_log_pressure(
    pressure_bar: np.ndarray,
    temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sort a curve and average temperatures at duplicate log pressures."""

    log_pressure = np.log10(np.asarray(pressure_bar, dtype=float))
    temperature = np.asarray(temperature_k, dtype=float)
    order = np.argsort(log_pressure)
    sorted_pressure = log_pressure[order]
    sorted_temperature = temperature[order]
    unique_pressure, inverse, counts = np.unique(
        sorted_pressure, return_inverse=True, return_counts=True
    )
    temperature_sum = np.zeros_like(unique_pressure)
    np.add.at(temperature_sum, inverse, sorted_temperature)
    return (
        unique_pressure,
        temperature_sum / counts,
        int(sorted_pressure.size - unique_pressure.size),
    )


def compare_temperature_profile(
    run: CompletedRun,
    reference: PublishedTemperatureReference,
    *,
    grid_size: int = TEMPERATURE_COMPARISON_GRID_SIZE,
) -> dict[str, Any]:
    """Compare T(P) on a uniform grid over the shared log-pressure range."""

    if run.case.identifier != reference.case_identifier:
        raise ValueError("Run and temperature reference case identifiers differ.")
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two.")
    model_log_pressure, model_temperature, model_duplicates = (
        _deduplicate_log_pressure(run.pressure_bar, run.temperature_k)
    )
    reference_log_pressure, reference_temperature, reference_duplicates = (
        _deduplicate_log_pressure(
            reference.pressure_bar, reference.temperature_k
        )
    )
    overlap_lower = max(model_log_pressure[0], reference_log_pressure[0])
    overlap_upper = min(model_log_pressure[-1], reference_log_pressure[-1])
    if not overlap_lower < overlap_upper:
        raise ValueError(
            f"Case {run.case.identifier!r} has no model/reference pressure overlap."
        )
    uniform_grid = np.linspace(overlap_lower, overlap_upper, int(grid_size))
    model_on_grid = np.interp(
        uniform_grid, model_log_pressure, model_temperature
    )
    reference_on_grid = np.interp(
        uniform_grid, reference_log_pressure, reference_temperature
    )
    difference = model_on_grid - reference_on_grid
    return {
        "availability": "available",
        "reference_contract": reference.reference_contract,
        "metric_contract": {
            "coordinate": "uniform_log10_pressure_bar",
            "grid_point_count": int(grid_size),
            "interpolation": "piecewise_linear_in_log10_pressure",
            "overlap_only": True,
            "duplicate_pressure_policy": "mean_temperature_before_interpolation",
            "vertex_weighting": False,
        },
        "model_input_point_count": int(run.pressure_bar.size),
        "reference_input_vertex_count": int(reference.pressure_bar.size),
        "model_duplicate_pressure_count": model_duplicates,
        "reference_duplicate_pressure_count": reference_duplicates,
        "overlap_log10_pressure_bar": [overlap_lower, overlap_upper],
        "overlap_pressure_bar": [10.0**overlap_lower, 10.0**overlap_upper],
        "temperature_error_k": {
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "mae": float(np.mean(np.abs(difference))),
            "sampled_maximum_absolute": float(np.max(np.abs(difference))),
        },
    }


def _missing_temperature_comparison(case_identifier: str) -> dict[str, Any]:
    return {
        "availability": "missing_for_case",
        "case_identifier": case_identifier,
        "reference_contract": None,
        "metric_contract": None,
        "temperature_error_k": None,
    }


def build_comparison_report(
    runs: Sequence[CompletedRun],
    temperature_references: Mapping[str, PublishedTemperatureReference] | None = None,
) -> dict[str, Any]:
    """Build machine-readable paper-target comparisons for completed runs."""

    if not runs:
        raise ValueError("At least one completed run is required.")
    references = (
        load_temperature_references()
        if temperature_references is None
        else dict(temperature_references)
    )
    cases = []
    for run in runs:
        transit = _finite_metric(run.summary, "transit_radius_earth")
        top_temperature = _finite_metric(run.summary, "top_temperature_k")
        config = run.summary["config"]
        targets = run.case.radius_targets
        reference = references.get(run.case.identifier)
        temperature_comparison = (
            _missing_temperature_comparison(run.case.identifier)
            if reference is None
            else compare_temperature_profile(run, reference)
        )
        cases.append(
            {
                "run_directory": str(run.directory),
                "case_identifier": run.case.identifier,
                "paper_figure": run.case.figure,
                "prescribed_number_ratio": dict(
                    run.case.prescribed_number_ratio
                ),
                "include_sio_s": run.case.include_sio_s,
                "validity_mode": config["validity_mode"],
                "gas_profile_contract": {
                    "normalization": "sum_over_explicit_solver_gas_species",
                    "implicit_neutral_atomic_curves_included": False,
                    "interpretation": (
                        "The paper panels display neutral atomic curves such "
                        "as H that are not explicit species in this ExoGibbs "
                        "network. The model gas panel is therefore not a "
                        "like-for-like published gas-curve comparison."
                    ),
                },
                "radius_comparison": {
                    "outer_rcb": _outer_rcb_comparison(run),
                    "transit_20_mbar": _radius_comparison(
                        transit, targets.transit_rearth
                    ),
                },
                "temperature_profile_comparison": temperature_comparison,
                "hydrogen_mass_fraction_comparison": (
                    _hydrogen_mass_fraction_comparison(run.summary)
                ),
                "fixed_boundary_model": {
                    "pressure_base_bar": config["pressure_base_bar"],
                    "luminosity_w": config["luminosity_w"],
                    "top_temperature_k": top_temperature,
                    "top_transport": run.transport[-1],
                },
                "amount_gauge_conversion_audit": {
                    "gas_amount_scale_source": "H_conservation_row",
                    "all_element_closure_relative_tolerance": (
                        CLOSURE_RELATIVE_TOLERANCE
                    ),
                    "all_element_closure_absolute_tolerance": (
                        CLOSURE_ABSOLUTE_TOLERANCE
                    ),
                    "maximum_absolute_closure_residual": (
                        run.density_conversion.maximum_absolute_closure_residual
                    ),
                    "maximum_tolerance_ratio": (
                        run.density_conversion.maximum_tolerance_ratio
                    ),
                    "number_density_unit": "cm^-3",
                },
                "source_provenance": run.summary.get("source_provenance"),
            }
        )
    return {
        "schema_version": COMPARISON_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_status": COMPARISON_CLAIM_STATUS,
        "paper_reference": PAPER_REFERENCE,
        "comparison_scope": (
            "Saved model gas and condensate profiles are rendered on "
            "paper-style axes but are not published-curve overlays. Published "
            "neutral atomic gas curves are unavailable, and model gas mixing "
            "ratios are normalized over the explicit solver species. Published "
            "temperature curves are vector-plot digitizations compared on a "
            "uniform shared log-pressure grid. The 20 mbar radius uses the "
            "scalar value stated in the paper. An outer RCB is compared only "
            "when the model has a top-connected non-convective region."
        ),
        "paper_constraints": {
            "core_mass_earth": PAPER_COMMON_INPUTS.core_mass_earth,
            "base_temperature_k": PAPER_COMMON_INPUTS.base_temperature_k,
            "equilibrium_temperature_k": (
                PAPER_COMMON_INPUTS.equilibrium_temperature_k
            ),
            "hydrogen_mass_fraction": PAPER_COMMON_INPUTS.hydrogen_mass_fraction,
            "top_pressure_bar": PAPER_COMMON_INPUTS.top_pressure_bar,
            "transit_pressure_bar": PAPER_COMMON_INPUTS.transit_pressure_bar,
            "pressure_fractional_decrement": (
                PAPER_COMMON_INPUTS.pressure_fractional_decrement
            ),
        },
        "known_unmatched_inputs": [
            "The model fixes pressure_base_bar and luminosity_w instead of "
            "shooting to the paper hydrogen mass fraction and equilibrium "
            "temperature.",
            "The paper absolute elemental abundances are not stated.",
            "The paper numerical opacity and conductivity closures are not stated.",
        ],
        "cases": cases,
        "artifacts": {
            "figure": OUTPUT_FIGURE_NAME,
            "report": OUTPUT_REPORT_NAME,
        },
    }


def _paper_style_species(case: PaperFigureCase):
    if case.figure == 1:
        return FIGURE_1_GASES, FIGURE_1_CONDENSATES
    return FIGURE_2_AND_5_GASES, FIGURE_2_AND_5_CONDENSATES


def _plot_transport_curve(
    axis,
    pressure: np.ndarray,
    values: np.ndarray,
    transport: Sequence[str],
    *,
    color,
    label: str,
) -> None:
    finite_values = np.asarray(values, dtype=float)
    axis.plot(finite_values, pressure, color=color, linewidth=0.8, alpha=0.9)
    convective = np.asarray(transport) == "convective"
    axis.plot(
        np.where(convective, finite_values, np.nan),
        pressure,
        color=color,
        linewidth=2.4,
        label=label,
    )


def _case_title(case: PaperFigureCase) -> str:
    ratio = ":".join(str(value) for _name, value in case.prescribed_number_ratio)
    names = ":".join(name for name, _value in case.prescribed_number_ratio)
    sio = ", SiO(s) on" if case.include_sio_s else ""
    return f"{names} = {ratio}{sio}"


def _plot_published_temperature(
    axis,
    reference: PublishedTemperatureReference,
) -> None:
    """Plot published vector segments without joining separate paths."""

    label_pending = True
    for segment in np.unique(reference.segment_index):
        selected = reference.segment_index == segment
        segment_transport = {
            value for value, keep in zip(reference.transport, selected) if keep
        }
        if len(segment_transport) != 1:
            raise ValueError(
                f"Published segment {segment} mixes transport regimes."
            )
        regime = segment_transport.pop()
        axis.plot(
            reference.temperature_k[selected],
            reference.pressure_bar[selected],
            color="tab:blue",
            linewidth=2.4 if regime == "convective" else 0.8,
            linestyle="--",
            label="published vector T(P)" if label_pending else None,
            zorder=3,
        )
        label_pending = False


def plot_comparison(
    path: Path,
    runs: Sequence[CompletedRun],
    temperature_references: Mapping[str, PublishedTemperatureReference] | None = None,
) -> None:
    """Write a paper-style three-row profile and scalar-radius comparison."""

    if not runs:
        raise ValueError("At least one completed run is required.")
    references = (
        load_temperature_references()
        if temperature_references is None
        else dict(temperature_references)
    )
    column_count = len(runs)
    figure, axes = plt.subplots(
        3,
        column_count,
        figsize=(5.2 * column_count, 11.5),
        squeeze=False,
        sharey=True,
    )
    pressure_max = max(float(np.max(run.pressure_bar)) for run in runs)
    pressure_min = min(float(np.min(run.pressure_bar)) for run in runs)
    color_cycle = tuple(plt.get_cmap("tab20").colors)
    for column, run in enumerate(runs):
        gas_axis, condensate_axis, temperature_axis = axes[:, column]
        gas_style, condensate_style = _paper_style_species(run.case)
        gas_index = {name: index for index, name in enumerate(run.gas_species)}
        condensate_index = {
            name: index for index, name in enumerate(run.condensate_species)
        }

        for style_index, (name, label) in enumerate(gas_style):
            if name not in gas_index:
                continue
            values = run.gas_mixing_ratio[:, gas_index[name]]
            values = np.where(values > 0.0, values, np.nan)
            _plot_transport_curve(
                gas_axis,
                run.pressure_bar,
                values,
                run.transport,
                color=color_cycle[style_index % len(color_cycle)],
                label=label,
            )
        for style_index, (name, label) in enumerate(condensate_style):
            if name not in condensate_index:
                continue
            values = run.condensate_number_density_cm3[
                :, condensate_index[name]
            ]
            values = np.where(values > 0.0, values, np.nan)
            _plot_transport_curve(
                condensate_axis,
                run.pressure_bar,
                values,
                run.transport,
                color=color_cycle[style_index % len(color_cycle)],
                label=label,
            )
        for axis in (gas_axis, condensate_axis):
            axis.text(
                0.98,
                0.97,
                "ExoExamples only\n(published curves not digitized)",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.35",
            )
        _plot_transport_curve(
            temperature_axis,
            run.pressure_bar,
            run.temperature_k,
            run.transport,
            color="black",
            label="ExoExamples",
        )
        reference = references.get(run.case.identifier)
        if reference is None:
            temperature_axis.text(
                0.97,
                0.03,
                "Published T(P) reference\nunavailable for this case",
                transform=temperature_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="tab:blue",
            )
            temperature_comparison = None
        else:
            _plot_published_temperature(temperature_axis, reference)
            temperature_comparison = compare_temperature_profile(run, reference)
        changes = np.asarray(
            [
                index
                for index in range(2, len(run.transport))
                if run.transport[index] != run.transport[index - 1]
            ],
            dtype=int,
        )
        if changes.size:
            temperature_axis.scatter(
                run.temperature_k[changes],
                run.pressure_bar[changes],
                color="black",
                s=14,
                zorder=4,
            )
        temperature_axis.scatter(
            [PAPER_COMMON_INPUTS.equilibrium_temperature_k],
            [PAPER_COMMON_INPUTS.top_pressure_bar],
            marker="*",
            s=80,
            color="tab:orange",
            label=r"paper $T_{eq}$ target",
            zorder=5,
        )

        model_transit = _finite_metric(run.summary, "transit_radius_earth")
        paper_transit = run.case.radius_targets.transit_rearth
        outer_rcb = _outer_rcb_comparison(run)
        if outer_rcb["model_rearth"] is None:
            outer_rcb_text = (
                rf"paper $R_{{rcb}}={outer_rcb['paper_rearth']:.2f}R_\oplus$; "
                "model analog unavailable\n"
                f"(model top: {outer_rcb['top_transport']})"
            )
        else:
            outer_rcb_text = (
                r"$R_{rcb}$ model/paper = "
                f"{outer_rcb['model_rearth']:.3f}/"
                f"{outer_rcb['paper_rearth']:.2f} "
                r"$R_\oplus$"
            )
        temperature_axis.text(
            0.97,
            0.97,
            (
                outer_rcb_text
                + "\n"
                +
                r"$R_t$ model/paper = "
                f"{model_transit:.3f}/{paper_transit:.2f} "
                r"$R_\oplus$"
                "\n"
                r"$\Delta R_t/R_t$ = "
                f"{(model_transit / paper_transit - 1.0):+.1%}"
            ),
            transform=temperature_axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )
        if temperature_comparison is not None:
            errors = temperature_comparison["temperature_error_k"]
            temperature_axis.text(
                0.97,
                0.03,
                (
                    r"uniform $\log_{10}P$ grid" "\n"
                    f"T RMSE/MAE = {errors['rmse']:.0f}/{errors['mae']:.0f} K\n"
                    "sampled max |dT| = "
                    f"{errors['sampled_maximum_absolute']:.0f} K"
                ),
                transform=temperature_axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="tab:blue",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.88,
                    "edgecolor": "0.75",
                },
            )

        gas_axis.set_title(_case_title(run.case))
        gas_axis.set_xscale("log")
        gas_axis.set_xlim(1.0e-18, 2.0)
        gas_axis.set_xlabel("Model gas mixing ratio")
        condensate_axis.set_xscale("log")
        condensate_axis.set_xlim(1.0e5, 1.0e24)
        condensate_axis.set_xlabel(
            r"Model condensate number density (cm$^{-3}$)"
        )
        temperature_axis.set_xlim(0.0, 4200.0)
        temperature_axis.set_xlabel("Temperature (K)")
        for axis, font_size, columns in (
            (gas_axis, 7, 2),
            (condensate_axis, 7, 2),
            (temperature_axis, 8, 1),
        ):
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=font_size, ncol=columns, loc="lower left")
        for axis in (gas_axis, condensate_axis, temperature_axis):
            axis.set_yscale("log")
            axis.set_ylim(pressure_max, pressure_min)
            axis.grid(alpha=0.18, which="both")

    figure.supylabel("Pressure (bar)")
    figure.suptitle(
        "Rocky Raccoon-like vs. Misener et al. (2026)\n"
        "Fixed-boundary diagnostic—not a reproduction",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_comparison(
    output_directory: Path,
    runs: Sequence[CompletedRun],
    temperature_references: Mapping[str, PublishedTemperatureReference] | None = None,
) -> dict[str, Any]:
    """Write the comparison PNG and its machine-readable report."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    references = (
        load_temperature_references()
        if temperature_references is None
        else dict(temperature_references)
    )
    report = build_comparison_report(runs, references)
    plot_comparison(destination / OUTPUT_FIGURE_NAME, runs, references)
    with (destination / OUTPUT_REPORT_NAME).open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone comparison CLI."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--run-directory",
        type=Path,
        action="append",
        required=True,
        help="Completed forward-run directory; repeat for additional columns.",
    )
    parser.add_argument(
        "--output-directory",
        "--output-dir",
        dest="output_directory",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--temperature-reference",
        type=Path,
        default=DEFAULT_TEMPERATURE_REFERENCE,
        help="Published vector T(P) reference CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the postprocessor without evaluating chemical equilibrium."""

    args = build_parser().parse_args(argv)
    try:
        runs = tuple(load_completed_run(path) for path in args.run_directory)
        references = load_temperature_references(args.temperature_reference)
        write_comparison(args.output_directory, runs, references)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Rocky Raccoon paper comparison failed: {error}", file=sys.stderr)
        return 1
    print("Rocky Raccoon paper comparison completed.")
    print(f"  Claim status: {COMPARISON_CLAIM_STATUS}")
    print(f"  Output directory: {args.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
