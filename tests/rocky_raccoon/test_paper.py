"""Contracts for the static Rocky Raccoon paper policy."""

from collections import Counter
from importlib.resources import files

import pytest

from examples.rocky_raccoon import paper


def _fastchem4_catalog_names(relative_path: str) -> tuple[str, ...]:
    resource = files("exogibbs").joinpath("data", relative_path)
    names = []
    for line in resource.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        definition = stripped.split("#", 1)[0]
        if ":" in definition:
            names.append(definition.split(maxsplit=1)[0])
    return tuple(names)


def test_appendix_gas_species_are_an_explicit_unique_70_entry_catalog():
    assert isinstance(paper.PAPER_GAS_SPECIES, tuple)
    assert len(paper.PAPER_GAS_SPECIES) == 70
    assert len(set(paper.PAPER_GAS_SPECIES)) == 70

    catalog = Counter(
        _fastchem4_catalog_names(
            paper.FASTCHEM4_ION_INCLUSIVE_GAS_CATALOG
        )
    )
    assert {
        species: catalog[species] for species in paper.PAPER_GAS_SPECIES
    } == {species: 1 for species in paper.PAPER_GAS_SPECIES}


def test_solver_gas_catalog_does_not_add_fastchem_reference_species():
    assert paper.PAPER_REPORTED_GAS_SPECIES is paper.PAPER_GAS_SPECIES
    assert paper.PAPER_SOLVER_GAS_SPECIES is paper.PAPER_GAS_SPECIES
    assert len(paper.PAPER_SOLVER_GAS_SPECIES) == 70
    assert len(set(paper.PAPER_SOLVER_GAS_SPECIES)) == 70
    assert not {"H1", "Mg1", "Si1", "O1", "C1", "e1-"} & set(
        paper.PAPER_SOLVER_GAS_SPECIES
    )
    assert "Si2" not in paper.PAPER_SOLVER_GAS_SPECIES
    assert "Si3" not in paper.PAPER_SOLVER_GAS_SPECIES


def test_canonical_and_sio_condensate_catalogs_match_appendix_policy():
    canonical = paper.PAPER_CANONICAL_CONDENSATES
    with_sio = paper.PAPER_CONDENSATES_WITH_SIO

    assert len(canonical) == len(set(canonical)) == 14
    assert len(with_sio) == len(set(with_sio)) == 15
    assert paper.SIO_CONDENSATE not in canonical
    assert set(with_sio) - set(canonical) == {paper.SIO_CONDENSATE}
    assert set(canonical) - set(with_sio) == set()

    catalog = Counter(
        _fastchem4_catalog_names(paper.FASTCHEM4_CONDENSATE_CATALOG)
    )
    assert {species: catalog[species] for species in with_sio} == {
        species: 1 for species in with_sio
    }


def test_figure_1_is_the_three_element_experiment():
    assert len(paper.FIGURE_1_CASES) == 3
    for case in paper.FIGURE_1_CASES:
        assert case.figure == 1
        assert case.physical_elements == ("H", "Si", "O")
        assert case.condensate_species == paper.FIGURE_1_CONDENSATES
        assert case.include_sio_s is False

    for case in paper.FIGURE_2_CASES + paper.FIGURE_5_CASES:
        assert case.physical_elements == ("H", "Mg", "Si", "O", "C")


def test_figure_cases_record_published_radius_targets():
    targets = {
        case.identifier: (
            case.radius_targets.outer_rcb_rearth,
            case.radius_targets.transit_rearth,
        )
        for case in paper.PAPER_FIGURE_CASES
    }
    assert targets == {
        "figure1_si_o_1_4": (1.55, 1.89),
        "figure1_si_o_1_2": (1.62, 2.49),
        "figure1_si_o_1_1": (1.56, 2.33),
        "figure2_mg_si_o_1_1_3": (1.63, 2.51),
        "figure2_mg_si_o_1_1_4": (1.59, 2.01),
        "figure5_mg_si_o_1_1_3_sio_off": (1.63, 2.51),
        "figure5_mg_si_o_1_1_3_sio_on": (1.63, 2.28),
    }
    assert paper.FIGURE_1_CASES[-1].radius_targets.transit_text_rearth == 2.34


def test_paper_case_lookup_is_canonical_and_rejects_unknown_identifiers():
    case = paper.paper_case_by_identifier("figure2_mg_si_o_1_1_3")

    assert case is paper.FIGURE_2_CASES[0]
    with pytest.raises(ValueError, match="Unknown or duplicate"):
        paper.paper_case_by_identifier("not_a_paper_case")


def test_paper_gamma_policy_and_common_inputs_are_explicit():
    assert paper.paper_gamma("H4Si1") == pytest.approx(1.3)
    assert paper.paper_gamma("H2O1") == pytest.approx(4.0 / 3.0)
    assert paper.paper_gamma("H2") == pytest.approx(7.0 / 5.0)

    inputs = paper.PAPER_COMMON_INPUTS
    assert inputs.core_mass_earth == 4.0
    assert inputs.base_temperature_k == 4000.0
    assert inputs.equilibrium_temperature_k == 1000.0
    assert inputs.hydrogen_mass_fraction == 0.03
    assert inputs.top_pressure_bar == 1.0e-3
    assert inputs.transit_pressure_bar == 2.0e-2
    assert paper.PAPER_PRESSURE_RATIO == pytest.approx(
        1.0 - inputs.pressure_fractional_decrement
    )

    assert inputs.absolute_element_abundances is None
    assert inputs.opacity_parameters is None
    assert inputs.conductivity_parameters is None
    assert all(
        case.absolute_element_abundances is None
        for case in paper.PAPER_FIGURE_CASES
    )
