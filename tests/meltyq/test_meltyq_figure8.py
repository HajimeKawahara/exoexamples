from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import examples.meltyq.meltyq_figure8 as figure8
from exogibbs.presets.magma_gas import MELTYQ_SPECIES
from exojax.database.molinfo import mean_molmass_manual

from examples.meltyq.meltyq_clear_forward import (
    HELIUM_MASS_FRACTION,
    project_h_he_for_density,
)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("rocky_core_mass_fraction", -0.1, "must lie in"),
        ("rocky_core_mass_fraction", 1.1, "must lie in"),
        ("max_outer_iterations", 0, "positive integer"),
        ("max_outer_iterations", True, "positive integer"),
        ("deep_nlayer", -1, "positive integer"),
        ("deep_nlayer", 1.5, "positive integer"),
    ),
)
def test_build_context_rejects_invalid_public_inputs(keyword, value, message):
    with pytest.raises(ValueError, match=message):
        figure8.build_context(**{keyword: value})


def test_density_projection_preserves_the_fixed_h_he_mass_fraction():
    molar_masses = jnp.asarray(
        [mean_molmass_manual(species) * 1.0e-3 for species in MELTYQ_SPECIES]
    )
    mole_fractions = jnp.asarray(
        [0.70, 0.01, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.20]
    )
    projected = np.asarray(
        project_h_he_for_density(mole_fractions, molar_masses)
    )
    h2_index = MELTYQ_SPECIES.index("H2")
    he_index = MELTYQ_SPECIES.index("He")
    h_he_mass = (
        projected[h2_index] * molar_masses[h2_index]
        + projected[he_index] * molar_masses[he_index]
    )
    helium_mass_fraction = (
        projected[he_index] * molar_masses[he_index] / h_he_mass
    )

    assert np.sum(projected) == pytest.approx(1.0)
    assert float(helium_mass_fraction) == pytest.approx(HELIUM_MASS_FRACTION)


def test_mole_fraction_validation_rejects_nonfinite_or_unnormalized_values():
    valid = np.full(len(MELTYQ_SPECIES), 1.0 / len(MELTYQ_SPECIES))
    np.testing.assert_allclose(
        figure8._validated_mole_fractions("test", valid),
        valid,
    )

    invalid = valid.copy()
    invalid[0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        figure8._validated_mole_fractions("test", invalid)

    with pytest.raises(RuntimeError, match="sum to one"):
        figure8._validated_mole_fractions("test", valid * 0.5)


def test_baseline_forward_path_converges_with_finite_outputs():
    context = figure8.build_context()
    result, continuation = figure8.solve_point(context, figure8.BASELINE)

    assert result.error == ""
    assert result.boundary_converged
    assert result.profile_converged
    assert result.base_converged
    assert np.isfinite(result.radius_base_rearth)
    assert continuation is not None
    np.testing.assert_allclose(np.sum(result.melt_mole_fractions), 1.0)
    np.testing.assert_allclose(np.sum(result.base_mole_fractions), 1.0)


def test_sweeps_share_one_baseline_and_cache_duplicate_coordinates(monkeypatch):
    calls = []

    def fake_solve_point(context, parameters, *, init=None):
        calls.append((parameters, init))
        result = figure8._empty_result(parameters)
        result.boundary_converged = True
        result.profile_converged = True
        result.base_converged = True
        result.melt_mole_fractions = np.full(
            len(MELTYQ_SPECIES),
            1.0 / len(MELTYQ_SPECIES),
        )
        result.base_mole_fractions = result.melt_mole_fractions.copy()
        result.radius_base_rearth = 2.0
        return result, SimpleNamespace(root_variables=np.zeros(4))

    monkeypatch.setattr(figure8, "solve_point", fake_solve_point)
    runs, cache, baseline = figure8.run_sweeps(SimpleNamespace())

    assert len(calls) == 72
    assert len(cache) == 72
    assert sum(len(panel) for panel in runs) == 77
    assert sum(
        point.result is baseline
        for panel in runs
        for point in panel
    ) == len(figure8.SWEEPS)


def test_results_csv_preserves_an_incomplete_point(tmp_path):
    context = SimpleNamespace(rocky_radius_rearth=1.5)
    result = figure8._empty_result(
        figure8.BASELINE,
        error="intentional failure",
    )
    sweep = figure8.SWEEPS[0]
    point = figure8.SweepPoint(sweep, sweep.values[0], result)
    path = tmp_path / "results.csv"

    figure8.write_results_csv(path, [[point]], context)

    text = path.read_text(encoding="utf-8")
    assert "intentional failure" in text
    assert ",nan," in text
