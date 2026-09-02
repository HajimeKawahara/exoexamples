# ExoExamples

Cross-package workflows for the ExoJAX, ExoGibbs, ExoEOS, and external packages.

- non-installable

## Rocky Raccoon-like deep-envelope column

`examples/rocky_raccoon/raccoon_like_forward.py` couples an exact 70-gas
Appendix network in ExoGibbs to ExoEOS ideal-mixture states and a local
two-candidate structure integrator. It is explicitly a Rocky Raccoon-like
model, not a reproduction of Misener et al. (2026). Neutral atomic reference
gases and a free-electron gas are not appended; `e-` is a zero-inventory
charge-constraint row.

Inspect the configured providers without solving a column:

```console
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/rocky_raccoon/raccoon_like_forward.py --check-inputs
```

Run the verified coarse three-level coupling smoke test:

```console
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/rocky_raccoon/raccoon_like_forward.py \
  --pressure-top-bar 1.5e5 --transit-pressure-bar 1.6e5 --pressure-ratio 0.8 \
  --output-dir outputs/rocky_raccoon_2026/raccoon_like_forward_smoke
```

The 0.8 pressure ratio is for runtime validation only. The default 0.99 grid
must be used with a convergence study for scientific results. Each trial gets
an independent gas-only numerical hint and source-problem provenance from the
accepted parent; condensate support and rainout inventory are never shared
between competing branches.

The authoritative CUDA default-grid run is complete under
`outputs/rocky_raccoon_2026/raccoon_like_forward_empty_support_rescue_gpu`.
All 1,903 accepted layers converged: one base, 1,879 convective, and 23
non-convective. The route counts are 1,462 lifecycle and 441 gas-only layers,
with 14 support changes. The column reached
`P = 0.000998090955700085 bar` and `T = 15.330699918575274 K`; its transit and
outer-RCB radii are `1.9066419361104325` and `1.4805555213405799` Earth radii.
This is a raccoon-like single-grid implementation result, not a paper
reproduction or a pressure-grid-convergence claim. The ordinary suite reports
53 passed and 21 skipped; the opt-in file contains one three-layer check and
20 exact one-layer regressions, including step 1383 with support `(5,)`.
ExoExamples does not propagate condensate support or skip a failed thermal
candidate; it fails closed and records every attempt in `run_status.json`.

See the [implementation documentation](docs/en/rocky_raccoon/raccoon_like_forward_en.rst)
for the chemistry contract, branch transaction, assumptions, outputs, tests,
and the current ExoGibbs regression status.

## MELTYQ-like clear forward model

`examples/meltyq/meltyq_clear_forward.py` connects a magma--gas boundary, a
non-ideal deep atmosphere, hydrostatic radius integration, and a minimal clear
transmission spectrum. It is a compact forward-model example rather than a
reproduction of the [MELTYQ retrieval](https://arxiv.org/abs/2605.08752).

Install the external rocky mass--radius dependency:

```console
python -m pip install exopie==2.1.0 matplotlib
```

[ExoPie](https://github.com/mplotnyko/exopie/) is a third-party package and is
not part of ExoFamily despite the similar name. Its interior-model methodology
is described by [Plotnykov & Valencia
(2024)](https://doi.org/10.1093/mnras/stae993). ExoExamples uses it only to
provide the rocky-body lower-boundary radius.

Run the example from the repository root:

```console
python examples/meltyq/meltyq_clear_forward.py
```

If JAX selects an accelerator that is not available on the host, set the
`JAX_PLATFORMS` environment variable to `cpu` before running the command.

The first run downloads the verified Chabrier--Debras EOS tables through
ExoEOS and the CO/H2--H2 opacity data through ExoJAX. ExoJAX data are cached
under `.database/`.

The example uses the external ExoPie rocky-body radius at a 33% core mass
fraction, FastChem4 gas
thermochemistry, pure-component Zhang--Duan fugacity coefficients, a fixed
`Y=0.275` H2--He density table, ideal N-bearing gases, a uniform 10-bar quench
composition, CO absorption, and H2--H2 collision-induced absorption. Clouds,
hazes, additional molecular opacities, disequilibrium chemistry, and retrieval
are outside its scope.

## MELTYQ Figure 8 comparison

`examples/meltyq/meltyq_figure8.py` evaluates the six one-parameter sweeps shown in
Figure 8 of the MELTYQ paper. It records the gas composition at the magma
surface and 10 bar, the hydrostatic radius at 10 bar, and convergence
diagnostics for every point. The published sampling is not documented in the
paper; the grids used here were inferred from the vector figure.

Run the comparison from the repository root:

```console
python examples/meltyq/meltyq_figure8.py
```

Results are written to `outputs/meltyq_figure8/results.csv`, with run metadata
and the six-panel plot in the same directory. Each sweep starts at the common
paper baseline and uses the previous converged magma--gas root independently
toward lower and higher parameter values. A failed solve remains a gap with
its diagnostics in the CSV; it is never interpolated into a curve. The first
point compiles reusable JAX boundary and structure kernels. Compiled kernels
are cached across processes in `.cache/jax`; set
`JAX_COMPILATION_CACHE_DIR` before launch to use another location.

The comparison uses the ExoGibbs FastChem4 thermochemistry preset. Before
solving, the script verifies that the selected thermochemistry changes between
1500 and 3000 K and stops if a temperature-clipped provider is supplied.

This is a same-input curve comparison, not an exact reproduction. The paper
does not publish its numerical Figure 8 data or deep-layer discretization, and
the external ExoPie rocky radius differs slightly from the Fortney et al. relation used
in MELTYQ. The metadata also records that the current ExoGibbs preset accepts
the melt inputs on elemental-C and atomic-N dilute mole-ratio bases; these
provider conventions must not be silently identified with differently labeled
paper quantities.

Detailed executable documentation is available as separate language editions:

- Japanese master: [notebook](docs/ja/meltyq/meltyq_figure8_forward_comparison_ja.ipynb),
  [reStructuredText](docs/ja/meltyq/meltyq_figure8_forward_comparison_ja.rst)
- English translation:
  [notebook](docs/en/meltyq/meltyq_figure8_forward_comparison_en.ipynb),
  [reStructuredText](docs/en/meltyq/meltyq_figure8_forward_comparison_en.rst)

## MELTYQ Figure 3 forward comparison

`examples/meltyq/meltyq_figure3.py` connects the converged magma/deep result to a
100-layer TauREx-NPoint-style upper atmosphere with configurable temperature
smoothing, five molecular opacities, H2--H2/H2--He CIA,
Rayleigh scattering, Lee haze, an opaque gray deck, the relevant TauREx
compatibility discretization, ExoJAX's native Simpson transmission solver,
and hash-pinned public JWST bins. The checked-in public
demo uses rounded Figure 9 posterior readings and is explicitly not the
unpublished maximum-likelihood Figure 3 spectrum.

ExoJAX owns the public configuration's ideal-gas hydrostatic profile, number
density, CIA interpolation, geometric layer optical depths, chord geometry,
annulus integration, reusable exact spectral-binning operators, pointwise
ESLOG resolution and grid sizing, and Diffgrid-to-teacher diagnostics. The
default `exojax_simpson` path uses
`ArtTransPure(integration="simpson")`, ExoJAX's variable-gravity atmosphere,
and zero CIA opacity outside each table's native coverage. The optional
`taurex_rectangle` compatibility path selects
`hydrostatic_radius_profile_ideal_gas(..., hydrostatic_scheme="layer_constant_gravity")`,
constructs one explicit
`OpaCIA(..., wavenumber_interpolation="interp")` contract, and passes cgs
absorber columns to ExoJAX's cross-section, log-CIA, and extinction layer-depth
APIs.
ExoExamples retains the comparison-specific
`TransmissionModel(new_path_method=False)` legacy chord lengths, rectangle
annulus areas, and the [pinned TauREx Rayleigh formulas](https://github.com/ucl-exoplanets/taurex3/blob/7b6e82a86d4675f140e9e59f3d1410a863251c03/src/taurex/util/scattering.py).
Molecular cross sections also come from ExoJAX. The
[pinned TauREx CIA source](https://github.com/ucl-exoplanets/taurex3/blob/7b6e82a86d4675f140e9e59f3d1410a863251c03/src/taurex/cia/hitrancia.py)
and [transmission source](https://github.com/ucl-exoplanets/taurex3/blob/7b6e82a86d4675f140e9e59f3d1410a863251c03/src/taurex/model/transmission.py)
define the translation target. This TauREx 3.2.0 source pin is not a claim
about the unpublished MELTYQ runtime revision; TauREx is not a runtime dependency.

Run the first, fast ExoMolOP CKD comparison with explicit download consent:

```console
python examples/meltyq/meltyq_figure3.py --fetch-public-data --allow-opacity-download --benchmark-repeats 3
```

Inspect external inputs before a forward solve:

```console
python examples/meltyq/meltyq_figure3.py --check-inputs
```

This is a lightweight audit: public-data paths and pinned hashes are verified,
while CIA-file presence and actual hashes are recorded;
each CKD table is also matched to its expected ExoMolOP basename, optional HDF5
`mol_name`, and molecular mass. For Diffgrid it hashes each complete NPZ and sidecar, validates
descriptor/metadata/user metadata, and loads only the small wavenumber,
pressure, and temperature grids needed for build-contract/teacher/resolution
checks. It does not
materialize the large cross-section tensors. Both lightweight and full
Diffgrid loading require coverage of the configured 0.65--12 micron endpoints
before the deep solve. A supplied Diffgrid reference CSV must declare the
`intrinsic_unoffset_model` contract and is checked for its header, finite
values, and wavelength order; `--reference-spectrum-sha256` optionally pins
its file hash. Point-wise reference comparison is rejected for CKD band
means. Later offline runs omit both download flags. The CKD path
uses R~1000 tables and a matching-g
perfect-correlation approximation. The paper-line-list-aligned main comparison
instead accepts hash-pinned R>=50000 `OpaDiffgrid` archives:

```console
python examples/meltyq/meltyq_figure3.py --opacity-mode diffgrid --diffgrid-manifest PATH_TO_FILLED_MANIFEST.json
```

The high-cost builder requires explicit permission before downloading opacity
sources and records line-database file inventories, NPZ/sidecar hashes, and
in-archive provenance:

```console
python examples/meltyq/meltyq_figure3_build_diffgrid.py --help
```

It builds one species per process on a measured Rmin>=50000 grid, using 100
pressure layers, 21 inverse-temperature nodes over 200--1200 K, requested H2
broadening, and `crit=0`. The strict default rejects a species for which the
requested `.broad` file is absent. In particular, the public CO/Li2015 source
has no H2 broadening file; RADIS would otherwise silently use its `.def`
defaults (`alpha_ref=0.07 cm-1 bar-1`, temperature exponent `0.5`). The
explicit `--allow-default-broadening-fallback` flag permits and records that
approximation. Because this policy is part of the cross-species build
contract, use the same flag for all five builds in one manifest. Before
saving, all 20 inverse-temperature-interval midpoint
isothermal profiles must have
finite teacher/Diffgrid cross sections and pass default absolute log-cross-section
gates of p99<=0.05 and maximum<=0.5; both thresholds have explicit CLI
overrides. ESLOG grid sizing and measurement use ExoJAX's pointwise resolution
option, while midpoint selection and error summaries use ExoJAX's Diffgrid
diagnostics; ExoExamples retains the thresholds, rejection policy, and
provenance. The loader revalidates all 20 recorded midpoint measurements rather
than trusting a passed status. This opacity-interpolation gate does not establish convergence of
observationally binned transit depths. Value and derivative tables require
about 4.562 GiB per species and 22.8 GiB for five, before line databases,
teachers, and compiler temporaries. CH4/YT34to10 declares about 34 billion
transitions. YT34to10 and BYTe stop at 12000 cm-1 (0.833 micron), and BYTe
completeness remains a specific limitation. The builder-generated manifest
locks exact spectral/pressure/temperature coordinate hashes and teacher
settings before each subsequent species build starts.

CKD band means are binned from their real wavelength-overlap edges with
ExoJAX's `band_mean_bin_operator`, without interpolating band centers.
Diffgrid point samples instead use `piecewise_linear_bin_operator` for exact
top-hat integration of their piecewise-linear reconstruction. Both paths use
one batched `apply_bin_operator` call across all scenarios and observation
bins. The code writes native and binned model spectra, public
data and model-side offsets, a comparison plot, and provenance/timing metadata
to `outputs/meltyq_figure3/`. It evaluates eight standalone-RT scenarios--the
total, aerosols, Rayleigh+CIA, and five molecule-only curves--in one stacked
JIT kernel, avoiding one compilation per component. Large Diffgrid tables are
dynamic inputs to a reusable opacity-interpolation JIT; only evaluated layer
cross sections are passed dynamically to the RT JIT, so the tables do not
become giant RT executable constants. Full-run metadata records
CKD, CIA, and optional-reference hashes plus lower-bound memory estimates.

Pending author-supplied numerical curves, the bilingual documentation also
contains a provisional comparison against the original Figure 3 raster in the
arXiv source bundle. A source-specific extractor records the black MELTYQ
curve, the six visible opacity components, and the T--P centerline in
`docs/meltyq/data/meltyq_figure3_raster_reference.csv`, including trace-ambiguity and
error-bar-overlap flags. This artifact has the separate contract
`published_raster_plot_digitization`; it is not passed to the
`intrinsic_unoffset_model` reference-spectrum interface. The comparison uses
the generated public CKD demo, shows raw and one-constant-aligned views, and
preserves metrics and input hashes in
`docs/meltyq/data/meltyq_figure3_raster_comparison_summary.json`. These are visual
discussion diagnostics, not a likelihood or an exact-reproduction claim.

Network-free checks cover equations, units, hash rejection, bin integration,
offset profiling, and component-kernel shape:

```console
JAX_PLATFORMS=cpu python -m pytest -q tests
```

They do not run the complete deep model, download real data, or establish an
exact MELTYQ reproduction. The author-dependent items are the
maximum-likelihood vector, exact temperature-smoothing and opacity-generation
settings/provenance, and machine-readable black curve. The generated Diffgrid
manifest is a local reproducibility record, not an author-supplied artifact.
If the authors confirm that mixed H2/He pressure broadening is required, the
tables must conditionally use an ExoJAX multi-broadener path or an external
teacher. This does not establish a further ExoFamily change requirement.
No additional ExoFamily package change is required for the current forward path.

Technical documentation, including the provisional raster comparison, is kept
as separate editions:

- [Japanese master](docs/ja/meltyq/meltyq_figure3_forward_preparation_ja.rst)
- [English translation](docs/en/meltyq/meltyq_figure3_forward_preparation_en.rst)

Build both Sphinx editions with:

```console
make -C docs html
```
