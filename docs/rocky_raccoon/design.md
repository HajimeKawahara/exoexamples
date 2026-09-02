# Rocky Raccoon-like ExoFamily design and implementation status

Status: the fixed-boundary ExoExamples coupler milestone is implemented. It
carries the machine-readable claim `raccoon_like_not_paper_reproduction` and
does not claim a reproduction of Misener et al. (2026). The authoritative CUDA
default-grid column is complete; pressure-grid convergence remains open.

The executable user documentation is
[`raccoon_like_forward_en.rst`](../en/rocky_raccoon/raccoon_like_forward_en.rst).
This document records the package boundary and the transition from resolved
one-layer provider regressions to default-column validation.

## Scope decision

The current objective is a Rocky Raccoon-like calculation using ExoFamily
providers, not recovery of the unpublished numerical solutions behind Figures
1, 2, or 5. The first milestone covers the deep atmospheric column above a
prescribed basal element composition. A paper-facing postprocessor now renders
completed model columns on the Figure 2/5 axes and compares their temperature
profiles and published scalar targets. This is an explicit diagnostic
comparison, not a change to the reproduction claim.

No new ExoStructure package is introduced. The coupler belongs in ExoExamples
until self-gravity, `dm/dr`, entropy/time evolution, or core EOS machinery is
shared by multiple workflows.

## Implemented ownership

| Responsibility | Owner |
| --- | --- |
| Exact paper-derived gas/condensate lists, presets, validity experiment, fixed-boundary structure stepping, branch transaction, provisional transport, CLI, and outputs | ExoExamples |
| Gas--condensate minimization, support closure, conservation gates, and rainout composition transition | ExoGibbs |
| Ideal-mixture density, heat capacity, and adiabatic gradient | ExoEOS |
| Upper-boundary construction, opacity, and radiative transfer | ExoJAX, deferred and not imported by the current workflow |
| Mg--Si--O melt activity and magma--gas basal inventory | ExoGibbs/ExoEOS, deferred |

## Chemistry contract

- The five-element model minimizes exactly the Appendix A list of 70 gas
  species. It does not use an element-only catalog filter.
- The formula matrix rows are `H`, `Mg`, `Si`, `O`, `C`, and `e-`; they have
  full row rank.
- `e-` is a zero-inventory charge-constraint row. Neither neutral atomic
  reference gases nor the free-electron gas `e1-` are appended for ExoGibbs.
- The canonical network has 14 condensates. `oxygen_poor_sio` adds `SiO(s)`
  as an independent fifteenth-species sensitivity.
- `paper_extrapolated` removes condensate upper-temperature bounds while
  preserving them as metadata. `strict_validity` enforces them. The current
  validity scope is condensates only.
- Rainout inventories are normalized compositions, not absolute retained
  element amounts.

## Coupling transaction

For each pressure step, ExoExamples computes convective and non-convective
candidate temperatures. Both ExoGibbs calls receive independent copies of the
same accepted incoming inventory. Each proposes its own equilibrium and
rainout output. The strict Equation (1) inequality selects a branch, and only
that branch's output is propagated. Equality selects the non-convective
branch. Both candidate solves must converge.

Each candidate also receives a freshly allocated gas-only numerical hint from
the same accepted parent transition. The hint records the temperature,
pressure, and incoming inventory of the accepted source problem as numerical
provenance. It does not carry condensate amounts, support, or a rainout result,
and ExoGibbs rediscovers the active support. The rejected structure candidate
never becomes a later initializer.

ExoGibbs owns the shared `regauge_gas_only_warm_start` conversion used by the
direct hint, inventory bridge, and native rainout profile. It preserves all
finite target-compatible gas-log ratios with one uniform gauge shift, including
linear-underflow traces, and floors only absent or depleted-incompatible
species. ExoExamples only attaches the accepted source point required for
bridge provenance.

ExoEOS `IdealGas` supplies the thermodynamic state. The current non-convective
closure uses constant Rosseland opacity and thermal conductivity. These and
the prescribed Si/H and C/H abundances are explicit raccoon-like assumptions,
not inferred reproductions of unpublished paper inputs.

The default pressure ratio is the paper-derived `0.99`. A ratio of `0.8` is
exposed only for the verified three-level runtime smoke test and is not a
scientific grid.

## Verified ExoExamples milestone

The deep-column smoke configuration runs from `2e5 bar` with a stopping
threshold of `1.5e5 bar` and `pressure_ratio=0.8`. It accepts three levels,
all using the 70-gas/14-condensate network, and exercises a condensate-support
change. The opt-in real-provider regression verifies convergence, charge
balance, full element reconstruction, and non-negative outgoing composition.
The opt-in file contains one three-layer provider test and 20 hard
passing one-layer regressions. These include the former `T=1433.764595 K`,
`P=8796.093022 bar` trace-Mg boundary and four subsequently isolated provider
boundaries. They cover caller-gauge-preserving exact-polish rescue,
optimizer-limit physical certification, alternative-basic-support rescue, and
the generic trace-capacity zero-barrier initializer. The former
pressure-step-378 and step-380 backend-parity states, the pressure-step-698
warm-parent state, the pressure-step-702 mixed-budget state, and the
pressure-step-774 support-release, step-999 optimizer-directed-release, and
step-1082 inventory-bridge states are also hard passing regressions. Eight
additional warm-parent regressions cover steps 1075, 1076, 1077, 1084, 1186,
1342, 1372, and 1383. Without the opt-in environment variable, the complete
Rocky Raccoon test directory currently reports 53 passed and 21 skipped.

The authoritative run accepted 1,903 layers, all converged: one base, 1,879
convective, and 23 non-convective. It used 1,462 lifecycle and 441 gas-only
routes and changed condensate support 14 times. The top state is
`P = 0.000998090955700085 bar`, `T = 15.330699918575274 K`; the transit
radius is `1.9066419361104325` Earth radii. The previously reported
`1.4805555213405799` Earth-radii boundary is a detached
convective-to-non-convective transition: the column becomes convective again
above it. It is a legacy transition diagnostic, not the outer boundary of a
top-connected non-convective region. The paper-analog outer RCB is unavailable
for this column. The artifacts are under
`outputs/rocky_raccoon_2026/raccoon_like_forward_empty_support_rescue_gpu`.
These are raccoon-like single-grid outputs, not reproduced paper values.

The CLI also records `profiles.csv`, `summary.json`, `profile.png`, and an
authoritative `run_status.json` for the latest attempt. Failed structure
candidates are retained as failures and are never interpolated or replaced.
ExoGibbs may use an audited intermediate inventory only as a numerical gas
seed; only an accepted exact endpoint becomes a layer or rainout transition.
The integrated summary reports `hydrogen_to_core_mass_ratio`; the denominator
is the fixed gravitating core mass, not the local gas mass used by the CSV's
per-layer `hydrogen_mass_fraction`.
Package versions are supplemented by scoped module paths, Git revisions,
dirty flags, and Python-source inventory hashes. A failed rerun does not delete
older completed artifacts, so downstream use must require a completed
`run_status.json`.

An optional `--accepted-layer-snapshot PATH` writes the latest committed layer
as an atomic, fully named NPZ. It retains exact gas log amounts and both
inventory vectors after a later candidate failure. It is diagnostic evidence,
not a restart file, and never supersedes `run_status.json`.

## Paper-facing comparison contract

`examples/rocky_raccoon/paper_comparison.py` is a read-only postprocessor for
saved forward columns. It checks `run_status.json`, the claim status, preset,
species order, output columns, and source provenance before plotting. A failed
or running directory is rejected; a partial oxygen-rich profile cannot be
presented as a completed Figure 2 result.

Generate the currently available Figure 2 comparison with:

```console
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/rocky_raccoon/paper_comparison.py \
  --run-directory outputs/rocky_raccoon_2026/comparison_oxygen_poor_gpu \
  --output-directory outputs/rocky_raccoon_2026/paper_comparison_figure2
```

Generate the completed Figure 5 SiO(s)-off/on comparison with:

```console
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python examples/rocky_raccoon/paper_comparison.py \
  --run-directory outputs/rocky_raccoon_2026/comparison_oxygen_poor_gpu \
  --run-directory outputs/rocky_raccoon_2026/comparison_oxygen_poor_sio_gpu \
  --output-directory outputs/rocky_raccoon_2026/paper_comparison_figure5
```

The postprocessor writes `paper_comparison.png` and
`paper_comparison.json`. Its plotted quantities have deliberately different
reference contracts:

- Gas mixing ratios and condensate number densities are ExoExamples outputs
  only. Gas fractions are normalized over the explicit solver species. The
  paper includes neutral atomic curves, including H, that are absent from this
  explicit ExoGibbs network. The PDF also does not give a reliable
  machine-readable species binding for every fragmented gas and condensate
  path, so no published composition overlay or residual is claimed.
- Condensate amounts are converted to number density through an
  ExoExamples-owned amount-gauge reconstruction. Its full element closure is
  audited and stored in the comparison JSON.
- Dashed temperature curves are coordinates extracted from the paper PDF's
  vector artwork. The checked-in
  `docs/rocky_raccoon/data/rocky_raccoon_temperature_vector_reference.csv` and
  companion JSON preserve the PDF hash, extraction contract, case identity,
  path regime, and row counts. They are not an author-provided data table and
  support visual/shape comparisons rather than likelihood evaluation.
- The 20-mbar transit radius is compared with the scalar stated in the paper.
  An outer RCB is compared only when the saved profile ends in a top-connected
  non-convective region. Detached internal transitions are not relabeled as
  the paper RCB.

The fixed-boundary calculation prescribes `pressure_base_bar` and
`luminosity_w`. The paper instead shoots those two quantities to its hydrogen
envelope mass fraction `f = 0.03` and equilibrium temperature
`Teq = 1000 K`. Its absolute elemental abundances and numerical opacity and
conductivity tables are also unavailable. Therefore profile and radius
differences describe the mismatch of the complete present closure; they are
not paper-reproduction errors or isolated chemistry errors.

The current execution boundary is explicit:

| Panel | Case | Status | `Rt` model / paper | T(P) RMSE / MAE / sampled max [K] | `fH` model / paper | Outer RCB model / paper |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 2 left / Figure 5 left | `oxygen_poor` | Completed | `1.90664194 / 2.51 Rearth` (`-24.0382%`) | `714.8793 / 629.2752 / 984.6299` | `0.03934380 / 0.03` | unavailable / `1.63 Rearth`; detached legacy transition `1.48056 Rearth` |
| Figure 2 right | `oxygen_rich` | ExoGibbs provider failure near `0.0919 bar` | unavailable | unavailable | unavailable | unavailable |
| Figure 5 right | `oxygen_poor_sio` | Completed | `1.87389604 / 2.28 Rearth` (`-17.8116%`) | `709.9582 / 620.6185 / 983.9805` | `0.03954282 / 0.03` | unavailable / `1.63 Rearth`; detached legacy transition `1.44802 Rearth` |

Direct replay of the oxygen-rich failure state succeeds on CPU with
`Mg(OH)2(s)` support but fails on CUDA with empty support. The divergence is
triggered by a backend-scale difference in the initial element potential. On
CUDA, the inventory-bridge midpoint nevertheless finds and certifies
`Mg(OH)2(s)` support, but the target retry retains only the gas warm start and
discards that support. Replaying the target with the certified support as an
initializer succeeds. The minimal generic ExoGibbs remedy is therefore to
carry certified bridge support into the target retry within the target
capacity, while leaving the final target KKT and physical audits authoritative.
ExoExamples must not hide this provider issue with a case-specific fallback.

Temperature errors use a 512-point uniform shared-log-pressure grid with
piecewise-linear interpolation in `log10(P)`. The maximum is sampled on that
grid, not a continuous global maximum. Both completed columns are convective
at the top and therefore have no paper-analog outer RCB.

Within this fixed-boundary closure, enabling SiO(s) changes the model transit
radius by `-0.0327459 Rearth` (`-1.72%` relative to the off case). The
published targets change by `-0.23 Rearth` (`-9.16%`), so the model's absolute
sensitivity is about `14.2%` of the published sensitivity. This is a closure
response diagnostic, not a reproduction score.

## Resolved ExoGibbs provider regressions

The default-column investigation isolated four exact one-layer states:

- `T = 1334.4049016146876 K`, `P = 6495.780683442079 bar`, normalized
  Mg inventory `= 1.9646e-14`: the rainout caller gauge is preserved and a
  closed finite-barrier state can initialize bounded exact polish;
- `T = 1561.8193557386803 K`, `P = 11290.04441816559 bar`: an
  evaluation-limit optimizer result is usable only when the independent full
  physical certificate passes;
- `T = 1269.1589798706555 K`, `P = 5643.1822694059156 bar`, normalized
  Mg inventory `= 3.7742e-15`: a bounded alternative-basic-support portfolio
  selects the full-rank `SiO2(s,l), MgSiO3(s,l)` basis and the existing closure
  certifies the final exact root.
- `T = 1173.1942732095774 K`, `P = 4132.5213914599017 bar`, normalized
  Mg inventory `= 7.5890e-17`: after an eligible finite-barrier restoration
  failure, a generic trace-capacity gate permits the preserved pre-PDIPM state
  to initialize one bounded exact zero-barrier closure; the unchanged internal
  and caller-gauge audits certify the final root.

All four states now pass as hard provider regressions. The initializer-only
gas-stationarity gate is `1e-5`, while every final zero-barrier and caller-gauge
physical audit retains the ordinary `1e-8` tolerances. An optimizer-limit root
cannot delete a phase, and all non-finite, physically uncertified, or otherwise
ineligible results fail closed.

These changes resolve the four earlier one-layer ExoGibbs blockers but do not
establish completion of the default column.

## Resolved ExoGibbs trace-capacity boundary

The exact pressure-step-386 state has:

- `T = 1173.1942732095774 K`;
- `P = 4132.5213914599017 bar`;
- normalized Mg inventory `= 7.5890e-17`.

With ExoGibbs merge commit `2c68aae`, its finite-barrier solve terminated with
`RESTORATION_MAX_ITER`. The original support
`SiO2(s,l), MgSiO3(s,l), MgO(s,l), Mg2SiO4(s,l)` has rank two because
`A_MgSiO3 = A_SiO2 + A_MgO` and
`A_Mg2SiO4 = A_SiO2 + 2 A_MgO`. Reducing it to a full-rank basis removes that
null space but not the more restrictive trace-capacity condition.

ExoGibbs merge commit `caf257b` resolves this state without an ExoExamples
workaround. For a support phase `j`, ExoGibbs computes the conservative
capacity

`c_j = min_{i in M, A^c_{ij} > 0} b_i / A^c_{ij}`,

where `M` contains only monotone conservation rows with non-negative gas and
condensate coefficients. Signed rows such as charge balance are excluded
because they do not provide an amount ceiling. If the ordinary terminal-state
initializer is unavailable after an eligible restoration failure and the
support contains a phase with `c_j` at or below the first finite-barrier
amount, the preserved pre-PDIPM state may initialize one bounded exact
zero-barrier closure.

The failed finite-barrier state remains diagnostic evidence and is not itself
accepted. Final acceptance still requires both the internal zero-barrier and
caller-gauge physical audits at the ordinary `1e-8` tolerances. The route is
selected from restoration status and capacity geometry, not from a species
name, temperature, or pressure. The exact state is retained as the hard
passing `test_resolved_default_column_trace_capacity_boundary` regression.

## Resolved ExoGibbs backend-parity regressions

The former A100 GPU pressure-step-378 boundary has:

- `T = 1188.1415292259892 K`;
- `P = 4478.5100542051532 bar`;
- normalized Mg inventory `= 2.1536e-16`; and
- lifecycle outcome `fixed_support_failed`.

The former CPU pressure-step-380 boundary has:

- `T = 1181.3459388985098 K`;
- `P = 4389.3877041264705 bar`;
- normalized Mg inventory `= 1.6632e-16`; and
- lifecycle outcome `fixed_support_failed`.

Current ExoGibbs converges for both exact inputs. They are retained without
backend conditions as the hard passing
`test_resolved_default_column_step_378` and
`test_resolved_default_column_step_380` regressions.

No ExoExamples-owned chemistry floor, grid change, support propagation, or
branch skip is introduced. The generic gas-hint floor belongs to the shared
ExoGibbs regauging policy described above.

## Resolved ExoGibbs guarded-restart boundary

Before the current guarded-restart fix, the latest unchanged A100 GPU default
run reached pressure step 698 at:

- `T = 480.4777949967222 K`;
- `P = 179.6370128930636 bar`; and
- normalized Mg inventory `= 2.403769699e-46`.

The exact regression starts from the accepted parent gas state, matching the
column's gas-only warm-start contract. ExoGibbs now derives capacity
regularization only from rows that are monotone across the joint gas and
condensate formula catalogs. A finite initializer-relative solve at its
evaluation limit, with positive active amounts but no local KKT certificate,
may seed one guarded dimensionless-unit-scaled restart. The restarted state
must still pass ordinary phase closure and the unchanged full physical and
caller-gauge audits. The exact state is the hard passing
`test_resolved_default_column_step_698_warm_parent` opt-in regression.

The subsequent unchanged A100 run passed this layer and stopped at pressure
step 702:

- `T = 475.01010900904657 K`;
- `P = 172.55859783339542 bar`; and
- normalized Mg inventory `= 6.300502379398082e-47`.

At this state, the positive H, Mg, Si, O, and C budgets need the existing
positive log-domain formulation, while the exactly zero signed charge budget
must remain linear. ExoGibbs now applies that mixed formulation to the existing
ordered alternative-basic-support portfolio after its normalized-linear
candidates are rejected. The selected support `(1, 8)`, corresponding to
`SiO2(s,l)` and `MgSiO3(s,l)`, passes the unchanged full and caller-gauge
physical audits. This regression deliberately starts cold and therefore
covers the cold fallback that also failed in the column; the production
column tries its independent parent-gas hint first. The exact state is
retained as the hard passing
`test_resolved_default_column_step_702_mixed_charge_budget` opt-in regression.

## Resolved ExoGibbs support-release boundary

The next unchanged CUDA default run passed step 702 and stopped at pressure
step 774:

- `T = 386.57556568831939 K`;
- `P = 83.689430815806617 bar`; and
- normalized Mg inventory `= 2.139116395339677e-56`.

At this state, the finite-barrier endpoint retains a trace-incompatible
condensate burden. The basic-support linear program fails, and the bounded
burden-preserving alternative portfolio exhausts its assigned work without a
certificate even though the element-budget gate passes. Spending the entire
shared work budget on those alternatives would leave no work for the
physically appropriate lower-dimensional face.

ExoGibbs now bounds the alternative portfolio so that work remains for one
generic support-release solve and the ordinary inactive-phase closure. The
first deterministic full-rank alternative supplies only the release
initializer. A mixed positive-log/signed-linear solve releases support
`(1, 4)` to `(1,)`; ordinary closure then adds phase 8 and certifies final
support `(1, 8)`, corresponding to `SiO2(s,l)` and `MgSiO3(s,l)`. Final
acceptance still requires the unchanged full and caller-gauge physical audits
within the original shared work limit. This route is selected from support
geometry, portfolio outcome, and available budget, not from a species name,
temperature, or pressure. The exact state is retained as the hard passing
`test_resolved_default_column_step_774_support_release` opt-in regression.

## Resolved ExoGibbs optimizer-directed-release boundary

The next unchanged CUDA default run passed step 774 and stopped at pressure
step 999:

- `T = 203.06986826073876 K`;
- `P = 8.7214641233652035 bar`;
- normalized Mg inventory `= 1.3681948091591687e-93`; and
- normalized Si inventory `= 4.137051394836369e-84`.

At this state, the burden-preserving basic-support and alternative-basis
searches do not reach a local root. A normalized-linear alternative solve has
a negative amount on one phase. ExoGibbs uses that sign only to select the
original non-negative builder basis as the source for its bounded proper-face
portfolio; rejected terminal amounts are never adopted. The mixed
positive-log/signed-linear face solve reaches support `(1, 8)` and returns to
the ordinary inactive-phase closure and unchanged physical audits. The exact
state is retained as the hard passing
`test_resolved_default_column_step_999_optimizer_directed_release` opt-in
regression.

## Resolved ExoGibbs inventory-bridge boundary

The following unchanged default run passed step 999 and stopped at pressure
step 1082:

- `T = 157.89357053396711 K`;
- `P = 3.7871329378560565 bar`;
- normalized Mg inventory `= 6.831754190721877e-112`; and
- normalized Si inventory `= 8.328995133274878e-119`.

The direct gas-only warm solve cannot enter the basin of the physical support
`(9,)`. Uniform continuation in temperature, pressure, and inventory was not
adopted because success was not monotone in its step size. Instead, ExoGibbs
uses the accepted source metadata to construct one inventory midpoint at the
exact target temperature and pressure. A row that is positive at both
endpoints is interpolated logarithmically; a row with a zero endpoint is
interpolated linearly.

The midpoint reaches support `(1, 8)` and must pass the ordinary lifecycle and
floorless budget certification. Only its gas state initializes an exact-target
retry, which reaches support `(9,)` and passes the same final audits. The
midpoint condensates, support, and proposed rainout inventory are discarded;
rainout propagation occurs only once at the exact endpoint. The bridge is
limited to two lifecycle calls, contains no species-specific condition, and
falls back cold if either stage is rejected. The exact state is retained as
the hard passing `test_default_column_step_1082_inventory_bridge` opt-in
regression.

## Resolved post-bridge continuation boundaries

Unchanged full-column attempts after the inventory-bridge change exposed eight
additional exact warm-parent boundaries. The pressure-step indices are grid
positions, not the chronological order in which the reruns exposed them. They
are retained as hard public provider regressions:

| Step | Candidate contract | Certified support |
| ---: | --- | --- |
| 1075, 1084 | finite gas log ratios survive linear underflow and target-inventory regauging | `(9,)` |
| 1076 | positive boundary face remains in the deterministic basic-support candidates | `(9,)` |
| 1077 | first convective warm-parent candidate after committed layer 1076 | `(9,)` |
| 1186 | convective sibling-basis transition after committed layer 1185 | `(5, 8)` |
| 1342 | non-convective warm-parent candidate after committed layer 1341 | `(1, 5)` |
| 1372 | convective candidate with a nonzero binary64-subnormal Si inventory | `(5, 1)` |
| 1383 | conditional exact closure from empty condensate support after committed layer 1382 | `(5,)` |

The tuple order is the provider's returned order; steps 1342 and 1372 have the
same support set. The fixtures preserve the accepted parent gas logs and source
provenance. They do not carry parent condensate amounts, support, or a rainout
proposal into the target solve.

At step 1372, `T = 69.84723203223899 K`,
`P = 0.20536053330940687 bar`, and the normalized Si target is
`9.108388204e-314`. The previously returned linear candidate missed this
nonzero target by about 4.83%, so the floorless rainout audit rejected it.
ExoGibbs now audits every nonzero budget row relative to its exact caller-gauge
target, including binary64-subnormal values, while retaining a scaled absolute
residual for the exactly zero charge row. A failed linear physical audit may
use the existing reduced-log-domain support search, but only a final exact
physical certificate can be accepted.

The corrected public step-1372 regression passes on CPU and CUDA with support
`(5, 1)`. Its Si reconstruction is bit-exact and its maximum floorless relative
budget residual is `1.699e-12`.

At step 1383, a gas-only candidate begins with empty condensate support. If its
caller-gauge physical audit fails, ExoGibbs conditionally runs the existing
bounded exact active-set closure from that empty support. Favorable phases may
be added, but the unchanged full physical and caller-gauge audits remain the
only acceptance authority. This generic, species-independent path closes on
support `(5,)`.

## Next ExoFamily milestone

- establish pressure-grid convergence for the completed fixed-boundary
  column; and
- quantify sensitivity to the provisional transport closure before treating
  the raccoon-like radii as scientific predictions.

## Deferred beyond fixed-column validation

- fixed-column parameter sweeps and any mass/temperature shooting;
- non-ideal density, EOS heat capacities, and fugacity experiments;
- magma--gas generation of the basal Mg--Si--O inventory;
- exact 10-bar ExoJAX stitching and spectra; and
- phase-aware retrieval.

Promotion of the structure coupler out of ExoExamples still requires a second
consumer or shared structure physics beyond this workflow.
