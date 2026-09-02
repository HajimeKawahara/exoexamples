# Rocky Raccoon published vector references

`rocky_raccoon_temperature_vector_reference.csv` contains the visible
temperature-profile vertices from both panels of Figure 2 and the right panel
of Figure 5 in:

William Misener et al., "Magnesium silicate condensation in sub-Neptune
envelopes: the fundamental link between chemistry, structure, and
observables," arXiv:2608.24873v1,
[doi:10.48550/arXiv.2608.24873](https://doi.org/10.48550/arXiv.2608.24873).

The PDF identifies the embedded vector figures as
`MgSiCOH_comparison.pdf` and `MgSiCOH_comparison_wSiO.pdf`. It contains no
attached machine-readable table, and the article gives no data or code
repository. The committed CSV is therefore a measurement of the published
vector artwork, not the authors' unpublished model output.

The source article is distributed under the
[Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).
The extractor converts paper pages 8 and 11 to SVG with `pdftocairo`, selects
the temperature paths by panel geometry and the exact blue stroke, and maps
their coordinates through the printed pressure and temperature ticks. Thin
and thick paths are recorded separately as `non_convective` and `convective`,
respectively. It rejects unexpected path counts and line-width sequences so a
changed or incompatible PDF fails visibly.

`rocky_raccoon_gas_vector_reference.csv` contains the visible gas mixing-ratio
polylines from the same three calculations.  Each of the 13 legend species has
a unique exact RGB value in the vector PDF, so species identity does not rely
on curve position or manual tracing.  Line width retains the paper's
`convective` and `non_convective` transport labels.  Pressure and
log10(mixing ratio) are calibrated from the printed axis ticks.

Run the extraction from the repository root with:

```console
python docs/rocky_raccoon/extract_rocky_raccoon_reference.py \
  refs/2608.24873v1.pdf \
  docs/rocky_raccoon/data/rocky_raccoon_temperature_vector_reference.csv \
  --gas-output-csv \
  docs/rocky_raccoon/data/rocky_raccoon_gas_vector_reference.csv
```

The source PDF SHA-256 is
`873084cecc24a22c0cee5fb4fdce5146ea9b88178137b06f3f931382744bae17`.
The committed CSV SHA-256 is
`21215c3ec184d42c28e1569f53247ee3bb14b0cc67ff236a00b73f94f270c8f9`.
The committed gas CSV SHA-256 is
`647264a7c4c9a075ef98f12caa5231ac3d70147812c509c73fb3ece35872e89d`.
The accompanying JSON files record the cases, exact species-color mapping,
paper radii, provenance, and row counts.

Figure 5 left is not duplicated in the CSV because the paper explicitly
identifies it as the same fiducial model shown in Figure 2 left. The three
stored profiles therefore represent the oxygen-poor, oxygen-rich, and
oxygen-poor-with-SiO(s) calculations exactly once.

The visible SVG vertices can be slightly simplified by the publication
pipeline. No scientific interpolation between sampled vertices is added, and
the data should be used for visual or shape comparisons rather than likelihood
calculations.  For gas curves, the only added points are geometric
intersections where a straight vector segment crosses the visible plot
boundary.  The gas plot floor is
log10(mixing ratio) = -18: a point on this floor is censored rather than an
equality measurement.  A curve that leaves and later re-enters the visible
rectangle is stored as separate runs, and the extractor never bridges the
hidden interval.  A species absent from a panel therefore means only that no
curve is visible above the plot floor, not that its abundance is zero.

Condensate curves remain excluded because their colors are reused between the
gas and condensate panels and the fragmented paths do not provide an equally
complete, unique species binding.

## Comparison contract

The comparison figure draws the model gas curves as solid lines and the visible
published vector segments as dashed lines. This absolute mixing-ratio overlay
is diagnostic rather than a like-for-like residual: the model is normalized by
the sum over its explicit solver gases, while the paper's total also includes
neutral atomic gases. Atomic H is consequently shown only on the paper side.

For a denominator-independent molecular comparison, the primary metric for
each shared molecule other than H2 is
`log10[(x_i/x_H2)_model] - log10[(x_i/x_H2)_paper]`. It is evaluated only where
the published species and H2 are visible and the model numerator is at least
`1e-18`. Floor contacts, lower model values, and hidden gaps remain censored,
and no metric interpolation crosses them. The report stores both the full
paper-visible coverage and the jointly visible fraction so excluded regions
remain explicit.

The H2-relative ratio cancels the different total-gas denominators, but it does
not make the calculations identical-input chemistry tests. Their temperature
profiles differ, their absolute basal abundances are not verified to match,
and their boundary closures differ. Reported residuals therefore measure the
mismatch of the complete present closure, not an isolated ExoGibbs chemistry
error or a paper-reproduction error.
