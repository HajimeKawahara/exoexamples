# Rocky Raccoon published temperature reference

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

Run the extraction from the repository root with:

```console
python docs/rocky_raccoon/extract_rocky_raccoon_reference.py \
  refs/2608.24873v1.pdf \
  docs/rocky_raccoon/data/rocky_raccoon_temperature_vector_reference.csv
```

The source PDF SHA-256 is
`873084cecc24a22c0cee5fb4fdce5146ea9b88178137b06f3f931382744bae17`.
The committed CSV SHA-256 is
`21215c3ec184d42c28e1569f53247ee3bb14b0cc67ff236a00b73f94f270c8f9`.
The accompanying JSON records the cases, paper radii, provenance, and row
counts.

Figure 5 left is not duplicated in the CSV because the paper explicitly
identifies it as the same fiducial model shown in Figure 2 left. The three
stored profiles therefore represent the oxygen-poor, oxygen-rich, and
oxygen-poor-with-SiO(s) calculations exactly once.

The visible SVG vertices can be slightly simplified by the publication
pipeline. No interpolation is added, and the data should be used for visual
or shape comparisons rather than likelihood calculations. Gas and condensate
curves are deliberately excluded: the PDF has colored legend text, but it
does not provide a machine-readable binding from every fragmented vector path
to a species name. Assigning those paths without source data would introduce
manual, case-specific assumptions.
