# MELTYQ Figure 8 vector reference

`meltyq_figure8_vector_reference.csv` contains the visible vector vertices of
the six curves panels in Figure 8 of:

Yuichi Ito and Quentin Changeat, "Coupling Magma--Ocean and Atmospheres in
Spectral Retrievals of Sub-Neptunes," *The Astrophysical Journal* 1006:37
(2026), [doi:10.3847/1538-4357/ae6917](https://doi.org/10.3847/1538-4357/ae6917).
The corresponding source bundle was obtained from
[arXiv:2605.08752](https://arxiv.org/abs/2605.08752).

The source article is distributed under the
[Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).
The CSV was extracted from `f8a_dep_pmelt_t.pdf` through
`f8f_dep_tb_t.pdf` in the article's arXiv source using
`../extract_meltyq_figure8_reference.py`.

The values are digitized plot coordinates, not the authors' unpublished
numerical calculation table. Curves outside the displayed abundance range
(`log10(x) < -10`) are unavailable. The documentation therefore excludes
floor-censored values from numerical error summaries.

The committed CSV SHA-256 is
`b5865f8f8a1c1f05acf26d5d7f4c2ac463c677c9e0bf57fd8f06578cfbbe2665`.
The extractor intentionally validates the expected color, stroke width, and
16 visible traces per panel, because its geometry constants are specific to
this source artwork.

# MELTYQ Figure 3 raster reference

`meltyq_figure3_raster_reference.csv` contains a calibrated digitization of
the visible top panels in Figure 3 of the same paper.  Figure 3 is a raster PNG
even in the arXiv source bundle, so these values are measurements of published
artwork rather than vector vertices or an author-supplied numerical spectrum.
Every row therefore uses the contract
`published_raster_plot_digitization`, which is intentionally incompatible with
the `intrinsic_unoffset_model` contract accepted by the forward-model CLI.

The source file is `f3_Combined_k2-18b.png` from
[arXiv:2605.08752](https://arxiv.org/abs/2605.08752), with dimensions
3597 by 1494 pixels and SHA-256
`3ca19cbe480878a8bf67d022cbe2eb6f0caa14733187c6b929a057886575ebe7`.
The source article is distributed under the Creative Commons Attribution 4.0
License cited above.  Run the source-specific extractor with:

```console
python docs/meltyq/extract_meltyq_figure3_reference.py \
  /path/to/f3_Combined_k2-18b.png \
  docs/meltyq/data/meltyq_figure3_raster_reference.csv
```

Printed ticks calibrate the logarithmic wavelength, linear transit-depth,
linear temperature, and logarithmic pressure axes.  The solid black MELTYQ
trace is followed with a continuity-constrained dark-pixel path under three
transition penalties.  The CSV records algorithm spread and possible overlap
with black observational error bars for each sample.  Exact palette pixels
provide the visible CH4, H2O, CO, CO2, aerosol, and Rayleigh+CIA curves.  The
black temperature-profile centerline is measured independently by raster row.
Missing component samples mean that the curve is hidden, dashed between exact
palette pixels, or outside the plotted range; they do not imply zero opacity.

One horizontal pixel corresponds to an approximate resolving power of 885,
one vertical spectrum pixel to 0.702 ppm, one temperature pixel to 8.55 K,
and one pressure pixel to 0.0164 dex.  Black observational error bars obscure
many columns, especially at long wavelengths.  The data are suitable for a
provisional visual and shape discussion, not an R=50,000 pointwise comparison,
a likelihood calculation, or a claim of reproducing the unpublished best-fit
state.  The committed CSV SHA-256 is
`f1ceff4374f9974fa72295666c08d004e21178dbd14ea1dd70b7cb8e4104bd4a`.

`meltyq_figure3_public_ckd_model.csv` and
`meltyq_figure3_public_ckd_metadata.json` are a matched, checked-in snapshot of
the public CKD demonstration used by the bilingual raster comparison. They
make `../build_meltyq_figure3_raster_comparison.py` runnable from a clean
checkout; ignored files under `outputs/` are not implicit inputs. Replacing the
snapshot is an explicit paired operation using `--model-output` and
`--metadata`, as documented in `../../README.md`.
