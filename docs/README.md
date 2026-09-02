# ExoExamples documentation

The Japanese master and English translation for MELTYQ are independent Sphinx
source trees:

- `ja/`: Japanese master
- `en/`: English translation

Both trees render the checked-in RST and figure assets generated from the
executed notebooks. The current Rocky Raccoon-like implementation note is
English-only under `en/rocky_raccoon/`. Building the documentation does not
execute notebooks or run a forward model.

The provisional Figure 3 comparison is also reproducible without ignored
runtime outputs. Its checked-in model spectrum and matching forward metadata
live under `meltyq/data/`. Regenerate the localized comparison figures and summary
with:

```console
python docs/meltyq/build_meltyq_figure3_raster_comparison.py
```

After deliberately producing a replacement public CKD run, update both
archives together by passing its model and metadata explicitly:

```console
python docs/meltyq/build_meltyq_figure3_raster_comparison.py \
  --model-output outputs/meltyq_figure3/model_spectra.csv \
  --metadata outputs/meltyq_figure3/metadata.json
```

The default command never reads or overwrites ignored files under `outputs/`.

Install the documentation dependencies and build both editions from the
repository root:

```console
python -m pip install -r docs/requirements.txt
make -C docs html
```

The entry points are:

- `docs/_build/html/ja/index.html`
- `docs/_build/html/en/index.html`

Build a single edition with `make -C docs html-ja` or
`make -C docs html-en`.

MELTYQ-specific support code and data live under `docs/meltyq/`; localized
sources and generated figures live under `docs/ja/meltyq/` and
`docs/en/meltyq/`. Regenerate an RST only after executing its corresponding
notebook with the commands recorded near the end of that notebook.
