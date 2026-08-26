SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?= -W --keep-going -n
PYTHON ?= python

.PHONY: html html-ja html-en

html: html-ja html-en
	$(PYTHON) meltyq/check_rendered_html.py

html-ja:
	$(MAKE) -C ja html SPHINXBUILD="$(SPHINXBUILD)" SPHINXOPTS="$(SPHINXOPTS)"

html-en:
	$(MAKE) -C en html SPHINXBUILD="$(SPHINXBUILD)" SPHINXOPTS="$(SPHINXOPTS)"
