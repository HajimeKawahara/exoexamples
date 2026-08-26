project = "ExoExamples"
copyright = "2026, ExoExamples contributors"
author = "ExoExamples contributors"
release = "development"

language = "ja"
root_doc = "index"
extensions = ["sphinx.ext.mathjax"]
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]
nitpicky = True

html_theme = "sphinx_rtd_theme"
html_title = "ExoExamples — 日本語"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["meltyq/meltyq_figure8_forward_comparison_ja.ipynb"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "sticky_navigation": True,
}

linkcheck_report_timeouts_as_broken = False
linkcheck_timeout = 20
linkcheck_retries = 2
