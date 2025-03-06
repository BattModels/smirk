# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from linkcode import linkcode_resolve

project = "smirk"
copyright = "2025, Alexius Wadell, Anoushka Bhutani, Venkat Viswanathan"
author = "Alexius Wadell, Anoushka Bhutani, Venkat Viswanathan"
release = f"v{importlib.metadata.version('smirk')}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinxarg.ext",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = []

viewcode_line_numbers = True

# -- MyST-NB -----------------------------------------------------------------
nb_execution_in_temp = True
nb_execution_mode = "cache"
nb_render_markdown_format = "myst"
myst_enable_extensions = [
    "fieldlist",
    "colon_fence",
]

# -- Autodoc ----------------------------------------------------------------
autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
templates_path = ["_templates"]
html_static_path = ["_static"]
html_theme = "furo"
html_logo = "_static/logo.svg"
html_copy_source = False
html_show_sourcelink = True
html_theme_options = {
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/BattModels/smirk",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
html_css_files = [
    "custom.css",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "tokenizers": ("https://huggingface.co/docs/tokenizers/main/en/", None),
}
