# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from recommonmark.parser import CommonMarkParser

# Add the Markdown parser.
source_parsers = {
    '.md': CommonMarkParser,
}

# Add '.md' to source suffixes.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

project = 'FlatNav'
copyright = '2024, Benjamin Ray Coleman, Blaise Munyampirwa, Vihan Lakshman'
author = 'Benjamin Ray Coleman, Blaise Munyampirwa, Vihan Lakshman'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
    "sphinx.ext.autodoc",
    "myst_parser"
]

# This is rendering the markdown files.
# Some of these might not be necessary though. I copied them from the official documentation.
# See here: https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


# Configure Breathe to find the Doxygen-generated XML files for the C++ code
breathe_projects = {
    "FlatNav": "./doxygen_output/xml"
}
breathe_default_project = "FlatNav"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Automatically add type annotations to the generated function signatures and descriptions. This
# means we don't have to manually add :type: annotations into the docstrings.
autodoc_typehints = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
