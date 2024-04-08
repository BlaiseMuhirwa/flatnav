# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 


# Make sure path to Python bindings is in the system path
PYTHON_LIB_PATH = os.path.join(os.getcwd(), "..", "flatnav_python", "build")
if not os.path.exists(PYTHON_LIB_PATH):
    raise FileNotFoundError(f"Python bindings not found at {PYTHON_LIB_PATH}."
                            f"This must be generated before building the documentation.")
    
sys.path.insert(0, PYTHON_LIB_PATH)


project = 'FlatNav'
copyright = '2024, Benjamin Ray Coleman, Blaise Munyampirwa, Vihan Lakshman'
author = 'Benjamin Ray Coleman, Blaise Munyampirwa, Vihan Lakshman'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "breathe",
]

# Configure Breathe to find the Doxygen-generated XML files for the C++ code
breathe_projects = {
    "MyProject": "xml"
}
breathe_default_project = "MyProject"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
