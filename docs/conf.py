"""Sphinx configuration for the everwillow documentation."""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "everwillow"
author = "everwillow developers"
copyright = f"{_dt.datetime.now().year}, {author}"

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}
autodoc_member_order = "bysource"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Enable Mermaid diagrams
myst_fence_as_directive = ["mermaid"]

nb_execution_mode = "off"

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "../images/logo.svg"
html_theme_options = {
    "home_page_in_toc": True,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "repository_url": "https://github.com/MoAly98/everwillow",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}
html_context = {"default_mode": "light"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "python/"]

html_title = " "  # everwillow documentation""
html_baseurl = "https://everwillow.readthedocs.io/"


def _skip_data_attributes(_app, what, _name, obj, skip, _options):
    """Skip raw class data attributes — already documented via Attributes: sections."""
    if skip:
        return True
    if what == "class" and not callable(obj):
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_data_attributes)
