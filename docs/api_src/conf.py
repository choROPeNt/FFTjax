import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))
os.environ.setdefault("JAX_ENABLE_X64", "1")

project = "FFTjax"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_markdown_builder",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"

master_doc = "index"

# headless: no HTML theme needed, this build only ever runs with `-b markdown`
html_theme = "alabaster"
