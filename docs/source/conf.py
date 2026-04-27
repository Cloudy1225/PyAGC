import datetime
import os.path as osp
import sys

import pyg_sphinx_theme
from pyg_sphinx_theme.extension.logo import logo_role

# Add pyagc to path
sys.path.insert(0, osp.abspath('../..'))
import pyagc

author = 'PyAGC Team'
project = 'pyagc'
version = pyagc.__version__
copyright = f'{datetime.datetime.now().year}, {author}'

sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'nbsphinx',
    'pyg',
]

# -- Mock imports --
autodoc_mock_imports = [
    # Triton（Needs GPU）
    'triton',
    'triton.language',
]

html_theme = 'pyg_sphinx_theme'
html_title = 'PyAGC Documentation'
html_short_title = 'PyAGC'

html_logo = '_static/img/logo.png'
html_favicon = '_static/img/favicon.png'
html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    'canonical_url': 'https://pyagc.readthedocs.io/',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

add_module_names = False
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
    'show-inheritance': True,
}

suppress_warnings = ['autodoc.import_object']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'torch_frame': ('https://pytorch-frame.readthedocs.io/en/latest/', None),
}

typehints_use_rtype = False
typehints_defaults = 'comma'


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, 'templates'):
        rst_context = {'pyagc': pyagc}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    """Setup sphinx application."""
    app.connect('source-read', rst_jinja_render)

    # Register custom inline-logo role
    app.add_role('pyagc', logo_role)

    # Custom CSS
    app.add_css_file('css/custom.css')

    # Do not drop type hints in signatures
    if 'autodoc-process-signature' in app.events.listeners:
        del app.events.listeners['autodoc-process-signature']
