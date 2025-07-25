"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# pylint: disable=redefined-builtin,invalid-name
# mypy: ignore-errors
from os import environ, listdir
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

# -- Project information -----------------------------------------------------

project = 'BOOMER'
copyright = '2020-2025, Michael Rapp et al.'
author = 'Michael Rapp et al.'

# The full version, including alpha/beta/rc tags
release = environ.get('PROJECT_VERSION', 'n/a')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx_design',
    'sphinxcontrib.spelling',
    'sphinxext.opengraph',
    'sphinx_inline_tabs',
    'sphinx_copybutton',
    'sphinx_favicon',
]

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

# Favicons
favicons = [{'href': 'favicon.svg'}]

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Aliases for external links
git_branch = environ.get('GIT_BRANCH', 'main')
builder = environ.get('SPHINX_BUILDER', 'html')

if builder == 'linkcheck':
    # Use GitHub's REST API when using the builder "linkcheck", because it comes with a less restrictive rate limit
    github_file_url = 'https://api.github.com/repos/mrapp-ke/MLRL-Boomer/contents/%s?ref=' + git_branch
    github_dir_url = github_file_url
else:
    github_file_url = 'https://github.com/mrapp-ke/MLRL-Boomer/blob/' + git_branch + '/%s'
    github_dir_url = 'https://github.com/mrapp-ke/MLRL-Boomer/tree/' + git_branch + '/%s'

extlinks = {
    'repo-file': (github_file_url, '%s'),
    'repo-dir': (github_dir_url, '%s'),
}

# Authentication headers to be used by builder "linkcheck"
linkcheck_request_headers = {}
github_token = environ.get('GITHUB_TOKEN')

if github_token:
    linkcheck_request_headers[r'^https://api.github.com/'] = {'Authorization': 'Token ' + github_token}

# Breathe configuration
breathe_projects = {
    file: Path('developer_guide', 'api', 'cpp', file, 'xml')
    for file in listdir(Path(__file__).resolve().parent.parent / 'cpp' / 'subprojects')
}
breathe_default_project = next(iter(breathe_projects.keys()))

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**/*.template.md']

# Suppress certain warnings.
suppress_warnings = [
    'duplicate_declaration.cpp',  # Suppress warnings about duplicate C++ declarations, such as namespaces
    'toc.not_included',  # Suppress warnings about files only included via {include} not being included in any toctree
]

# Exclude the following Python modules from the API documentation.
autodoc_mock_imports = [
    'sklearn',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'furo'
html_title = project + ' ' + release

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# A list of paths that contain extra files not directly related to the
# documentation, such as robots.txt or .htaccess. Relative paths are taken
# as relative to the configuration directory. They are copied to the output
# directory. They will overwrite any existing file of the same name.
html_extra_path = []
