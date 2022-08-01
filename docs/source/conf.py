# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import datetime
import ocnn


# -- Project information

project = 'ocnn-pytorch'
author = 'Peng-Shuai Wang'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)
release = ocnn.__version__
version = ocnn.__version__


# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

autosummary_generate = True
templates_path = ['_templates']

source_suffix = '.rst'
master_doc = 'index'

# doctest_default_flags = doctest.NORMALIZE_WHITESPACE
autodoc_member_order = 'bysource'
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://pytorch.org/docs/master', None),
}


# -- Options for HTML output

html_logo = '_static/img/ocnn.png'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
    'navigation_depth': 2,
}


# -- Options for EPUB output

epub_show_urls = 'footnote'


# -- Misc

add_module_names = False


def setup(app):
  def skip(app, what, name, obj, skip, options):
    members = [
        '__init__',
        '__repr__',
        '__weakref__',
        '__dict__',
        '__module__',
    ]
    return True if name in members else skip

  def rst_jinja_render(app, docname, source):
    src = source[0]
    rst_context = {'ocnn': ocnn}
    rendered = app.builder.templates.render_string(src, rst_context)
    source[0] = rendered

  app.connect('autodoc-skip-member', skip)
  app.connect("source-read", rst_jinja_render)
