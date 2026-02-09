# PyAGC Documentation

This directory contains the Sphinx documentation for PyAGC.

## Building Documentation

### Prerequisites

1. Install PyAGC from source:
```bash
cd ..
pip install -e .
```

2. Install Sphinx and dependencies:
```bash
pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
pip install sphinx sphinx-autodoc-typehints sphinx-copybutton nbsphinx
```

### Build HTML Documentation

```bash
cd docs
make html
```

The documentation will be available at `build/html/index.html`.

### Build PDF Documentation (Optional)

```bash
make latexpdf
```

### Live Preview

For live preview during development:

```bash
pip install sphinx-autobuild
make livehtml
```

Then open http://127.0.0.1:8000 in your browser.

## Documentation Structure

```
docs/
├── source/
│   ├── _static/          # Static files (CSS, images)
│   ├── _templates/       # Custom templates
│   ├── notes/           # Getting started guides
│   ├── tutorial/        # Step-by-step tutorials
│   ├── modules/         # API reference
│   ├── conf.py          # Sphinx configuration
│   └── index.rst        # Main page
├── Makefile             # Unix build commands
├── make.bat             # Windows build commands
└── README.md           # This file
```

## Contributing to Documentation

### Adding New Tutorials

1. Create a new `.rst` file in `source/tutorial/`
2. Add it to the toctree in `source/index.rst`
3. Follow the existing tutorial format

### Adding API Documentation

API docs are auto-generated from docstrings. To add documentation for a new module:

1. Ensure all classes/functions have proper docstrings
2. Add the module to the appropriate file in `source/modules/`
3. Rebuild the documentation

### Docstring Format

We use NumPy-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description of function.
    
    Longer description with more details.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
    
    Returns
    -------
    result : float
        Description of return value
    
    Examples
    --------
    >>> result = my_function(5, "test")
    >>> print(result)
    42.0
    """
    pass
```

## Style Guidelines

### ReStructuredText (RST) Basics

**Headings:**
```rst
Main Title
==========

Section
-------

Subsection
~~~~~~~~~~
```

**Code Blocks:**
```rst
.. code-block:: python

    import pyagc
    data = pyagc.data.get_dataset('Cora')
```

**Lists:**
```rst
- Item 1
- Item 2
  - Nested item
```

**Links:**
```rst
`External Link <https://example.com>`_
:doc:`Internal Link <path/to/doc>`
:class:`pyagc.models.DAEGC`
```

**Admonitions:**
```rst
.. note::
    This is a note.

.. warning::
    This is a warning.

.. tip::
    This is a tip.
```

## Troubleshooting

### Common Issues

**1. Module import errors**

Make sure PyAGC is installed in editable mode:
```bash
pip install -e ..
```

**2. Theme not found**

Install the PyG theme:
```bash
pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
```

**3. autodoc warnings**

These are usually caused by missing dependencies or import errors in the source code. Check that all optional dependencies are installed.

**4. Build errors on Windows**

Use `make.bat` instead of `make`:
```cmd
make.bat html
```

## Deployment

Documentation is automatically built and deployed to Read the Docs on each commit to the main branch.

### Manual Deployment

To deploy to GitHub Pages:

```bash
make html
cd build/html
git init
git add -A
git commit -m "Deploy documentation"
git push -f git@github.com:Cloudy1225/PyAGC.git main:gh-pages
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)

## Contact

For questions about the documentation:
- Open an issue: https://github.com/Cloudy1225/PyAGC/issues
- Email: pyagc-dev@example.com
```

## 23. Requirements File for Documentation (`docs/requirements.txt`)

```txt
# Sphinx and extensions
sphinx>=5.0.0
sphinx-autodoc-typehints>=1.19.0
sphinx-copybutton>=0.5.0
nbsphinx>=0.9.0
sphinx-autobuild>=2021.3.14

# PyG theme
git+https://github.com/pyg-team/pyg_sphinx_theme.git

# Core dependencies
torch>=2.0.0
torch-geometric>=2.4.0

# Optional dependencies for examples
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.4.0

# For building PDF (optional)
# latexmk
# texlive-latex-extra
```

## 24. GitHub Actions Workflow for Documentation (`.github/workflows/docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
        pip install -r docs/requirements.txt
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Check for warnings
      run: |
        cd docs
        make html SPHINXOPTS="-W --keep-going"
    
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/build/html/
    
    - name: Deploy to GitHub Pages
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build/html
```

## 25. ReadTheDocs Configuration (`.readthedocs.yaml`)

```yaml
# .readthedocs.yaml
# Read the Docs configuration file

version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.9"
  jobs:
    post_install:
      - pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - dev
    - requirements: docs/requirements.txt

sphinx:
  configuration: docs/source/conf.py
  fail_on_warning: false

formats:
  - pdf
  - epub
```

## Summary

This comprehensive documentation setup provides:

1. **Complete API Reference**: Auto-generated from docstrings for all modules
2. **Tutorials**: Step-by-step guides for common tasks
   - Quickstart
   - ECO Framework
   - Custom encoders
   - Custom cluster heads
   - Scalability techniques

3. **Installation Guide**: Multiple installation methods
4. **Resources**: Papers, datasets, tools, and community links
5. **Professional Theme**: Using PyG Sphinx theme for consistency
6. **CI/CD**: Automatic building and deployment
7. **Searchable**: Full-text search functionality
8. **Responsive**: Works on mobile and desktop

To build the documentation:

```bash
cd docs
pip install git+https://github.com/pyg-team/pyg_sphinx_theme.git
pip install -r requirements.txt
make html
```

Then open `docs/build/html/index.html` in your browser.