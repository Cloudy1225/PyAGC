Installation
============

PyAGC is available for :python:`Python >= 3.10`, :pytorch:`PyTorch >= 2.6.0`, :pyg:`PyTorch Geometric >= 2.7.0`.

.. note::
   We do not recommend installation as a root user on your system :python:`Python`.
   Please setup a virtual environment, *e.g.*, via `venv <https://virtualenv.pypa.io/en/latest>`_, :conda:`null` `Anaconda/Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_, or create a `Docker image <https://www.docker.com/>`_.

Install from PyPI
-----------------

The easiest way to install PyAGC is via pip:

.. code-block:: bash

   pip install pyagc

Install from Source
-------------------

For the latest development version:

.. code-block:: bash

    git clone https://github.com/Cloudy1225/PyAGC.git
    cd PyAGC
    pip install -e .

With Optional Dependencies
---------------------------

For Triton-accelerated clustering:

.. code-block:: bash

    pip install pyagc[triton]

For full development setup including testing:

.. code-block:: bash

    pip install pyagc[dev]

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

    import pyagc
    print(pyagc.__version__)

    # Check available modules
    from pyagc import models, clusters, encoders
    print("PyAGC is successfully installed!")
