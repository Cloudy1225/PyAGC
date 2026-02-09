:github_url: https://github.com/Cloudy1225/PyAGC

PyAGC Documentation
===================

:pyagc:`null` **PyAGC** (PyTorch Attributed Graph Clustering) is a comprehensive, modular library for attributed graph clustering built on :pytorch:`null` `PyTorch <https://pytorch.org>`_ and :pyg:`null` `PyTorch Geometric <https://www.pyg.org/>`_. It provides a unified framework for implementing, evaluating, and comparing state-of-the-art graph clustering algorithms at scale.

.. image:: https://img.shields.io/pypi/v/pyagc.svg
   :target: https://pypi.org/project/pyagc/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/Cloudy1225/PyAGC/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/Cloudy1225/PyAGC.svg?style=social
   :target: https://github.com/Cloudy1225/PyAGC
   :alt: GitHub Stars

Key Features
------------

- 📊 **Diverse Dataset Collection**: A diverse benchmark spanning 5 orders of magnitude across multiple domains. Features both academic benchmarks and real-world industrial datasets with heterogeneous attributes and varying structural properties.
- 🧩 **Unified Algorithm Framework**: Implements 20+ SOTA AGC methods unified under the Encode-Cluster-Optimize framework. Covers the full spectrum from traditional approaches to cutting-edge deep learning methods, with modular components enabling easy experimentation and method composition.
- 📏 **Holistic Evaluation Protocol**: Goes beyond standard supervised metrics by incorporating unsupervised structural quality metrics and comprehensive efficiency profiling. Addresses the real-world scenario where ground-truth labels are unavailable.
- 🚀 **Production-Grade Scalability**: Breaks the scalability barrier with GPU-accelerated clustering and mini-batch training support. Successfully scales deep clustering methods to graphs with 111 million nodes on a single 32GB GPU, making industrial deployment feasible.
- 🛠️ **Developer-Friendly Design**: Built on PyTorch and PyTorch Geometric with a clean, modular architecture. Features plug-and-play encoders, cluster heads, and optimization strategies. Configuration-driven experiments via YAML files ensure full reproducibility.
- 📖 **Complete Documentation & Reproducibility**: Provides extensive documentation, standardized data loaders, unified preprocessing pipelines, and reproducible experiment configurations. Open-source codebase with detailed tutorials enabling researchers and practitioners to quickly prototype, benchmark, and deploy AGC solutions.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   notes/installation
   notes/introduction

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial/quickstart
   tutorial/eco_framework
   tutorial/custom_cluster_head
   tutorial/scalability

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   modules/root
   modules/clusters
   modules/data
   modules/encoders
   modules/metrics
   modules/models
   modules/transforms
   modules/utils

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
