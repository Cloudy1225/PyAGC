pyagc.metrics
=============

.. contents:: Contents
    :local:

The :mod:`pyagc.metrics` package provides a unified and optimized suite of evaluation metrics for attributed graph clustering.
It includes both **label-based metrics** that compare predicted clusters against ground-truth labels, and **structural metrics** that evaluate clustering quality based on the graph topology alone.

.. code-block:: python

    from pyagc.metrics import label_metrics, structure_metrics

    # Compute label-based metrics:
    label_results = label_metrics(y_true, y_pred, metrics=('NMI', 'ARI', 'ACC', 'F1'))
    # >>> {'NMI': 0.85, 'ARI': 0.72, 'ACC': 0.89, 'F1': 0.87}

    # Compute structural metrics:
    struct_results = structure_metrics(edge_index, y_pred, metrics=('Mod', 'Cond'))
    # >>> {'Mod': 0.45, 'Cond': 0.32}

Label-Based Metrics
-------------------

.. currentmodule:: pyagc.metrics

Label-based metrics require ground-truth cluster labels and measure the agreement between predicted assignments and the true partition.
For metrics that require label alignment (*e.g.*, accuracy and Macro-F1), the predicted clusters are automatically aligned with the true labels using the **Hungarian algorithm** (`linear_sum_assignment <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html>`_) to account for arbitrary label permutations.

The unified entry point :func:`label_metrics` supports computing any combination of the following metrics in a single call:

- **NMI** — Normalized Mutual Information (`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html>`__)
- **ARI** — Adjusted Rand Index (`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html>`__)
- **Homo** — Homogeneity Score (`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html>`__)
- **Comp** — Completeness Score (`sklearn <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html>`__)
- **ACC** — Clustering Accuracy (with Hungarian matching)
- **F1** — Macro-F1 Score (with Hungarian matching)

.. code-block:: python

    from pyagc.metrics import label_metrics

    # Compute all default metrics (NMI, ARI, ACC, F1):
    results = label_metrics(y_true, y_pred)

    # Compute a specific subset:
    results = label_metrics(y_true, y_pred, metrics=('NMI', 'ACC'))

    # Single metric (pass as string):
    results = label_metrics(y_true, y_pred, metrics='NMI')

.. note::
    Both :class:`torch.Tensor` and :class:`numpy.ndarray` inputs are supported.
    Tensors are automatically detached and moved to CPU before computation.

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   label_metrics

.. autofunction:: label_metrics

Structural Metrics
------------------

.. currentmodule:: pyagc.metrics

Structural metrics evaluate the quality of a graph clustering using only the graph topology and the predicted partition, **without requiring ground-truth labels**.
These metrics are particularly useful for evaluating clustering on graphs where ground-truth communities are unavailable, or for complementing label-based evaluation with topology-aware measures.

The unified entry point :func:`structure_metrics` supports computing any combination of the following metrics:

- **Mod** — `Modularity <https://en.wikipedia.org/wiki/Modularity_(networks)>`_: Measures the fraction of edges within communities minus the expected fraction under a random null model. Higher values indicate stronger community structure.
- **Cond** — `Conductance <https://en.wikipedia.org/wiki/Conductance_(graph)>`_: Measures the ratio of inter-cluster edges to total edges. Lower values indicate better-separated communities.

Both metrics support a **vectorized** mode (enabled by default) for efficient computation on large graphs, as well as a loop-based fallback for debugging and validation.

.. code-block:: python

    from pyagc.metrics import structure_metrics, modularity, conductance

    # Compute all structural metrics via the unified interface:
    results = structure_metrics(edge_index, y_pred, metrics=('Mod', 'Cond'))

    # Compute individual metrics directly:
    mod = modularity(edge_index, y_pred)
    cond = conductance(edge_index, y_pred)

    # Use loop-based computation (useful for debugging):
    results = structure_metrics(edge_index, y_pred, vectorized=False)

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   structure_metrics
   modularity
   conductance

.. autofunction:: structure_metrics

.. autofunction:: modularity

.. autofunction:: conductance
