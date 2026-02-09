pyagc.clusters
==============

.. contents:: Contents
    :local:

The :mod:`pyagc.clusters` package provides modular **Cluster Head** implementations for attributed graph clustering.
Following the :doc:`Encode-Cluster-Optimize <../tutorial/eco_framework>` framework, a cluster head transforms latent node embeddings :math:`\mathbf{Z} \in \mathbb{R}^{N \times H}` into a cluster assignment matrix :math:`\mathbf{P} \in \mathbb{R}^{N \times K}`:

.. math::
    \mathbf{P} = \mathcal{C}(\mathbf{Z}; \Theta_{\mathcal{C}})

All cluster heads inherit from :class:`~pyagc.clusters.BaseClusterHead`, ensuring a consistent interface for clustering, loss computation, and assignment retrieval.
This modular design allows any encoder backbone to be easily paired with different clustering mechanisms — simply swap the cluster head to isolate the effect of the clustering objective:

.. code-block:: python

    from pyagc.clusters import DECClusterHead, DMoNClusterHead, KMeansClusterHead

    # Differentiable prototypical clustering:
    dec_head = DECClusterHead(n_clusters=7, n_features=256)

    # Differentiable discriminative clustering:
    dmon_head = DMoNClusterHead(n_clusters=7, n_features=256)

    # Post-hoc discrete clustering:
    kmeans_head = KMeansClusterHead(n_clusters=7, backend="torch")

Based on the differentiability of the cluster projection, which dictates the feasibility of end-to-end optimization, the cluster heads are organized into two categories:

- **Differentiable Cluster Heads** treat clustering as a differentiable layer within a neural network, allowing gradients to flow from the clustering loss back to the encoder. This includes discriminative approaches (*e.g.*, :class:`~pyagc.clusters.DMoNClusterHead`, :class:`~pyagc.clusters.MinCutClusterHead`, :class:`~pyagc.clusters.NeuromapClusterHead`) and prototypical approaches (*e.g.*, :class:`~pyagc.clusters.DECClusterHead`, :class:`~pyagc.clusters.DinkClusterHead`, :class:`~pyagc.clusters.INCClusterHead`).

- **Discrete Cluster Heads** apply a non-differentiable clustering algorithm to fixed representations, decoupling clustering from representation learning (*e.g.*, :class:`~pyagc.clusters.KMeansClusterHead`).

Base Class
----------

.. currentmodule:: pyagc.clusters

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   BaseClusterHead

.. autoclass:: BaseClusterHead
   :members:

Differentiable Cluster Heads
----------------------------

.. currentmodule:: pyagc.clusters

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   DECClusterHead
   DinkClusterHead
   DMoNClusterHead
   INCClusterHead
   MinCutClusterHead
   NeuromapClusterHead
   SBMClusterHead
   SBMMatchClusterHead

.. autoclass:: DECClusterHead
   :members:

.. autoclass:: DinkClusterHead
   :members:

.. autoclass:: DMoNClusterHead
   :members:

.. autoclass:: INCClusterHead
   :members:

.. autoclass:: MinCutClusterHead
   :members:

.. autoclass:: NeuromapClusterHead
   :members:

.. autoclass:: SBMClusterHead
   :members:

.. autoclass:: SBMMatchClusterHead
   :members:

Discrete Cluster Heads
----------------------

.. currentmodule:: pyagc.clusters

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   KMeansClusterHead

.. autoclass:: KMeansClusterHead
   :members:

GPU-Accelerated KMeans
----------------------

To overcome the computational bottleneck of CPU-based KMeans when :math:`N > 10^6`, :mod:`pyagc.clusters` provides GPU-accelerated KMeans implementations using PyTorch and `OpenAI Triton <https://github.com/triton-lang/triton>`_, achieving multi-fold speedups over CPU baselines:

.. code-block:: python

    from pyagc.clusters import TorchKMeans, TritonKMeans

    # PyTorch-based GPU KMeans:
    kmeans = TorchKMeans(n_clusters=7, metric="euclidean", max_iter=300)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

    # Triton-accelerated GPU KMeans (requires CUDA):
    kmeans = TritonKMeans(n_clusters=7, metric="euclidean", max_iter=300)
    kmeans.fit(embeddings)
    labels = kmeans.labels_

.. currentmodule:: pyagc.clusters

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TorchKMeans
   TritonKMeans

.. autoclass:: TorchKMeans
   :members:

.. autoclass:: TritonKMeans
   :members:
