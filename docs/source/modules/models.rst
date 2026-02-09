pyagc.models
============

.. contents:: Contents
    :local:

The :mod:`pyagc.models` module implements the high-level model classes that
orchestrate the interaction between encoders and cluster heads within the
:doc:`Encode-Cluster-Optimize <../tutorial/eco_framework>` framework. Each model defines its own
optimization strategy, managing the forward pass and the computation of
joint losses.

Models are organized into the following categories:

- **Base Classes**: Abstract interfaces for all models (:class:`BaseModel`,
  :class:`TrainableModel`, :class:`ClusteringModel`).
- **Traditional Methods**: Attribute-only or structure-only baselines
  (e.g., :class:`Node2Vec`).
- **Non-Parametric Methods**: Training-free models using fixed graph filters
  (e.g., :class:`SGC`, :class:`SSGC`, :class:`NAFS`, :class:`SAGSC`,
  :class:`S2CAG`).
- **Deep Decoupled Methods**: Models where the encoder is pre-trained with a
  self-supervised objective and clustering is applied post-hoc
  (e.g., :class:`GAE`, :class:`DGI`, :class:`CCASSG`, :class:`S3GC`,
  :class:`NS4GC`, :class:`MAGI`).
- **Deep Joint Methods**: End-to-end models that jointly optimize
  representation learning and clustering
  (e.g., :class:`DAEGC`, :class:`DinkNet`, :class:`DMoN`, :class:`MinCut`,
  :class:`Neuromap`, :class:`GCSBM`).

.. currentmodule:: pyagc.models

Base Classes
------------

.. autosummary::
   :nosignatures:

   LossOutput
   BaseModel
   TrainableModel
   ClusteringModel

.. autoclass:: LossOutput
   :members:

.. autoclass:: BaseModel
   :members:

.. autoclass:: TrainableModel
   :members:

.. autoclass:: ClusteringModel
   :members:

Traditional Methods
-------------------

.. autosummary::
   :nosignatures:

   Node2Vec

.. autoclass:: Node2Vec
   :members:
   :undoc-members:

Non-Parametric Methods
----------------------

These methods construct node representations via fixed filtering operations
over the graph topology without learnable encoder parameters. Clustering is
typically performed post-hoc on the resulting embeddings.

.. autosummary::
   :nosignatures:

   SGC
   SSGC
   NAFS
   SAGSC
   S2CAG

.. autoclass:: SGC
   :members:
   :undoc-members:

.. autoclass:: SSGC
   :members:
   :undoc-members:

.. autoclass:: NAFS
   :members:
   :undoc-members:

.. autoclass:: SAGSC
   :members:
   :undoc-members:

.. autoclass:: S2CAG
   :members:
   :undoc-members:

Deep Decoupled Methods
----------------------

In decoupled methods, the encoder is pre-trained using a self-supervised
objective :math:`\mathcal{L}_{\text{rep}}`, and a discrete clustering
algorithm (e.g., KMeans) is applied to the learned embeddings post-hoc.
The clustering process is **not** part of the training loop.

.. autosummary::
   :nosignatures:

   GAE
   ARGA
   DGI
   CCASSG
   GBT
   S3GC
   NS4GC
   MAGI

.. autoclass:: GAE
   :members:
   :undoc-members:

.. autoclass:: ARGA
   :members:
   :undoc-members:

.. autoclass:: DGI
   :members:
   :undoc-members:

.. autoclass:: CCASSG
   :members:
   :undoc-members:

.. autoclass:: GBT
   :members:
   :undoc-members:

.. autoclass:: S3GC
   :members:
   :undoc-members:

.. autoclass:: NS4GC
   :members:
   :undoc-members:

.. autoclass:: MAGI
   :members:
   :undoc-members:

Deep Joint Methods
------------------

In joint methods, the encoder and cluster head are optimized **end-to-end**.
The clustering objective :math:`\mathcal{L}_{\text{clust}}` provides
supervisory signals that directly refine the learned representations.
These models inherit from :class:`ClusteringModel` and output cluster
assignments directly.

.. autosummary::
   :nosignatures:

   DAEGC
   DinkNet
   DMoN
   MinCut
   Neuromap
   GCSBM

.. autoclass:: DAEGC
   :members:
   :undoc-members:

.. autoclass:: DinkNet
   :members:
   :undoc-members:

.. autoclass:: DMoN
   :members:
   :undoc-members:

.. autoclass:: MinCut
   :members:
   :undoc-members:

.. autoclass:: Neuromap
   :members:
   :undoc-members:

.. autoclass:: GCSBM
   :members:
   :undoc-members:
