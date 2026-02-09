pyagc.transforms
================

.. contents:: Contents
    :local:

The :mod:`pyagc.transforms` package provides graph augmentation transforms commonly used
in self-supervised and contrastive graph learning methods within the ECO framework.
These transforms generate augmented views of the input graph by randomly perturbing
node features and/or graph topology, which serve as the basis for contrastive or
reconstruction-based representation learning objectives.

All transforms inherit from :class:`~torch_geometric.transforms.BaseTransform` and are
registered as **functional transforms** via PyTorch Geometric's transform registry,
enabling seamless composition with other PyG transforms.

.. code-block:: python

    from pyagc.transforms import GSSLTransform, RandomDropEdge, RandomMaskFeat

    # Unified GSSL augmentation (feature masking + edge dropping):
    transform = GSSLTransform(p_feat_mask=0.3, p_edge_drop=0.3)
    augmented_data = transform(data)

    # Or use individual transforms and compose them:
    from torch_geometric.transforms import Compose

    transform = Compose([
        RandomDropEdge(p=0.3),
        RandomMaskFeat(p=0.3),
    ])
    augmented_data = transform(data)

    # Functional transform resolution via string name:
    from torch_geometric.transforms import AddSelfLoops
    import torch_geometric.transforms as T

    transform = T.Compose([
        T.functional_transform('random_drop_edge')(p=0.2),
        T.functional_transform('random_mask_feat')(p=0.3),
    ])

These augmentations are used extensively by contrastive and self-supervised AGC methods,
including `GRACE <https://arxiv.org/abs/2006.04131>`_,
`CCA-SSG <https://arxiv.org/abs/2106.12484>`_,
and `BGRL <https://arxiv.org/abs/2102.06514>`_.
Both homogeneous and heterogeneous graphs are supported.

GSSL Transform
--------------

The :class:`GSSLTransform` provides a unified augmentation pipeline that combines
random feature masking and random edge dropping in a single transform. This is the
recommended transform for graph self-supervised learning, as it applies both
augmentations consistently and only retains the specified node and edge attributes.

.. code-block:: python

    from pyagc.transforms import GSSLTransform

    # Create two augmented views with different augmentation strengths:
    transform_1 = GSSLTransform(p_feat_mask=0.3, p_edge_drop=0.2)
    transform_2 = GSSLTransform(p_feat_mask=0.4, p_edge_drop=0.4)

    view_1 = transform_1(data)
    view_2 = transform_2(data)

.. currentmodule:: pyagc.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   GSSLTransform

Random Drop Edge
----------------

The :class:`RandomDropEdge` transform randomly removes edges from the graph with a
given probability, following the `"DropEdge" <https://arxiv.org/abs/1907.10903>`_ paper.
Associated edge attributes are optionally dropped alongside the removed edges.
This augmentation helps alleviate over-smoothing in deep GNNs and serves as a
topology-level perturbation for contrastive learning.

.. code-block:: python

    from pyagc.transforms import RandomDropEdge

    # Drop 30% of edges:
    transform = RandomDropEdge(p=0.3)
    augmented_data = transform(data)

    # Drop edges along with custom edge attributes:
    transform = RandomDropEdge(p=0.3, edge_attrs=["edge_attr", "edge_weight"])
    augmented_data = transform(data)

.. currentmodule:: pyagc.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   RandomDropEdge

Random Mask Feature
-------------------

The :class:`RandomMaskFeat` transform randomly masks (zeros out) columns of node
and/or edge feature tensors, following the `"Graph Contrastive Learning with
Augmentations" <https://arxiv.org/abs/2010.13902>`_ paper. This provides a
feature-level perturbation that encourages the encoder to learn robust representations
invariant to missing attributes.

.. code-block:: python

    from pyagc.transforms import RandomMaskFeat

    # Mask 30% of node feature columns:
    transform = RandomMaskFeat(p=0.3)
    augmented_data = transform(data)

    # Mask both node and edge features:
    transform = RandomMaskFeat(
        p=0.3,
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
    )
    augmented_data = transform(data)

.. currentmodule:: pyagc.transforms

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   RandomMaskFeat
