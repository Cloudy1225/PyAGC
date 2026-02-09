pyagc.encoders
==============

.. contents:: Contents
    :local:

The :mod:`pyagc.encoders` package provides the **Representation Encoding** module in the :doc:`Encode-Cluster-Optimize <../tutorial/eco_framework>` framework.
The encoder :math:`\mathcal{E}` fuses structural topology and node attributes into a latent representation space :math:`\mathbf{Z} \in \mathbb{R}^{N \times H}`:

.. math::
    \mathbf{Z} = \mathcal{E}(\mathbf{A}, \mathbf{X}; \Theta_{\mathcal{E}})

Following the `"Classic GNNs are Strong Baselines" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024),
we provide **TunedGNN** — enhanced implementations of standard GNNs with critical improvements including residual connections,
pre-linear transformations, flexible normalization, and optimized dropout strategies.
These encoders support both full-batch processing for small graphs and neighbor-sampling-based mini-batching for massive graphs.

In addition, we re-export standard GNN backbones from `PyTorch Geometric <https://pyg.org/>`_,
including :class:`~torch_geometric.nn.models.basic_gnn.GCN`, :class:`~torch_geometric.nn.models.basic_gnn.GraphSAGE`,
:class:`~torch_geometric.nn.models.basic_gnn.GAT`, :class:`~torch_geometric.nn.models.basic_gnn.GIN`,
as well as graph transformers :class:`~torch_geometric.nn.models.SGFormer` and :class:`~torch_geometric.nn.models.Polynormer`.

This design allows any clustering head from :mod:`pyagc.clusters` to be easily paired with varying encoder backbones
without code duplication — simply change the encoder specification in the configuration file.

.. code-block:: python

    from pyagc.encoders import TunedGCN, TunedGAT, create_tuned_gnn

    # Create a tuned GCN encoder directly:
    encoder = TunedGCN(
        in_channels=1433,
        hidden_channels=256,
        num_layers=3,
        out_channels=128,
        dropout=0.5,
        norm="batch_norm",
        residual=True,
    )

    # Or use the factory function for convenience:
    encoder = create_tuned_gnn(
        "gcn",
        in_channels=1433,
        hidden_channels=256,
        num_layers=3,
        out_channels=128,
        dropout=0.5,
        norm="batch_norm",
        residual=True,
    )

    # Create a tuned GAT with multiple attention heads:
    encoder = create_tuned_gnn(
        "gat",
        in_channels=1433,
        hidden_channels=256,
        num_layers=3,
        out_channels=128,
        heads=4,
        concat=True,
        dropout=0.6,
        norm="layer_norm",
    )

    # Incompatible parameters are automatically filtered:
    encoder = create_tuned_gnn(
        "gcn",
        in_channels=1433,
        hidden_channels=256,
        num_layers=3,
        heads=4,  # ignored for GCN, with a warning
    )

Tuned GNN Models
----------------

The **TunedGNN** family provides enhanced versions of standard GNN architectures with
hyperparameters tuned for optimal node-level performance. Key improvements over vanilla
PyG implementations include:

- **Residual connections** — especially beneficial for heterophilous graphs and deeper networks.
- **Pre-linear transformation** — optional linear layer before the first GNN layer.
- **Flexible normalization** — supports :obj:`"batch_norm"` (recommended for large graphs) and :obj:`"layer_norm"` (for smaller graphs).
- **Optimized dropout** — applied at configurable positions in the network.
- **Jumping Knowledge** — optional aggregation across layers (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).

.. currentmodule:: pyagc.encoders

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TunedGNN
   TunedGCN
   TunedGraphSAGE
   TunedGAT
   TunedGIN
   TunedPNA
   TunedEdgeCNN

Factory Function
----------------

The :func:`create_tuned_gnn` factory function provides a convenient way to instantiate
any tuned GNN model by name. It automatically inspects the target model's signature and
filters out incompatible parameters, so you can safely pass all hyperparameters without
worrying about compatibility across different GNN types.

.. currentmodule:: pyagc.encoders

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   create_tuned_gnn

.. autofunction:: create_tuned_gnn

PyG Backbone Re-exports
------------------------

For convenience, :mod:`pyagc.encoders` also re-exports the following standard GNN
and graph transformer models from `PyTorch Geometric <https://pyg.org/>`_.
These can be used as drop-in encoder backbones within the ECO framework:

.. code-block:: python

    from pyagc.encoders import GCN, GraphSAGE, GAT, GIN, SGFormer, Polynormer

    # Use a standard PyG GCN as encoder:
    encoder = GCN(
        in_channels=1433,
        hidden_channels=256,
        num_layers=2,
        out_channels=128,
    )

    # Use a graph transformer as encoder:
    encoder = SGFormer(
        in_channels=1433,
        hidden_channels=256,
        out_channels=128,
        num_layers=3,
    )

**Standard GNNs:**

.. currentmodule:: pyagc.encoders

.. autosummary::
   :nosignatures:

   GCN
   GraphSAGE
   GAT
   GIN
   PNA
   EdgeCNN

.. note::
    These classes are imported directly from :mod:`torch_geometric.nn.models`.
    See the `PyG documentation <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models>`_
    for full API details.

**Graph Transformers:**

.. autosummary::
   :nosignatures:

   SGFormer
   Polynormer

.. note::
    These classes are imported directly from :mod:`torch_geometric.nn.models`.
    See the `PyG documentation <https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models>`_
    for full API details.
