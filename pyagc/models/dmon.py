import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.clusters import DMoNClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs


class DMoN(ClusteringModel):
    r"""
    The Deep Modularity Network (DMoN) is proposed in the
    `"Graph Clustering with Graph Neural Networks"
    <https://arxiv.org/abs/2006.16904>`_ paper (Tsitsulin et al., JMLR 2023).

    This model performs **unsupervised graph clustering** by combining a
    graph encoder (e.g., GCN, GraphSAGE) with the
    :class:`~pyagc.clusters.DMoNClusterHead`. The encoder produces
    node embeddings :math:`\mathbf{Z}`, which are then projected into
    soft cluster assignments :math:`\mathbf{S}`. The model jointly optimizes
    the modularity-based and collapse regularization objectives to learn
    meaningful community structures in the graph.

    The optimization objective consists of two losses:

    **(1) Spectral modularity loss:**

    .. math::
        \mathcal{L}_m = - \frac{1}{2m}
        \mathrm{Tr}(\mathbf{S}^\top \mathbf{B} \mathbf{S})

    where :math:`\mathbf{B} = \mathbf{A} - \frac{\mathbf{d}\mathbf{d}^\top}{2m}`
    is the modularity matrix and :math:`m` is the total number of edges.
    This term encourages nodes that are more densely connected than random
    to be assigned to the same cluster.

    **(2) Collapse regularization loss:**

    .. math::
        \mathcal{L}_c = \frac{\sqrt{K}}{N} \left\| \sum_i \mathbf{S}_i^\top \right\|_F - 1

    which prevents degenerate solutions by penalizing unbalanced or
    collapsed cluster assignments.

    The final training objective is a weighted combination of the two terms:

    .. math::
        \mathcal{L} = \mathcal{L}_m + \lambda \mathcal{L}_c

    where :math:`\lambda` controls the strength of the regularization.

    Args:
        encoder (torch.nn.Module): Node encoder that outputs node embeddings.
        n_features (int): Feature dimension of the encoder outputs.
        n_clusters (int): Number of clusters.
        lam (float, optional): Regularization coefficient for the
            collapse loss :math:`\mathcal{L}_c`. (default: :obj:`1.0`)
    """
    def __init__(self, encoder: nn.Module, n_features: int, n_clusters: int, lam: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.lam = lam

        # DMoN clustering head
        self.head = DMoNClusterHead(n_clusters, n_features)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        self.head.reset_cluster_centers()

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Compute node embeddings via the encoder."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Predict hard cluster assignments from current parameters."""
        z = self.embed(*args, **kwargs)
        return self.head.cluster(z)

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the DMoN loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        loss_m, loss_c = self.head(z=z, edge_index=edge_index)
        loss = loss_m + self.lam * loss_c

        return LossOutput(
            total=loss,
            components={
                'modularity': loss_m.item(),
                'collapse': loss_c.item()
            }
        )

    def loss_batch(self, batch: Data):
        r"""DMoN currently does not support mini-batch training."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch training.")

    def __repr__(self):
        return f"{self.__class__.__name__}(lam={self.lam}, encoder={self.encoder.__class__.__name__})"
