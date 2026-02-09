from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.clusters import MinCutClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs


class MinCut(ClusteringModel):
    r"""
    The MinCut model is based on
    `"Spectral Clustering in Graph Neural Networks for Graph Pooling"
    <https://arxiv.org/abs/1907.00481>`_ (Bianchi et al., 2019).

    It performs **unsupervised graph clustering** by coupling a graph encoder
    (e.g., GCN, GraphSAGE) with the :class:`~pyagc.clusters.MinCutClusterHead`.
    The encoder produces node embeddings :math:`\mathbf{Z}`, which are projected
    into soft cluster assignments :math:`\mathbf{S}`. The model jointly
    optimizes the MinCut objective and the orthogonality regularizer to yield
    compact, well-separated clusters.

    The optimization objective consists of two losses:

    **(1) MinCut loss:**

    .. math::
        \mathcal{L}_{\text{mincut}} = -
        \frac{\mathrm{Tr}(\mathbf{S}^\top \mathbf{A} \mathbf{S})}
             {\mathrm{Tr}(\mathbf{S}^\top \mathbf{D} \mathbf{S})}

    encouraging large within-cluster connectivity relative to cluster volume.

    **(2) Orthogonality regularization:**

    .. math::
        \mathcal{L}_{\text{ortho}} =
        \left\| \frac{\mathbf{S}^\top \mathbf{S}}{\|\mathbf{S}^\top \mathbf{S}\|_F}
        - \frac{\mathbf{I}_K}{\sqrt{K}} \right\|_F

    encouraging near-orthogonal cluster assignment columns to avoid collapse.

    The final training objective is a weighted combination of the two terms:

    .. math::
        \mathcal{L} = \mathcal{L}_{\text{mincut}} + \lambda \mathcal{L}_{\text{ortho}}

    where :math:`\lambda` controls the strength of the orthogonality regularization.

    Args:
        encoder (torch.nn.Module): Node encoder that outputs node embeddings.
        n_features (int): Feature dimension of the encoder outputs.
        n_clusters (int): Number of clusters.
        lam (float, optional): Regularization coefficient for the
            orthogonality loss :math:`\mathcal{L}_{\text{ortho}}`. (default: :obj:`1.0`)
        temperature (float, optional): Softmax temperature used in the MinCut head. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        encoder: nn.Module,
        n_features: int,
        n_clusters: int,
        lam: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.lam = lam

        # MinCut clustering head
        self.head = MinCutClusterHead(n_clusters=n_clusters, n_features=n_features, temperature=temperature)

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

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs: Any) -> LossOutput:
        r"""
        Computes the MinCut loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        loss_mincut, loss_ortho = self.head(z=z, edge_index=edge_index)
        loss = loss_mincut + self.lam * loss_ortho

        return LossOutput(
            total=loss,
            components={
                'mincut': loss_mincut.item(),
                'ortho': loss_ortho.item()
            }
        )

    def loss_batch(self, batch: Data, **kwargs: Any):
        r"""MinCut currently does not support mini-batch training."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch training.")

    def __repr__(self):
        return (f"{self.__class__.__name__}(lam={self.lam}, "
                f"encoder={self.encoder.__class__.__name__})")
