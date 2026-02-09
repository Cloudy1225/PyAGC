from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.clusters import NeuromapClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs


class Neuromap(ClusteringModel):
    r"""
    The **Neuromap** model implements the differentiable map equation from
    `"The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks"
    <https://arxiv.org/abs/2310.01144>`_ (Blöcker et al., NeurIPS 2024).

    This model performs **unsupervised graph clustering** by combining a graph encoder
    (e.g., GCN, GraphSAGE) with the :class:`~pyagc.clusters.NeuromapClusterHead`.

    The encoder produces node embeddings :math:`\mathbf{Z}`, which are projected into
    soft cluster assignments :math:`\mathbf{S}`. The model then minimizes the differentiable
    **Map Equation loss**, which measures the expected description length of random walks
    on the graph according to the Minimum Description Length (MDL) principle:

    .. math::

        \mathcal{L}(A, S) = q \log q
        - \sum_m q_m \log q_m
        - \sum_m m_{\text{exit}} \log m_{\text{exit}}
        - \sum_u p_u \log p_u
        + \sum_m p_m \log p_m

    where all terms are derived from the soft community flow structure induced by :math:`S`.

    This loss naturally balances model complexity and data fit without explicit regularization,
    allowing Neuromap to infer the number of effective communities automatically.

    Args:
        encoder (torch.nn.Module): Node encoder producing node embeddings.
        n_features (int): Feature dimension of encoder outputs.
        n_clusters (int): Maximum number of clusters to consider.
        lam (float, optional): Regularization coefficient for the collapse loss. (default: :obj:`1.0`)
        alpha (float, optional): Teleportation probability. (default: :obj:`0.15`)
        n_iters (int, optional): Power iteration steps for stationary distribution. (default: :obj:`100`)
    """

    def __init__(self, encoder: nn.Module, n_features: int, n_clusters: int,
                 lam: float = 1.0, alpha: float = 0.15, n_iters: int = 100):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.lam = lam
        self.alpha = alpha
        self.n_iters = n_iters

        # Neuromap clustering head
        self.head = NeuromapClusterHead(
            n_clusters=n_clusters,
            n_features=n_features,
            alpha=alpha,
            n_iters=n_iters,
        )

    def reset_parameters(self):
        r"""Resets all learnable parameters of the encoder and the cluster head."""
        reset(self.encoder)
        self.head.reset_cluster_centers()
        self.head.F = None
        self.head.p = None
        self.head.p_log_p = None

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Compute node embeddings via the encoder."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tensor:
        r"""Predicts hard cluster assignments from current parameters."""
        z = self.embed(*args, **kwargs)
        return self.head.cluster(z)

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs: Any) -> LossOutput:
        r"""
        Computes the Neuromap loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)
        codelength, collapse_loss = self.head(z=z, edge_index=edge_index)
        loss = codelength + self.lam * collapse_loss

        return LossOutput(
            total=loss,
            components={
                'codelength': codelength.item(),
                'collapse': collapse_loss.item()
            }
        )

    def loss_batch(self, batch: Data, **kwargs: Any):
        r"""Neuromap currently does not support mini-batch training."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support batch training.")

    def __repr__(self):
        return f"{self.__class__.__name__}(encoder={self.encoder.__class__.__name__}, n_clusters={self.n_clusters})"
