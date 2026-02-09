from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.clusters import SBMClusterHead, SBMMatchClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs


class GCSBM(ClusteringModel):
    r"""
    Stochastic Block Model (GCSBM) based clustering model from the paper
    `"Differentiable Community Detection with Graph Neural Networks and Stochastic Block Models"
    <https://openreview.net/forum?id=T1vdfm1THf>`_ (Arliss & Mueller, LoG 2025).

    This model learns node embeddings via a GNN encoder and performs clustering by
    maximizing the likelihood of an GCSBM-based generative model. It supports multiple
    GCSBM variants including Bernoulli, Poisson, and their degree-corrected versions.

    The model optimizes:

    .. math::
        \mathcal{L} = m^{-1} \mathcal{L}_{GCSBM} + \alpha \|\mathbf{1}_k - \mathrm{diag}(\hat{\Theta})\|_F

    where :math:`\mathcal{L}_{GCSBM}` is the negative log-likelihood, :math:`m` is the number
    of edges, and :math:`\alpha` controls regularization strength.

    **Supported Variants:**

    - :obj:`'bernoulli'`: Standard Bernoulli GCSBM
    - :obj:`'poisson'`: Partial Poisson GCSBM (suitable for simple graphs)
    - :obj:`'bernoulli-dc'`: Degree-corrected Bernoulli GCSBM
    - :obj:`'poisson-dc'`: Degree-corrected Poisson GCSBM
    - :obj:`'match'`: Graph Matching variant (fastest)

    Args:
        encoder (torch.nn.Module): Node encoder that outputs node embeddings.
        n_features (int): Feature dimension of the encoder outputs.
        n_clusters (int): Number of clusters.
        variant (str, optional): GCSBM variant to use. Options: :obj:`'bernoulli'`,
            :obj:`'poisson'`, :obj:`'bernoulli-dc'`, :obj:`'poisson-dc'`, :obj:`'match'`.
            (default: :obj:`'bernoulli'`)
        eta (float, optional): Negative sampling ratio. (default: :obj:`3.0`)
        alpha (float, optional): Regularization strength. (default: :obj:`1.0`)
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_features: int,
        n_clusters: int,
        variant: str = 'bernoulli',
        eta: float = 3.0,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.variant = variant
        self.eta = eta
        self.alpha = alpha

        # Create appropriate clustering head
        if variant == 'match':
            self.head = SBMMatchClusterHead(n_clusters, n_features)
        else:
            self.head = SBMClusterHead(n_clusters, n_features, variant=variant, eta=eta)

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
        Computes the GCSBM loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)

        likelihood_loss, reg_loss = self.head(z=z, edge_index=edge_index)
        total_loss = likelihood_loss + self.alpha * reg_loss

        return LossOutput(
            total=total_loss,
            components={
                'likelihood': likelihood_loss.item(),
                'reg': (self.alpha * reg_loss).item()
            }
        )

    def loss_batch(self, batch: Data, **kwargs: Any) -> LossOutput:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch.batch_size]

        # Extract edges within the batch
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        m = batch_edge_index.size(1)
        if m == 0:
            # No edges in batch, return zero loss
            device = z.device
            return LossOutput(
                total=torch.tensor(0.0, device=device),
                components={'likelihood': 0.0, 'reg': 0.0}
            )

        likelihood_loss, reg_loss = self.head(z=z, edge_index=batch_edge_index)
        total_loss = likelihood_loss + self.alpha * reg_loss

        return LossOutput(
            total=total_loss,
            components={
                'likelihood': likelihood_loss.item(),
                'reg': (self.alpha * reg_loss).item()
            }
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}(variant={self.variant}, "
                f"n_clusters={self.n_clusters}, alpha={self.alpha}, "
                f"encoder={self.encoder.__class__.__name__})")
