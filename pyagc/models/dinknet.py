from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset

from pyagc.clusters import DinkClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-15


class DinkNet(ClusteringModel):
    r"""The Dink-Net model from the
    `"Dink-Net: Neural Clustering on Large Graphs"
    <https://arxiv.org/abs/2305.18405>`_ paper (Liu et al., ICML 2023).

    Dink-Net unifies representation learning and clustering optimization via:

    1. **Node Discriminate Module**: Learns discriminative features by distinguishing
       original vs. augmented nodes
    2. **Neural Clustering Module**: Optimizes clustering via dilation (push centers apart)
       and shrink (pull nodes to centers) losses

    The total loss is:

    .. math::
        \mathcal{L} = \mathcal{L}_\text{dilation} + \mathcal{L}_\text{shrink} + \alpha \mathcal{L}_\text{discri}

    Args:
        encoder (torch.nn.Module): The encoder module (e.g., GCN, GraphSAGE).
        projector (torch.nn.Module): The projection head for discriminative learning.
        n_clusters (int): Number of clusters.
        hidden_channels (int): Hidden dimension of node embeddings.
        corruption (Callable, optional): Corruption function for data augmentation.
            If None, uses random feature shuffling. (default: :obj:`None`)
        alpha (float, optional): Trade-off weight for discriminative loss.
            (default: :obj:`1e-10`)
    """

    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module,
            n_clusters: int,
            hidden_channels: int,
            corruption: Optional[Callable] = None,
            alpha: float = 1e-10,
    ):
        super(DinkNet, self).__init__()
        self.encoder = encoder
        self.projector = projector
        self.n_clusters = n_clusters
        self.hidden_channels = hidden_channels
        self.alpha = alpha

        # Default corruption: feature shuffle
        if corruption is None:
            corruption = lambda x, edge_index: (
                x[torch.randperm(x.size(0))], edge_index
            )
        self.corruption = corruption

        # Initialize cluster head (will be set after pretraining)
        self.cluster_head = DinkClusterHead(n_clusters, self.hidden_channels)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.projector)
        self.cluster_head.reset_cluster_centers()

    def embed(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self.encoder(x, edge_index, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Returns cluster assignments."""
        z = self.embed(x, edge_index, **kwargs)
        z = F.normalize(z, p=2, dim=-1)
        return self.cluster_head.cluster(z, soft=False)

    def _discriminate_loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Computes discriminative loss for node discrimination."""
        batch_size = kwargs.get('batch_size', -1)

        # Original embeddings
        z1 = self.embed(x, edge_index, **kwargs)
        z1 = z1[:batch_size] if batch_size > 0 else z1
        z1_proj = self.projector(z1)
        g1 = z1_proj.sum(dim=-1)  # (n_samples,)

        # Corrupted embeddings
        x_corrupt, edge_index_corrupt = self.corruption(x, edge_index)
        z2 = self.embed(x_corrupt, edge_index_corrupt, **kwargs)
        z2 = z2[:batch_size] if batch_size > 0 else z2
        z2_proj = self.projector(z2)
        g2 = z2_proj.sum(dim=-1)  # (n_samples,)

        # Binary cross-entropy
        loss = (F.softplus(-g1) + F.softplus(g2)).mean() / 2

        return loss

    def pretrain_loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Computes pretraining loss (discriminative only)."""
        return self._discriminate_loss(x, edge_index, **kwargs)

    def loss(self, x: Tensor, edge_index: Tensor, pretrain: bool = False, **kwargs) -> LossOutput:
        r"""
        Computes the total loss.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            pretrain (bool, optional): If True, only compute discriminative loss.
                (default: :obj:`False`)

        Returns:
            LossOutput containing total loss and individual components.
        """
        loss_discri = self._discriminate_loss(x, edge_index, **kwargs)
        if pretrain:
            return LossOutput(
                total=loss_discri,
                components={'discri': loss_discri.item()}
            )

        # Fine-tuning: discriminative + clustering losses
        z = self.embed(x, edge_index, **kwargs)
        z = F.normalize(z, p=2, dim=-1)

        loss_dilation, loss_shrink = self.cluster_head(z)

        total_loss = loss_dilation + loss_shrink + self.alpha * loss_discri

        return LossOutput(
            total=total_loss,
            components={
                'dilation': loss_dilation.item(),
                'shrink': loss_shrink.item(),
                'discri': loss_discri.item()
            }
        )

    def loss_batch(self, batch: Data, pretrain: bool = False) -> LossOutput:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.
            pretrain (bool, optional): If True, only compute discriminative loss.

        Returns:
            LossOutput containing total loss and individual components.
        """
        batch_size = batch.batch_size

        loss_discri = self._discriminate_loss(batch.x, batch.edge_index, batch_size=batch_size)

        if pretrain:
            return LossOutput(
                total=loss_discri,
                components={'discri': loss_discri.item()}
            )

        # Fine-tuning with batch
        z = self.embed(batch.x, batch.edge_index)
        z = z[:batch_size]  # Only seed nodes
        z = F.normalize(z, p=2, dim=-1)

        loss_dilation, loss_shrink = self.cluster_head(z)

        total_loss = loss_dilation + loss_shrink + self.alpha * loss_discri

        return LossOutput(
            total=total_loss,
            components={
                'dilation': loss_dilation.item(),
                'shrink': loss_shrink.item(),
                'discri': loss_discri.item()
            }
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'n_clusters={self.n_clusters}, '
                f'alpha={self.alpha})')
