from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models import InnerProductDecoder
from torch_geometric.utils import negative_sampling

from pyagc.clusters import DECClusterHead
from pyagc.models.base import ClusteringModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-15


class DAEGC(ClusteringModel):
    r"""Deep Attentional Embedded Graph Clustering model from the
    `"Attributed Graph Clustering: A Deep Attentional Embedding Approach"
    <https://arxiv.org/abs/1906.06532>`_ paper (Wang et al., IJCAI 2019).

    DAEGC jointly optimizes graph embedding and clustering through:

    1. **Graph Attentional Autoencoder**: Learns representations by encoding
       both structure and content with attention mechanism, then reconstructs
       the graph structure via inner product decoder.

    2. **Self-training Clustering**: Uses confident cluster assignments as soft
       labels to guide the optimization, iteratively refining clustering results.

    The total loss combines reconstruction and clustering objectives:

    .. math::
        \mathcal{L} = \mathcal{L}_r + \gamma \mathcal{L}_c

    where:

    - :math:`\mathcal{L}_r` is the binary cross-entropy reconstruction loss
    - :math:`\mathcal{L}_c = KL(P||Q)` is the clustering loss with Student's t-distribution
    - :math:`\gamma` is the clustering coefficient

    Args:
        encoder (torch.nn.Module): The graph attention encoder (typically GAT-based).
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to :class:`InnerProductDecoder`.
            (default: :obj:`None`)
        n_clusters (int): Number of clusters.
        hidden_channels (int): Hidden dimension of node embeddings.
        gamma (float, optional): Weight for clustering loss. (default: :obj:`10.0`)
        update_interval (int, optional): Number of iterations between target
            distribution updates. (default: :obj:`5`)
    """

    def __init__(
            self,
            encoder: nn.Module,
            n_clusters: int,
            hidden_channels: int,
            decoder: Optional[nn.Module] = None,
            gamma: float = 10.0,
            update_interval: int = 5,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.n_clusters = n_clusters
        self.hidden_channels = hidden_channels
        self.gamma = gamma
        self.update_interval = update_interval

        # Initialize cluster head (DEC-style)
        self.cluster_head = DECClusterHead(n_clusters, hidden_channels)

        # Track training iterations for target distribution updates
        self.register_buffer('iteration', torch.tensor(0))

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)
        self.cluster_head.reset_cluster_centers()

    def embed(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Computes node embeddings via the encoder."""
        return self.encoder(x, edge_index, **filter_kwargs(self.encoder.forward, kwargs))

    def decode(self, z: Tensor, edge_index: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Reconstructs edge probabilities via the decoder."""
        return self.decoder(z, edge_index, sigmoid=sigmoid)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Returns cluster assignments (hard labels)."""
        z = self.embed(x, edge_index, **kwargs)
        return self.cluster_head.cluster(z, soft=False)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:
        r"""Computes the binary cross-entropy reconstruction loss.

        Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def pretrain_loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Computes pretraining loss (reconstruction only).

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            Reconstruction loss.
        """
        z = self.embed(x, edge_index, **kwargs)
        return self.recon_loss(z, edge_index)

    def loss(
            self,
            x: Tensor,
            edge_index: Tensor,
            pretrain: bool = False,
            **kwargs
    ) -> LossOutput:
        r"""Computes the total loss.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
            pretrain (bool, optional): If :obj:`True`, only compute reconstruction
                loss for pretraining. (default: :obj:`False`)

        Returns:
            LossOutput containing total loss and individual components.
        """
        z = self.embed(x, edge_index, **kwargs)

        # Reconstruction loss
        loss_recon = self.recon_loss(z, edge_index)

        if pretrain:
            return LossOutput(
                total=loss_recon,
                components={'recon': loss_recon.item()}
            )

        # Determine whether to update target distribution P
        # Update every `update_interval` iterations to maintain stability
        update_target = (self.iteration % self.update_interval == 0)

        # Clustering loss (KL divergence between Q and P)
        loss_cluster = self.cluster_head(z, update_target=update_target)

        # Update iteration counter
        if self.training:
            self.iteration += 1

        total_loss = loss_recon + self.gamma * loss_cluster

        return LossOutput(
            total=total_loss,
            components={
                'recon': loss_recon.item(),
                'cluster': loss_cluster.item()
            }
        )

    def loss_batch(self, batch: Data, pretrain: bool = False) -> LossOutput:
        r"""Computes loss for a mini-batch.

        Args:
            batch (Data): A mini-batch from the loader.
            pretrain (bool, optional): If :obj:`True`, only compute reconstruction
                loss for pretraining.

        Returns:
            LossOutput containing total loss and individual components.
        """
        batch_size = batch.batch_size

        # Get embeddings for all nodes in the batch (including neighbors)
        z = self.embed(batch.x, batch.edge_index)

        # Slice to seed nodes only
        z_seed = z[:batch_size]

        # Extract edges within the batch
        edge_mask = (batch.edge_index[0] < batch_size) & (batch.edge_index[1] < batch_size)
        batch_edge_index = batch.edge_index[:, edge_mask]

        # Reconstruction loss
        loss_recon = self.recon_loss(z_seed, batch_edge_index)

        if pretrain:
            return LossOutput(
                total=loss_recon,
                components={'recon': loss_recon.item()}
            )

        # Determine whether to update target distribution
        update_target = (self.iteration % self.update_interval == 0)

        # Clustering loss
        loss_cluster = self.cluster_head(z_seed, update_target=update_target)

        # Increment iteration counter
        if self.training:
            self.iteration += 1

        total_loss = loss_recon + self.gamma * loss_cluster

        return LossOutput(
            total=total_loss,
            components={
                'recon': loss_recon.item(),
                'cluster': loss_cluster.item()
            }
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'n_clusters={self.n_clusters}, '
                f'gamma={self.gamma})')
