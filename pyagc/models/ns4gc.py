from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs


class NS4GC(TrainableModel):
    r"""
    The NS4GC (Node Similarity-guided Contrastive Graph Clustering) model is proposed in the
    `"Reliable Node Similarity Matrix Guided Contrastive Graph Clustering"
    <https://arxiv.org/abs/2408.03765>`_ paper (Liu et al., TKDE 2024).

    NS4GC aims to learn node representations suitable for clustering via contrastive learning
    guided by a node similarity matrix computed from two augmented views of the graph.

    The model uses the following contrastive loss function:

        .. math::
            \mathcal{L} = \mathcal{L}_{\text{ali}} + \lambda \mathcal{L}_{\text{nei}} + \gamma \mathcal{L}_{\text{spa}}

    where:

    - :math:`\boldsymbol{S} = \boldsymbol{Z}^{(1)} (\boldsymbol{Z}^{(2)})^{\top}` is the similarity matrix
      computed by inner product between two views;
    - :math:`\mathcal{L}_{\text{ali}} = -\frac{1}{N} \sum_i \boldsymbol{S}_{ii}`
      encourages aligned embeddings from two views;
    - :math:`\mathcal{L}_{\text{nei}} = -\frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \boldsymbol{S}_{ij}`
      enforces consistency for neighbors;
    - :math:`\mathcal{L}_{\text{spa}} = \mathbb{E}_{(i,j) \notin \mathcal{E}} \sigma((\boldsymbol{S}_{ij} - s) / \tau)`
      encourages sparsity of similarity values between non-neighbor pairs.
      :math:`\sigma(x) = 1 / (1 + \exp(-x))` is the sigmoid function.

    Args:
        encoder (torch.nn.Module): The encoder shared across both views.
        transform1 (torch_geometric.transforms.BaseTransform): The 1-st graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The 2-nd graph view transformation.
        s (float): Threshold for the sparsity loss term (default: :obj:`0.6`).
        tau (float): Temperature for the sigmoid function in sparsity loss (default: :obj:`0.1`).
        lam (float): Weight for the neighborhood consistency term (default: :obj:`1.0`).
        gam (float): Weight for the sparsity loss term (default: :obj:`1.0`).

    Example:
        >>> from pyagc.models import NS4GC
        >>> from pyagc.transforms import GSSLTransform
        >>> from pyagc.encoders import create_tuned_gnn
        >>>
        >>> # Create encoder
        >>> encoder = create_tuned_gnn('gcn', in_channels=128, hidden_channels=64, num_layers=2)
        >>>
        >>> # Create augmentation transforms
        >>> transform1 = GSSLTransform(p_feat_mask=0.2, p_edge_drop=0.3)
        >>> transform2 = GSSLTransform(p_feat_mask=0.2, p_edge_drop=0.3)
        >>>
        >>> # Create model
        >>> model = NS4GC(
        ...     encoder=encoder,
        ...     transform1=transform1,
        ...     transform2=transform2,
        ...     s=0.6,
        ...     tau=0.1,
        ...     lam=1.0,
        ...     gam=1.0
        ... )
    """

    def __init__(self, encoder: Module, transform1: BaseTransform, transform2: BaseTransform,
                 s: float = 0.6, tau: float = 0.1, lam: float = 1.0, gam: float = 1.0):
        super(NS4GC, self).__init__()
        self.encoder = encoder
        self.transform1 = transform1
        self.transform2 = transform2
        self.s = s
        self.tau = tau
        self.lam = lam
        self.gam = gam

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        r"""Generates embeddings from two graph augmentations."""
        data1 = self.transform1(*args, **kwargs)
        data2 = self.transform2(*args, **kwargs)
        z1 = self.encoder(**data1)
        z2 = self.encoder(**data2)
        return z1, z2

    def _compute_loss(self, z1: Tensor, z2: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Computes the total loss and its individual components based on two-view embeddings.

        Args:
            z1 (torch.Tensor): The first node embeddings.
            z2 (torch.Tensor): The second node embeddings.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            Tuple of (total_loss, alignment_loss, neighbor_loss, sparsity_loss).
        """
        device = z1.device
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        S = z1 @ z2.T
        num_nodes = z1.size(0)
        loss_ali = -torch.diag(S).mean()

        mask = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        src, dst = edge_index
        mask[src, dst] = False

        loss_nei = -S[src, dst].mean()

        S_spa = torch.masked_select(S, mask)
        S_spa = torch.sigmoid((S_spa - self.s) / self.tau)
        loss_spa = S_spa.mean()

        loss = loss_ali + self.lam * loss_nei + self.gam * loss_spa
        return loss, loss_ali, loss_nei, loss_spa

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the NS4GC loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(x, edge_index, **kwargs)
        loss, loss_ali, loss_nei, loss_spa = self._compute_loss(z1, z2, edge_index)

        return LossOutput(
            total=loss,
            components={
                'ali': loss_ali.item(),
                'nei': loss_nei.item(),
                'spa': loss_spa.item()
            }
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""
        Computes loss for a mini-batch with seed node slicing.

        Args:
            batch (Data): A mini-batch from the loader.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(batch.x, batch.edge_index)
        z1 = z1[:batch.batch_size]
        z2 = z2[:batch.batch_size]

        # Extract edges within the batch
        # input_nodes = torch.arange(batch.batch_size, device=batch.x.device)
        # batch_edge_index, _ = subgraph(input_nodes, batch.edge_index, edge_attr=None, relabel_nodes=False)
        batch_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
        batch_edge_index = batch.edge_index[:, batch_mask]

        loss, loss_ali, loss_nei, loss_spa = self._compute_loss(z1, z2, batch_edge_index)

        return LossOutput(
            total=loss,
            components={
                'ali': loss_ali.item(),
                'nei': loss_nei.item(),
                'spa': loss_spa.item()
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
