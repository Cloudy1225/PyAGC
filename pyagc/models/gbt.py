from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import off_diagonal, filter_kwargs


class GBT(TrainableModel):
    r"""
    The G-BT (Graph Barlow Twins) model is proposed in the
    `"Graph Barlow Twins: A Self-Supervised Representation Learning Framework for Graphs"
    <https://arxiv.org/abs/2106.02466>`_ paper (Bielak et al., KBS 2022).

    G-BT learns node representations by maximizing the similarity between two embeddings of the same node under
    different augmentations, while reducing redundancy between dimensions via a decorrelation objective.

    The loss function is:

    .. math::
        \mathcal{L} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_{i}\sum_{j \ne i} \mathcal{C}_{ij}^2

    where:

    - :math:`\mathcal{C}` is the cross-correlation matrix between normalized embeddings from two views:
      :math:`\mathcal{C}_{ij} = \sum_b \tilde{z}^{(1)}_{b,i} \tilde{z}^{(2)}_{b,j}`.
    - :math:`\tilde{z}` denotes the batch-normalized node embedding.
    - :math:`\lambda` balances invariance and redundancy reduction.

    Args:
        encoder (torch.nn.Module): The shared encoder used to produce node embeddings.
        transform1 (torch_geometric.transforms.BaseTransform): The first graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The second graph view transformation.
        lam (float): Weight for redundancy reduction term. (default: :obj:`5e-3`)
    """

    def __init__(self, encoder: nn.Module, transform1: BaseTransform, transform2: BaseTransform, lam: float = 5e-3):
        super().__init__()
        self.encoder = encoder
        self.transform1 = transform1
        self.transform2 = transform2
        self.lam = lam

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

    def _compute_loss(self, z1: Tensor, z2: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes the Graph Barlow Twins loss.

        Args:
            z1 (torch.Tensor): First view embeddings.
            z2 (torch.Tensor): Second view embeddings.

        Returns:
            Tuple of (total_loss, invariance_term, redundancy_term).
        """
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        c = (z1.T @ z2) / z1.shape[0]  # dim x dim

        inv = torch.diagonal(c).add_(-1).pow_(2).sum()  # invariance loss
        red = off_diagonal(c).pow_(2).sum()  # redundancy reduction loss
        loss = inv + self.lam * red
        return loss, inv, red

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the GBT loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(x, edge_index, **kwargs)
        loss, inv, red = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'red': red.item()
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
        loss, inv, red = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'red': red.item()
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
