from typing import Tuple

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform

from pyagc.models import TrainableModel, LossOutput
from pyagc.utils import off_diagonal, filter_kwargs


class CCASSG(TrainableModel):
    r"""
    The CCA-SSG (Canonical Correlation Analysis-inspired Self-Supervised GNN) model is proposed in the
    `From Canonical Correlation Analysis to Self-supervised Graph Neural Networks
    <https://arxiv.org/abs/2106.12484>`_ paper (Zhang et al., NeurIPS 2021).

    CCASSG aims to learn node representations by maximizing the agreement between embeddings from
    two augmented graph views while decorrelating the feature dimensions of each view.

    The loss function combines an invariance term and a decorrelation term:

    .. math::
        \mathcal{L} = \underbrace{\| \tilde{\boldsymbol{Z}}^{(1)} - \tilde{\boldsymbol{Z}}^{(2)} \|_F^2}_{\text{invariance}} +
        \lambda \left( \| \tilde{\boldsymbol{Z}}^{(1)\top} \tilde{\boldsymbol{Z}}^{(1)} - \boldsymbol{I} \|_F^2 + \| \tilde{\boldsymbol{Z}}^{(2)\top} \tilde{\boldsymbol{Z}}^{(2)} - \boldsymbol{I} \|_F^2 \right)

    where:

    - :math:`\tilde{\boldsymbol{Z}}^{(i)}` is the column-normalized embedding of view :math:`i`;
    - :math:`\boldsymbol{I}` is the identity matrix;
    - :math:`\lambda` controls the strength of decorrelation regularization.

    Args:
        encoder (torch.nn.Module): The encoder shared across both views.
        transform1 (torch_geometric.transforms.BaseTransform): The 1-st graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The 2-nd graph view transformation.
        lam (float): Weight of decorrelation loss (default: :obj:`1e-3`).
    """

    def __init__(self, encoder: nn.Module, transform1: BaseTransform, transform2: BaseTransform, lam: float = 1e-3):
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
        Computes the CCA-SSG loss.

        Args:
            z1 (torch.Tensor): First view embeddings.
            z2 (torch.Tensor): Second view embeddings.

        Returns:
            Tuple of (total_loss, invariance_term, decorrelation_term).
        """
        z1 = (z1 - z1.mean(0)) / z1.std(0)
        z2 = (z2 - z2.mean(0)) / z2.std(0)

        inv = -(z1 * z2).sum() / z1.shape[0]  # invariance loss

        C1 = z1.T @ z1 / z1.shape[0]  # dim x dim
        C2 = z2.T @ z2 / z2.shape[0]  # dim x dim

        dec = (off_diagonal(C1).pow(2) + off_diagonal(C2).pow(2)).sum()  # decorrelation reduction loss

        loss = inv + self.lam * dec
        return loss, inv, dec

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the CCA-SSG loss with multiple components.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(x, edge_index, **kwargs)
        loss, inv, dec = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'dec': dec.item()
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
        loss, inv, dec = self._compute_loss(z1, z2)

        return LossOutput(
            total=loss,
            components={
                'inv': inv.item(),
                'dec': dec.item()
            }
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"