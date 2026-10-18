from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_remaining_self_loops

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs


class SCGC(TrainableModel):
    r"""
    Simple Contrastive Graph Clustering with a neighbor-oriented structural loss
    (`"Simple Contrastive Graph Clustering" <https://arxiv.org/abs/2205.07865>`_,
    Liu et al., IEEE TNNLS 2023).

    This implementation follows the PyAGC decoupled contrastive pipeline (shared encoder,
    two graph augmentations, post-hoc K-Means). The objective matches the SCGC neighbor
    contrast on :math:`\hat{\boldsymbol{A}} = \boldsymbol{A} + \boldsymbol{I}`, but uses
    separate means over positive and negative pairs for better optimization on sparse graphs:

        .. math::
            \mathcal{L} = \underbrace{\frac{1}{|\hat{E}|}\sum_{(i,j)\in \hat{E}}
            \left(\boldsymbol{S}_{ij} - 1\right)^2}_{\text{pos}}
            + \underbrace{\frac{1}{N^2 - |\hat{E}|}\sum_{(i,j)\notin \hat{E}}
            \boldsymbol{S}_{ij}^2}_{\text{neg}}

    where :math:`\boldsymbol{S} = \boldsymbol{Z}^{(1)} (\boldsymbol{Z}^{(2)})^{\top}` with
    row-wise :math:`\ell_2` normalization.

    Args:
        encoder (torch.nn.Module): The encoder shared across both views.
        transform1 (torch_geometric.transforms.BaseTransform): The 1-st graph view transformation.
        transform2 (torch_geometric.transforms.BaseTransform): The 2-nd graph view transformation.
    """

    def __init__(
        self,
        encoder: Module,
        transform1: BaseTransform,
        transform2: BaseTransform,
    ):
        super(SCGC, self).__init__()
        self.encoder = encoder
        self.transform1 = transform1
        self.transform2 = transform2

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

    def _compute_loss(
        self, z1: Tensor, z2: Tensor, edge_index: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes the SCGC structural contrastive loss.

        Args:
            z1 (torch.Tensor): The first node embeddings.
            z2 (torch.Tensor): The second node embeddings.
            edge_index (torch.Tensor): Edge indices of the original graph.

        Returns:
            Tuple of (total_loss, positive_term, negative_term).
        """
        device = z1.device
        num_nodes = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        S = z1 @ z2.T

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
        src, dst = edge_index

        loss_pos_term = ((S[src, dst] - 1) ** 2).mean()

        mask = torch.ones((num_nodes, num_nodes), device=device, dtype=torch.bool)
        mask[src, dst] = False
        loss_neg_term = S[mask].pow(2).mean()

        loss = loss_pos_term + loss_neg_term
        return loss, loss_pos_term, loss_neg_term

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""
        Computes the SCGC loss.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            LossOutput containing total loss and individual components.
        """
        z1, z2 = self(x, edge_index, **kwargs)
        loss, loss_pos, loss_neg = self._compute_loss(z1, z2, edge_index)

        return LossOutput(
            total=loss,
            components={
                'pos': loss_pos.item(),
                'neg': loss_neg.item(),
            },
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

        batch_mask = (
            (batch.edge_index[0] < batch.batch_size)
            & (batch.edge_index[1] < batch.batch_size)
        )
        batch_edge_index = batch.edge_index[:, batch_mask]

        loss, loss_pos, loss_neg = self._compute_loss(z1, z2, batch_edge_index)

        return LossOutput(
            total=loss,
            components={
                'pos': loss_pos.item(),
                'neg': loss_neg.item(),
            },
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(encoder={self.encoder})"
