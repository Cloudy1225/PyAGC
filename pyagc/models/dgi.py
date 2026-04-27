import copy
from typing import Callable, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset, uniform

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs

EPS = 1e-15


def default_corruption(x: Tensor, edge_index: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
    r"""Default corruption function: randomly shuffle node features.

    This is the most commonly used corruption strategy in DGI, which disrupts
    the correspondence between node features and graph structure by permuting
    the feature matrix.

    Args:
        x (Tensor): Node feature matrix of shape :obj:`[num_nodes, num_features]`.
        edge_index (Tensor): Graph connectivity in COO format.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of corrupted features and original edge_index.
    """
    return x[torch.randperm(len(x))], edge_index


def default_summary(z: Tensor, *args, **kwargs) -> Tensor:
    r"""Default summary function: global mean pooling with sigmoid activation.

    This readout function computes a graph-level representation by averaging
    all node embeddings and applying sigmoid activation.

    Args:
        z (Tensor): Node embeddings of shape :obj:`[num_nodes, hidden_channels]`.
        *args: Additional positional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Graph-level summary vector of shape :obj:`[hidden_channels]`.
    """
    return z.mean(dim=0).sigmoid()


class DGI(TrainableModel):
    r"""The Deep Graph Infomax (DGI) model from the
    `"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
    paper (Veličković et al., ICLR 2019)based on user-defined encoder and summary model :math:`\mathcal{E}`
    and :math:`\mathcal{R}` respectively, and a corruption function
    :math:`\mathcal{C}`.

    DGI maximizes mutual information between patch representations and
    corresponding high-level summaries of the graph by training a discriminator
    to distinguish between positive samples (real node-graph pairs) and
    negative samples (corrupted node-graph pairs).

    This implementation is adapted from: `pyg/deep_graph_infomax
    <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/deep_graph_infomax.py>`_.

    Args:
        hidden_channels (int): The latent space dimensionality.
        encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
        summary (callable, optional): The readout function :math:`\mathcal{R}`
            that computes graph-level representations. If :obj:`None`, uses
            global mean pooling with sigmoid activation. (default: :obj:`None`)
        corruption (callable, optional): The corruption function :math:`\mathcal{C}`
            that generates negative samples. If :obj:`None`, uses random feature
            shuffling. (default: :obj:`None`)

    Example:
        >>> from pyagc.models import DGI
        >>> from torch_geometric.nn import GCN
        >>>
        >>> # With default summary and corruption
        >>> encoder = GCN(in_channels=128, hidden_channels=64, num_layers=2)
        >>> model = DGI(hidden_channels=64, encoder=encoder)
        >>>
        >>> # With custom functions
        >>> def custom_corruption(x, edge_index):
        ...     return x + torch.randn_like(x) * 0.1, edge_index
        >>>
        >>> def custom_summary(z):
        ...     return z.max(dim=0)[0]
        >>>
        >>> model = DGI(
        ...     hidden_channels=64,
        ...     encoder=encoder,
        ...     summary=custom_summary,
        ...     corruption=custom_corruption
        ... )
    """

    def __init__(
            self,
            hidden_channels: int,
            encoder: Module,
            summary: Optional[Callable] = None,
            corruption: Optional[Callable] = None,
    ):
        super(DGI, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder

        # Use default functions if not provided
        self.summary = summary if summary is not None else default_summary
        self.corruption = corruption if corruption is not None else default_corruption

        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))
        uniform(self.hidden_channels, self.weight)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.summary)
        uniform(self.hidden_channels, self.weight)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation.
        """
        pos_z = self.embed(*args, **kwargs)

        cor = self.corruption(*args, **kwargs)
        cor = cor if isinstance(cor, tuple) else (cor,)
        cor_args = cor[:len(args)]
        cor_kwargs = copy.copy(kwargs)
        for key, value in zip(kwargs.keys(), cor[len(args):]):
            cor_kwargs[key] = value

        neg_z = self.embed(*cor_args, **cor_kwargs)

        summary = self.summary(pos_z, *args, **kwargs)

        return pos_z, neg_z, summary

    def discriminate(self, z: Tensor, summary: Tensor,
                     sigmoid: bool = True) -> Tensor:
        r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
        the probability scores assigned to this patch-summary pair.

        Args:
            z (torch.Tensor): The latent space.
            summary (torch.Tensor): The summary vector.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, x: Tensor, edge_index: Tensor, **kwargs) -> LossOutput:
        r"""Computes the mutual information maximization objective."""
        pos_z, neg_z, summary = self(x, edge_index, **kwargs)

        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        return LossOutput(
            total=pos_loss + neg_loss,
            components={'pos': pos_loss.item(), 'neg': neg_loss.item()}
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""Computes loss for a mini-batch with seed node slicing."""
        pos_z, neg_z, summary = self(batch.x, batch.edge_index)
        pos_z = pos_z[:batch.batch_size]
        neg_z = neg_z[:batch.batch_size]
        summary = self.summary(pos_z)

        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean()

        return LossOutput(
            total=pos_loss + neg_loss,
            components={'pos': pos_loss.item(), 'neg': neg_loss.item()}
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_channels={self.hidden_channels})'
