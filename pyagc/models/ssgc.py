from typing import Optional

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm

from pyagc.models.base import BaseModel
from pyagc.utils import filter_kwargs


class SSGC(MessagePassing, BaseModel):
    r"""The non-parametric simple spectral graph convolutional (SSGC) operator from the
    `"Simple Spectral Graph Convolution" <https://openreview.net/forum?id=CYO5T-YjWZV>`_ paper
    (Zhu, Hao, and Piotr Koniusz, ICLR 2021).
    This implementation is adapted from: `pyg/ssg_conv
    <https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/ssg_conv.py>`_.

    .. math::
        \mathbf{X}^{\prime} = \frac{1}{K} \sum_{k=1}^K\left((1-\alpha)
        {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^k
        \mathbf{X}+\alpha \mathbf{X}\right),

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        alpha (float): Teleport probability :math:`\alpha \in [0, 1]`.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\frac{1}{K} \sum_{k=1}^K\left((1-\alpha)
            {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2} \right)}^k \mathbf{X}+
            \alpha \mathbf{X}\right)` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{in})`
    """

    _cached_x: Optional[Tensor]

    def __init__(self, alpha: float, K: int = 1,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.alpha = alpha
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self._cached_x = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_x = None

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings."""
        return self(*args, **filter_kwargs(self.forward, kwargs))

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        cache = self._cached_x
        if cache is None:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)
            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim), False,
                    self.add_self_loops, self.flow, dtype=x.dtype)

            h = x * self.alpha
            for _ in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
                h = h + (1 - self.alpha) / self.K * x

            if self.cached:
                self._cached_x = h
        else:
            h = cache.detach()

        return h

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'
