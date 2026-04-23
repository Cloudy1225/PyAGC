from typing import Callable, List, Optional, Union, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear as PyGLinear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class H2GCNConv(MessagePassing):
    r"""The approximate neighborhood aggregation operator inspired by the
    `"Beyond Homophily in Graph Neural Networks: Current Limitations and
    Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper.

    This operator is a scalable approximation of the neighborhood aggregation
    stage of :obj:`H2GCN`, where 1-hop and 2-hop information are obtained by
    repeated propagation over the same normalized adjacency matrix.

    Instead of explicitly constructing strict 1-hop and strict 2-hop
    neighborhoods, this layer computes:

    .. math::
        \mathbf{X}^{\prime} =
        \left( \mathbf{A}\mathbf{X} \ \Vert \ \mathbf{A}(\mathbf{A}\mathbf{X}) \right),

    where :math:`\mathbf{A}` denotes the normalized adjacency matrix.

    More specifically, let :math:`\mathbf{\hat{A}}` denote the normalized
    propagation matrix. The layer computes:

    .. math::
        \mathbf{X}^{(1)} = \mathbf{\hat{A}} \mathbf{X}

    and

    .. math::
        \mathbf{X}^{(2)} = \mathbf{\hat{A}} \mathbf{X}^{(1)}
        = \mathbf{\hat{A}} (\mathbf{\hat{A}} \mathbf{X}),

    and returns:

    .. math::
        \mathbf{X}^{\prime} =
        \left( \mathbf{X}^{(1)} \Vert \mathbf{X}^{(2)} \right).

    In contrast to the original :obj:`H2GCN` formulation, this implementation
    does not explicitly isolate strict 2-hop neighbors. Instead, it uses
    repeated propagation as a memory- and computation-efficient approximation
    that is more suitable for large graphs.

    Args:
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of the normalized adjacency matrix on first
            execution, and will use the cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add
            self-loops to the input graph before normalization.
            Since :obj:`H2GCN` does not explicitly mix ego representations
            during intermediate aggregation, this is usually set to
            :obj:`False`. (default: :obj:`False`)
        normalize (bool, optional): Whether to compute symmetric
            normalization coefficients on-the-fly. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, 2F)`
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index,
                        edge_weight,
                        x.size(self.node_dim),
                        improved=False,
                        add_self_loops=self.add_self_loops,
                        flow=self.flow,
                        dtype=x.dtype,
                    )
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        out_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out_2 = self.propagate(edge_index, x=out_1, edge_weight=edge_weight)

        return torch.cat([out_1, out_2], dim=-1)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)


class H2GCN(torch.nn.Module):
    r"""The approximate :obj:`H2GCN` model for node classification.

    This model follows the three-stage design of :obj:`H2GCN`:

    **1. Feature embedding stage**

    A graph-agnostic linear transformation is applied to the input node
    features:

    .. math::
        \mathbf{R}^{(0)} = \sigma(\mathbf{X} \mathbf{W}_{e})

    where :math:`\sigma` is a non-linear activation function.

    **2. Neighborhood aggregation stage**

    The embedded node features are repeatedly updated for :math:`K` rounds
    using :class:`H2GCNConv`. At each round, the representation is updated as:

    .. math::
        \mathbf{R}^{(k)} =
        \left( \mathbf{A}\mathbf{R}^{(k-1)} \Vert
        \mathbf{A}(\mathbf{A}\mathbf{R}^{(k-1)}) \right)

    where :math:`\mathbf{A}` denotes the normalized adjacency matrix.

    **3. Final representation and classification stage**

    The final node representation concatenates all intermediate
    representations:

    .. math::
        \mathbf{R}^{(\mathrm{final})} =
        \left( \mathbf{R}^{(0)} \Vert \mathbf{R}^{(1)} \Vert \cdots \Vert
        \mathbf{R}^{(K)} \right),

    and a linear transform is applied on top:

    .. math::
        \mathbf{Y} = \mathbf{R}^{(\mathrm{final})} \mathbf{W}_{c}.

    In contrast to the original :obj:`H2GCN` formulation, this implementation
    uses :math:`\mathbf{A}\mathbf{X}` and :math:`\mathbf{A}(\mathbf{A}\mathbf{X})`
    as scalable approximations of 1-hop and 2-hop neighborhood aggregation.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of the feature embedding
            :math:`\mathbf{R}^{(0)}`.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of neighborhood aggregation rounds
            :math:`K`. Must be greater than or equal to :obj:`1`.
        dropout (float, optional): Dropout probability. (default: :obj:`0.5`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        cached (bool, optional): If set to :obj:`True`, each convolution layer
            will cache the normalized adjacency matrix on first execution.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add
            self-loops to the input graph before normalization.
            (default: :obj:`False`)
        normalize (bool, optional): Whether to compute symmetric
            normalization coefficients on-the-fly. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the transform will not
            learn an additive bias. (default: :obj:`True`)

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** logits :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.5,
        act: Union[str, Callable, None] = "relu",
        act_kwargs: Optional[Dict[str, Any]] = None,
        cached: bool = False,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"'{self.__class__.__name__}' requires "
                             f"'num_layers' to be greater than or equal to 1 "
                             f"(got {num_layers})")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.embed = PyGLinear(
            in_channels,
            hidden_channels,
            bias=True,
            weight_initializer='glorot',
        )

        self.convs = ModuleList()
        for k in range(num_layers):
            self.convs.append(
                H2GCNConv(
                    cached=cached,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                )
            )

        final_in_channels = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.transform = Linear(final_in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.embed.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.transform.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        xs: List[Tensor] = []

        x = self.embed(x)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        xs.append(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        x = torch.cat(xs, dim=-1)
        x = self.transform(x)

        return x
