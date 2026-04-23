# This code is adapted from the following source:
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/sgformer.py


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.attention import SGFormerAttention
from torch_geometric.nn.models.sgformer import GraphModule
from torch_geometric.utils import to_dense_batch


class SGModule(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_heads=1,
        dropout=0.5,
    ):
        super().__init__()

        self.attns = torch.nn.ModuleList()
        self.fcs = torch.nn.ModuleList()
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.LayerNorm(hidden_channels))
        for _ in range(num_layers):
            self.attns.append(
                SGFormerAttention(hidden_channels, num_heads, hidden_channels))
            self.bns.append(torch.nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for attn in self.attns:
            attn.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x: Tensor, batch: Optional[Tensor] = None):
        # If batch is provided, sort it as to_dense_batch requires sorted batch;
        # if batch is None, to_dense_batch treats all nodes as a single graph.
        if batch is not None:
            batch, indices = batch.sort(stable=True)
            rev_perm = torch.empty_like(indices)
            rev_perm[indices] = torch.arange(len(indices), device=indices.device)
            x = x[indices]

        x, mask = to_dense_batch(x, batch)
        layer_ = []

        x = self.fcs[0](x)
        x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)

        for i, attn in enumerate(self.attns):
            x = attn(x, mask)
            x = (x + layer_[i]) / 2.
            x = self.bns[i + 1](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        x_mask = x[mask]

        # Reverse the sorting only if reordering was applied
        if batch is not None:
            x_mask = x_mask[rev_perm]

        return x_mask


class SGFormer(torch.nn.Module):
    r"""The sgformer module from the
    `"SGFormer: Simplifying and Empowering Transformers for
    Large-Graph Representations"
    <https://arxiv.org/abs/2306.10759>`_ paper.

    SGFormer integrates a **global attention module** and a **GNN module**
    to jointly capture:
    - global all-pair node interactions (Transformer-style attention)
    - local structural information (GNN message passing)

    **1. Simplified Global Attention**

    Given input node features :math:`Z^{(0)} \in \mathbb{R}^{N \times d}`:

    .. math::

        Q = f_Q(Z^{(0)}), \quad
        K = f_K(Z^{(0)}), \quad
        V = f_V(Z^{(0)})

    Normalize:

    .. math::

        \tilde{Q} = \frac{Q}{\|Q\|_F}, \quad
        \tilde{K} = \frac{K}{\|K\|_F}

    Define diagonal normalization:

    .. math::

        D = \operatorname{diag}\left(1 + \frac{1}{N} \tilde{Q}(\tilde{K}^\top \mathbf{1}) \right)

    The attention output is:

    .. math::

        Z = \beta D^{-1} \left( V + \frac{1}{N} \tilde{Q}(\tilde{K}^\top V) \right)
        + (1 - \beta) Z^{(0)}

    This formulation achieves **linear complexity :math:`O(N)`**
    compared to :math:`O(N^2)` in standard Transformers :contentReference[oaicite:0]{index=0}.

    **2. GNN-based Local Propagation**

    Structural information is incorporated via a GNN:

    .. math::

        Z_{\text{gnn}} = \mathrm{GN}(Z^{(0)}, A)

    where :math:`A` is the adjacency matrix.

    **3. Aggregation Strategy**

    The global and local representations are combined as:

    **(a) Weighted sum (add):**

    .. math::

        Z_{\text{out}} = (1 - \alpha) Z + \alpha Z_{\text{gnn}}

    **(b) Concatenation (cat):**

    .. math::

        Z_{\text{out}} = [Z \, \| \, Z_{\text{gnn}}]

    **4. Output Layer**

    .. math::

        \hat{Y} = f_O(Z_{\text{out}})

    where :math:`f_O` is a linear projection.

    Args:
        in_channels (int): Input channels.
        hidden_channels (int): Hidden channels.
        out_channels (int): Output channels.
        trans_num_layers (int): The number of layers for all-pair attention.
            (default: :obj:`2`)
        trans_num_heads (int): The number of heads for attention.
            (default: :obj:`1`)
        trans_dropout (float): Global dropout rate.
            (default: :obj:`0.5`)
        gnn_num_layers (int): The number of layers for GNN.
            (default: :obj:`3`)
        gnn_dropout (float): GNN dropout rate.
            (default: :obj:`0.5`)
        graph_weight (float): The weight balance global and gnn module.
            (default: :obj:`0.5`)
        aggregate (str): Aggregate type.
            (default: :obj:`add`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        trans_num_layers: int = 2,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        gnn_num_layers: int = 3,
        gnn_dropout: float = 0.5,
        graph_weight: float = 0.5,
        aggregate: str = 'add',
    ):
        super().__init__()
        self.trans_conv = SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
        )
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = torch.nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = torch.nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

        self.out_channels = out_channels

    def reset_parameters(self) -> None:
        self.trans_conv.reset_parameters()
        self.graph_conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
        """
        x1 = self.trans_conv(x, batch)
        x2 = self.graph_conv(x, edge_index)
        if self.aggregate == 'add':
            x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
        else:
            x = torch.cat((x1, x2), dim=1)
        return self.fc(x)
