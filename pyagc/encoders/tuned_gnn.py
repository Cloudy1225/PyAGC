# Adapted from:
# 1. https://github.com/LUOyk1999/tunedGNN/blob/main/medium_graph/model.py
# 2. https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/basic_gnn.py

import copy
import inspect
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn.conv import (
    EdgeConv,
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils._trim_to_layer import TrimToLayer


class TunedGNN(torch.nn.Module):
    r"""An enhanced GNN model with tuned hyperparameters based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    This implementation incorporates critical improvements identified in the paper:
    - Residual connections for deeper networks and heterophilous graphs
    - Pre-linear transformation option
    - Flexible normalization (LayerNorm/BatchNorm)
    - Optimized dropout strategies
    - Support for deeper architectures (up to 10-15 layers)

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_last (bool, optional): If set to :obj:`True`, applies activation
            function to the final output. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): If set to :obj:`True`, applies a linear
            transformation before the first GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    supports_edge_weight: Final[bool]
    supports_edge_attr: Final[bool]
    supports_norm_batch: Final[bool]

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_layers: int,
            out_channels: Optional[int] = None,
            dropout: float = 0.0,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_last: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
            residual: bool = False,
            pre_linear: bool = False,
            jk: Optional[str] = None,
            **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.residual = residual
        self.pre_linear = pre_linear

        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.jk_mode = jk
        self.act_first = act_first
        self.act_last = act_last
        self.norm = norm if isinstance(norm, str) else None
        self.norm_kwargs = norm_kwargs

        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = hidden_channels

        # Pre-linear transformation (optional)
        if self.pre_linear:
            self.lin_in = Linear(in_channels, hidden_channels)
            conv_in_channels = hidden_channels
        else:
            conv_in_channels = in_channels

        # Initialize convolutional layers
        self.convs = ModuleList()

        # First layer
        if num_layers > 1:
            self.convs.append(
                self.init_conv(conv_in_channels, hidden_channels, **kwargs))
            if isinstance(conv_in_channels, (tuple, list)):
                conv_in_channels = (hidden_channels, hidden_channels)
            else:
                conv_in_channels = hidden_channels

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                self.init_conv(conv_in_channels, hidden_channels, **kwargs))
            if isinstance(conv_in_channels, (tuple, list)):
                conv_in_channels = (hidden_channels, hidden_channels)
            else:
                conv_in_channels = hidden_channels

        # Last layer
        if out_channels is not None and jk is None:
            self._is_conv_to_out = True
            self.convs.append(
                self.init_conv(conv_in_channels, out_channels, **kwargs))
        else:
            self.convs.append(
                self.init_conv(conv_in_channels, hidden_channels, **kwargs))

        # Residual connection linear layers
        if self.residual:
            self.res_lins = ModuleList()

            # Handle pre-linear case
            if self.pre_linear:
                res_in = hidden_channels
            else:
                res_in = in_channels

            # First residual layer
            if num_layers > 1:
                if isinstance(res_in, int):
                    self.res_lins.append(Linear(res_in, hidden_channels))
                else:
                    self.res_lins.append(Linear(res_in[0], hidden_channels))
                res_in = hidden_channels

            # Hidden residual layers
            for _ in range(num_layers - 2):
                self.res_lins.append(Linear(res_in, hidden_channels))

            # Last residual layer
            if out_channels is not None and jk is None:
                self.res_lins.append(Linear(res_in, out_channels))
            else:
                self.res_lins.append(Linear(res_in, hidden_channels))

        # Normalization layers
        self.norms = ModuleList()
        norm_layer = normalization_resolver(
            norm,
            hidden_channels,
            **(norm_kwargs or {}),
        )
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.supports_norm_batch = False
        if hasattr(norm_layer, 'forward'):
            norm_params = inspect.signature(norm_layer.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params

        for _ in range(num_layers - 1):
            self.norms.append(copy.deepcopy(norm_layer))

        if jk is not None:
            self.norms.append(copy.deepcopy(norm_layer))
        else:
            self.norms.append(torch.nn.Identity())

        # Jumping Knowledge
        if jk is not None and jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if jk is not None:
            if jk == 'cat':
                in_channels_jk = num_layers * hidden_channels
            else:
                in_channels_jk = hidden_channels
            self.lin = Linear(in_channels_jk, self.out_channels)

        # We define `trim_to_layer` functionality as a module such that we can
        # still use `to_hetero` on-top.
        self._trim = TrimToLayer()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.pre_linear:
            self.lin_in.reset_parameters()

        for conv in self.convs:
            conv.reset_parameters()

        if self.residual:
            for res_lin in self.res_lins:
                res_lin.reset_parameters()

        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

        if hasattr(self, 'jk'):
            self.jk.reset_parameters()

        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: Adj,
            edge_weight: OptTensor = None,
            edge_attr: OptTensor = None,
            batch: OptTensor = None,
            batch_size: Optional[int] = None,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            num_sampled_nodes_per_hop (List[int], optional): The number of
                sampled nodes per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
            num_sampled_edges_per_hop (List[int], optional): The number of
                sampled edges per hop.
                Useful in :class:`~torch_geometric.loader.NeighborLoader`
                scenarios to only operate on minimal-sized representations.
                (default: :obj:`None`)
        """
        if (num_sampled_nodes_per_hop is not None
                and isinstance(edge_weight, Tensor)
                and isinstance(edge_attr, Tensor)):
            raise NotImplementedError("'trim_to_layer' functionality does not "
                                      "yet support trimming of both "
                                      "'edge_weight' and 'edge_attr'")

        # Pre-linear transformation
        if self.pre_linear:
            x = self.lin_in(x)
            x = self.dropout(x)

        xs: List[Tensor] = []

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Trim to layer for mini-batch training
            if (not torch.jit.is_scripting()
                    and num_sampled_nodes_per_hop is not None):
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Store input for residual connection
            x_res = x

            # Convolution
            if self.supports_edge_weight and self.supports_edge_attr:
                x = conv(x, edge_index, edge_weight=edge_weight,
                         edge_attr=edge_attr)
            elif self.supports_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            elif self.supports_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)

            # Residual connection
            if self.residual:
                x = x + self.res_lins[i](x_res)

            # Apply normalization and activation for all layers except potentially the last
            if i < self.num_layers - 1 or self.jk_mode is not None:
                if self.act is not None and self.act_first:
                    x = self.act(x)

                if self.supports_norm_batch:
                    x = norm(x, batch, batch_size)
                else:
                    x = norm(x)

                if self.act is not None and not self.act_first:
                    x = self.act(x)

                x = self.dropout(x)

                if hasattr(self, 'jk'):
                    xs.append(x)

        # Jumping Knowledge aggregation
        x = self.jk(xs) if hasattr(self, 'jk') else x

        # Final linear transformation
        x = self.lin(x) if hasattr(self, 'lin') else x

        # Apply activation to final output if requested
        if self.act is not None and self.act_last :
            x = self.act(x)

        return x

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Adj,
        batch_size: int,
    ) -> Tensor:
        """Inference for a single layer."""
        x_res = x
        x = self.convs[layer](x, edge_index)[:batch_size]

        if self.residual:
            x = x + self.res_lins[layer](x_res)[:batch_size]

        if layer == self.num_layers - 1 and self.jk_mode is None:
            return x

        if self.act is not None and self.act_first:
            x = self.act(x)
        if self.norms is not None:
            x = self.norms[layer](x)
        if self.act is not None and not self.act_first:
            x = self.act(x)
        if layer == self.num_layers - 1:
            if hasattr(self, 'lin'):
                x = self.lin(x)
            # Apply act_last after final linear transformation
            if self.act is not None and self.act_last:
                x = self.act(x)

        return x

    @torch.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        device: Optional[Union[str, torch.device]] = None,
        embedding_device: Union[str, torch.device] = 'cpu',
        progress_bar: bool = False,
        cache: bool = False,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This is an efficient way to compute the output embeddings for all
        nodes in the graph.
        Only applicable in case :obj:`jk=None` or `jk='last'`.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loader
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loader.num_neighbors = [-1]`.
            device (torch.device, optional): The device to run the GNN on.
                (default: :obj:`None`)
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            progress_bar (bool, optional): If set to :obj:`True`, will print a
                progress bar during computation. (default: :obj:`False`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)
        """
        assert self.jk_mode is None or self.jk_mode == 'last'
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training
        # assert not loader.shuffle  # TODO (matthias) does not work :(
        if progress_bar:
            pbar = tqdm(total=len(self.convs) * len(loader))
            pbar.set_description('Inference')

        x_all = loader.data.x.to(embedding_device)

        # Pre-linear transformation
        if self.pre_linear:
            x_all = self.lin_in(x_all)

        if cache:

            # Only cache necessary attributes:
            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, 'adj_t'):
                    kwargs['adj_t'] = data.adj_t
                else:
                    kwargs['edge_index'] = data.edge_index

                return Data.from_dict(kwargs)

            loader = CachedLoader(loader, device=device, transform=transform)

        for i in range(self.num_layers):
            xs: List[Tensor] = []
            for batch in loader:
                x = x_all[batch.n_id].to(device)
                batch_size = batch.batch_size
                if hasattr(batch, 'adj_t'):
                    edge_index = batch.adj_t.to(device)
                else:
                    edge_index = batch.edge_index.to(device)

                x = self.inference_per_layer(i, x, edge_index, batch_size)
                xs.append(x.to(embedding_device))

                if progress_bar:
                    pbar.update(1)

            x_all = torch.cat(xs, dim=0)

        if progress_bar:
            pbar.close()

        return x_all

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers}, '
                f'residual={self.residual})')


class TunedGCN(TunedGNN):
    r"""Tuned Graph Convolutional Network based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Key improvements over standard GCN:
    - Residual connections (especially beneficial for deep networks)
    - Flexible normalization (BatchNorm/LayerNorm)
    - Optimized dropout strategies
    - Optional pre-linear transformation

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): Apply linear transformation before first
            GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality,
            while default will not.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)


class TunedGraphSAGE(TunedGNN):
    r"""Tuned GraphSAGE Network based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)


class TunedGIN(TunedGNN):
    r"""Tuned Graph Isomorphism Network based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): Apply linear transformation before first
            GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return GINConv(mlp, **kwargs)


class TunedGAT(TunedGNN):
    r"""Tuned Graph Attention Network based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
        concat (bool, optional): Concatenate attention heads. (default: :obj:`True`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): Apply linear transformation before first
            GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)


class TunedPNA(TunedGNN):
    r"""Tuned Principal Neighbourhood Aggregation Network based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): Apply linear transformation before first
            GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.PNAConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return PNAConv(in_channels, out_channels, **kwargs)


class TunedEdgeCNN(TunedGNN):
    r"""Tuned EdgeCNN (Dynamic Graph CNN) based on
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function.
            Recommended: :obj:`"batch_norm"` for large graphs, :obj:`"layer_norm"`
            for smaller graphs. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs.
            (default: :obj:`False`)
        pre_linear (bool, optional): Apply linear transformation before first
            GNN layer. (default: :obj:`False`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.EdgeConv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        mlp = MLP(
            [2 * in_channels, out_channels, out_channels],
            act=self.act,
            act_first=self.act_first,
            norm=self.norm,
            norm_kwargs=self.norm_kwargs,
        )
        return EdgeConv(mlp, **kwargs)


# Factory function for convenient model creation
def create_tuned_gnn(
        gnn_type: str,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_last: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        residual: bool = False,
        pre_linear: bool = False,
        jk: Optional[str] = None,
        **kwargs
) -> TunedGNN:
    r"""Factory function to create tuned GNN models with recommended defaults.

    This function provides an easy way to create tuned GNN models with
    hyperparameters optimized based on empirical findings from the
    `"Classic GNNs are Strong Baselines: Reassessing GNNs for
    Node Classification" <https://arxiv.org/abs/2406.08993>`_ paper (Luo et al., NeurIPS 2024).

    The function automatically filters out incompatible parameters for each GNN type
    by inspecting the model's signature, so you can safely pass all parameters
    without worrying about compatibility.

    Args:
        gnn_type (str): Type of GNN. Options: :obj:`"gcn"`, :obj:`"sage"`,
            :obj:`"gat"`, :obj:`"gatv2"`, :obj:`"gin"`, :obj:`"pna"`,
            :obj:`"edgecnn"`.
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
            Recommendation: 2-6 for homophilous graphs, 6-15 for heterophilous.
        out_channels (int, optional): Output size. If not set, will use
            :obj:`hidden_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. Paper findings suggest
            0.2-0.7 range works well. (default: :obj:`0.0`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_last (bool, optional): If set to :obj:`True`, applies activation
            function to the final output. Useful for tasks requiring non-linear
            final representations. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): Normalization type. Options:
            :obj:`"batch_norm"`, :obj:`"layer_norm"`. Paper recommends
            BatchNorm for large graphs, LayerNorm for smaller graphs.
            (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        residual (bool, optional): If set to :obj:`True`, applies residual
            connections. Especially beneficial for heterophilous graphs and
            deeper networks. (default: :obj:`False`)
        pre_linear (bool, optional): If set to :obj:`True`, applies a linear
            transformation before the first GNN layer. (default: :obj:`False`)
        jk (str, optional): Jumping Knowledge mode. Options: :obj:`None`,
            :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`.
            Paper shows this is optional but can help in some cases.
            (default: :obj:`None`)
        **kwargs: Additional GNN-specific arguments. These will be automatically
            filtered based on the GNN type. Common options include:

            - :obj:`heads` (int): Number of attention heads (GAT/GATv2 only)
            - :obj:`concat` (bool): Concatenate attention heads (GAT/GATv2 only)
            - :obj:`v2` (bool): Use GATv2 variant (GAT only, auto-set for gatv2)
            - :obj:`add_self_loops` (bool): Add self-loops to adjacency matrix
            - :obj:`normalize` (bool): Apply normalization (GCN only)
            - :obj:`improved` (bool): Use improved GCN formulation (GCN only)
            - :obj:`cached` (bool): Cache normalized edge weights (GCN only)
            - :obj:`bias` (bool): Add bias parameters
            - :obj:`aggr` (str): Aggregation scheme (e.g., "mean", "max", "add")
            - :obj:`aggregators` (List[str]): Aggregation functions (PNA only)
            - :obj:`scalers` (List[str]): Scaling functions (PNA only)
            - :obj:`deg` (Tensor): Degree histogram for normalization (PNA only)
            - :obj:`edge_dim` (int): Edge feature dimensionality (GAT/GATv2/EdgeCNN)
            - :obj:`fill_value` (float or str): Value for self-loops

    Returns:
        TunedGNN: The initialized tuned GNN model.

    Examples:
        >>> # Create a tuned GCN for homophilous graphs
        >>> model = create_tuned_gnn(
        ...     'gcn', in_channels=128, hidden_channels=256,
        ...     num_layers=3, out_channels=10, dropout=0.5,
        ...     norm='batch_norm'
        ... )

        >>> # Create a tuned GCN for heterophilous graphs (deeper + residual)
        >>> model = create_tuned_gnn(
        ...     'gcn', in_channels=128, hidden_channels=256,
        ...     num_layers=10, out_channels=10, dropout=0.5,
        ...     norm='batch_norm', residual=True, pre_linear=True
        ... )

        >>> # Create a tuned GAT with multiple attention heads
        >>> model = create_tuned_gnn(
        ...     'gat', in_channels=128, hidden_channels=256,
        ...     num_layers=3, out_channels=10, heads=4, concat=True,
        ...     dropout=0.6, norm='layer_norm'
        ... )

        >>> # Create a tuned model with custom activation
        >>> model = create_tuned_gnn(
        ...     'sage', in_channels=128, hidden_channels=256,
        ...     num_layers=3, act='elu', act_first=True,
        ...     norm='batch_norm', residual=True
        ... )

        >>> # Create a model with Jumping Knowledge
        >>> model = create_tuned_gnn(
        ...     'gcn', in_channels=128, hidden_channels=256,
        ...     num_layers=4, out_channels=10, jk='cat',
        ...     norm='layer_norm'
        ... )

        >>> # Pass all parameters - incompatible ones are automatically filtered
        >>> model = create_tuned_gnn(
        ...     'gcn', in_channels=128, hidden_channels=256,
        ...     num_layers=3, heads=4  # 'heads' will be ignored for GCN
        ... )
    """
    gnn_type = gnn_type.lower()

    model_map = {
        'gcn': TunedGCN,
        'sage': TunedGraphSAGE,
        'graphsage': TunedGraphSAGE,
        'gat': TunedGAT,
        'gatv2': TunedGAT,
        'gin': TunedGIN,
        'pna': TunedPNA,
        'edgecnn': TunedEdgeCNN,
    }

    if gnn_type not in model_map:
        raise ValueError(f"Unknown GNN type: {gnn_type}. "
                         f"Available options: {list(model_map.keys())}")

    model_class = model_map[gnn_type]

    # Get valid parameters by inspecting the model class __init__ signature
    init_signature = inspect.signature(model_class.__init__)
    valid_params = set(init_signature.parameters.keys()) - {'self'}

    # Build the complete parameter dictionary
    all_params = {
        'in_channels': in_channels,
        'hidden_channels': hidden_channels,
        'num_layers': num_layers,
        'out_channels': out_channels,
        'dropout': dropout,
        'act': act,
        'act_first': act_first,
        'act_last': act_last,
        'act_kwargs': act_kwargs,
        'norm': norm,
        'norm_kwargs': norm_kwargs,
        'residual': residual,
        'pre_linear': pre_linear,
        'jk': jk,
    }

    # Add kwargs
    all_params.update(kwargs)

    # GATv2-specific handling (set v2=True before filtering)
    if gnn_type == 'gatv2':
        all_params['v2'] = True

    # Filter to only include valid parameters for this model class
    filtered_params = {k: v for k, v in all_params.items() if k in valid_params}

    # Optional: Log filtered parameters for debugging
    filtered_out = set(all_params.keys()) - set(filtered_params.keys())
    if filtered_out:
        import warnings
        warnings.warn(
            f"The following parameters are not applicable to {gnn_type.upper()} "
            f"and will be ignored: {filtered_out}",
            UserWarning,
            stacklevel=2
        )

    return model_class(**filtered_params)


__all__ = [
    'TunedGNN',
    'TunedGCN',
    'TunedGraphSAGE',
    'TunedGAT',
    'TunedGIN',
    'TunedPNA',
    'TunedEdgeCNN',
    'create_tuned_gnn',
]
