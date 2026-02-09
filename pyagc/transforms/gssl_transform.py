from typing import List, Optional, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('gssl_transform')
class GSSLTransform(BaseTransform):
    r"""Applies random feature masking and random edge dropping for
    Graph Self-Supervised Learning (functional name: :obj:`gssl_transform`).

    This transform is commonly used in graph self-supervised learning methods
    such as `GRACE <https://arxiv.org/abs/2006.04131>`_,
    `CCA-SSG <https://arxiv.org/abs/2106.12484>`_, and
    `BGRL <https://arxiv.org/abs/2102.06514>`_.

    For each node attribute in :obj:`node_attrs`, randomly masks features.
    For each edge attribute in :obj:`edge_attrs`, randomly drops edges.

    Works for both homogeneous and heterogeneous graphs.

    Only keeps specified node attributes and edge attributes in the returned data.

    Args:
        p_feat_mask (float, optional): Probability of masking node features. (default: :obj:`0.5`)
        p_edge_drop (float, optional): Probability of dropping edges. (default: :obj:`0.5`)
        node_attrs (List[str], optional): Node attributes to transform and keep. (default: :obj:`["x"]`)
        edge_attrs (List[str], optional): Edge attributes to transform and keep. (default: :obj:`["edge_attr"]`)
    """
    def __init__(
        self,
        p_feat_mask: float = 0.5,
        p_edge_drop: float = 0.5,
        node_attrs: Optional[List[str]] = ["x"],
        edge_attrs: Optional[List[str]] = ["edge_attr"],
    ):
        for p in (p_feat_mask, p_edge_drop):
            if p < 0. or p > 1.:
                raise ValueError(f'Masking ratio has to be between 0 and 1 '
                                 f'(got {p}')
        self.p_feat_mask = p_feat_mask
        self.p_edge_drop = p_edge_drop
        self.node_attrs = node_attrs
        self.edge_attrs = edge_attrs

    def _mask_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_feat_mask == 0.0 or x.numel() == 0:
            return x
        mask = torch.rand_like(x) < self.p_feat_mask
        x = x.clone()
        x[mask] = 0
        return x

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_attrs: List[Optional[torch.Tensor]]
    ) -> (torch.Tensor, List[Optional[torch.Tensor]]):
        if self.p_edge_drop == 0.0 or edge_index.numel() == 0:
            return edge_index, edge_attrs
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) >= self.p_edge_drop

        edge_index = edge_index[:, mask]
        new_edge_attrs = []
        for edge_attr in edge_attrs:
            if edge_attr is not None:
                new_edge_attrs.append(edge_attr[mask])
            else:
                new_edge_attrs.append(None)
        return edge_index, new_edge_attrs

    def __call__(self, *args, **kwargs) -> Union[dict, Data, HeteroData]:
        r"""
        Supports both Data object and separate arguments.

        If called with a Data object:
            transform(data) -> Data

        If called with separate arguments:
            transform(x, edge_index, ...) -> dict with transformed values
        """
        # Case 1: Called with Data/HeteroData object
        if len(args) == 1 and isinstance(args[0], (Data, HeteroData)):
            return self.forward(args[0])

        # Case 2: Called with separate arguments (x, edge_index, ...)
        # Reconstruct from args and kwargs
        result = {}

        # Handle positional args (assumed to be x, edge_index in order)
        if len(args) >= 1:
            x = args[0]
            result['x'] = self._mask_features(x)

        if len(args) >= 2:
            edge_index = args[1]
            edge_attrs_values = [kwargs.get(attr) for attr in self.edge_attrs]
            edge_index, new_edge_attrs = self._drop_edges(edge_index, edge_attrs_values)
            result['edge_index'] = edge_index

            for attr, value in zip(self.edge_attrs, new_edge_attrs):
                if value is not None:
                    result[attr] = value

        # Handle kwargs
        for key, value in kwargs.items():
            if key == 'x' and 'x' not in result:
                result['x'] = self._mask_features(value)
            elif key == 'edge_index' and 'edge_index' not in result:
                edge_attrs_values = [kwargs.get(attr) for attr in self.edge_attrs]
                edge_index, new_edge_attrs = self._drop_edges(value, edge_attrs_values)
                result['edge_index'] = edge_index
                for attr, val in zip(self.edge_attrs, new_edge_attrs):
                    if val is not None:
                        result[attr] = val
            elif key not in self.edge_attrs and key not in result:
                result[key] = value

        return result

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        out = data.__class__()

        if isinstance(data, Data):
            # Mask node attributes
            for attr in self.node_attrs:
                if hasattr(data, attr):
                    value = getattr(data, attr)
                    out[attr] = self._mask_features(value)

            # Drop edges and corresponding edge attributes
            edge_attrs_values = [getattr(data, attr, None) for attr in self.edge_attrs]
            edge_index, new_edge_attrs = self._drop_edges(
                data.edge_index, edge_attrs_values
            )
            out.edge_index = edge_index
            for attr, value in zip(self.edge_attrs, new_edge_attrs):
                if value is not None:
                    out[attr] = value

        elif isinstance(data, HeteroData):
            for node_type in data.node_types:
                for attr in self.node_attrs:
                    if attr in data[node_type]:
                        out[node_type][attr] = self._mask_features(data[node_type][attr])

            for edge_type in data.edge_types:
                edge_attrs_values = [data[edge_type].get(attr, None) for attr in self.edge_attrs]
                edge_index, new_edge_attrs = self._drop_edges(
                    data[edge_type].edge_index, edge_attrs_values
                )
                out[edge_type].edge_index = edge_index
                for attr, value in zip(self.edge_attrs, new_edge_attrs):
                    if value is not None:
                        out[edge_type][attr] = value

        return out
