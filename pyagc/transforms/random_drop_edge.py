from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_drop_edge')
class RandomDropEdge(BaseTransform):
    r"""Randomly drops edges in the graph (functional name:
    :obj:`random_drop_edge`), as described in the `"DropEdge:
    Towards Deep Graph Convolutional Networks on Node Classification"
    <https://arxiv.org/abs/1907.10903>`_ paper.
    Optionally, drops associated edge attributes.

    Args:
        p (float, optional): The probability of dropping each edge.
            Must be between 0 and 1. Default is :obj:`0.5`.
        edge_attrs (List[str], optional): The names of edge attributes to drop
            alongside dropped edges. Default is :obj:`["edge_attr"]`.
        inplace (bool, optional): If set to :obj:`False`, will clone the input
            data before applying the transform. Default is :obj:`False`.
    """
    def __init__(
        self,
        p: float = 0.5,
        edge_attrs: Optional[List[str]] = ["edge_attr"],
        inplace: bool = False,
    ):
        if p < 0. or p > 1.:
            raise ValueError(f'Masking ratio has to be between 0 and 1 '
                             f'(got {p}')

        self.p = p
        self.edge_attrs = edge_attrs or []
        self.inplace = inplace

    def _drop_edge_index(self, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.p
        return edge_index[:, keep_mask], keep_mask

    def _process_store(self, store):
        if 'edge_index' not in store:
            return

        edge_index = store['edge_index']
        if edge_index.numel() == 0:
            return

        # Apply random edge dropping
        edge_index_new, keep_mask = self._drop_edge_index(edge_index)
        store['edge_index'] = edge_index_new

        # Drop corresponding edge attributes, if specified
        for attr in self.edge_attrs:
            if attr in store and store[attr].size(0) == keep_mask.size(0):
                store[attr] = store[attr][keep_mask]

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        if self.p == 0.0:
            return data  # fast path

        if not self.inplace:
            data = data.clone()

        for store in data.edge_stores:
            self._process_store(store)

        return data
