from typing import List, Union, Optional

from torch import Tensor
from torch.distributions import Bernoulli
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random_mask_feat')
class RandomMaskFeat(BaseTransform):
    r"""Randomly masks columns of node and/or edge feature tensors (functional name:
    :obj:`random_mask_feat`), as described in the `"Graph Contrastive Learning
    with Augmentations" <https://arxiv.org/abs/2010.13902>`_ paper.

    Args:
        p (float, optional): The probability of masking a feature column.
            Default is :obj:`0.5`.
        node_attrs (List[str], optional): The names of node features to mask.
            Default is :obj:`["x"]`.
        edge_attrs (List[str], optional): The names of edge features to mask.
            Default is :obj:`[]`.
        inplace (bool, optional): If set to :obj:`False`, will clone the input
            data object and feature tensors before applying the transform.
            Default is :obj:`False`.
    """
    def __init__(
        self,
        p: float = 0.5,
        node_attrs: Optional[List[str]] = ["x"],
        edge_attrs: Optional[List[str]] = [],
        inplace: bool = False,
    ):
        if p < 0. or p > 1.:
            raise ValueError(f'Masking ratio has to be between 0 and 1 '
                             f'(got {p}')

        self.p = p
        self.node_attrs = node_attrs or []
        self.edge_attrs = edge_attrs or []
        self.inplace = inplace
        self.dist = Bernoulli(p)

    def _mask_attrs(self, stores, attrs):
        r"""Applies random column masking to the given attributes in each store."""
        for store in stores:
            for attr in attrs:
                if attr in store:
                    feat: Tensor = store[attr]
                    if feat.numel() == 0:
                        continue  # Skip empty tensors
                    mask = self.dist.sample((feat.size(-1),)).bool().to(feat.device)
                    feat[:, mask] = 0
                    store[attr] = feat

    def forward(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        r"""Applies random feature masking to node and edge features."""

        # Fast path: skip transform if masking probability is zero
        if self.p == 0.0:
            return data

        # Clone the input graph if not applying in-place
        if not self.inplace:
            data = data.clone()

        # Mask node and edge feature attributes
        self._mask_attrs(data.node_stores, self.node_attrs)
        self._mask_attrs(data.edge_stores, self.edge_attrs)

        return data
