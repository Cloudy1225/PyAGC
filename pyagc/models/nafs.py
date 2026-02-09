from typing import Optional, List, Union
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected, add_remaining_self_loops
import torch.nn.functional as F

from pyagc.models.base import BaseModel


class NAFS(BaseModel):
    r"""Node-Adaptive Feature Smoothing (NAFS) from the
    `"NAFS: A Simple yet Tough-to-beat Baseline for Graph Representation Learning"
    <https://arxiv.org/abs/2206.08583>`_ paper (Zhang et al., ICML 2022).

    NAFS is a training-free method that constructs node representations by adaptively
    smoothing node features over different propagation steps. It addresses two key
    limitations of GNNs: over-smoothing and scalability.

    The method consists of two operations:

    **(1) Node-Adaptive Feature Smoothing:**

    For each smoothing step :math:`k`, compute:

    .. math::
        X^{(k)} = \hat{A}^k X

    where :math:`\hat{A} = \tilde{D}^{r-1} \tilde{A} \tilde{D}^{-r}` is the normalized
    adjacency matrix with parameter :math:`r`.

    Then adaptively combine them using smoothing weights:

    .. math::
        \hat{X} = \sum_{k=0}^{K} W^{(k)} X^{(k)}

    where :math:`W^{(k)}` are diagonal weight matrices computed based on the
    over-smoothing distance:

    .. math::
        D_i(k) = \text{dist}([X^{(k)}]_i, X_i)

    **(2) Feature Ensemble:**

    Combine smoothed features from different :math:`r` values:

    .. math::
        Z = \bigoplus_{t=1}^{T} \hat{X}^{(t)}

    where :math:`\bigoplus` can be concatenation, mean, or max pooling.

    Args:
        K (int): Maximum number of smoothing steps. (default: :obj:`20`)
        r_list (List[float], optional): List of normalization parameters for feature
            ensemble. (default: :obj:`[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]`)
        ensemble (str, optional): Ensemble strategy, one of :obj:`'mean'`, :obj:`'max'`,
            or :obj:`'concat'`. (default: :obj:`'concat'`)
        distance (str, optional): Distance function for computing over-smoothing
            distance, one of :obj:`'cosine'` or :obj:`'euclidean'`.
            (default: :obj:`'cosine'`)

    Example:
        >>> from pyagc.models import NAFS
        >>> from pyagc.data import get_dataset
        >>>
        >>> # Load data
        >>> x, edge_index, y = get_dataset('Cora', root='./data')
        >>>
        >>> # Create model
        >>> model = NAFS(K=20, ensemble='concat')
        >>>
        >>> # Get embeddings
        >>> embeddings = model.embed(x, edge_index)
        >>>
        >>> # Use for clustering
        >>> from pyagc.clusters import KMeansClusterHead
        >>> kmeans = KMeansClusterHead(n_clusters=7)
        >>> pred = kmeans.fit_predict(embeddings)
    """

    def __init__(
        self,
        K: int = 20,
        r_list: Optional[List[float]] = None,
        ensemble: str = 'concat',
        distance: str = 'cosine',
    ):
        super().__init__()
        self.K = K
        self.r_list = r_list if r_list is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.ensemble = ensemble
        self.distance = distance

        if ensemble not in ('mean', 'max', 'concat'):
            raise ValueError(f"Invalid ensemble strategy: {ensemble}")
        if distance not in ('cosine', 'euclidean'):
            raise ValueError(f"Invalid distance function: {distance}")

    def _normalize_adj(
        self,
        edge_index: Adj,
        num_nodes: int,
        r: float = 0.5
    ) -> Tensor:
        r"""Normalize adjacency matrix with parameter r.

        .. math::
            \hat{A} = \tilde{D}^{r-1} \tilde{A} \tilde{D}^{-r}
        """
        # Add self-loops
        edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
        edge_index = to_undirected(edge_index)

        # Compute degree
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))

        # Compute normalization: D^{r-1} * A * D^{-r}
        deg_inv_left = deg.pow(r - 1)
        deg_inv_left[torch.isinf(deg_inv_left)] = 0.0

        deg_inv_right = deg.pow(-r)
        deg_inv_right[torch.isinf(deg_inv_right)] = 0.0

        # Create normalized sparse adjacency
        edge_weight = deg_inv_left[row] * deg_inv_right[col]

        adj_normalized = torch.sparse_coo_tensor(
            edge_index,
            edge_weight,
            size=(num_nodes, num_nodes)
        ).coalesce()

        return adj_normalized

    def _compute_distance(self, x: Tensor, x_smoothed: Tensor) -> Tensor:
        r"""Compute over-smoothing distance between original and smoothed features."""
        if self.distance == 'cosine':
            # Cosine similarity as distance (1 - similarity)
            x_norm = F.normalize(x, p=2, dim=1)
            x_smoothed_norm = F.normalize(x_smoothed, p=2, dim=1)
            similarity = (x_norm * x_smoothed_norm).sum(dim=1)
            # Convert to distance (higher means farther from original)
            return 1.0 - similarity
        else:  # euclidean
            return torch.norm(x - x_smoothed, p=2, dim=1)

    def _compute_smoothing_weights(
        self,
        features_list: List[Tensor],
        x_original: Tensor
    ) -> Tensor:
        r"""Compute node-adaptive smoothing weights using softmax normalization.

        Args:
            features_list: List of smoothed features at different steps
            x_original: Original node features

        Returns:
            Weight matrix of shape :obj:`(n_nodes, K+1)`
        """
        n_nodes = x_original.size(0)
        K = len(features_list) - 1

        # Compute distances for all smoothing steps
        distances = []
        for x_k in features_list:
            dist = self._compute_distance(x_original, x_k)
            distances.append(dist)

        # Stack distances: (n_nodes, K+1)
        distances = torch.stack(distances, dim=1)

        # Apply softmax to get weights
        weights = F.softmax(distances, dim=1)

        return weights

    def _feature_smoothing(
        self,
        x: Tensor,
        edge_index: Adj,
        r: float = 0.5
    ) -> Tensor:
        r"""Perform node-adaptive feature smoothing for a given r value."""
        num_nodes = x.size(0)

        # Get normalized adjacency matrix
        adj_norm = self._normalize_adj(edge_index, num_nodes, r)

        # Compute smoothed features at different steps
        features_list = [x]  # X^(0)
        x_current = x

        for _ in range(self.K):
            # X^(k) = A * X^(k-1)
            x_current = torch.sparse.mm(adj_norm, x_current)
            features_list.append(x_current)

        # Compute adaptive weights
        weights = self._compute_smoothing_weights(features_list, x)

        # Weighted combination
        output = torch.zeros_like(x)
        for k, x_k in enumerate(features_list):
            output += weights[:, k:k+1] * x_k

        return output

    @torch.no_grad()
    def embed(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        r"""Generate node embeddings using NAFS.

        Args:
            x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
            edge_index (Tensor): Edge indices.

        Returns:
            Node embeddings of shape :obj:`(num_nodes, num_features * T)` if
            ensemble='concat', otherwise :obj:`(num_nodes, num_features)`.
        """
        smoothed_features = []

        # Feature smoothing with different r values
        for r in self.r_list:
            x_smoothed = self._feature_smoothing(x, edge_index, r)
            smoothed_features.append(x_smoothed)

        # Feature ensemble
        if self.ensemble == 'mean':
            output = torch.stack(smoothed_features).mean(dim=0)
        elif self.ensemble == 'max':
            output = torch.stack(smoothed_features).max(dim=0)[0]
        else:  # concat
            output = torch.cat(smoothed_features, dim=1)

        return output

    def forward(self, x: Tensor, edge_index: Adj, **kwargs) -> Tensor:
        r"""Alias for :meth:`embed`."""
        return self.embed(x, edge_index, **kwargs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'K={self.K}, '
                f'r_list={self.r_list}, '
                f'ensemble={self.ensemble})')
