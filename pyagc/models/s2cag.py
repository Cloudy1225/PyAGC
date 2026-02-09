from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import to_undirected, add_remaining_self_loops, degree

from pyagc.models.base import BaseModel


def compute_normalized_matrices(
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int
) -> tuple[Tensor, Tensor]:
    r"""
    Computes normalized adjacency matrix and normalized feature matrix.

    Args:
        x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
        edge_index (Tensor): Edge indices of shape :obj:`(2, num_edges)`.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        Tuple of (normalized transition matrix, normalized features).
    """
    # Ensure self-loops and undirected
    edge_index = add_remaining_self_loops(edge_index, num_nodes=num_nodes)[0]
    edge_index = to_undirected(edge_index)

    # Compute degree
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=x.dtype)

    # Normalized adjacency: P_hat = D^{-1} A
    deg_inv = deg.pow(-1)
    deg_inv[torch.isinf(deg_inv)] = 0

    # Build normalized transition matrix (sparse)
    edge_weight = deg_inv[row]
    P_hat = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes)
    ).coalesce().to(x.device)

    # Normalize P_hat row-wise
    row_sums = torch.sparse.sum(P_hat, dim=1).to_dense()
    row_sums[row_sums == 0] = 1.0

    # Create normalized P_hat
    edge_weight_normalized = edge_weight / row_sums[row]
    P_hat_normalized = torch.sparse_coo_tensor(
        edge_index,
        edge_weight_normalized,
        size=(num_nodes, num_nodes)
    ).to(x.device)

    # Normalize features: X_hat_i = X_i / sqrt(sum_j X_i · X_j^T)
    # Avoid computing x @ x.T by computing row-wise norms efficiently
    # ||X_i||_2 for each row
    x_row_norms = torch.norm(x, p=2, dim=1, keepdim=True)
    x_row_norms[x_row_norms == 0] = 1.0
    x_normalized = x / x_row_norms

    return P_hat_normalized, x_normalized


def power_iteration(
        P_hat: Tensor,
        x_hat: Tensor,
        T: int,
        alpha: float
) -> Tensor:
    r"""
    Computes Normalized Smoothed Representations (NSR) via power iteration.

    .. math::
        Z = \sum_{t=0}^{T} \frac{(1-\alpha)\alpha^t}{1-\alpha^{T+1}} \hat{P}^t \hat{X}

    Args:
        P_hat (Tensor): Normalized transition matrix (sparse).
        x_hat (Tensor): Normalized feature matrix.
        T (int): Number of propagation steps.
        alpha (float): Decay factor.

    Returns:
        Node representations Z of shape :obj:`(num_nodes, num_features)`.
    """
    factor = (1 - alpha) / (1 - alpha ** (T + 1))
    Z = factor * x_hat

    for t in range(1, T + 1):
        x_hat = torch.sparse.mm(P_hat, x_hat)
        Z = Z + alpha ** t * factor * x_hat

    return Z


class S2CAG(BaseModel):
    r"""
    The S²CAG (Spectral Subspace Clustering for Attributed Graphs) model from the
    `"Spectral Subspace Clustering for Attributed Graphs"
    <https://arxiv.org/abs/2411.11074>`_ paper (Lin et al., KDD 2025).

    S²CAG performs unsupervised graph clustering by computing Normalized Smoothed
    Representations (NSR) via graph Laplacian smoothing, then extracting cluster
    structures through truncated SVD and spectral rounding.

    The model optimizes the following objective:

    .. math::
        \min_{S} \|Z - SZ\|_F^2 + \|S\|_* + \|S^T S - I\|_F^2

    where :math:`Z` is the NSR matrix, :math:`S` is the self-expressive matrix,
    and :math:`\|\cdot\|_*` denotes the nuclear norm.

    Theoretically, S²CAG is equivalent to minimizing the total conductance of
    clusters on an affinity graph constructed from NSR.

    Args:
        n_clusters (int): Number of clusters.
        T (int, optional): Number of propagation steps. (default: :obj:`10`)
        alpha (float, optional): Decay factor for smoothing. (default: :obj:`0.9`)
        tau (int, optional): Number of iterations for SVD approximation. (default: :obj:`7`)
        oversampling (int, optional): Oversampling parameter for randomized SVD.
            (default: :obj:`10`)

    Example:
        >>> from pyagc.models import S2CAG
        >>> from pyagc.data import get_dataset
        >>>
        >>> # Load data
        >>> x, edge_index, y = get_dataset('Cora', root='./data')
        >>> n_clusters = int(y.max()) + 1
        >>>
        >>> # Create model
        >>> model = S2CAG(n_clusters=n_clusters, T=20, alpha=0.9)
        >>>
        >>> # Get cluster assignments
        >>> clusters = model.embed(x, edge_index)
    """

    def __init__(
            self,
            n_clusters: int,
            T: int = 10,
            alpha: float = 0.9,
            tau: int = 7,
            oversampling: int = 10
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.T = T
        self.alpha = alpha
        self.tau = tau
        self.oversampling = oversampling

    def _compute_nsr(
            self,
            x: Tensor,
            edge_index: Tensor,
            num_nodes: Optional[int] = None
    ) -> Tensor:
        r"""Computes Normalized Smoothed Representations (NSR)."""
        if num_nodes is None:
            num_nodes = x.size(0)

        P_hat, x_hat = compute_normalized_matrices(x, edge_index, num_nodes)
        Z = power_iteration(P_hat, x_hat, self.T, self.alpha)
        return Z

    def _randomized_svd(
            self,
            Z: Tensor,
            k: int,
            n_iter: Optional[int] = None
    ) -> Tensor:
        r"""
        Computes top-k left singular vectors via randomized SVD.

        Args:
            Z (Tensor): Input matrix of shape :obj:`(n, d)`.
            k (int): Number of singular vectors to compute.
            n_iter (int, optional): Number of power iterations.

        Returns:
            Top-k left singular vectors of shape :obj:`(n, k)`.
        """
        if n_iter is None:
            n_iter = self.tau

        n, d = Z.shape
        o = self.oversampling
        k_total = min(k + o, min(n, d))

        # Random initialization
        Q = torch.randn(d, k_total, device=Z.device, dtype=Z.dtype)

        # Power iterations
        for _ in range(n_iter):
            Q = Z @ Q
            Q = Z.T @ Q

        Q = Z @ Q

        # QR decomposition
        Q, _ = torch.linalg.qr(Q)

        # Project and compute SVD
        B = (Z.T @ Q).T
        U_hat, _, _ = torch.linalg.svd(B, full_matrices=False)

        # Compute final left singular vectors
        U = Q @ U_hat

        # Skip first trivial vector and return top-k
        return U[:, 1:k + 1]

    def embed(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""
        Computes cluster assignments via S²CAG.

        Args:
            x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
            edge_index (Tensor): Edge indices of shape :obj:`(2, num_edges)`.

        Returns:
            Node embeddings of shape :obj:`(num_nodes, n_clusters)`.
        """
        num_nodes = x.size(0)

        # Compute NSR
        Z = self._compute_nsr(x, edge_index, num_nodes)

        # Compute top-k left singular vectors
        Y = self._randomized_svd(Z, self.n_clusters)

        return Y

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Alias for :meth:`embed`."""
        return self.embed(x, edge_index, **kwargs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'n_clusters={self.n_clusters}, '
                f'T={self.T}, '
                f'alpha={self.alpha})')


class MS2CAG(BaseModel):
    r"""
    The M-S²CAG (Modularity-based S²CAG) model from the
    `"Spectral Subspace Clustering for Attributed Graphs"
    <https://arxiv.org/abs/2411.11074>`_ paper (Lin et al., KDD 2025).

    M-S²CAG extends S²CAG by incorporating modularity maximization into the
    subspace clustering framework. It optimizes:

    .. math::
        \max_C \text{trace}(C^T(\hat{Z}\hat{Z}^T - \gamma \frac{\omega\omega^T}{\omega^T 1})C)

    where :math:`\hat{Z}` is normalized NSR, :math:`\omega` represents degree-like
    weights, and :math:`\gamma` controls the balance between intra-cluster and
    inter-cluster connectivity.

    Args:
        n_clusters (int): Number of clusters.
        T (int, optional): Number of propagation steps. (default: :obj:`10`)
        alpha (float, optional): Decay factor for smoothing. (default: :obj:`0.9`)
        gamma (float, optional): Modularity parameter. (default: :obj:`0.9`)
        tau (int, optional): Number of subspace iterations. (default: :obj:`50`)

    Example:
        >>> from pyagc.models import MS2CAG
        >>> from pyagc.data import get_dataset
        >>>
        >>> # Load data
        >>> x, edge_index, y = get_dataset('Cora', root='./data')
        >>> n_clusters = int(y.max()) + 1
        >>>
        >>> # Create model
        >>> model = MS2CAG(n_clusters=n_clusters, T=20, alpha=0.9, gamma=1.0)
        >>>
        >>> # Get cluster assignments
        >>> clusters = model.embed(x, edge_index)
    """

    def __init__(
            self,
            n_clusters: int,
            T: int = 10,
            alpha: float = 0.9,
            gamma: float = 0.9,
            tau: int = 50
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.T = T
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau

    def _compute_nsr(
            self,
            x: Tensor,
            edge_index: Tensor,
            num_nodes: Optional[int] = None
    ) -> Tensor:
        r"""Computes Normalized Smoothed Representations (NSR)."""
        if num_nodes is None:
            num_nodes = x.size(0)

        P_hat, x_hat = compute_normalized_matrices(x, edge_index, num_nodes)
        Z = power_iteration(P_hat, x_hat, self.T, self.alpha)
        return Z

    def embed(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""
        Computes cluster assignments via M-S²CAG.

        Args:
            x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
            edge_index (Tensor): Edge indices of shape :obj:`(2, num_edges)`.

        Returns:
            Cluster assignments of shape :obj:`(num_nodes,)`.
        """
        num_nodes = x.size(0)

        # Compute NSR
        Z = self._compute_nsr(x, edge_index, num_nodes)

        # Normalize Z: avoid computing Z @ Z.T explicitly
        # Use row-wise L2 normalization instead
        Z_norms = torch.norm(Z, p=2, dim=1, keepdim=True)
        Z_norms[Z_norms == 0] = 1.0
        Z_hat = Z / Z_norms

        # # Compute omega (degree-like weights): row sums of Z_hat @ Z_hat.T
        # omega = (Z_hat @ Z_hat.T).sum(dim=1, keepdim=True)
        # omega_sum = omega.sum()

        # Since Z_hat @ Z_hat.T is symmetric and row sums equal column sums:
        # omega_i = sum_j (Z_hat_i · Z_hat_j) = Z_hat @ (Z_hat.sum(dim=0))
        Z_hat_col_sum = Z_hat.sum(dim=0)  # (d,)
        omega = Z_hat @ Z_hat_col_sum  # (n,)
        omega = omega.unsqueeze(1)  # (n, 1)
        omega_sum = omega.sum()

        # Initialize Q with random orthonormal matrix
        Q, _ = torch.linalg.qr(torch.randn(num_nodes, self.n_clusters, device=x.device))

        # Subspace iterations
        for _ in range(self.tau):
            # H = Z_hat @ Z_hat^T @ Q - gamma * omega @ (omega^T @ Q) / omega_sum
            H = Z_hat @ (Z_hat.T @ Q)
            H = H - self.gamma * omega @ (omega.T @ Q) / omega_sum

            # Orthonormalize
            Q, _ = torch.linalg.qr(H)

        return Q

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        r"""Alias for :meth:`embed`."""
        return self.embed(x, edge_index, **kwargs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'n_clusters={self.n_clusters}, '
                f'T={self.T}, '
                f'alpha={self.alpha}, '
                f'gamma={self.gamma})')


def snem_rounding(vectors: Tensor, n_clusters: int, T: int = 100) -> Tensor:
    r"""
    SNEM (Spectral Network Embedding) rounding algorithm for clustering.

    This algorithm iteratively refines cluster assignments by alternating between
    soft assignment based on similarity to cluster prototypes and re-normalization.

    Args:
        vectors (Tensor): Node embeddings of shape :obj:`(num_nodes, n_features)`.
        n_clusters (int): Number of clusters.
        T (int, optional): Number of iterations. (default: :obj:`100`)

    Returns:
        Cluster assignments of shape :obj:`(num_nodes,)`.

    Reference:
        From S²CAG paper and SNEM algorithm.
    """
    device = vectors.device
    n_samples = vectors.size(0)

    # Initialize with hard assignment
    labels = vectors.argmax(dim=1)

    # Create one-hot encoding
    C = torch.zeros(n_samples, n_clusters, device=device, dtype=vectors.dtype)
    C[torch.arange(n_samples, device=device), labels] = 1.0

    # Normalize columns
    col_sums = C.sum(dim=0).sqrt()
    col_sums[col_sums == 0] = 1.0
    C = C / col_sums.unsqueeze(0)

    # Iterative refinement
    for _ in range(T):
        # Compute prototype matrix Q = V^T @ C
        Q = vectors.T @ C  # (n_features, n_clusters)

        # Soft assignment: V @ Q
        soft_assign = vectors @ Q  # (n_samples, n_clusters)

        # Hard assignment
        labels = soft_assign.argmax(dim=1)

        # Create one-hot encoding
        C = torch.zeros(n_samples, n_clusters, device=device, dtype=vectors.dtype)
        C[torch.arange(n_samples, device=device), labels] = 1.0

        # Normalize columns
        col_sums = C.sum(dim=0).sqrt()
        col_sums[col_sums == 0] = 1.0
        C = C / col_sums.unsqueeze(0)

    return labels


# from scipy.sparse import csc_matrix
# import numpy as np
#
# def snem_rounding_numpy(vectors: Tensor, n_clusters: int, T: int = 100) -> Tensor:
#     r"""
#     SNEM (Spectral Network Embedding) rounding algorithm for clustering.
#
#     This algorithm iteratively refines cluster assignments by alternating between
#     soft assignment based on similarity to cluster prototypes and re-normalization.
#
#     Args:
#         vectors (Tensor): Node embeddings of shape :obj:`(num_nodes, n_features)`.
#         n_clusters (int): Number of clusters.
#         T (int, optional): Number of iterations. (default: :obj:`100`)
#
#     Returns:
#         Cluster assignments of shape :obj:`(num_nodes,)`.
#
#     Reference:
#         From S²CAG paper and SNEM algorithm.
#     """
#     vectors = vectors.detach().cpu().numpy().astype(np.float64)
#     n_samples, n_feats = vectors.shape
#
#     # Initialize with hard assignment
#     labels = vectors.argmax(axis=1)
#     vectors_discrete = csc_matrix(
#         (np.ones(len(labels)), (np.arange(n_samples), labels)),
#         shape=(n_samples, n_clusters)
#     )
#
#     # Normalize columns (convert to array for element-wise operations)
#     vectors_discrete_array = vectors_discrete.toarray()
#     vectors_sum = np.sqrt(vectors_discrete_array.sum(axis=0))
#     vectors_sum[vectors_sum == 0] = 1.0
#     vectors_discrete_array = vectors_discrete_array / vectors_sum[np.newaxis, :]
#
#     # Iterative refinement
#     for _ in range(T):
#         # Compute prototype matrix Q
#         Q = vectors.T @ vectors_discrete_array
#         Q = np.asarray(Q)
#
#         # Soft assignment
#         vectors_discrete_new = vectors @ Q
#
#         # Hard assignment
#         labels = vectors_discrete_new.argmax(axis=1)
#         vectors_discrete_array = np.zeros((n_samples, n_clusters))
#         vectors_discrete_array[np.arange(n_samples), labels] = 1.0
#
#         # Normalize columns
#         vectors_sum = np.sqrt(vectors_discrete_array.sum(axis=0))
#         vectors_sum[vectors_sum == 0] = 1.0
#         vectors_discrete_array = vectors_discrete_array / vectors_sum[np.newaxis, :]
#
#     return torch.tensor(labels, dtype=torch.long)
