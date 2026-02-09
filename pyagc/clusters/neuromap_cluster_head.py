from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_torch_coo_tensor

from pyagc.clusters import BaseClusterHead

EPS = 1e-15


def _mk_smart_teleportation_flow(
    A: Tensor, alpha: float = 0.15, n_iters: int = 100, device: str = "cpu"
) -> Tuple[Tensor, Tensor]:
    r"""
    Construct the smart teleportation flow matrix F and the stationary
    node visit probabilities p as described in the Neuromap paper.

    Args:
        A (torch.Tensor): Adjacency matrix of shape (N, N)
        alpha (float, optional): Teleportation probability. Default: 0.15
        n_iters (int, optional): Number of power iterations. Default: 100
        device (str, optional): Device for computation. Default: "cpu"

    Returns:
        - F (torch.Tensor): Flow matrix of shape (N, N)
        - p (torch.Tensor): Stationary node visit probabilities of shape (N,)
    """
    # --- Build transition matrix T ---
    T = torch.nan_to_num(
        A.T * (torch.sum(A, dim=1) ** (-1.0)).to_dense(), nan=0.0
    ).T.to(device)

    # --- Distribution according to in-degrees ---
    e_v = (torch.sum(A, dim=0) / torch.sum(A)).to_dense().to(device)

    # --- Power iteration for stationary distribution ---
    p = e_v
    for _ in range(n_iters):
        p = alpha * e_v + (1 - alpha) * (p @ T)

    # --- Smart teleportation flow matrix ---
    F = alpha * A / torch.sum(A) + (1 - alpha) * (p * T.T).T

    return F, p


def _mk_smart_teleportation_flow_sparse(
    A: torch.sparse_coo_tensor, alpha: float = 0.15, n_iters: int = 100
) -> Tuple[Tensor, Tensor]:
    r"""
    Construct the smart teleportation flow matrix F (sparse)
    and stationary node visit probabilities p.

    Args:
        A (torch.sparse_coo_tensor): Sparse adjacency matrix of shape (N, N)
        alpha (float, optional): Teleportation probability. Default: 0.15
        n_iters (int, optional): Number of power iterations. Default: 100

    Returns:
        - F (torch.sparse_coo_tensor): Sparse flow matrix
        - p (torch.Tensor): Stationary node visit probabilities (N,)
    """
    assert A.is_sparse, "A must be torch.sparse_coo_tensor"
    device = A.device
    A = A.coalesce()

    # --- Compute out-degree for each node ---
    row_sum = torch.sparse.sum(A, dim=1).to_dense()
    row_inv = torch.nan_to_num(row_sum.pow(-1), nan=0.0)

    ## --- Build transition matrix T = D^{-1}A ---
    T = torch.sparse_coo_tensor(
        A.indices(),
        A.values() * row_inv[A.indices()[0]],
        size=A.shape,
        device=device
    ).coalesce()  # must be coalesced before accessing values()

    # --- Teleportation distribution e_v based on in-degree ---
    e_v = torch.sparse.sum(A, dim=0).to_dense()
    e_v = e_v / e_v.sum()

    # --- Power iteration for stationary distribution p ---
    p = e_v.clone()
    for _ in range(n_iters):
        p_new = alpha * e_v + (1 - alpha) * torch.sparse.mm(T.T, p.unsqueeze(1)).squeeze(1)
        if torch.allclose(p_new, p, rtol=1e-6, atol=1e-9):
            break
        p = p_new

    # --- Smart teleportation flow matrix ---
    total_A = torch.sparse.sum(A)
    p_values = p[A.indices()[0]]
    F_values = alpha * A.values() / total_A + (1 - alpha) * p_values * T.values()

    F = torch.sparse_coo_tensor(
        A.indices(), F_values, size=A.shape, device=device
    ).coalesce()

    return F, p


class NeuromapClusterHead(BaseClusterHead):
    r"""
    Neuromap Clustering Head from the paper
    `"The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks"
    <https://arxiv.org/abs/2310.01144>`_ paper (Blöcker et al., NeurIPS 2024).

    This module implements a differentiable version of the map equation
    for end-to-end optimization with (graph) neural networks.

    It learns soft cluster assignments :math:`\mathbf{S}` via a linear projection
    from node embeddings :math:`\mathbf{Z}`, and computes the Neuromap loss
    (expected per-step description length) following the Minimum Description Length principle:

    .. math::

        \mathcal{L}(A, S) = q \log q
        - \sum_m q_m \log q_m
        - \sum_m m_{\text{exit}} \log m_{\text{exit}}
        - \sum_u p_u \log p_u
        + \sum_m p_m \log p_m

    where all quantities are computed from the soft cluster assignment matrix.

    Args:
        n_clusters (int): Maximum number of clusters.
        n_features (int): Feature dimension of input node embeddings.
        alpha (float, optional): Teleportation probability for flow. Default: 0.15.
        n_iters (int, optional): Number of power iterations for stationary distribution. Default: 100.
    """

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        alpha: float = 0.15,
        n_iters: int = 100,
    ):
        super(NeuromapClusterHead, self).__init__()

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha
        self.n_iters = n_iters

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

        # Cached flow matrix and p may be set externally (lazy initialization)
        self.F = None
        self.p = None
        self.p_log_p = None

    def reset_cluster_centers(self, cluster_centers: Tensor = None) -> None:
        r"""
        Manually sets the cluster centers.

        Args:
            cluster_centers (torch.Tensor, optional):
                Tensor of shape :obj:`(n_clusters, n_features)` to initialize
                the cluster centers. If None, use Xavier uniform initialization.
        """
        if cluster_centers is not None:
            assert cluster_centers.shape == (self.n_clusters, self.n_features)
            with torch.no_grad():
                self.cluster_centers.copy_(cluster_centers)
        else:
            nn.init.xavier_uniform_(self.cluster_centers)

    def build_flow(self, edge_index: Tensor, N: int):
        """Construct sparse flow matrix F and stationary distribution p."""
        device = edge_index.device
        A = to_torch_coo_tensor(edge_index, size=(N, N)).to(device)
        self.F, self.p = _mk_smart_teleportation_flow_sparse(A, alpha=self.alpha, n_iters=self.n_iters)
        self.p_log_p = torch.sum(self.p * torch.nan_to_num(torch.log2(self.p), nan=0.0))

    def forward0(self, z: Tensor, edge_index: Tensor) -> Tensor:
        r"""
        Compute the Neuromap (Map Equation) loss given node embeddings and adjacency.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            torch.Tensor: Map equation loss (codelength)
        """
        N = z.size(0)

        # If flow not built yet, initialize it
        if self.F is None or self.p is None:
            self.build_flow(edge_index, N)

        # === Step 1: Compute soft assignments ===
        sim = torch.matmul(z, self.cluster_centers.t())  # (N, K)
        S = torch.softmax(sim, dim=-1)

        # === Step 2: Pool flow to community level (sparse ops) ===
        # (1) F @ S
        FS = torch.sparse.mm(self.F, S)  # (N, K)
        # (2) Sᵀ(FS)
        C = torch.matmul(S.T, FS)  # (K, K)
        diag_C = torch.diag(C)

        # === Step 3: Compute map equation quantities ===
        q = 1.0 - torch.trace(C)
        q_m = torch.sum(C, dim=1) - diag_C
        m_exit = torch.sum(C, dim=0) - diag_C
        p_m = q_m + torch.sum(C, dim=0)

        # === Step 4: Map equation codelength ===
        codelength = (
            torch.sum(q * torch.nan_to_num(torch.log2(q), nan=0.0))
            - torch.sum(q_m * torch.nan_to_num(torch.log2(q_m), nan=0.0))
            - torch.sum(m_exit * torch.nan_to_num(torch.log2(m_exit), nan=0.0))
            - self.p_log_p
            + torch.sum(p_m * torch.nan_to_num(torch.log2(p_m), nan=0.0))
        )

        return codelength

    def forward(self, z: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Compute the Neuromap (Map Equation) loss given node embeddings and adjacency.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Map equation loss (codelength) and collapse_loss
        """
        N = z.size(0)

        # If flow not built yet, initialize it
        if self.F is None or self.p is None:
            self.build_flow(edge_index, N)

        # === Step 1: Compute soft assignments ===
        sim = torch.matmul(z, self.cluster_centers.t())  # (N, K)
        S = torch.softmax(sim, dim=-1)

        # === Step 2: Pool flow to community level (sparse ops) ===
        # (1) F @ S
        FS = torch.sparse.mm(self.F, S)  # (N, K)
        # (2) Sᵀ(FS)
        C = torch.matmul(S.T, FS)  # (K, K)
        diag_C = torch.diag(C)

        # === Step 3: Compute map equation quantities ===
        q = 1.0 - torch.trace(C)
        q_m = torch.sum(C, dim=1) - diag_C
        m_exit = torch.sum(C, dim=0) - diag_C
        p_m = q_m + torch.sum(C, dim=0)

        # === Step 4: Map equation codelength ===
        codelength = (
            torch.sum(q * torch.nan_to_num(torch.log2(q), nan=0.0))
            - torch.sum(q_m * torch.nan_to_num(torch.log2(q_m), nan=0.0))
            - torch.sum(m_exit * torch.nan_to_num(torch.log2(m_exit), nan=0.0))
            - self.p_log_p
            + torch.sum(p_m * torch.nan_to_num(torch.log2(p_m), nan=0.0))
        )

        cluster_sizes = S.sum(dim=0)  # (K,)
        collapse_loss = (cluster_sizes.norm() * (self.n_clusters ** 0.5) / N) - 1

        return codelength, collapse_loss

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""
        Predicts cluster assignments.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.
            soft (bool, optional):
                If True, returns the soft assignment matrix;
                if False, returns hard cluster assignments. (default: :obj:`False`)

        Returns:
            - If :obj:`soft` is False, :obj:`(n_samples,)` tensor of cluster indices.
            - If :obj:`soft` is True, :obj:`(n_samples, n_clusters)` tensor of probabilities.
        """
        sim = torch.matmul(z, self.cluster_centers.t())

        if soft:
            return sim.softmax(dim=-1)
        else:
            return sim.argmax(dim=-1)
