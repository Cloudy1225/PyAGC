from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import degree, negative_sampling

from pyagc.clusters import BaseClusterHead

EPS = 1e-15


class SBMClusterHead(BaseClusterHead):
    r"""
    Stochastic Block Model (SBM) Clustering Head from the paper
    `"Differentiable Community Detection with Graph Neural Networks and Stochastic Block Models"
    <https://openreview.net/forum?id=T1vdfm1THf>`_ (Arliss & Mueller, LoG 2025).

    This head learns cluster assignments by maximizing the likelihood of an SBM-based generative model.
    It supports both Bernoulli and Poisson variants, with optional degree correction.

    The cluster assignment matrix :math:`\mathbf{P} \in [0,1]^{N \times K}` is obtained via softmax
    transformation of the similarity between node embeddings and learnable cluster centers, and the
    structure matrix :math:`\mathbf{\Theta} \in \mathbb{R}^{K \times K}` is estimated via MLE as:

    .. math::
        \hat{\Theta}_{ij} = \frac{M_{ij}}{n_i n_j}

    where :math:`M_{ij}` is the number of edges between communities :math:`i` and :math:`j`,
    and :math:`n_i` is the number of nodes in community :math:`i`.

    **Loss Functions:**

    **(1) Bernoulli SBM:**

    .. math::
        \mathcal{L}_B = -\sum_{(u,v) \in E} \ln(\pi_{uv}) - \eta^{-1} \sum_{(u,v) \notin E} \ln(1 - \pi_{uv})

    where :math:`\pi_{uv} = \mathbf{P}_u^T \hat{\Theta} \mathbf{P}_v`.

    **(2) Poisson SBM:**

    .. math::
        \mathcal{L}_P = -\sum_{(u,v) \in E} [\ln(\pi_{uv}) - \pi_{uv}] + \eta^{-1} \sum_{(u,v) \notin E} \pi_{uv}

    **(3) Degree-Corrected variants:**

    For degree correction, the expected value becomes :math:`\phi_u \phi_v \mathbf{P}_u^T \hat{\Theta} \mathbf{P}_v`,
    where:

    .. math::
        \hat{\phi}_u = (\mathbf{P}_u^T \mathbf{n}) \frac{d_u}{\mathbf{P}_u^T \boldsymbol{\delta}}

    with :math:`\boldsymbol{\delta}` being the sum of degrees in each community.

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of input node embeddings.
        variant (str, optional): SBM variant to use. Options: :obj:`'bernoulli'`, :obj:`'poisson'`,
            :obj:`'bernoulli-dc'`, :obj:`'poisson-dc'`. (default: :obj:`'bernoulli'`)
        eta (float, optional): Negative sampling ratio (number of negative samples per positive edge).
            (default: :obj:`3.0`)
    """

    def __init__(
            self,
            n_clusters: int,
            n_features: int,
            variant: str = 'bernoulli',
            eta: float = 3.0,
    ):
        super().__init__()
        if variant not in ('bernoulli', 'poisson', 'bernoulli-dc', 'poisson-dc'):
            raise ValueError(f"Invalid variant: '{variant}'. Expected one of: "
                             "'bernoulli', 'poisson', 'bernoulli-dc', 'poisson-dc'")

        self.n_clusters = n_clusters
        self.n_features = n_features
        self.variant = variant
        self.eta = eta
        self.degree_corrected = variant.endswith('-dc')

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

    def reset_cluster_centers(self, cluster_centers: Optional[Tensor] = None) -> None:
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

    def _estimate_structure_matrix(
            self,
            P: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        r"""
        Estimates the structure matrix :math:`\hat{\Theta}` using MLE.

        Args:
            P (torch.Tensor): Soft partition matrix of shape :obj:`(N, K)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            Structure matrix of shape :obj:`(K, K)`.
        """
        # Community sizes: n = sum_u P_u  (K,)
        n = P.sum(dim=0).clamp(min=EPS)  # (K,)

        # Edge count matrix: M_ij = sum_{(u,v) in E} P_ui * P_vj
        # Efficient implementation: M = P[src].T @ P[dst]
        src, dst = edge_index
        M = P[src].T @ P[dst]  # (K, K)

        # MLE: Theta_ij = M_ij / (n_i * n_j)
        Theta = M / (torch.outer(n, n).clamp(min=EPS))

        return Theta

    def _estimate_degree_correction(
            self,
            P: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        r"""
        Estimates the degree correction vector :math:`\hat{\phi}` using MLE.

        Args:
            P (torch.Tensor): Soft partition matrix of shape :obj:`(N, K)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            Degree correction vector of shape :obj:`(N,)`.
        """
        N = P.size(0)
        device = P.device

        # Node degrees
        d = degree(edge_index[0], N, dtype=P.dtype).view(-1, 1)  # (N, 1)

        # Community sizes
        n = P.sum(dim=0).clamp(min=EPS)  # (K,)

        # Sum of degrees per community: delta_i = sum_u P_ui * d_u
        delta = (P.T @ d).squeeze(-1).clamp(min=EPS)  # (K,)

        # phi_u = (P_u^T n) * d_u / (P_u^T delta)
        P_n = (P @ n.unsqueeze(-1)).squeeze(-1)  # (N,)
        P_delta = (P @ delta.unsqueeze(-1)).squeeze(-1).clamp(min=EPS)  # (N,)

        phi = P_n * d.squeeze(-1) / P_delta

        # For Bernoulli variant, clamp to [0, 1]
        if 'bernoulli' in self.variant:
            phi = phi.clamp(max=1.0)

        return phi

    def forward(
            self,
            z: Tensor,
            edge_index: Tensor,
            num_neg_samples: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes the SBM loss.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.
            num_neg_samples (int, optional): Number of negative samples. If None,
                uses :obj:`eta * num_edges`. (default: :obj:`None`)

        Returns:
            Tuple of (likelihood_loss, regularization_loss).
        """
        N = z.size(0)
        device = z.device

        # Compute soft partition via similarity to cluster centers
        sim = torch.matmul(z, self.cluster_centers.t())  # (N, K)
        P = torch.softmax(sim, dim=-1)  # (N, K)

        # Estimate structure matrix
        Theta = self._estimate_structure_matrix(P, edge_index)

        # Estimate degree correction if needed
        phi = None
        if self.degree_corrected:
            phi = self._estimate_degree_correction(P, edge_index)

        # Number of positive and negative samples
        num_pos = edge_index.size(1)
        if num_neg_samples is None:
            num_neg_samples = int(self.eta * num_pos)

        # Sample negative edges
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=N, num_neg_samples=num_neg_samples
        )

        # Compute edge probabilities for positive and negative edges
        src_pos, dst_pos = edge_index
        src_neg, dst_neg = neg_edge_index

        # Efficient probability computation for sampled edges only
        if phi is not None:
            pi_pos = (P[src_pos] * (Theta @ P[dst_pos].T).T).sum(dim=1)
            pi_pos = phi[src_pos] * pi_pos * phi[dst_pos]

            pi_neg = (P[src_neg] * (Theta @ P[dst_neg].T).T).sum(dim=1)
            pi_neg = phi[src_neg] * pi_neg * phi[dst_neg]
        else:
            pi_pos = (P[src_pos] * (Theta @ P[dst_pos].T).T).sum(dim=1)
            pi_neg = (P[src_neg] * (Theta @ P[dst_neg].T).T).sum(dim=1)

        pi_pos = pi_pos.clamp(min=EPS, max=1.0 - EPS)
        pi_neg = pi_neg.clamp(min=EPS, max=1.0 - EPS)

        # Compute loss based on variant
        if self.variant.startswith('bernoulli'):
            loss_pos = -torch.log(pi_pos).mean()
            loss_neg = -torch.log(1 - pi_neg).mean()
        else:  # poisson
            loss_pos = -(torch.log(pi_pos) - pi_pos).mean()
            loss_neg = pi_neg.mean()
        likelihood_loss = loss_pos + loss_neg / self.eta

        # Regularization: encourage diagonal dominance
        reg_loss = torch.norm(1.0 - Theta.diag())

        return likelihood_loss, reg_loss

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""
        Predicts cluster assignments.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            soft (bool, optional): If True, returns soft assignments; otherwise hard assignments.
                (default: :obj:`False`)

        Returns:
            Cluster assignments.
        """
        sim = torch.matmul(z, self.cluster_centers.t())
        if soft:
            return sim.softmax(dim=-1)
        else:
            return sim.argmax(dim=-1)


class SBMMatchClusterHead(BaseClusterHead):
    r"""
    Graph Matching SBM Clustering Head from the paper
    `"Differentiable Community Detection with Graph Neural Networks and Stochastic Block Models"
    <https://openreview.net/forum?id=T1vdfm1THf>`_ (Arliss & Mueller, LoG 2025).

    This variant uses the Graph Matching objective, which aligns the graph with its
    community representation by minimizing:

    .. math::
        \mathcal{L}_{Match} = -\mathrm{tr}(\mathbf{A}^T \mathbf{P} \hat{\Theta} \mathbf{P}^T)

    This approach exploits matrix sparsity and is significantly faster than edge sampling methods.

    Args:
        n_clusters (int): Number of clusters.
        n_features (int): Feature dimension of input node embeddings.
    """

    def __init__(self, n_clusters: int, n_features: int):
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features

        # Cluster centers are learnable parameters.
        self.cluster_centers = nn.Parameter(torch.empty(n_clusters, n_features))
        self.reset_cluster_centers()

    def reset_cluster_centers(self, cluster_centers: Optional[Tensor] = None) -> None:
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

    def _estimate_structure_matrix(
            self,
            P: Tensor,
            edge_index: Tensor
    ) -> Tensor:
        r"""Estimates the structure matrix using MLE."""
        n = P.sum(dim=0).clamp(min=EPS)

        src, dst = edge_index
        M = P[src].T @ P[dst]  # (K, K)

        Theta = M / (torch.outer(n, n).clamp(min=EPS))
        return Theta

    def forward(self, z: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Computes the Graph Matching loss.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, E)`.

        Returns:
            Tuple of (likelihood_loss, regularization_loss).
        """
        # Compute soft partition via similarity to cluster centers
        sim = torch.matmul(z, self.cluster_centers.t())  # (N, K)
        P = torch.softmax(sim, dim=-1)  # (N, K)

        # Estimate structure matrix
        Theta = self._estimate_structure_matrix(P, edge_index)

        # Compute A^T @ P efficiently using sparse operations
        # A^T @ P means for each node u, sum P_v over all v that connect to u
        src, dst = edge_index
        A_T_P = torch.zeros_like(P)
        A_T_P.index_add_(0, dst, P[src])

        # Compute trace(A^T @ P @ Theta @ P^T) efficiently
        # = trace(P^T @ A^T @ P @ Theta)
        # = sum_i sum_j (P^T @ A^T @ P)_ij * Theta_ji
        P_T_A_T_P = P.T @ A_T_P  # (K, K)
        likelihood_loss = -torch.trace(P_T_A_T_P @ Theta) / edge_index.size(1)

        # Regularization: encourage diagonal dominance
        reg_loss = torch.norm(1.0 - Theta.diag())

        return likelihood_loss, reg_loss

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""
        Predicts cluster assignments.

        Args:
            z (torch.Tensor): Node embeddings of shape :obj:`(N, F)`.
            soft (bool, optional): If True, returns soft assignments; otherwise hard assignments.
                (default: :obj:`False`)

        Returns:
            Cluster assignments.
        """
        sim = torch.matmul(z, self.cluster_centers.t())
        if soft:
            return sim.softmax(dim=-1)
        else:
            return sim.argmax(dim=-1)
