from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from pyagc.models.base import BaseModel
from pyagc.models.sgc import SGC


class SAGSC(BaseModel):
    r"""
    The Scalable Attributed-Graph Subspace Clustering (SAGSC) model from the
    `"Scalable Attributed-Graph Subspace Clustering"
    <https://ojs.aaai.org/index.php/AAAI/article/view/25918>`_ paper (Fettal et al., AAAI 2023).

    SAGSC consists of two stages:

    **(1) Node Embedding Generation (SGC):**

    .. math::
        H = S^p X

    where :math:`S = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}` is the normalized adjacency
    and :math:`p` is the propagation order.

    **(2) Implicit Subspace Clustering:**

    SAGSC avoids learning a dense coefficient matrix :math:`C \in \mathbb{R}^{n \times n}`
    by imposing a low–rank factorization:

    .. math::
        C = U U^\top,\quad U^\top U = I_k.

    :math:`U` is obtained from the top-:math:`k` left singular vectors of :math:`H`.

    To construct a **non–negative affinity matrix**, SAGSC uses a *quadratic kernel*
    applied row-wise:

    .. math::
        Q = \Phi(U), \qquad M = Q Q^\top \ge 0.

    Then the normalized matrix

    .. math::
        \tilde{Q} = Q D^{-1/2},

    is decomposed via SVD, and the rows of the top :math:`k` singular vectors
    (excluding the trivial first one) are clustered using :math:`k`-means.

    SAGSC performs *implicit spectral clustering* without explicitly forming
    the affinity matrix :math:`M \in \mathbb{R}^{n \times n}`.

    Args:
        p (int): Power of the SGC propagation matrix.
        k (int): Number of clusters (also feature dimension of encoder outputs).
        bias (float): Bias term in the quadratic feature map :math:`\Phi`.
    """

    def __init__(
        self,
        p: int,
        k: int,
        bias: float = 2 ** -0.5,   # same as authors' code
    ):
        super().__init__()
        self.p = p
        self.k = k
        self.bias = bias

        # SGC encoder (no parameters, only propagation)
        self.encoder = SGC(K=p, cached=False)

    # --------------------------------------------------------------------------
    # Feature map Phi(U): quadratic kernel (row-wise)
    # --------------------------------------------------------------------------
    def _quad_feature_map(self, U: Tensor) -> Tensor:
        r"""
        Applies a row-wise quadratic polynomial feature map :math:`\Phi` used in SAGSC.

        For each row :math:`\mathbf{u} \in \mathbb{R}^k`:

        .. math::
            \Phi(\mathbf{u}) = [
                u_1^2, \ldots, u_k^2,
                \sqrt{2} u_1 u_2, \ldots, \sqrt{2} u_{k-1} u_k,
                \sqrt{2b} u_1, \ldots, \sqrt{2b} u_k,
                b
            ]

        producing a feature dimensionality :math:`m = \frac{(k+2)(k+1)}{2}`.

        Args:
            U (Tensor): Input matrix of shape :obj:`(n, k)`.

        Returns:
            Transformed features of shape :obj:`(n, m)`.
        """
        n, k = U.size()
        device = U.device

        # squared terms
        sq = U ** 2  # (n, k)

        # Cross terms: u_i * u_j for i < j
        cross_terms = []
        for i in range(k):
            for j in range(i + 1, k):
                cross_terms.append((U[:, i] * U[:, j]) * (2.0 ** 0.5))
        cross_terms = torch.stack(cross_terms, dim=1) if len(cross_terms) > 0 else None

        # linear terms
        linear = U * (2.0 * self.bias) ** 0.5

        # bias term
        bias_col = torch.full((n, 1), self.bias, device=device)

        parts = [sq]
        if cross_terms is not None:
            parts.append(cross_terms)
        parts.append(linear)
        parts.append(bias_col)

        return torch.cat(parts, dim=1)

    # --------------------------------------------------------------------------
    # Core SAGSC embedding
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def embed(self,  *args, **kwargs) -> Tensor:
        r"""
        Computes the SAGSC latent representation used for clustering.

        Steps:
            1. :math:`H = S^p X` (SGC propagation)
            2. :math:`U =` top-:math:`k` left singular vectors of :math:`H`
            3. :math:`Q = \Phi(U)`
            4. :math:`D =` row sum vector of :math:`Q Q^\top` (implemented implicitly)
            5. :math:`\tilde{Q} = Q D^{-1/2}`
            6. :math:`Z =` singular vectors 2 to :math:`k+1` of :math:`\tilde{Q}`

        Args:
            x (Tensor): Node features of shape :obj:`(n, d)`.
            edge_index (Adj): Edge indices.
            edge_weight (Tensor, optional): Edge weights.

        Returns:
            Spectral embedding of shape :obj:`(n, k)`.
        """
        # -------- Step 1: SGC propagation ----------
        H = self.encoder.embed( *args, **kwargs)    # (n, d)

        # -------- Step 2: top-k left singular vectors ----------
        # H = U Σ Vᵀ → want U[:, :k]
        U, _, _ = torch.svd_lowrank(H, q=self.k)
        # U: (n, k)

        # -------- Step 3: quadratic kernel feature map ----------
        Q = self._quad_feature_map(U)  # (n, m)

        # -------- Step 4: compute D = row sums of M = Q Q^T ----------
        # D_i = sum_j M[i,j] = (Q_i · Q_j)_j = ||Q_i||^2
        D = Q.pow(2).sum(dim=1) + 1e-10        # (n,)
        D_inv_sqrt = torch.where(D > 0, D.rsqrt(), torch.zeros_like(D))

        # -------- Step 5: row normalization ----------
        Q_tilde = Q * D_inv_sqrt.unsqueeze(1)   # (n, m)

        # -------- Step 6: obtain spectral embedding ----------
        # Singular vectors 2 to k+1 (skip first)
        U2, _, _ = torch.svd_lowrank(Q_tilde, q=self.k + 1)
        Z = U2[:, 1:self.k+1]   # (n, k)

        return Z

    # --------------------------------------------------------------------------
    # Same as embed()
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor] = None):
        r"""
        Returns the spectral embedding.
        """
        return self.embed(x, edge_index, edge_weight)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(p={self.p}, k={self.k})'
