import torch
from torch import Tensor
from typing import Optional

from sklearn.cluster import KMeans
from pyagc.clusters import TorchKMeans
from pyagc.clusters import BaseClusterHead
from pyagc.utils import pairwise_squared_distance


class KMeansClusterHead(BaseClusterHead):
    r"""The K-Means clustering head with fixed cluster centers.

    This module performs clustering using the :class:`~pyagc.cluster.TorchKMeans`
    or :class:`sklearn.cluster.KMeans` algorithm, and stores the resulting
    cluster centers for inference. Once fitted, the :meth:`cluster` method can be
    used to assign new points based on the stored centers.

    .. note::
        This class does not learn trainable parameters and does not define
        a clustering loss. It is typically used for post-hoc or plug-in clustering.

    Args:
        n_clusters (int): Number of clusters.
        backend (str, optional): The backend to use for K-Means, either :obj:`"torch"`
            or :obj:`"triton"` or :obj:`"sklearn"`. (default: :obj:`"torch"`)
        n_init (int, optional): Number of K-Means initializations to run.
            (default: :obj:`10`)
        max_iter (int, optional): Maximum number of iterations per K-Means run.
            (default: :obj:`300`)
        random_state (int, optional): Random seed.
            (default: :obj:`None`)
    """

    def __init__(
        self,
        n_clusters: int,
        backend: str = "torch",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        if backend not in ("torch", "triton", "sklearn"):
            raise ValueError(f"Invalid backend: '{backend}'. Expected 'torch', 'triton' or 'sklearn'")

        self.n_clusters = n_clusters
        self.backend = backend
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.register_buffer("cluster_centers", torch.empty(0))

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            "KMeansClusterHead does not support loss computation via `forward`."
        )

    @torch.no_grad()
    def fit_predict(self, z: Tensor) -> Tensor:
        r"""Performs k-means clustering on the input data and returns cluster labels.

        Args:
            z (torch.Tensor): The input data of shape :obj:`(n_samples, n_features)`.

        Returns:
            Cluster assignments of shape :obj:`(n_samples,)`.
        """
        if self.backend == "torch":
            kmeans = TorchKMeans(
                metric='euclidean',
                init='k-means++',
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=False,
            )
            labels = kmeans.fit_predict(z)
            self.cluster_centers = kmeans.cluster_centers_.detach()
        elif self.backend == "triton":
            from pyagc.clusters.triton_kmeans import TritonKMeans
            kmeans = TritonKMeans(
                metric='euclidean',
                init='k-means++',
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=False,
                dtype=z.dtype,
                device=z.device,
            )
            labels = kmeans.fit_predict(z)
            self.cluster_centers = kmeans.cluster_centers_.detach()
        else:
            kmeans = KMeans(
                init='k-means++',
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=False,
            )
            labels_np = kmeans.fit_predict(z.detach().cpu().numpy())
            labels = torch.tensor(labels_np, dtype=torch.long, device=z.device)
            centers = torch.tensor(kmeans.cluster_centers_,
                                   dtype=z.dtype, device=z.device)
            self.cluster_centers = centers

        return labels

    @torch.no_grad()
    def cluster(self, z: Tensor, soft: bool = False) -> Tensor:
        r"""Assigns samples to clusters based on fixed cluster centers.

        This function computes the squared Euclidean distance to each center and
        returns either hard assignments or soft probabilities.

        Args:
            z (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.
            soft (bool, optional):
                If True, returns the soft assignment matrix;
                if False, returns hard cluster assignments. (default: :obj:`False`)

        Returns:
            - If :obj:`soft` is False, :obj:`(n_samples,)` tensor of cluster indices.
            - If :obj:`soft` is True, :obj:`(n_samples, n_clusters)` tensor of probabilities.
        """
        if self.cluster_centers.numel() == 0:
            raise RuntimeError("Must call `fit_predict` before using `cluster`.")

        dist = pairwise_squared_distance(z, self.cluster_centers)  # (n_samples, n_clusters)

        if soft:
            return (-dist.sqrt()).softmax(dim=-1)  # smaller distance => higher score
        else:
            return dist.argmin(dim=-1)  # assign to nearest cluster center
