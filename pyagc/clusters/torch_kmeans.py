import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
import warnings as _warnings
from typing import Callable, Optional, Tuple, Union


def _distributed_sync(tensor: Tensor) -> Tensor:
    r"""Synchronizes tensors across all distributed workers."""
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.stack(tensors_gather)


def _pairwise_cosine(x1: Tensor, x2: Tensor, pairwise: bool = True) -> Tensor:
    r"""Computes pairwise Cosine distances."""
    x1 = F.normalize(x1)
    x2 = F.normalize(x2)
    if not pairwise:
        return 1 - (x1 * x2).sum(dim=-1)
    return 1 - x1.mm(x2.T)

def _pairwise_dot(x1: Tensor, x2: Tensor, pairwise: bool = True) -> Tensor:
    r"""Computes pairwise Dot distances."""
    if not pairwise:
        return - (x1 * x2).sum(dim=-1)
    return - x1.mm(x2.T)

def _pairwise_euclidean(x1: Tensor, x2: Tensor, pairwise: bool = True) -> Tensor:
    r"""Computes pairwise Euclidean distances."""
    if not pairwise:
        return (x1 - x2).pow(2).sum(dim=-1).sqrt()
    return torch.cdist(x1, x2, p=2.)


def _stable_cumsum(arr: Tensor, dim: Optional[int] = None, rtol=1e-05, atol=1e-08) -> Tensor:
    r"""Performs a numerically stable cumulative sum."""
    if dim is None:
        arr = arr.flatten()
        dim = 0
    out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
    expected = torch.sum(arr, dim=dim, dtype=torch.float64)
    if not torch.all(torch.isclose(out[-1], expected, rtol=rtol, atol=atol, equal_nan=True)):
        _warnings.warn('cumsum was found to be unstable: its last element does not correspond to sum',
                       RuntimeWarning)
    return out


def _kmeans_plusplus(X: Tensor, n_clusters: int, random_state: int, pairwise_distance: Callable,
                     n_local_trials: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Computational component for k-means++ initialization."""
    n_samples, n_features = X.size()

    generator = torch.Generator(device=str(X.device)).manual_seed(random_state)
    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(torch.log(torch.tensor(n_clusters)).item())

    # Pick first center randomly and track index of point
    #     center_id = random_state.randint(n_samples)
    center_id = torch.randint(n_samples, (1,), generator=generator, device=X.device)

    indices = torch.full((n_clusters,), -1, dtype=torch.int, device=X.device)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = pairwise_distance(centers[0, None], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        #         rand_vals = random_state.random_sample(n_local_trials) * current_pot
        rand_vals = torch.rand(n_local_trials, generator=generator, device=X.device) * current_pot
        candidate_ids = torch.searchsorted(_stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        candidate_ids.clamp_(max=closest_dist_sq.numel() - 1)

        # Compute distances to center candidates
        distance_to_candidates = pairwise_distance(X[candidate_ids], X)

        # Update closest distances squared and potential for each candidate
        torch.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=-1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


class TorchKMeans:
    r"""A PyTorch-based KMeans clustering implementation supporting both Euclidean
    and Cosine distance metrics, with optional distributed training.
    This implementation is adapted from: `Hzzone/torch_clustering
    <https://github.com/Hzzone/torch_clustering>`_.

    Args:
        metric (str, optional): Distance metric to use: ``'euclidean'`` or ``'cosine'``.
            (default: ``'euclidean'``)
        init (str or torch.Tensor, optional): Method for initialization:
            ``'k-means++'``, ``'random'`` or user-specified tensor of shape
            :obj:`(n_clusters, n_features)`. (default: ``'k-means++'``)
        random_state (int, optional): Random seed for initialization. (default: ``None``)
        n_clusters (int, optional): Number of clusters. (default: ``8``)
        n_init (int, optional): Number of times the algorithm will be run with different
            centroid seeds. (default: ``10``)
        max_iter (int, optional): Maximum number of iterations of the k-means algorithm
            for a single run. (default: ``300``)
        tol (float, optional): Relative tolerance with regards to inertia to declare convergence.
            (default: ``1e-4``)
        distributed (bool, optional): Whether to use distributed training. (default: ``False``)
        verbose (bool, optional): Whether to print progress information. (default: ``False``)
    """
    @torch.no_grad()
    def __init__(self,
                 metric: str = 'euclidean',
                 init: Union[str, Tensor] = 'k-means++',
                 random_state: Optional[int] = None,
                 n_clusters: int = 8,
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 distributed: bool = False,
                 verbose: bool = False):
        self.metric = metric.lower()
        if metric not in {'euclidean', 'cosine'}:
            raise ValueError(
                'Invalid metric value. Must be either "euclidean" or "cosine".'
                ' But got "{}".'.format(metric)
            )
        self.distance_metric = {'euclidean': _pairwise_euclidean, 'cosine': _pairwise_cosine}[metric]
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_: Optional[Tensor] = None
        self.init = init
        if isinstance(self.init, torch.Tensor):
            self.n_init = 1
        if random_state is None:
            random_state = 0
        self.random_state = random_state
        self.is_root_worker = not dist.is_initialized() or dist.get_rank() == 0
        self.verbose = verbose and self.is_root_worker
        self.distributed = distributed and dist.is_initialized()
        if self.verbose and self.distributed:
            print('Perform K-means in distributed mode.')
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

    @torch.no_grad()
    def initialize(self, X: Tensor, random_state: int) -> Tensor:
        r"""Initializes the cluster centers.

        Args:
            X (torch.Tensor): The input data of shape :obj:`(n_samples, n_features)`.
            random_state (int): The random seed.

        Returns:
            Initialized cluster centers of shape :obj:`(n_clusters, n_features)`.
        """
        num_samples = X.size(0)
        if isinstance(self.init, str):
            generator = torch.Generator().manual_seed(random_state)
            if self.init == 'random':
                indices = torch.randperm(num_samples, generator=generator)[:self.n_clusters]
                init_state = X[indices]
            elif self.init == 'k-means++':
                init_state, _ = _kmeans_plusplus(X, n_clusters=self.n_clusters,
                                                 random_state=random_state,
                                                 pairwise_distance=self.distance_metric)
            else:
                raise NotImplementedError(f"Unknown init method: {self.init}")
        elif isinstance(self.init, Tensor):
            init_state = self.init.to(X)
        else:
            raise NotImplementedError
        return init_state

    @torch.no_grad()
    def fit_predict(self, X: Tensor) -> Tensor:
        r"""Performs k-means clustering on the input data and returns cluster labels.

        Args:
            X (torch.Tensor): The input data of shape :obj:`(n_samples, n_features)`.

        Returns:
            Cluster assignments of shape :obj:`(n_samples,)`.
        """
        tol = torch.mean(torch.var(X, dim=0)) * self.tol

        min_inertia, best_states, best_labels = float('inf'), None, None

        random_states = torch.arange(self.n_init * self.world_size) + self.random_state
        random_states = random_states[self.rank::self.world_size]

        self.stats = {'state': [], 'inertia': [], 'label': []}

        for n_init in range(self.n_init):
            random_state = int(random_states[n_init])
            old_state = self.initialize(X, random_state=random_state)
            old_labels, inertia = self._predict(X, old_state)

            labels = old_labels

            progress_bar = tqdm.tqdm(total=self.max_iter, disable=not self.verbose)

            for n_iter in range(self.max_iter):
                # Compute new cluster centers
                state = torch.zeros_like(old_state)
                counts = torch.zeros(self.n_clusters, dtype=X.dtype, device=X.device) + 1e-6
                counts.index_add_(0, labels, torch.ones_like(labels, dtype=X.dtype))
                state.index_add_(0, labels, X)
                state = state / counts.view(-1, 1)

                # Compute new labels and inertia
                labels, inertia = self._predict(X, state)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_states, best_labels = state, labels

                if self.verbose:
                    progress_bar.set_description(
                        f'nredo {n_init + 1}/{self.n_init:02d}, iteration {n_iter:03d} with inertia {inertia:.2f}')
                    progress_bar.update(1)

                if torch.equal(labels, old_labels):
                    if self.verbose:
                        print(f"Converged at iteration {n_iter}: strict convergence.")
                    break
                else:
                    center_shift_tot = self.distance_metric(old_state, state, pairwise=False).sum()
                    if center_shift_tot <= tol:
                        if self.verbose:
                            print(f"Converged at iteration {n_iter}: center shift "
                                  f"{center_shift_tot:.2e} within tolerance {tol:.2e}.")
                        break

                old_labels[:] = labels
                old_state = state

            progress_bar.close()
            self.stats['state'].append(old_state)
            self.stats['inertia'].append(inertia)
            self.stats['label'].append(old_labels)

        self.stats['state'] = torch.stack(self.stats['state'])
        self.stats['inertia'] = torch.tensor(self.stats['inertia'])
        self.stats['label'] = torch.stack(self.stats['label'])

        if self.distributed:
            min_inertia = _distributed_sync(torch.tensor(min_inertia))
            best_idx = torch.argmin(min_inertia).item()
            dist.broadcast(best_labels, src=best_idx)
            dist.broadcast(best_states, src=best_idx)
            self.stats['state'] = _distributed_sync(self.stats['state'])
            self.stats['inertia'] = _distributed_sync(self.stats['inertia'])
            self.stats['label'] = _distributed_sync(self.stats['label'])

        if self.verbose:
            print(f"Final min inertia {min_inertia.item():.2f}.")

        self.cluster_centers_ = best_states
        return best_labels

    @torch.no_grad()
    def _predict(self, X: Tensor, cluster_centers_: Tensor = None) -> Tuple[Tensor, float]:
        r"""Assigns each sample in :obj:`X` to the nearest cluster center.

        Args:
            X (torch.Tensor): Input data of shape :obj:`(n_samples, n_features)`.
            cluster_centers_ (torch.Tensor, optional): Precomputed cluster centers.
                If :obj:`None`, uses :obj:`self.cluster_centers_`.

        Returns:
            1. Cluster labels of shape :obj:`(num_nodes,)`.
            2. Total inertia (float scalar).
        """
        if cluster_centers_ is None:
            cluster_centers_ = self.cluster_centers_

        dist_mat = self.distance_metric(X, cluster_centers_)
        dists, labels = dist_mat.min(dim=1)
        inertia = dists.sum().item()
        return labels, inertia

        # split_size = min(4096, X.size(0))
        # all_labels = []
        # inertia = 0.0
        #
        # for chunk in X.split(split_size, dim=0):
        #     dist_mat = self.distance_metric(chunk, cluster_centers_)
        #     dists, labels = dist_mat.min(dim=1)
        #     inertia += dists.sum().item()
        #     all_labels.append(labels)
        #
        # return torch.cat(all_labels, dim=0), inertia

    @torch.no_grad()
    def predict(self, X: Tensor, soft: bool = False) -> Tensor:
        r"""Assigns samples to clusters based on fixed cluster centers.

        This function computes the squared Euclidean distance to each center and
        returns either hard assignments or soft probabilities.

        Args:
            X (torch.Tensor): Input tensor of shape :obj:`(n_samples, n_features)`.
            soft (bool, optional):
                If True, returns the soft assignment matrix;
                if False, returns hard cluster assignments. (default: :obj:`False`)

        Returns:
            - If :obj:`soft` is False, :obj:`(n_samples,)` tensor of cluster indices.
            - If :obj:`soft` is True, :obj:`(n_samples, n_clusters)` tensor of probabilities.
        """
        if self.cluster_centers_ is None or self.cluster_centers_.numel() == 0:
            raise RuntimeError("Must call `fit_predict` before using `cluster`.")

        dists = self.distance_metric(X, self.cluster_centers_)  # (n_samples, n_clusters)

        if soft:
            return (-dists.sqrt()).softmax(dim=-1)  # smaller distance => higher score
        else:
            return dists.argmin(dim=-1)  # assign to nearest cluster center

    def __repr__(self) -> str:
        """String representation of the TorchKMeans object."""
        return (
            f"TritonKMeans(metric={self.metric!r}, "
            f"init={self.init!r}, "
            f"n_clusters={self.n_clusters}, "
            f"n_init={self.n_init}, "
            f"max_iter={self.max_iter}, "
            f"tol={self.tol}, "
            f"random_state={self.random_state}, "
            f"verbose={self.verbose})"
        )

if __name__ == '__main__':
    clustering_model = TorchKMeans(metric='euclidean',
                                     init='k-means++',
                                     random_state=0,
                                     n_clusters=1000,
                                     n_init=10,
                                     max_iter=300,
                                     tol=1e-4,
                                     distributed=False,
                                     verbose=True)
    X = torch.randn(1280, 16)
    clustering_model.fit_predict(X)
