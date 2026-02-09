from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import nn, Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


@dataclass
class LossOutput:
    r"""
    Unified loss output format for training.

    This class encapsulates both the total loss (used for backpropagation)
    and individual loss components (used for logging and monitoring).

    Args:
        total (Tensor): The total loss scalar for backpropagation.
        components (Dict[str, float]): Dictionary of individual loss components
            for logging purposes, e.g., :obj:`{'reconstruction': 0.5, 'kl': 0.3}`.

    Example:
        >>> loss_output = LossOutput(
        ...     total=torch.tensor(1.5),
        ...     components={'ali': 0.8, 'nei': 0.5, 'spa': 0.2}
        ... )
        >>> print(loss_output.log_string("Epoch 01: "))
        Epoch 01: Loss: 1.5000, ALI: 0.8000, NEI: 0.5000, SPA: 0.2000
    """
    total: Tensor
    components: Dict[str, float]

    def __float__(self) -> float:
        r"""Enables :obj:`float(loss_output)` for compatibility."""
        return float(self.total.item())

    def log_string(self, prefix: str = "") -> str:
        r"""
        Generates a formatted log string for printing.

        Args:
            prefix (str, optional): Prefix string to prepend. (default: :obj:`""`)

        Returns:
            Formatted string like :obj:`"Loss: 1.50, ALI: 0.80, NEI: 0.50"`.
        """
        parts = [f"Loss: {self.total.item():.4f}"]
        for name, value in self.components.items():
            parts.append(f"{name.upper()}: {value:.4f}")
        return f"{prefix}{', '.join(parts)}"


class BaseModel(ABC, nn.Module):
    r"""
    Base interface for all PyAGC models.

    All models must implement :meth:`embed` to produce node embeddings.
    For trainable models, inherit from :class:`TrainableModel` instead.

    Example:
        >>> class MyEncoder(BaseModel):
        ...     def embed(self, x, edge_index):
        ...         return self.encoder(x, edge_index)

    See Also:
        - :class:`TrainableModel`: For models with loss computation
        - :class:`ClusteringModel`: For end-to-end clustering
    """

    @abstractmethod
    def embed(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""
        Returns node embeddings.

        Args:
            For graph-based models: x, edge_index, ...
            For lookup-based models: batch (node indices)

        The output shape and type depend on the specific model implementation.
        Typically returns a :obj:`Tensor` of shape :obj:`(num_nodes, hidden_dim)`.
        """
        pass

    def reset_parameters(self):
        r"""Resets learnable parameters. Override when needed."""
        pass

    @torch.no_grad()
    def infer_full(self, data: Data) -> Any:
        r"""
        Full-graph inference: returns embeddings or predictions for all nodes.

        Args:
            data (Data): Input graph data.

        Returns:
            Node embeddings or predictions, typically of shape :obj:`(num_nodes, *)`.
        """
        self.eval()
        return self.embed(**data)

    @torch.no_grad()
    def infer_batch(self, loader: NeighborLoader, verbose: bool = True) -> Any:
        r"""
        Mini-batch inference over a NeighborLoader.

        For node-level outputs, concatenates only the seed nodes of each batch.

        Args:
            loader (NeighborLoader): Mini-batch data loader.
            verbose (bool, optional): If :obj:`True`, displays a progress bar.
                (default: :obj:`True`)

        Returns:
            Node embeddings or predictions for all nodes in the loader.
        """
        self.eval()
        all_z = []
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=loader.data.num_nodes)
            pbar.set_description("Inference stage")

        for batch in loader:
            batch = batch.to(device)
            z = self.embed(**batch)
            all_z.append(z[:batch.batch_size].cpu())
            if verbose:
                pbar.update(batch.batch_size)

        if verbose:
            pbar.close()

        return torch.cat(all_z, dim=0).to(device)


class TrainableModel(BaseModel):
    r"""
    Base class for trainable graph models.

    Subclasses must implement the :meth:`loss` method. This class provides
    default implementations for :meth:`train_full` and :meth:`train_batch`
    that handle both single-loss and multi-component loss outputs.

    The :meth:`loss` method can return either:
        - A single :obj:`Tensor` for simple losses
        - A :class:`LossOutput` object for losses with multiple components
    """
    def __init__(self):
        super().__init__()
        self.logger = None  # Can be set externally

    def set_logger(self, logger):
        r"""Sets a custom logger for training output."""
        self.logger = logger

    @abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> Union[Tensor, LossOutput]:
        r"""
        Computes the training loss.

        Returns:
            Either a scalar loss :obj:`Tensor`, or a :class:`LossOutput` object
            containing the total loss and individual components.
        """
        pass

    def train_full(self, data: Data, optimizer: torch.optim.Optimizer,
                   epoch: int, verbose: bool = True, **loss_kwargs: Any) -> float:
        r"""
        Runs one epoch of full-batch training.

        Args:
            data (Data): The input full graph data.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)
            **loss_kwargs: Additional keyword arguments passed to :meth:`loss`.

        Returns:
            Loss value of the epoch.
        """
        self.train()
        optimizer.zero_grad()

        # Merge data attributes with loss_kwargs
        loss_output = self.loss(**{**data, **loss_kwargs})

        # Handle both single Tensor and LossOutput returns
        if isinstance(loss_output, LossOutput):
            loss = loss_output.total
            log_str = loss_output.log_string(f"Epoch: {epoch:03d} ")
        else:
            loss = loss_output
            log_str = f"Epoch: {epoch:03d} Loss: {loss.item():.4f}"

        loss.backward()
        optimizer.step()

        if verbose:
            self.logger.info(log_str) if self.logger else print(log_str)

        return float(loss.item())

    def train_batch(self, loader: NeighborLoader, optimizer: torch.optim.Optimizer,
                    epoch: int, verbose: bool = True, **loss_kwargs: Any) -> float:
        r"""
        Runs one epoch of mini-batch training.

        Args:
            loader (NeighborLoader): The mini-batch loader.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)
            **loss_kwargs: Additional keyword arguments passed to :meth:`loss_batch`.

        Returns:
            Average loss value of the epoch.
        """
        self.train()

        if loader.input_nodes is None:
            num_nodes = loader.data.num_nodes
        else:
            num_nodes = loader.input_nodes.size(0)

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=num_nodes)
            pbar.set_description(f'Epoch {epoch:03d}')

        # Accumulate loss components across batches
        total_loss = 0.0
        components_sum = {}
        device = next(self.parameters()).device

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Pass loss_kwargs to loss_batch
            loss_output = self.loss_batch(batch, **loss_kwargs)

            # Handle both single Tensor and LossOutput returns
            if isinstance(loss_output, LossOutput):
                loss = loss_output.total
                for name, value in loss_output.components.items():
                    components_sum[name] = components_sum.get(name, 0.0) + value
            else:
                loss = loss_output

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if verbose:
                pbar.update(batch.batch_size)

        if verbose:
            pbar.close()

        # Compute average values
        num_batches = len(loader)
        avg_loss = total_loss / num_batches

        # Print loss information
        if verbose:
            if components_sum:
                parts = [f"Loss: {avg_loss:.4f}"]
                for name, value in components_sum.items():
                    avg_value = value / num_batches
                    parts.append(f"{name.upper()}: {avg_value:.4f}")
                log_str = f"Epoch: {epoch:03d} {', '.join(parts)}"
            else:
                log_str = f"Epoch: {epoch:03d} Loss: {avg_loss:.4f}"
            self.logger.info(log_str) if self.logger else print(log_str)

        return avg_loss

    def loss_batch(self, batch: Data, **kwargs: Any) -> Union[Tensor, LossOutput]:
        r"""
        Computes loss for a mini-batch.

        Default implementation simply calls :meth:`loss`. Subclasses can override
        this method to handle batch-specific logic (e.g., slicing to seed nodes).

        Args:
            batch (Data): A mini-batch from the loader.
            **kwargs: Additional keyword arguments.

        Returns:
            Loss output, same format as :meth:`loss`.
        """
        return self.loss(**{**batch, **kwargs})


class ClusteringModel(TrainableModel):
    r"""
    Base class for end-to-end clustering models.

    This class is designed for models that directly output cluster assignments
    (e.g., DMoN, MinCut) rather than embeddings. It provides a unified interface
    for clustering tasks by overriding the :meth:`infer_full` and :meth:`infer_batch`
    methods to return cluster assignments directly.

    Subclasses should implement:
        - :meth:`embed`: Returns node embeddings
        - :meth:`forward`: Returns hard cluster assignments
        - :meth:`loss`: Computes clustering loss
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""
        Returns hard cluster assignments.

        Returns:
            Cluster assignments of shape :obj:`(num_nodes,)`.
        """
        pass

    @torch.no_grad()
    def infer_full(self, data: Data) -> Tensor:
        r"""
        Full-graph inference: returns cluster assignments for all nodes.

        Args:
            data (Data): Input graph data.

        Returns:
            Cluster assignments of shape :obj:`(num_nodes,)`.
        """
        self.eval()
        return self.forward(**data)

    @torch.no_grad()
    def infer_batch(self, loader: NeighborLoader, verbose: bool = True) -> Tensor:
        r"""
        Mini-batch inference over a NeighborLoader.

        Args:
            loader (NeighborLoader): Mini-batch data loader.
            verbose (bool, optional): If :obj:`True`, displays a progress bar.
                (default: :obj:`True`)

        Returns:
            Cluster assignments for all nodes in the loader.
        """
        self.eval()
        all_pred = []
        device = next(self.parameters()).device if list(self.parameters()) else 'cpu'

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=loader.data.num_nodes)
            pbar.set_description("Inference stage")

        for batch in loader:
            batch = batch.to(device)
            pred = self.forward(**batch)
            all_pred.append(pred[:batch.batch_size])
            if verbose:
                pbar.update(batch.batch_size)

        if verbose:
            pbar.close()

        return torch.cat(all_pred, dim=0)

    @torch.no_grad()
    def initialize_cluster_centers(self, data: Data, num_layers: int, train_idx: Tensor = None, batch_size: int = 4096,
                                   fan_out: int = -1, method: str = 'kmeans', verbose: bool = True):
        r"""
        Initialize cluster centers using K-Means.

        Supports two modes:
        1. Small graphs: Use all nodes for initialization
        2. Large graphs: Use only training nodes or mini-batch inference

        Args:
            data (Data): Full graph data.
            num_layers (int): Number of encoder layers.
            train_idx (torch.Tensor, optional): Training node indices.
                If provided, only use these nodes for initialization.
                (default: :obj:`None`)
            batch_size (int, optional): Batch size for mini-batch inference.
                (default: :obj:`4096`)
            fan_out (int, optional): Number of sampled neighbors.
                (default: :obj:`-1`)
            method (str, optional): Initialization method. (default: :obj:`'kmeans'`)
            verbose (bool, optional): If :obj:`True`, prints initializing progress.
                (default: :obj:`True`)
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            if not isinstance(data, Data):
                raise TypeError("data must be a torch_geometric.data.Data object")
            # Determine which nodes to use for initialization
            use_train_only = train_idx is not None and len(train_idx) < data.num_nodes
            input_nodes = train_idx if use_train_only else None
            num_nodes = len(train_idx) if use_train_only else data.num_nodes

            if verbose:
                log_str = f"Initializing cluster centers using {'subset' if use_train_only else 'all'} {num_nodes} nodes..."
                self.logger.info(log_str) if self.logger else print(log_str)

            # Try full-batch embedding first
            try:
                if use_train_only:
                    from torch_geometric.utils import subgraph
                    edge_index_subset, _ = subgraph(
                        train_idx,
                        data.edge_index,
                        relabel_nodes=True,
                        num_nodes=data.num_nodes
                    )
                    x = data.x[train_idx].to(device)
                    edge_index = edge_index_subset.to(device)
                else:
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)

                z = self.embed(x, edge_index)

            except RuntimeError as e:
                # print(f"[Warning] Full-batch embedding failed: {e}")
                # print(f"Using mini-batch inference (batch_size={batch_size})...")

                # Mini-batch inference with NeighborLoader
                from torch_geometric.loader import NeighborLoader
                loader = NeighborLoader(
                    data,
                    input_nodes=input_nodes,
                    num_neighbors=[fan_out] * num_layers,
                    batch_size=batch_size,
                    shuffle=False
                )

                all_z = []
                for batch in loader:
                    batch = batch.to(device)
                    z_batch = self.embed(batch.x, batch.edge_index)
                    all_z.append(z_batch[:batch.batch_size].cpu())
                z = torch.cat(all_z, dim=0).to(device)

            # Normalize embeddings
            z = torch.nn.functional.normalize(z, p=2, dim=-1)

            # Run K-Means
            if method == 'kmeans':
                # print(f"Running K-Means on {z.size(0)} embeddings...")
                from pyagc.clusters.kmeans_cluster_head import KMeansClusterHead
                kmeans = KMeansClusterHead(
                    n_clusters=self.n_clusters,
                    backend='torch',
                    n_init=1,
                    max_iter=300
                )
                kmeans.fit_predict(z)

                # Set cluster centers
                self.cluster_head.reset_cluster_centers(
                    kmeans.cluster_centers.detach().to(device)
                )
            else:
                raise ValueError(f"Unknown initialization method: {method}")

        # print(f"✓ Cluster centers initialized: shape={self.cluster_head.cluster_centers.shape}")
