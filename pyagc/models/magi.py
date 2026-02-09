from typing import Union, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.inits import reset
from torch_geometric.sampler import NodeSamplerInput
# from torch_sparse import SparseTensor
from torch_geometric.typing import SparseTensor

from pyagc.models.base import TrainableModel, LossOutput
from pyagc.utils import filter_kwargs


class MAGI(TrainableModel):
    r"""The MAGI (Modularity-Aware Graph clustering via contrastive learnIng) model
    from the `"Revisiting Modularity Maximization for Graph Clustering: A Contrastive
    Learning Perspective" <https://arxiv.org/abs/2406.14288>`_ paper (Liu et al., KDD 2024).

    MAGI establishes the connection between modularity maximization and graph contrastive
    learning, where positive and negative samples are naturally guided by the modularity
    matrix. The model uses a community-aware pretext task based on two-stage random walks
    to capture high-order proximity within communities.

    The loss function follows InfoNCE-style contrastive learning:

    .. math::
        \mathcal{L} = -\sum_{v \in \mathcal{B}} \sum_{u \in \mathcal{M}^+_v}
        \log \frac{\exp(\mathbf{z}_v^\top \mathbf{z}_u / \tau)}
        {\sum_{u \in \mathcal{M}^+_v} \exp(\mathbf{z}_v^\top \mathbf{z}_u / \tau) +
        \sum_{u' \in \mathcal{M}^-_v} \exp(\mathbf{z}_v^\top \mathbf{z}_{u'} / \tau)}

    where:

    - :math:`\mathcal{M}^+_v = \{u \mid B_{vu} > 0\}` are positive samples (same community).
    - :math:`\mathcal{M}^-_v = \{u \mid B_{vu} \leq 0\}` are negative samples (different communities).
    - :math:`B_{vu}` is the modularity coefficient computed via two-stage random walks.
    - :math:`\tau` is the temperature parameter.

    Args:
        encoder (torch.nn.Module): The GNN encoder module.
        tau (float, optional): Temperature parameter for contrastive loss.
            (default: :obj:`0.5`)
        scale_embeddings (bool, optional): Whether to apply min-max scaling to embeddings
            before normalization. (default: :obj:`True`)
    """

    def __init__(
        self,
        encoder: Module,
        tau: float = 0.5,
        scale_embeddings: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.scale_embeddings = scale_embeddings

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)

    def embed(self, *args, **kwargs) -> Tensor:
        r"""Computes node embeddings.

        Returns:
            Node embeddings of shape :obj:`(num_nodes, hidden_dim)`.
        """
        z = self.encoder(*args, **filter_kwargs(self.encoder.forward, kwargs))
        if self.scale_embeddings:
            z = self._scale(z)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def _scale(self, z: Tensor) -> Tensor:
        r"""Applies min-max scaling to embeddings."""
        z_max = z.max(dim=-1, keepdim=True)[0]
        z_min = z.min(dim=-1, keepdim=True)[0]
        z_std = (z - z_min) / (z_max - z_min + 1e-20)
        return z_std

    def _compute_loss(
            self,
            z: Tensor,
            pos_pairs: Tensor,
    ) -> Tensor:
        r"""Computes the InfoNCE contrastive loss based on positive sample pairs.

        Args:
            z (torch.Tensor): Normalized embeddings of shape :obj:`(num_nodes, hidden_dim)`.
            pos_pairs (torch.Tensor): Positive sample pairs of shape :obj:`(2, num_pos_edges)`,
                where each column :obj:`[i, j]` indicates that nodes :obj:`i` and :obj:`j`
                are in the same community.

        Returns:
            Scalar loss tensor.
        """
        # Compute full similarity matrix: S[i,j] = z_i^T z_j
        sim = torch.matmul(z, z.T) / self.tau  # (num_nodes, num_nodes)

        # For numerical stability, subtract max along each row
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        # Compute exponentials of similarities
        exp_sim = torch.exp(sim)  # (num_nodes, num_nodes)

        # Compute denominator: sum over all non-self nodes
        # Use subtraction to exclude diagonal (self-loops) without in-place operations
        denom = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim).unsqueeze(1)  # (num_nodes, 1)

        # Extract positive pairs
        src, dst = pos_pairs  # (num_pos_edges,)

        # Compute log probabilities for positive pairs
        log_prob = sim[src, dst] - torch.log(denom[src].squeeze(-1) + 1e-15)

        # Average over all positive pairs
        loss = -log_prob.mean()

        return loss

    def loss(self, x: Tensor, edge_index: Tensor, pos_pairs: Tensor, **kwargs) -> LossOutput:
        r"""Computes the MAGI contrastive loss for full-graph training.

        Args:
            x (torch.Tensor): Node feature matrix of shape :obj:`(num_nodes, num_features)`.
            edge_index (torch.Tensor): Edge indices of shape :obj:`(2, num_edges)`.
            pos_pairs (torch.Tensor): Precomputed positive sample pairs of shape
                :obj:`(2, num_pos_edges)` based on modularity matrix.
            **kwargs: Additional arguments for the encoder.

        Returns:
            LossOutput containing the total loss.
        """
        z = self.embed(x, edge_index, **kwargs)
        loss = self._compute_loss(z, pos_pairs)

        return LossOutput(
            total=loss,
            components={'contrastive': loss.item()}
        )

    def loss_batch(self, batch: Data) -> LossOutput:
        r"""Computes loss for a mini-batch with expanded seed nodes.

        In mini-batch training, the positive pairs are defined only among the
        expanded batch seed nodes (obtained via two-stage random walks).

        Args:
            batch (Data): A mini-batch from the MAGI loader containing:
                - :obj:`x`: Node features (including neighbors)
                - :obj:`edge_index`: Sampled edges
                - :obj:`pos_pairs`: Positive pairs for expanded batch seed nodes
                - :obj:`expanded_batch_size`: Number of expanded seed nodes

        Returns:
            LossOutput containing the total loss.
        """
        z = self.embed(batch.x, batch.edge_index)

        # Slice embeddings to expanded batch seed nodes
        z_batch = z[:batch.expanded_batch_size]

        # Positive pairs are already relative to expanded batch nodes
        loss = self._compute_loss(z_batch, batch.pos_pairs)

        return LossOutput(
            total=loss,
            components={'contrastive': loss.item()}
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"encoder={self.encoder}, "
                f"tau={self.tau})")


class MAGIRandomWalkSampler:
    r"""Two-stage random walk sampler for MAGI model.

    This sampler implements the two-stage random walk strategy described in
    the MAGI paper to construct mini-batch modularity matrices and positive
    sample pairs.

    Stage 1 (S1): Sample multiple sub-communities by performing random walks
    from root nodes and filtering nodes visited more than the mean frequency.

    Stage 2 (S2): Compute similarity matrix via random walks on the expanded
    batch and derive modularity-based positive/negative pairs.

    Args:
        adj (torch_sparse.SparseTensor): Full graph adjacency matrix in SparseTensor format.
        num_walks (int, optional): Number of random walks per node (wt in paper).
            (default: :obj:`20`)
        walk_length (int, optional): Length of each random walk (wl in paper).
            (default: :obj:`4`)
    """

    def __init__(
        self,
        adj: SparseTensor,
        num_walks: int = 20,
        walk_length: int = 4,
    ):
        self.adj = adj.cpu()
        self.num_walks = num_walks
        self.walk_length = walk_length

    def stage1_expand_batch(self, root_nodes: Tensor) -> Tensor:
        r"""Stage 1: Expand root nodes to sub-communities via random walks.

        For each root node, perform random walks and keep nodes that are visited
        more frequently than the average.

        Args:
            root_nodes (torch.Tensor): Root node indices of shape :obj:`(num_roots,)`.

        Returns:
            Expanded batch nodes of shape :obj:`(expanded_size,)`.
        """
        num_roots = root_nodes.size(0)

        # Repeat root nodes for multiple walks
        start_nodes = root_nodes.repeat_interleave(self.num_walks)  # (num_roots * num_walks,)

        # Perform random walks (excluding starting nodes)
        walks = self.adj.random_walk(start_nodes, self.walk_length)

        # Handle both tuple and tensor returns
        if isinstance(walks, tuple):
            walks = walks[0]

        walks = walks[:, 1:]  # Exclude start node, shape: (num_roots * num_walks, walk_length)

        # Reshape to (num_roots, num_walks * walk_length)
        walks = walks.reshape(num_roots, -1)

        # Collect expanded batch
        expanded_batch = []
        for i in range(num_roots):
            rw_nodes, counts = torch.unique(walks[i], return_counts=True)
            mean_count = counts.float().mean()

            # Keep nodes visited more than mean
            mask = counts > mean_count
            selected = rw_nodes[mask].tolist()
            expanded_batch.extend(selected)

        # Add original root nodes
        expanded_batch.extend(root_nodes.tolist())

        # Remove duplicates and convert to tensor
        expanded_batch = torch.tensor(list(set(expanded_batch)), dtype=torch.long)

        return expanded_batch

    def stage2_compute_positive_pairs(self, expanded_batch: Tensor) -> Tensor:
        r"""Stage 2: Compute positive sample pairs based on modularity matrix.

        Perform random walks on expanded batch nodes to compute a similarity matrix,
        then derive modularity coefficients and identify positive pairs.

        Args:
            expanded_batch (torch.Tensor): Expanded batch node indices of shape
                :obj:`(expanded_size,)`.

        Returns:
            Positive sample pairs of shape :obj:`(2, num_pos_pairs)`, where each column
            :obj:`[i, j]` contains local indices (relative to expanded_batch) of nodes
            in the same community.
        """
        batch_size = expanded_batch.size(0)

        # Repeat for multiple walks
        start_nodes = expanded_batch.repeat_interleave(self.num_walks)

        # Perform random walks
        walks = self.adj.random_walk(start_nodes, self.walk_length)

        if isinstance(walks, tuple):
            walks = walks[0]

        walks = walks[:, 1:]  # Exclude start node
        walks = walks.reshape(batch_size, -1)

        # Build visit count matrix
        row_indices = []
        col_indices = []
        values = []

        # Create mapping from global to local indices
        global_to_local = {int(node): i for i, node in enumerate(expanded_batch)}

        for i in range(batch_size):
            visited_nodes, counts = torch.unique(walks[i], return_counts=True)

            for node, count in zip(visited_nodes, counts):
                node_int = int(node)
                if node_int in global_to_local:
                    j = global_to_local[node_int]
                    row_indices.append(i)
                    col_indices.append(j)
                    values.append(int(count))

        # Build sparse visit matrix
        if len(values) > 0:
            row = torch.tensor(row_indices, dtype=torch.long)
            col = torch.tensor(col_indices, dtype=torch.long)
            val = torch.tensor(values, dtype=torch.float)

            # Compute similarity matrix S[i,j] = visit_count(i->j) / total_visits(i)
            row_sums = torch.zeros(batch_size)
            row_sums.scatter_add_(0, row, val)

            # Normalize by row sums
            sim_values = val / (row_sums[row] + 1e-15)

            # Compute modularity matrix: B[i,j] = S[i,j] - 1/|batch|
            mod_values = sim_values - (1.0 / batch_size)

            # Extract positive pairs (B[i,j] > 0)
            pos_mask = mod_values > 0
            pos_row = row[pos_mask]
            pos_col = col[pos_mask]

            pos_pairs = torch.stack([pos_row, pos_col], dim=0)  # (2, num_pos_pairs)
        else:
            # Fallback: no valid pairs found, return empty tensor
            pos_pairs = torch.empty((2, 0), dtype=torch.long)

        return pos_pairs

    def sample(self, root_nodes: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Performs two-stage sampling for a batch of root nodes.

        Args:
            root_nodes (torch.Tensor): Root node indices of shape :obj:`(num_roots,)`.

        Returns:
            Tuple of:
                - Expanded batch nodes of shape :obj:`(expanded_size,)`
                - Positive sample pairs of shape :obj:`(2, num_pos_pairs)`
        """
        # Stage 1: Expand batch
        expanded_batch = self.stage1_expand_batch(root_nodes)

        # Stage 2: Compute positive pairs
        pos_pairs = self.stage2_compute_positive_pairs(expanded_batch)

        return expanded_batch, pos_pairs


class MAGINeighborLoader(NeighborLoader):
    r"""A specialized neighbor loader for MAGI that performs two-stage random walk
    sampling to construct mini-batches with community-aware structure.

    This loader extends :class:`~torch_geometric.loader.NeighborLoader` by:
        1. Expanding initial seed nodes to sub-communities (Stage 1)
        2. Computing positive sample pairs based on modularity (Stage 2)
        3. Sampling neighbors for the expanded batch nodes

    Args:
        data (Data or HeteroData): The graph data object.
        num_neighbors (List[int]): Number of neighbors to sample per layer.
        num_walks (int, optional): Number of random walks per node for MAGI sampling.
            (default: :obj:`20`)
        walk_length (int, optional): Length of random walks. (default: :obj:`4`)
        batch_size (int, optional): Number of seed nodes per batch.
            (default: :obj:`128`)
        **kwargs: Additional arguments for :class:`NeighborLoader`.

    Example:
        >>> from torch_geometric.datasets import Planetoid
        >>> from pyagc.models.magi import MAGINeighborLoader
        >>> data = Planetoid(root='data', name='Cora')[0]
        >>> loader = MAGINeighborLoader(
        ...     data,
        ...     num_neighbors=[10, 10],
        ...     num_walks=20,
        ...     walk_length=4,
        ...     batch_size=128,
        ... )
        >>> for batch in loader:
        ...     # batch.x: node features
        ...     # batch.edge_index: sampled edges
        ...     # batch.pos_pairs: positive pairs for contrastive loss
        ...     # batch.expanded_batch_size: size of expanded batch
        ...     pass
    """

    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: List[int],
        num_walks: int = 20,
        walk_length: int = 4,
        batch_size: int = 128,
        **kwargs,
    ):
        # Initialize parent NeighborLoader
        # We'll override the sampling behavior in collate_fn
        super().__init__(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            **kwargs,
        )

        # Build adjacency matrix for random walks
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes
        self.adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(num_nodes, num_nodes),
        )
        self.adj.fill_value_(1.0)

        # Initialize MAGI random walk sampler
        self.magi_sampler = MAGIRandomWalkSampler(
            adj=self.adj,
            num_walks=num_walks,
            walk_length=walk_length,
        )

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Data:
        r"""Modified collate function that performs MAGI two-stage sampling.

        Args:
            index (Tensor or List[int]): Indices of seed nodes in this batch.

        Returns:
            Data object containing sampled subgraph with additional attributes:
                - :obj:`pos_pairs`: Positive sample pairs
                - :obj:`expanded_batch_size`: Number of expanded seed nodes
        """
        if not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)

        # Get original seed nodes
        input_data: NodeSamplerInput = self.input_data[index]
        original_seed_nodes = input_data.node

        # Stage 1: Expand batch via random walks
        expanded_batch, pos_pairs = self.magi_sampler.sample(original_seed_nodes)

        expanded_batch_size = expanded_batch.size(0)

        # Create new input data with expanded batch as seed nodes
        expanded_input_data = NodeSamplerInput(
            # input_id=torch.arange(expanded_batch_size),
            input_id=expanded_batch,
            node=expanded_batch,
            time=None,
            input_type=input_data.input_type,
        )

        # Perform neighbor sampling on expanded batch
        out = self.node_sampler.sample_from_nodes(expanded_input_data)

        if self.filter_per_worker:
            out = self.filter_fn(out)

        # Add MAGI-specific attributes
        out.pos_pairs = pos_pairs
        out.expanded_batch_size = expanded_batch_size
        out.batch_size = index.size(0)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'num_walks={self.magi_sampler.num_walks}, '
                f'walk_length={self.magi_sampler.walk_length})')


def precompute_full_graph_positive_pairs(
    data: Data,
    num_walks: int = 100,
    walk_length: int = 2,
) -> Tensor:
    r"""Precomputes positive sample pairs for full-graph training.

    This function performs the two-stage random walk on all nodes in the graph
    and returns positive pairs based on the modularity matrix.

    Args:
        data (Data): The graph data object.
        num_walks (int, optional): Number of random walks per node.
            (default: :obj:`100`)
        walk_length (int, optional): Length of random walks.
            (default: :obj:`2`)

    Returns:
        Positive sample pairs of shape :obj:`(2, num_pos_pairs)`.

    Example:
        >>> from torch_geometric.datasets import Planetoid
        >>> data = Planetoid(root='data', name='Cora')[0]
        >>> pos_pairs = precompute_full_graph_positive_pairs(data)
        >>> data.pos_pairs = pos_pairs  # Store in data object
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index.cpu()

    # Build adjacency matrix
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj.fill_value_(1.0)

    # Stage 1: For full graph, all nodes are considered (no expansion needed)
    all_nodes = torch.arange(num_nodes)

    # Stage 2: Compute similarity via random walks
    start_nodes = all_nodes.repeat_interleave(num_walks)
    walks = adj.random_walk(start_nodes, walk_length)

    if isinstance(walks, tuple):
        walks = walks[0]

    walks = walks[:, 1:]  # Exclude start node
    walks = walks.reshape(num_nodes, -1)

    # Build visit count matrix
    row_indices = []
    col_indices = []
    values = []

    for i in range(num_nodes):
        visited_nodes, counts = torch.unique(walks[i], return_counts=True)

        for node, count in zip(visited_nodes, counts):
            j = int(node)
            if j < num_nodes:  # Valid node
                row_indices.append(i)
                col_indices.append(j)
                values.append(int(count))

    # Build sparse similarity matrix
    row = torch.tensor(row_indices, dtype=torch.long)
    col = torch.tensor(col_indices, dtype=torch.long)
    val = torch.tensor(values, dtype=torch.float)

    # Compute row sums for normalization
    row_sums = torch.zeros(num_nodes)
    row_sums.scatter_add_(0, row, val)

    # Similarity matrix: S[i,j] = visit_count(i->j) / total_visits(i)
    sim_values = val / (row_sums[row] + 1e-15)

    # Modularity matrix: B[i,j] = S[i,j] - 1/N
    mod_values = sim_values - (1.0 / num_nodes)

    # Extract positive pairs (B[i,j] > 0)
    pos_mask = mod_values > 0
    pos_row = row[pos_mask]
    pos_col = col[pos_mask]

    pos_pairs = torch.stack([pos_row, pos_col], dim=0)

    return pos_pairs
