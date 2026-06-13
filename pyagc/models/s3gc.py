import math
from typing import Optional, Tuple, Literal, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.index import index2ptr
from torch_geometric.nn.inits import reset
from torch_geometric.typing import WITH_PYG_LIB
from torch_geometric.utils import degree, add_remaining_self_loops, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

from pyagc.models.base import TrainableModel, LossOutput


def ppr_diffusion_weights(k: int, alpha: float = 0.2) -> Tensor:
    r"""
    Computes PPR (Personalized PageRank) diffusion weights.

    .. math::
        w_i = \alpha(1-\alpha)^i

    Args:
        k (int): Number of hops.
        alpha (float, optional): Teleport probability. (default: :obj:`0.2`)

    Returns:
        Tensor of shape :obj:`(k+1,)` containing weights.
    """
    weights = torch.tensor([alpha * ((1 - alpha) ** i) for i in range(k + 1)])
    return weights


def heat_diffusion_weights(k: int, t: float = 5.0) -> Tensor:
    r"""
    Computes heat kernel diffusion weights.

    .. math::
        w_i = \frac{e^{-t} \cdot t^i}{i!}

    Args:
        k (int): Number of hops.
        t (float, optional): Diffusion time. (default: :obj:`5.0`)

    Returns:
        Tensor of shape :obj:`(k+1,)` containing weights.
    """
    weights = torch.tensor([
        (math.exp(-t) * (t ** i)) / math.factorial(i)
        for i in range(k + 1)
    ])
    return weights


def compute_normalized_adjacency(
        edge_index: Tensor,
        num_nodes: int,
        add_self_loops: bool = True
) -> Tensor:
    r"""
    Computes the symmetric normalized adjacency matrix in COO format.

    .. math::
        \tilde{A} = D^{-1/2} A D^{-1/2}

    where :math:`A` is the adjacency matrix with optional self-loops.

    Args:
        edge_index (Tensor): Edge indices of shape :obj:`(2, num_edges)`.
        num_nodes (int): Number of nodes in the graph.
        add_self_loops (bool, optional): Whether to add self-loops.
            (default: :obj:`True`)

    Returns:
        Sparse normalized adjacency matrix.
    """
    if add_self_loops:
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)

    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(num_nodes, num_nodes)
    ).to(edge_index.device)


def compute_diffusion_matrix(
        normalized_adj: Tensor,
        x: Tensor,
        k: int = 2,
        method: Literal['ppr', 'heat', 'custom'] = 'ppr',
        coefs: Optional[Tensor] = None,
        **kwargs
) -> Tensor:
    r"""
    Computes the :math:`k`-hop diffusion matrix:

    .. math::
        S_k X = \sum_{i=0}^{k} \alpha_i \tilde{A}^i X

    where :math:`\tilde{A}` is the normalized adjacency matrix.

    Args:
        normalized_adj (Tensor): Sparse normalized adjacency matrix.
        x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
        k (int, optional): Number of hops. (default: :obj:`2`)
        method (str, optional): Diffusion method, one of :obj:`['ppr', 'heat', 'custom']`.
            (default: :obj:`'ppr'`)
        coefs (Tensor, optional): Custom diffusion coefficients of shape :obj:`(k+1,)`.
            Required if method is :obj:`'custom'`. (default: :obj:`None`)
        **kwargs: Additional arguments for PPR (alpha) or heat (t) methods.

    Returns:
        Diffusion features of shape :obj:`(num_nodes, num_features)`.

    Example:
        >>> # PPR diffusion
        >>> SX = compute_diffusion_matrix(normalized_adj, x, k=2, method='ppr', alpha=0.2)
        >>>
        >>> # Heat diffusion
        >>> SX = compute_diffusion_matrix(normalized_adj, x, k=3, method='heat', t=5.0)
        >>>
        >>> # Custom weights
        >>> weights = torch.tensor([0.5, 0.3, 0.2])
        >>> SX = compute_diffusion_matrix(normalized_adj, x, k=2, method='custom', coefs=weights)
    """
    # Determine diffusion weights
    if method == 'ppr':
        alpha = kwargs.get('alpha', 0.2)
        weights = ppr_diffusion_weights(k, alpha)
    elif method == 'heat':
        t = kwargs.get('t', 5.0)
        weights = heat_diffusion_weights(k, t)
    elif method == 'custom':
        if coefs is None:
            raise ValueError("Must provide 'coefs' when method='custom'")
        weights = coefs
        if weights.size(0) != k + 1:
            raise ValueError(f"coefs must have length {k + 1}, got {weights.size(0)}")
    else:
        raise ValueError(f"Unknown diffusion method: {method}. Choose from ['ppr', 'heat', 'custom']")

    weights = weights.to(x.device)

    # Compute diffusion
    result = weights[0] * x
    current = x

    for i in range(1, k + 1):
        current = torch.sparse.mm(normalized_adj, current)
        result = result + weights[i] * current

    return result


class SmallS3GCEncoder(nn.Module):
    r"""
    Encoder for small-scale graphs (e.g., Cora).

    Uses simple linear transformations without hidden layers.

    Architecture:
        :math:`\bar{X} = \tilde{A}X\Theta_1 + \S_kX\Theta_2 + I`
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_nodes: int):
        super().__init__()
        self.w1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w2 = nn.Linear(in_channels, hidden_channels, bias=True)

        self.iden = nn.Parameter(torch.randn(num_nodes, hidden_channels, dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)
        nn.init.normal_(self.iden)

    def forward(self, AX: Tensor, SX: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        if indices is not None:
            AX = AX[indices]
            SX = SX[indices]
            iden = self.iden[indices]
        else:
            iden = self.iden

        return F.normalize(
            self.w1(AX) + self.w2(SX) + iden,
            p=2, dim=1
        )


class MediumS3GCEncoder(nn.Module):
    r"""
    Encoder for medium-scale graphs (e.g., ogbn-arxiv).

    Uses 2-layer MLPs for better expressiveness.

    Architecture:
        :math:`\bar{X} = \text{PReLU}(W_2 \cdot \text{PReLU}(W_1 \tilde{A}X)) +
        \text{PReLU}(W_4 \cdot \text{PReLU}(W_3 S_kX)) + I`
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_nodes: int):
        super().__init__()
        self.w1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.w3 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w4 = nn.Linear(hidden_channels, hidden_channels, bias=True)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.iden = nn.Parameter(torch.randn(num_nodes, hidden_channels, dtype=torch.float))

        self.reset_parameters()

    def reset_parameters(self):
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)
        self.w3.bias.data.fill_(0.0)
        self.w4.bias.data.fill_(0.0)
        nn.init.normal_(self.iden)

    def forward(self, AX: Tensor, SX: Tensor, indices: Optional[Tensor] = None) -> Tensor:
        if indices is not None:
            AX = AX[indices]
            SX = SX[indices]
            iden = self.iden[indices]
        else:
            iden = self.iden

        return F.normalize(
            self.w2(self.prelu1(self.w1(AX))) +
            self.w4(self.prelu2(self.w3(SX))) +
            iden,
            p=2, dim=1
        )


class LargeS3GCEncoder(nn.Module):
    r"""
    Encoder for large-scale graphs (e.g., ogbn-papers100M).

    Uses 2-layer MLPs for better expressiveness.
    Unlike Medium encoder, this doesn't store node embeddings internally -
    they are managed externally by the S3GC model for memory efficiency.

    Architecture:
        :math:`\bar{X} = \text{PReLU}(W_2 \cdot \text{PReLU}(W_1 \tilde{A}X)) +
        \text{PReLU}(W_4 \cdot \text{PReLU}(W_3 S_kX)) + I`

    where I (identity embeddings) are passed as an argument rather than stored.
    """

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.w1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.w3 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.w4 = nn.Linear(hidden_channels, hidden_channels, bias=True)

        self.prelu1 = nn.PReLU(hidden_channels)
        self.prelu2 = nn.PReLU(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.w1.bias.data.fill_(0.0)
        self.w2.bias.data.fill_(0.0)
        self.w3.bias.data.fill_(0.0)
        self.w4.bias.data.fill_(0.0)

    def forward(self, AX: Tensor, SX: Tensor, iden: Tensor) -> Tensor:
        r"""
        Forward pass with explicit iden parameter.

        Args:
            AX (Tensor): Pre-multiplied features :math:`\tilde{A}X` of shape
                :obj:`(batch_size, in_channels)`.
            SX (Tensor): Diffusion features :math:`S_kX` of shape
                :obj:`(batch_size, in_channels)`.
            iden (Tensor): Learnable node embeddings of shape
                :obj:`(batch_size, hidden_channels)`.

        Returns:
            Normalized embeddings of shape :obj:`(batch_size, hidden_channels)`.
        """
        return F.normalize(
            self.w2(self.prelu1(self.w1(AX))) +
            self.w4(self.prelu2(self.w3(SX))) +
            iden,
            p=2, dim=1
        )


class S3GC(TrainableModel):
    r"""
    The S3GC (Scalable Self-Supervised Graph Clustering) model from the
    `"S3GC: Scalable Self-Supervised Graph Clustering"
    <https://openreview.net/forum?id=ldl2V3vLZ5>`_ paper (Devvrit et al., NeurIPS 2022).

    S3GC uses a simple GCN-based encoder combined with contrastive learning
    to learn clusterable node representations. The architecture adapts based
    on graph scale:

    - **Small** (< 50K nodes): Simple linear encoder
    - **Medium** (50K - 3M nodes): 2-layer MLP encoder
    - **Large** (> 3M nodes): 2-layer MLP with memory-efficient embeddings

    The encoder combines:

    1. Direct attribute transformation: :math:`f(\tilde{A}X)`
    2. Diffusion-based transformation: :math:`g(S_kX)`
    3. Learnable node embeddings: :math:`I`

    Training uses SimCLR-style contrastive loss where nodes sampled via
    random walks are positives, and randomly sampled nodes are negatives.

    Args:
        edge_index (Tensor): Edge indices of the graph.
        num_nodes (int): Number of nodes in the graph.
        in_channels (int): Input feature dimension.
        hidden_channels (int): Output embedding dimension.
        walk_length (int): Length of random walks for positive sampling.
        context_size (int): Context size for random walk (should be :math:`\leq` walk_length).
        walks_per_node (int, optional): Number of random walks per node. (default: :obj:`1`)
        num_negative_samples (int, optional): Number of negative samples per positive.
            (default: :obj:`1`)
        p (float, optional): Return parameter for random walk. (default: :obj:`1.0`)
        q (float, optional): In-out parameter for random walk. (default: :obj:`1.0`)
        scale (str, optional): Graph scale, one of :obj:`['small', 'medium', 'large', 'auto']`.
            If :obj:`'auto'`, automatically determined by num_nodes. (default: :obj:`'auto'`)

    Example:
        >>> from pyagc.models.s3gc import S3GC, precompute_features
        >>> from pyagc.data import get_dataset
        >>>
        >>> # Load data
        >>> x, edge_index, y = get_dataset('Cora', root='./data')
        >>>
        >>> # Precompute features
        >>> AX, SX = precompute_features(x, edge_index, x.size(0), method='ppr')
        >>>
        >>> # Create model (automatically detects 'small' scale)
        >>> model = S3GC(
        ...     edge_index=edge_index,
        ...     num_nodes=x.size(0),
        ...     in_channels=x.size(1),
        ...     hidden_channels=256,
        ...     walk_length=3,
        ...     context_size=3
        ... )
        >>>
        >>> # Set precomputed features
        >>> model.set_precomputed_features(AX, SX)
        >>>
        >>> # Train
        >>> loader = model.loader(batch_size=2708, shuffle=True)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> for epoch in range(1, 201):
        ...     loss = model.train_epoch(loader, optimizer, epoch)
    """

    def __init__(
        self,
        edge_index: Tensor,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        num_negative_samples: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        scale: Literal['small', 'medium', 'large', 'auto'] = 'auto',
        keep_embeddings_on_cpu: bool = False,
    ):
        super().__init__()

        if not WITH_PYG_LIB:
            raise ImportError(f"'{self.__class__.__name__}' "
                              f"requires 'pyg-lib>=0.6.0'")
        self.random_walk_fn = torch.ops.pyg.random_walk

        if walk_length < context_size:
            raise ValueError(
                f"walk_length ({walk_length}) must be >= context_size ({context_size})"
            )

        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # Determine scale
        if scale == 'auto':
            if num_nodes < 50_000:
                scale = 'small'
            elif num_nodes < 3_000_000:
                scale = 'medium'
            else:
                scale = 'large'

        self.scale = scale
        self.keep_embeddings_on_cpu = keep_embeddings_on_cpu

        # Create appropriate encoder
        if scale == 'small':
            self.encoder = SmallS3GCEncoder(in_channels, hidden_channels, num_nodes)
        elif scale == 'medium':
            self.encoder = MediumS3GCEncoder(in_channels, hidden_channels, num_nodes)
        else:  # large
            self.encoder = LargeS3GCEncoder(in_channels, hidden_channels)
            # Embeddings are managed separately for memory efficiency
            self.iden_embedding = nn.Embedding(num_nodes, hidden_channels, sparse=True)

        # Convert to CSR format for efficient random walk
        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.walk_length = walk_length - 1  # Adjusted for implementation
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q

        # Store precomputed matrices (will be set externally)
        self.register_buffer('_AX', torch.empty(0), persistent=False)
        self.register_buffer('_SX', torch.empty(0), persistent=False)
        self._features_on_cpu = False

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters."""
        reset(self.encoder)
        if self.scale == 'large':
            nn.init.normal_(self.iden_embedding.weight)

    def set_precomputed_features(self, AX: Tensor, SX: Tensor, keep_on_cpu: bool = False):
        r"""
        Sets precomputed :math:`\tilde{A}X` and :math:`S_kX` matrices.

        This should be called before training to avoid recomputing these
        matrices in every forward pass.

        Args:
            AX (Tensor): Precomputed :math:`\tilde{A}X` of shape :obj:`(num_nodes, in_channels)`.
            SX (Tensor): Precomputed :math:`S_kX` of shape :obj:`(num_nodes, in_channels)`.
            keep_on_cpu (bool): If True, keep features on CPU to save GPU memory
        """
        if keep_on_cpu:
            # Keep on CPU, will transfer batches during training
            self._AX = AX.cpu()
            self._SX = SX.cpu()
            self._features_on_cpu = True
        else:
            device = next(self.encoder.parameters()).device
            self._AX = AX.to(device)
            self._SX = SX.to(device)
            self._features_on_cpu = False

    def embed(self, indices: Optional[Tensor] = None) -> Tensor:
        r"""
        Returns node embeddings.

        Args:
            indices (Tensor, optional): Node indices. If :obj:`None`, returns
                embeddings for all nodes. (default: :obj:`None`)

        Returns:
            Node embeddings of shape :obj:`(num_nodes, hidden_channels)` or
            :obj:`(len(indices), hidden_channels)`.
        """
        if self._AX.numel() == 0 or self._SX.numel() == 0:
            raise RuntimeError(
                "Must call `set_precomputed_features()` before `embed()`."
            )

        if self.scale == 'large':
            # For large graphs, use embedding layer
            if indices is None:
                indices = torch.arange(self.num_nodes, device=self._AX.device)

            # Ensure indices are on correct device
            if self._features_on_cpu:
                indices_cpu = indices.cpu() if indices.device.type != 'cpu' else indices
                AX = self._AX[indices_cpu]
                SX = self._SX[indices_cpu]

                # Move features to encoder device
                encoder_device = next(self.encoder.parameters()).device
                AX = AX.to(encoder_device)
                SX = SX.to(encoder_device)
            else:
                AX = self._AX[indices]
                SX = self._SX[indices]

            # Handle embeddings - load to GPU on demand
            if self.keep_embeddings_on_cpu:
                # Embeddings are on CPU, load batch to GPU
                indices_cpu = indices.cpu() if indices.device.type != 'cpu' else indices
                iden = self.iden_embedding(indices_cpu)
                # Move to same device as encoder
                encoder_device = next(self.encoder.parameters()).device
                iden = iden.to(encoder_device)
            else:
                # Embeddings are on GPU
                indices_embed = indices.to(self.iden_embedding.weight.device)
                iden = self.iden_embedding(indices_embed)

            return self.encoder(AX, SX, iden)
        else:
            # For small/medium graphs, iden is part of encoder
            if self._features_on_cpu and indices is not None:
                device = next(self.encoder.parameters()).device
                AX = self._AX[indices].to(device)
                SX = self._SX[indices].to(device)
                return self.encoder(AX, SX, indices)
            else:
                return self.encoder(self._AX, self._SX, indices)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        r"""Samples positive nodes via biased random walks."""
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(
            self.rowptr.cpu(),
            self.col.cpu(),
            batch.cpu(),
            self.walk_length,
            self.p,
            self.q
        )
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])

        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        r"""Samples negative nodes randomly or cluster-aware."""
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        # Random negative sampling
        rw = torch.randint(
            self.num_nodes,
            (batch.size(0), self.walk_length * self.num_negative_samples),
            device=batch.device
        )
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])

        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        r"""Samples positive and negative random walks for a batch."""
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loader(self, **kwargs) -> DataLoader:
        r"""
        Creates a DataLoader for S3GC training.

        Returns:
            DataLoader that samples nodes and generates positive/negative walks.
        """
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor, node_mapping: Tensor) -> LossOutput:
        r"""
        Computes the SimCLR-style contrastive loss.

        Args:
            pos_rw (Tensor): Positive random walks of shape :obj:`(num_walks, context_size)`.
            neg_rw (Tensor): Negative random walks of shape :obj:`(num_walks, context_size)`.
            node_mapping (Tensor): Mapping from global node IDs to batch-local IDs.

        Returns:
            LossOutput containing total loss and individual components.
        """
        # Map global node IDs to batch-local IDs
        pos_rw_mapped = F.embedding(pos_rw.view(-1), node_mapping.view(-1, 1)).view(pos_rw.size())
        neg_rw_mapped = F.embedding(neg_rw.view(-1), node_mapping.view(-1, 1)).view(neg_rw.size())

        # Get unique nodes and compute their embeddings
        unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))

        # For CPU embeddings, need to handle device transfer
        if self.scale == 'large' and self.keep_embeddings_on_cpu:
            # Keep unique on original device for consistency
            device = unique.device
            embeddings = self.embed(indices=unique)  # This will handle CPU->GPU transfer
            embeddings = embeddings.to(device)  # Ensure on correct device
        else:
            embeddings = self.embed(indices=unique)

        # Rest of loss computation remains the same
        start_pos, rest_pos = pos_rw_mapped[:, 0], pos_rw_mapped[:, 1:].contiguous()
        h_start_pos = F.embedding(start_pos, embeddings).view(
            pos_rw_mapped.size(0), 1, self.hidden_channels
        )
        h_rest_pos = F.embedding(rest_pos.view(-1), embeddings).view(
            pos_rw_mapped.size(0), -1, self.hidden_channels
        )
        out_pos = (h_start_pos * h_rest_pos).sum(dim=-1)
        pos_loss = torch.logsumexp(out_pos, dim=-1)

        start_neg, rest_neg = neg_rw_mapped[:, 0], neg_rw_mapped[:, 1:].contiguous()
        h_start_neg = F.embedding(start_neg, embeddings).view(
            neg_rw_mapped.size(0), 1, self.hidden_channels
        )
        h_rest_neg = F.embedding(rest_neg.view(-1), embeddings).view(
            neg_rw_mapped.size(0), -1, self.hidden_channels
        )
        out_neg = (h_start_neg * h_rest_neg).sum(dim=-1)
        neg_loss = torch.logsumexp(out_neg, dim=-1)

        neg_loss = torch.logsumexp(
            torch.cat((neg_loss.view(-1, 1), pos_loss.view(-1, 1)), dim=-1),
            dim=-1
        )

        total = -torch.mean(torch.exp(pos_loss - neg_loss))

        return LossOutput(
            total=total,
            components={
                'pos': pos_loss.mean().item(),
                'neg': neg_loss.mean().item()
            }
        )

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        verbose: bool = True
    ) -> float:
        r"""
        Runs one epoch of S3GC training using the custom DataLoader.

        Args:
            loader (DataLoader): S3GC data loader created via :meth:`loader`.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)

        Returns:
            Average loss value of the epoch.
        """
        self.train()

        if self._AX.numel() == 0 or self._SX.numel() == 0:
            raise RuntimeError(
                "Must call `set_precomputed_features()` before training."
            )

        device = next(self.parameters()).device
        mapping = torch.zeros(self.num_nodes, dtype=torch.long, device=device)

        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        num_batches = 0

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=len(loader))
            pbar.set_description(f'Epoch {epoch:02d}')

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            # Get unique nodes in this batch
            unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
            mapping.scatter_(0, unique, torch.arange(unique.size(0), device=device))

            loss_output = self.loss(pos_rw, neg_rw, mapping)
            loss_output.total.backward()
            optimizer.step()

            total_loss += loss_output.total.item()
            total_pos += loss_output.components['pos']
            total_neg += loss_output.components['neg']
            num_batches += 1

            if verbose:
                pbar.update(1)

        if verbose:
            pbar.close()

        avg_loss = total_loss / num_batches
        avg_pos = total_pos / num_batches
        avg_neg = total_neg / num_batches

        if verbose:
            print(f"Epoch: {epoch:02d} Loss: {avg_loss:.4f}, POS: {avg_pos:.4f}, NEG: {avg_neg:.4f}")

        return avg_loss

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'num_nodes={self.num_nodes}, '
            f'scale={self.scale}, '
            f'in_channels={self.in_channels}, '
            f'hidden_channels={self.hidden_channels})'
        )

def precompute_features(
        x: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        method: Literal['ppr', 'heat', 'custom'] = 'ppr',
        k_hop: int = 2,
        coefs: Optional[Tensor] = None,
        add_self_loops: bool = True,
        verbose: bool = True,
        **kwargs
) -> Tuple[Tensor, Tensor]:
    r"""
    Precomputes :math:`\tilde{A}X` and :math:`S_kX` for S3GC.

    This function should be called before training to avoid recomputing
    these matrices in every forward pass.

    Args:
        x (Tensor): Node features of shape :obj:`(num_nodes, num_features)`.
        edge_index (Tensor): Edge indices of shape :obj:`(2, num_edges)`.
        num_nodes (int): Number of nodes in the graph.
        method (str, optional): Diffusion method, one of :obj:`['ppr', 'heat', 'custom']`.
            (default: :obj:`'ppr'`)
        k_hop (int, optional): Number of hops for diffusion. (default: :obj:`2`)
        coefs (Tensor, optional): Custom diffusion coefficients for :obj:`method='custom'`.
            (default: :obj:`None`)
        add_self_loops (bool, optional): Whether to add self-loops. (default: :obj:`True`)
        verbose (bool, optional): Whether to print progress. (default: :obj:`True`)
        **kwargs: Additional arguments for specific diffusion methods:
            - :obj:`alpha` (float): PPR teleport probability (default: 0.2)
            - :obj:`t` (float): Heat diffusion time (default: 5.0)

    Returns:
        Tuple of (:math:`\tilde{A}X`, :math:`S_kX`), both of shape
        :obj:`(num_nodes, num_features)`.

    Example:
        >>> from pyagc.models.s3gc import precompute_features
        >>> from pyagc.data import get_dataset
        >>>
        >>> # Load data
        >>> x, edge_index, y = get_dataset('Cora', root='./data')
        >>>
        >>> # PPR diffusion (default)
        >>> AX, SX = precompute_features(x, edge_index, x.size(0), method='ppr', k_hop=2)
        >>>
        >>> # Heat diffusion
        >>> AX, SX = precompute_features(x, edge_index, x.size(0), method='heat', k_hop=3, t=5.0)
        >>>
        >>> # Custom weights
        >>> weights = torch.tensor([0.5, 0.3, 0.2])
        >>> AX, SX = precompute_features(x, edge_index, x.size(0), method='custom',
        ...                              k_hop=2, alpha=weights)
    """
    if verbose:
        print(f"Precomputing features using {method} diffusion with k={k_hop}...")

    # Compute normalized adjacency
    if verbose:
        print("Computing normalized adjacency matrix...")
    normalized_adj = compute_normalized_adjacency(
        edge_index, num_nodes, add_self_loops
    )

    # Compute AX
    if verbose:
        print("Computing AX...")
    AX = torch.sparse.mm(normalized_adj, x)

    # Compute diffusion matrix SX
    if verbose:
        print(f"Computing diffusion matrix SX with {method} method...")
    SX = compute_diffusion_matrix(
        normalized_adj, x,
        k=k_hop,
        method=method,
        coefs=coefs,
        add_self_loops=add_self_loops,
        **kwargs
    )

    if verbose:
        print("Feature precomputation completed!")

    return AX, SX


class CompositeOptimizer:
    r"""
    A wrapper optimizer that combines multiple optimizers into one.

    It exposes a unified interface (`zero_grad`, `step`, `state_dict`,
    `load_state_dict`) so it can be used exactly like a single PyTorch
    optimizer. It does not inherit from `Optimizer` to keep the
    implementation lightweight and flexible.

    The internal optimizers can be of any type (Adam, SparseAdam, SGD, ...).

    Example:
        >>> sparse_opt = torch.optim.SparseAdam(...)
        >>> dense_opt = torch.optim.Adam(...)
        >>> optimizer = CompositeOptimizer(sparse=sparse_opt, dense=dense_opt)
    """

    def __init__(self, **optimizers: torch.optim.Optimizer):
        r"""
        Args:
            **optimizers: Arbitrary number of optimizers passed as
                keyword arguments. The key names will be preserved
                in `state_dict()` for saving/loading.
        """
        self.optimizers = optimizers  # dict: name -> optimizer

    def zero_grad(self):
        r"""Clears gradients of all wrapped optimizers."""
        for opt in self.optimizers.values():
            opt.zero_grad()

    def step(self):
        r"""Performs a step for each wrapped optimizer."""
        for opt in self.optimizers.values():
            opt.step()

    def state_dict(self):
        r"""
        Returns a state_dict containing the state of all
        wrapped optimizers, keyed by the names provided at initialization.
        """
        return {name: opt.state_dict() for name, opt in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        r"""
        Loads the state_dict for each wrapped optimizer.

        Args:
            state_dict (dict): A state dictionary produced by `state_dict()`.
        """
        for name, opt in self.optimizers.items():
            if name not in state_dict:
                raise KeyError(f"Missing optimizer state for key '{name}'")
            opt.load_state_dict(state_dict[name])
