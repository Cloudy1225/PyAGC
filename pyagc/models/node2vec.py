from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.typing import WITH_PYG_LIB
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

from pyagc.models.base import TrainableModel, LossOutput


class Node2Vec(TrainableModel):
    r"""The Node2Vec model from the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper where random walks of
    length :obj:`walk_length` are sampled in a given graph, and node embeddings
    are learned via negative sampling optimization.

    Args:
        edge_index (torch.Tensor): The edge indices.
        embedding_dim (int): The size of each embedding vector.
        walk_length (int): The walk length.
        context_size (int): The actual context size which is considered for
            positive samples. This parameter increases the effective sampling
            rate by reusing samples across different source nodes.
        walks_per_node (int, optional): The number of walks to sample for each
            node. (default: :obj:`1`)
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        num_negative_samples (int, optional): The number of negative samples to
            use for each positive sample. (default: :obj:`1`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
            weight matrix will be sparse. (default: :obj:`False`)
        cpu_embedding (bool, optional): If set to :obj:`True`, stores embeddings
            on CPU and only moves required embeddings to GPU during training.
            Essential for very large graphs. (default: :obj:`False`)
    """

    def __init__(
            self,
            edge_index: Tensor,
            embedding_dim: int,
            walk_length: int,
            context_size: int,
            walks_per_node: int = 1,
            p: float = 1.0,
            q: float = 1.0,
            num_negative_samples: int = 1,
            num_nodes: Optional[int] = None,
            sparse: bool = False,
            cpu_embedding: bool = False,
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

        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col

        self.EPS = 1e-15

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.cpu_embedding = cpu_embedding
        self.compute_device = torch.device('cpu')

        self.embedding = Embedding(
            self.num_nodes,
            embedding_dim,
            sparse=sparse
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.embedding.reset_parameters()

    def to(self, device):
        """Override to method to handle CPU embedding mode."""
        if self.cpu_embedding:
            # Keep embeddings on CPU, but remember the compute device
            self.compute_device = torch.device(device) if not isinstance(device, torch.device) else device
            # Don't move embedding parameters
            return self
        else:
            # Standard behavior: move everything to device
            self.compute_device = torch.device(device) if not isinstance(device, torch.device) else device
            return super().to(device)

    def embed(self, batch: Optional[Tensor] = None, device: Optional[torch.device] = None) -> Tensor:
        r"""
        Returns the embeddings for the nodes in :obj:`batch`.

        Args:
            batch (torch.Tensor, optional): Node indices. If :obj:`None`,
                returns embeddings for all nodes. (default: :obj:`None`)
            device (torch.device, optional): Target device for the embeddings.
                If None, uses self.compute_device for cpu_embedding mode,
                or the embedding's device for standard mode. (default: :obj:`None`)

        Returns:
            Node embeddings of shape :obj:`(num_nodes, embedding_dim)` or
            :obj:`(batch_size, embedding_dim)`.
        """
        # Determine target device
        if device is None:
            if self.cpu_embedding:
                # For CPU embedding mode, default to compute_device for consistency
                target_device = self.compute_device
            else:
                # For standard mode, use embedding's device
                target_device = self.embedding.weight.device
        else:
            target_device = device

        if self.cpu_embedding:
            # For CPU embedding mode, we need to handle this differently
            if batch is None:
                # Return all embeddings
                emb = self.embedding.weight.data
                # Move to target device if requested
                if target_device.type == 'cuda' and target_device != torch.device('cpu'):
                    return emb.to(target_device)
                return emb
            else:
                # Get embeddings for batch
                if batch.device.type != 'cpu':
                    batch_cpu = batch.cpu()
                else:
                    batch_cpu = batch

                emb = self.embedding.weight[batch_cpu]

                # Move to target device if requested
                if target_device.type == 'cuda' and target_device != torch.device('cpu'):
                    return emb.to(target_device)
                return emb
        else:
            # Standard mode
            emb = self.embedding.weight if batch is None else self.embedding.weight[batch]

            # Move to target device if different
            if emb.device != target_device:
                return emb.to(target_device)
            return emb

    def loader(self, **kwargs) -> DataLoader:
        r"""
        Creates a DataLoader for training Node2Vec.

        Returns:
            DataLoader that samples positive and negative random walks.
        """
        return DataLoader(range(self.num_nodes), collate_fn=self.sample, **kwargs)

    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        r"""Samples positive random walks."""
        batch = batch.repeat(self.walks_per_node)
        rw = self.random_walk_fn(self.rowptr, self.col, batch,
                                 self.walk_length, self.p, self.q)
        if not isinstance(rw, Tensor):
            rw = rw[0]

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        r"""Samples negative random walks."""
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        r"""Samples positive and negative random walks for a batch of nodes."""
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> LossOutput:
        r"""
        Computes the loss given positive and negative random walks.

        Args:
            pos_rw (torch.Tensor): Positive random walks of shape
                :obj:`(num_walks, context_size)`.
            neg_rw (torch.Tensor): Negative random walks of shape
                :obj:`(num_walks, context_size)`.

        Returns:
            LossOutput containing total loss and individual components.
        """
        # Positive loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim)
        h_start, h_rest = h_start.to(self.compute_device), h_rest.to(self.compute_device)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim)
        h_start, h_rest = h_start.to(self.compute_device), h_rest.to(self.compute_device)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        total = pos_loss + neg_loss

        return LossOutput(
            total=total,
            components={
                'pos': pos_loss.item(),
                'neg': neg_loss.item()
            }
        )

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    epoch: int, verbose: bool = True) -> float:
        r"""
        Runs one epoch of Node2Vec training using the custom DataLoader.

        Args:
            loader (DataLoader): Node2Vec data loader created via :meth:`loader`.
            optimizer (torch.optim.Optimizer): The optimizer.
            epoch (int): Current epoch number.
            verbose (bool, optional): If :obj:`True`, prints training progress.
                (default: :obj:`True`)

        Returns:
            Average loss value of the epoch.
        """
        self.train()

        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        num_batches = 0

        if verbose:
            from tqdm import tqdm
            pbar = tqdm(total=len(loader))
            pbar.set_description(f'Epoch {epoch:03d}')

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()

            # Use the tracked compute device
            if not self.cpu_embedding:
                pos_rw = pos_rw.to(self.compute_device)
                neg_rw = neg_rw.to(self.compute_device)

            loss_output = self.loss(pos_rw, neg_rw)
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
            log_str = f"Epoch: {epoch:03d} Loss: {avg_loss:.4f}, POS: {avg_pos:.4f}, NEG: {avg_neg:.4f}"
            self.logger.info(log_str) if self.logger else print(log_str)

        return avg_loss

    def __repr__(self) -> str:
        emb_location = "CPU" if self.cpu_embedding else "GPU"
        compute_loc = f"compute_on={self.compute_device}"
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)}, '
                f'embedding_on={emb_location}, {compute_loc})')
