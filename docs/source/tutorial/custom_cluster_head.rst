Creating Custom Cluster Heads
==============================

This tutorial demonstrates how to create custom cluster heads in PyAGC. We'll walk through the implementation of a new clustering mechanism and integrate it into the training pipeline.

Overview
--------

PyAGC provides a modular architecture where cluster heads are standalone components that can be easily swapped. All cluster heads inherit from ``BaseClusterHead`` and implement two key methods:

- ``forward()``: Computes the clustering loss during training
- ``cluster()``: Generates cluster assignments during inference

The BaseClusterHead Interface
------------------------------

.. code-block:: python

    from pyagc.clusters import BaseClusterHead
    import torch.nn as nn

    class BaseClusterHead(nn.Module):
        def forward(self, *args, **kwargs):
            """Compute clustering loss for training"""
            raise NotImplementedError

        def cluster(self, z, soft=False):
            """
            Generate cluster assignments

            Args:
                z: Node embeddings (N, F)
                soft: If True, return probabilities; if False, return hard labels

            Returns:
                Cluster assignments (N,) or probabilities (N, K)
            """
            raise NotImplementedError

Example 1: Implementing a Simple Distance-Based Cluster Head
-------------------------------------------------------------

Let's create a simple cluster head that assigns nodes to the nearest cluster center:

.. code-block:: python

    import torch
    import torch.nn as nn
    from pyagc.clusters import BaseClusterHead

    class SimpleClusterHead(BaseClusterHead):
        def __init__(self, n_clusters, n_features):
            super().__init__()
            self.n_clusters = n_clusters
            self.n_features = n_features

            # Learnable cluster centers
            self.cluster_centers = nn.Parameter(
                torch.empty(n_clusters, n_features)
            )
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.xavier_uniform_(self.cluster_centers)

        def forward(self, z):
            """
            Compute clustering loss as mean squared distance

            Args:
                z: Node embeddings (N, F)

            Returns:
                Scalar loss
            """
            # Compute pairwise distances: (N, K)
            dist = torch.cdist(z, self.cluster_centers, p=2)

            # Assign to nearest cluster
            assignments = dist.argmin(dim=-1)  # (N,)

            # Loss: mean distance to assigned cluster
            loss = dist[torch.arange(len(z)), assignments].mean()

            return loss

        @torch.no_grad()
        def cluster(self, z, soft=False):
            """Generate cluster assignments"""
            dist = torch.cdist(z, self.cluster_centers, p=2)

            if soft:
                # Convert distances to similarities via softmax
                sim = -dist  # Negate so smaller distance = higher similarity
                return torch.softmax(sim, dim=-1)
            else:
                return dist.argmin(dim=-1)

Example 2: Integrating with the DMoN Model
-------------------------------------------

Now let's see how to use our custom cluster head with an existing model. We'll modify the DMoN model to use our ``SimpleClusterHead``:

.. code-block:: python

    from pyagc.models import DMoN
    from pyagc.encoders import GCN

    # Create encoder
    encoder = GCN(
        in_channels=dataset.num_features,
        hidden_channels=64,
        num_layers=2,
    )

    # Create model with custom cluster head
    model = DMoN(
        encoder=encoder,
        n_features=64,
        n_clusters=7
    )

    # Replace the default cluster head
    model.cluster_head = SimpleClusterHead(
        n_clusters=7,
        n_features=64
    )

Example 3: Creating a Graph-Aware Cluster Head
-----------------------------------------------

For more sophisticated clustering, we can create a cluster head that considers graph structure. Here's an example inspired by DMoN:

.. code-block:: python

    from torch_geometric.utils import degree

    class GraphAwareClusterHead(BaseClusterHead):
        def __init__(self, n_clusters, n_features, alpha=1.0):
            super().__init__()
            self.n_clusters = n_clusters
            self.alpha = alpha  # Trade-off parameter

            self.cluster_centers = nn.Parameter(
                torch.empty(n_clusters, n_features)
            )
            nn.init.xavier_uniform_(self.cluster_centers)

        def forward(self, z, edge_index):
            """
            Combine embedding distance with graph structure

            Args:
                z: Node embeddings (N, F)
                edge_index: Graph edges (2, E)

            Returns:
                Scalar loss
            """
            N = z.size(0)

            # Soft assignment matrix
            S = torch.matmul(z, self.cluster_centers.t())  # (N, K)
            S = torch.softmax(S, dim=-1)

            # Embedding loss: distance to assigned cluster
            dist = torch.cdist(z, self.cluster_centers, p=2)
            assignments = S.argmax(dim=-1)
            embed_loss = dist[torch.arange(N), assignments].mean()

            # Structure loss: edge homogeneity
            # Edges should connect nodes in the same cluster
            src, dst = edge_index
            structure_loss = -torch.sum(S[src] * S[dst]) / edge_index.size(1)

            # Combined loss
            loss = embed_loss + self.alpha * structure_loss

            return loss

        @torch.no_grad()
        def cluster(self, z, soft=False):
            sim = torch.matmul(z, self.cluster_centers.t())
            if soft:
                return torch.softmax(sim, dim=-1)
            else:
                return sim.argmax(dim=-1)

Example 4: Using Custom Cluster Head in Training
-------------------------------------------------

Here's a complete example showing how to train a model with a custom cluster head:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from pyagc.data import get_dataset
    from pyagc.encoders import GCN
    from pyagc.models import ClusteringModel

    # Load dataset
    x, edge_index, y = get_dataset('Cora')
    data = Data(x=x, edge_index=edge_index)

    # Create model
    encoder = GCN(in_channels=x.size(1), hidden_channels=64, num_layers=2)
    cluster_head = GraphAwareClusterHead(n_clusters=7, n_features=64)

    # Simple wrapper model
    class MyClusteringModel(ClusteringModel):
        def __init__(self, encoder, cluster_head):
            super().__init__()
            self.encoder = encoder
            self.cluster_head = cluster_head

        def forward(self, data):
            z = self.encoder(data.x, data.edge_index)
            loss = self.cluster_head(z, data.edge_index)
            return loss

        def infer_full(self, data):
            z = self.encoder(data.x, data.edge_index)
            return self.cluster_head.cluster(z, soft=False)

    model = MyClusteringModel(encoder, cluster_head)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        loss = model(data)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

    # Inference
    model.eval()
    pred = model.infer_full(data)
    print(f'Predicted clusters: {pred}')

Comparing with Existing Implementations
----------------------------------------

Let's compare our simple implementation with PyAGC's built-in ``DECClusterHead``:

.. code-block:: python

    from pyagc.clusters import DECClusterHead

    # Our simple cluster head
    simple_head = SimpleClusterHead(n_clusters=7, n_features=64)

    # PyAGC's DEC cluster head
    dec_head = DECClusterHead(n_clusters=7, n_features=64, alpha=1.0)

    # Both can be used interchangeably
    z = torch.randn(100, 64)  # Random embeddings

    # Compute loss
    loss_simple = simple_head(z)
    loss_dec = dec_head(z)

    # Get predictions
    pred_simple = simple_head.cluster(z, soft=False)
    pred_dec = dec_head.cluster(z, soft=False)

    print(f'Simple loss: {loss_simple:.4f}, DEC loss: {loss_dec:.4f}')

Key Takeaways
-------------

1. **Inherit from BaseClusterHead**: This ensures compatibility with PyAGC's training pipelines
2. **Implement forward() and cluster()**: These are the only required methods
3. **Use learnable parameters**: Initialize cluster centers or other parameters as ``nn.Parameter``
4. **Support soft and hard assignments**: The ``cluster()`` method should handle both cases
5. **Consider graph structure**: For attributed graph clustering, incorporating edge information often improves results

Next Steps
----------

- Check out the `benchmark scripts <https://github.com/Cloudy1225/PyAGC/benchmark>`_ for complete training examples
- Experiment with different loss formulations and assignment strategies for your specific use case
- Scale your cluster head to large graphs with :doc:`mini-batch training <scalability>`
- Review existing implementations in the :doc:`clusters module <../modules/clusters>`
- Check out the :doc:`ECO framework tutorial <eco_framework>` for design patterns
