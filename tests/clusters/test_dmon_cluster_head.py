import torch
from torch_geometric.utils import erdos_renyi_graph, to_undirected

from pyagc.clusters import DMoNClusterHead


def test_kmeans_cluster_head():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Create a random synthetic graph
    num_nodes = 2000
    num_edges = 8000
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=num_edges / (num_nodes ** 2))
    edge_index = to_undirected(edge_index).to(device)

    # 2. Create random node embeddings
    n_features = 64
    z = torch.randn(num_nodes, n_features, device=device)

    # 3. Initialize DMoNClusterHead
    n_clusters = 10
    head = DMoNClusterHead(n_clusters, n_features).to(device)

    # 4. Forward pass
    L_s, L_c = head(z, edge_index)

    # 5. Combine losses for backprop
    total_loss = L_s + 0.1 * L_c
    total_loss.backward()

    # 6. Print results
    print("=== DMoNClusterHead Test ===")
    print(f"Num nodes:     {num_nodes}")
    print(f"Num clusters:  {n_clusters}")
    print(f"Spectral loss: {L_s.item():.6f}")
    print(f"Collapse loss: {L_c.item():.6f}")
    print(f"Grad norm (centers): {head.cluster_centers.grad.norm():.6f}")

    # 7. Predict cluster assignments
    clusters = head.cluster(z)
    print(f"Predicted clusters (unique): {clusters.unique().numel()} / {n_clusters}")
