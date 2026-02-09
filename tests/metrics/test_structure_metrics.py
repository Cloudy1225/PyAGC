import numpy as np
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix

from pyagc.metrics import modularity, conductance


## See https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py#L93
def modularity_sp(adjacency, clusters):
  """Computes graph modularity.

  Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

  Returns:
    The value of graph modularity.
    https://en.wikipedia.org/wiki/Modularity_(networks)
  """
  degrees = adjacency.sum(axis=0).A1
  n_edges = degrees.sum()  # Note that it's actually 2*n_edges.
  result = 0
  for cluster_id in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    adj_submatrix = adjacency[cluster_indices, :][:, cluster_indices]
    degrees_submatrix = degrees[cluster_indices]
    result += np.sum(adj_submatrix) - (np.sum(degrees_submatrix)**2) / n_edges
  return result / n_edges


## See https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/metrics.py#L115
def conductance_sp(adjacency, clusters):
  """Computes graph conductance as in Yang & Leskovec (2012).

  Args:
    adjacency: Input graph in terms of its sparse adjacency matrix.
    clusters: An (n,) int cluster vector.

  Returns:
    The average conductance value of the graph clusters.
  """
  inter = 0  # Number of inter-cluster edges.
  intra = 0  # Number of intra-cluster edges.
  cluster_indices = np.zeros(adjacency.shape[0], dtype=bool)
  for cluster_id in np.unique(clusters):
    cluster_indices[:] = 0
    cluster_indices[np.where(clusters == cluster_id)[0]] = 1
    adj_submatrix = adjacency[cluster_indices, :]
    inter += np.sum(adj_submatrix[:, cluster_indices])
    intra += np.sum(adj_submatrix[:, ~cluster_indices])
  return intra / (inter + intra)



def test_metrics_equivalence():
    # ==== Step 1: Load a small graph (Cora) ====
    data = KarateClub()[0]
    data.edge_index = to_undirected(data.edge_index)
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    clusters_th = data.y
    clusters_np = data.y.numpy()

    # ==== Step 2: Compute metrics using scipy ====
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    sp_mod = modularity_sp(adjacency=adj, clusters=clusters_np)
    sp_cond = conductance_sp(adjacency=adj, clusters=clusters_np)

    # ==== Step 3: Compute metrics using PyG ====
    pyg_mod = modularity(edge_index, clusters_th)
    pyg_cond = conductance(edge_index, clusters_th)

    # ==== Step 4: Print comparison ====
    print(f"[Modularity] scipy: {sp_mod:.6f}, pyg: {pyg_mod:.6f}")
    print(f"[Conductance] scipy: {sp_cond:.6f}, pyg: {pyg_cond:.6f}")

    # ==== Step 5: Assert close ====
    assert np.allclose(sp_mod, pyg_mod, rtol=1e-5), "Modularity mismatch!"
    assert np.allclose(sp_cond, pyg_cond, rtol=1e-5), "Conductance mismatch!"
