import torch

from pyagc.clusters.neuromap_cluster_head import _mk_smart_teleportation_flow, _mk_smart_teleportation_flow_sparse


def test_sparse_vs_dense(n=5, density=0.4, alpha=0.15, seed=0):
    """
    Compare the sparse and dense implementations to ensure they produce
    equivalent results.
    """
    torch.manual_seed(seed)

    # Generate random sparse adjacency matrix
    A_dense = torch.rand(n, n)
    A_dense[A_dense > density] = 0.0  # control sparsity
    A_dense.fill_diagonal_(0.0)       # remove self-loops
    A_sparse = A_dense.to_sparse_coo()

    # Run both versions
    F_dense, p_dense = _mk_smart_teleportation_flow(A_dense, alpha=alpha)
    F_sparse, p_sparse = _mk_smart_teleportation_flow_sparse(A_sparse, alpha=alpha)

    # Convert sparse result to dense for comparison
    F_sparse_dense = F_sparse.to_dense()

    # Check closeness
    assert torch.allclose(p_dense, p_sparse, rtol=1e-5, atol=1e-8), \
        f"p mismatch: max diff {(p_dense - p_sparse).abs().max().item()}"

    assert torch.allclose(F_dense, F_sparse_dense, rtol=1e-5, atol=1e-8), \
        f"F mismatch: max diff {(F_dense - F_sparse_dense).abs().max().item()}"
