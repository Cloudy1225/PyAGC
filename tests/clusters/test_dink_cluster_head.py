import torch
from pyagc.clusters import DinkClusterHead


def test_dink_cluster_head():
    torch.manual_seed(42)
    n_features = 16
    n_clusters = 4
    n_samples = 32

    model = DinkClusterHead(n_features=n_features, n_clusters=n_clusters)

    # Test forward pass and loss computation
    x = torch.randn(n_samples, n_features)
    loss = model(x)
    assert loss.requires_grad, "Loss should require grad"
    assert loss.item() > 0, "Loss should be positive"

    # Test reset_cluster_centers
    old_centers = model.cluster_centers.detach().clone()
    model.reset_cluster_centers()
    new_centers = model.cluster_centers.detach().clone()
    assert not torch.equal(old_centers, new_centers), \
        "Cluster centers should change after reset"

    # Test hard clustering
    hard_assignments = model.cluster(x, soft=False)
    assert hard_assignments.shape == (n_samples,), \
        f"Expected shape (n_samples,), got {hard_assignments.shape}"

    # Test soft clustering
    soft_assignments = model.cluster(x, soft=True)
    assert soft_assignments.shape == (n_samples, n_clusters), \
        f"Expected shape (n_samples, n_clusters), got {soft_assignments.shape}"
