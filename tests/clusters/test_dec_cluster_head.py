import torch
from pyagc.clusters import DECClusterHead


def test_initialization():
    n_features = 10
    n_clusters = 5
    model = DECClusterHead(n_features=n_features, n_clusters=n_clusters)
    assert model.cluster_centers.shape == (n_clusters, n_features)


def test_reset_cluster_centers():
    n_features = 10
    n_clusters = 5
    model = DECClusterHead(n_features=n_features, n_clusters=n_clusters)
    new_centers = torch.randn(n_clusters, n_features)
    model.reset_cluster_centers(new_centers)
    assert torch.allclose(model.cluster_centers.data, new_centers)


def test_forward_loss():
    n_features = 10
    n_clusters = 5
    model = DECClusterHead(n_features=n_features, n_clusters=n_clusters)
    x = torch.randn(100, n_features)
    loss = model(x)
    assert loss.dim() == 0  # scalar
    assert loss > 0


def test_predict_soft_hard():
    n_features = 10
    n_clusters = 5
    model = DECClusterHead(n_features=n_features, n_clusters=n_clusters)
    x = torch.randn(20, n_features)

    soft_labels = model.predict(x, soft=True)
    hard_labels = model.predict(x, soft=False)

    assert soft_labels.shape == (20, n_clusters)  # soft output
    assert torch.allclose(soft_labels.sum(dim=1), torch.ones(20), atol=1e-5)  # softmax like (sum=1)

    assert hard_labels.shape == (20,)  # hard output
    assert hard_labels.max() < n_clusters and hard_labels.min() >= 0

