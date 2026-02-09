import torch
import pytest

from pyagc.clusters import KMeansClusterHead


@pytest.mark.parametrize("backend", ["torch", "sklearn"])
def test_kmeans_cluster_head(backend):
    torch.manual_seed(42)

    n_samples, n_features, n_clusters = 100, 16, 5
    z = torch.randn(n_samples, n_features)

    head = KMeansClusterHead(n_clusters=n_clusters, backend=backend)

    # Test fit_predict
    labels = head.fit_predict(z)
    assert labels.shape == (n_samples,)
    assert labels.min() >= 0 and labels.max() < n_clusters
    assert head.cluster_centers.shape == (n_clusters, n_features)

    # Test hard cluster prediction
    pred = head.cluster(z)
    assert pred.shape == (n_samples,)
    assert torch.allclose(pred.float(), pred.float().round())  # ensure ints

    # Test soft cluster prediction
    soft_pred = head.cluster(z, soft=True)
    assert soft_pred.shape == (n_samples, n_clusters)
    assert torch.allclose(soft_pred.sum(dim=1), torch.ones(n_samples), atol=1e-5)

    # Test that cluster is deterministic after fit
    pred_2 = head.cluster(z)
    assert torch.equal(pred, pred_2)
