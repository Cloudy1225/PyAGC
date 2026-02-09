import torch
from sklearn.datasets import make_blobs
from pyagc.clusters import TorchKMeans


def test_torch_kmeans():
    # Create synthetic data
    X_np, y_np = make_blobs(n_samples=100, n_features=5, centers=4, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32)

    # Test Euclidean distance
    kmeans_euc = TorchKMeans(
        n_clusters=4,
        n_init=5,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        metric='euclidean',
        verbose=False
    )
    labels_euc = kmeans_euc.fit_predict(X)

    assert labels_euc.shape[0] == X.shape[0], "Mismatch in number of labels."
    assert kmeans_euc.cluster_centers_.shape == (4, 5), "Cluster centers shape mismatch."
    assert torch.all(labels_euc >= 0) and torch.all(labels_euc < 4), "Invalid cluster labels."

    print(f"Euclidean KMeans Final Inertia: {kmeans_euc.stats['inertia'].min().item():.4f}")

    # Test Cosine distance
    kmeans_cos = TorchKMeans(
        n_clusters=4,
        n_init=5,
        max_iter=100,
        tol=1e-4,
        random_state=42,
        metric='cosine',
        verbose=False
    )
    labels_cos = kmeans_cos.fit_predict(X)

    assert labels_cos.shape[0] == X.shape[0], "Mismatch in number of labels."
    assert kmeans_cos.cluster_centers_.shape == (4, 5), "Cluster centers shape mismatch."
    assert torch.all(labels_cos >= 0) and torch.all(labels_cos < 4), "Invalid cluster labels."

    print(f"Cosine KMeans Final Inertia: {kmeans_cos.stats['inertia'].min().item():.4f}")
