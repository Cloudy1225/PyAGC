import torch
import numpy as np
from pyagc.metrics import label_metrics

def test_label_metrics():
    # Ground truth and predicted clusters (unordered)
    y_true_np = np.array([0, 0, 1, 1, 2, 2])
    y_pred_np = np.array([1, 1, 2, 2, 0, 0])  # Permuted clustering labels

    y_true_torch = torch.tensor(y_true_np)
    y_pred_torch = torch.tensor(y_pred_np)

    # Single metric (string)
    nmi = label_metrics(y_true_np, y_pred_np, metrics='NMI')['NMI']
    assert isinstance(nmi, float), "NMI should be a float"

    # Multiple metrics (tuple), numpy input
    scores = label_metrics(y_true_np, y_pred_np, metrics=('NMI', 'ARI', 'ACC', 'F1'))
    assert len(scores) == 4
    assert all(isinstance(score, float) for metric, score in scores.items())

    # Torch input
    scores_torch = label_metrics(y_true_torch, y_pred_torch, metrics=('NMI', 'ARI', 'ACC'))
    assert len(scores_torch) == 3
    assert all(isinstance(score, float) for metric, score in scores.items())

    # ACC should be 1.0 after Hungarian matching
    acc = label_metrics(y_true_np, y_pred_np, metrics='ACC')['ACC']
    assert acc == 1.0, f"Expected 1.0 ACC, got {acc}"

    # Invalid metric should raise ValueError
    try:
        _ = label_metrics(y_true_np, y_pred_np, metrics='INVALID_METRIC')
    except ValueError as e:
        assert "Invalid metric(s)" in str(e)
    else:
        assert False, "Expected ValueError for invalid metric"

    print("All tests passed for label_metrics.")
