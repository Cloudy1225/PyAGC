import warnings

import numpy as np
import torch
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    accuracy_score,
    f1_score,
)
from scipy.optimize import linear_sum_assignment
from typing import Union, Tuple, Dict


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _hungarian_match(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    r"""Align the predicted clusters with the true labels using the Hungarian algorithm.

    This function uses the Hungarian algorithm to find the optimal assignment
    of predicted clusters to true labels based on maximizing the overlap between
    the true labels and predicted cluster assignments.

    Args:
        y_true (np.ndarray): True labels of shape :obj:`(n_samples,)`.
        y_pred (np.ndarray): Predicted cluster labels of shape :obj:`(n_samples,)`.

    Returns:
        Aligned predicted labels of shape :obj:`(n_samples,)`.
    """
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    if len(true_labels) != len(pred_labels):
        warnings.warn(
            f"Number of predicted clusters ({len(pred_labels)}) differs from "
            f"true labels ({len(true_labels)}). Results may be unreliable."
        )

    # Create cost matrix
    cost_matrix = np.zeros((len(true_labels), len(pred_labels)))
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            cost_matrix[i, j] = np.sum((y_true == t) & (y_pred == p))

    # Use the Hungarian algorithm (linear sum assignment) to find the optimal label mapping
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    # Map predicted label values to true label values
    label_map = {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}

    # Apply mapping safely
    y_pred_aligned = np.array([label_map.get(label, label) for label in y_pred])
    return y_pred_aligned


VALID_METRICS = ('NMI', 'ARI', 'Homo', 'Comp', 'ACC', 'F1')


def label_metrics(
        y_true: Union[torch.Tensor, np.ndarray],
        y_pred: Union[torch.Tensor, np.ndarray],
        metrics: Union[str, Tuple[str, ...]] = ('NMI', 'ARI', 'ACC', 'F1')
) -> Dict[str, float]:
    r"""Compute clustering evaluation metrics.

    If accuracy or Macro-F1 score is requested, it performs alignment of predicted
    clusters with true labels using the Hungarian algorithm to account for label mismatches.

    Args:
        y_true (torch.Tensor or np.ndarray): True labels of shape :obj:`(n_samples,)`.
        y_pred (torch.Tensor or np.ndarray): Predicted cluster labels of shape :obj:`(n_samples,)`.
        metrics (str or tuple of str, optional): The metrics to compute.
            Can be one or more of :obj:`('NMI', 'ARI', 'Homo', 'Comp', 'ACC', 'F1')`.
            Default is :obj:`('NMI', 'ARI', 'ACC', 'F1')`.

    Returns:
        Dictionary mapping metric names to their computed values.

    Example:
        >>> result = label_metrics(y_true, y_pred, metrics=('NMI', 'ARI', 'ACC'))
        >>> print(result)
        {'NMI': 0.85, 'ARI': 0.72, 'ACC': 0.89}
    """
    # Validate metrics argument
    if isinstance(metrics, str):
        metrics = (metrics,)
    invalid_metrics = [metric for metric in metrics if metric not in VALID_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"Invalid metric(s): {', '.join(invalid_metrics)}. "
            f"Valid metrics are: {', '.join(VALID_METRICS)}.")

    # Convert torch tensors to numpy arrays if needed
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # Initialize results dictionary
    results = {}

    # If accuracy or Macro-F1 is needed, perform alignment only once
    if 'ACC' in metrics or 'F1' in metrics:
        y_pred = _hungarian_match(y_true, y_pred)

    # Compute selected metrics
    if 'NMI' in metrics:
        results['NMI'] = normalized_mutual_info_score(y_true, y_pred)
    if 'ARI' in metrics:
        results['ARI'] = adjusted_rand_score(y_true, y_pred)
    if 'Homo' in metrics:
        results['Homo'] = homogeneity_score(y_true, y_pred)
    if 'Comp' in metrics:
        results['Comp'] = completeness_score(y_true, y_pred)
    if 'ACC' in metrics:
        results['ACC'] = accuracy_score(y_true, y_pred)
    if 'F1' in metrics:
        results['F1'] = f1_score(y_true, y_pred, average='macro')

    return results
