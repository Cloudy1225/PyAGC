import warnings
from collections import Counter
from itertools import combinations, permutations

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from pyagc.metrics import label_metrics

pytestmark = pytest.mark.filterwarnings("ignore:Number of predicted clusters")


def exhaustive_accuracy(truth, prediction):
    contingency = Counter(zip(truth, prediction))
    true_labels = set(truth)
    pred_labels = set(prediction)
    size = min(len(true_labels), len(pred_labels))
    return max(
        sum(contingency[pair] for pair in zip(assignment, subset))
        for subset in combinations(pred_labels, size)
        for assignment in permutations(true_labels, size)
    ) / len(truth)


@pytest.mark.parametrize(
    ("truth", "prediction", "expected_f1"),
    [
        pytest.param([0] * 4, [7, 7, 7, 0], 3 / 7, id="numeric-collision"),
        pytest.param([0] * 4, [7, 7, 7, 8], 3 / 7, id="equivalent-relabeling"),
        pytest.param(["dog"] * 4, ["x", "x", "x", "dog"], 3 / 7, id="string-collision"),
        # The two unmatched clusters each contribute their own zero F1.
        pytest.param(
            [-4] * 6, [90, 90, 90, 90, -4, 11], 4 / 15, id="two-unmatched-clusters"
        ),
        # The dominant class gets F1 of 2/3. Both missed true classes contribute zero.
        pytest.param(
            [-5, -5, -5, 11, 11, 90], [11] * 6, 2 / 9, id="fewer-predicted-clusters"
        ),
        pytest.param([-8, -8, 11, 11], [11, 11, -8, -8], 1.0, id="exact-permutation"),
    ],
)
def test_alignment_against_one_to_one_oracle(truth, prediction, expected_f1):
    y_true = np.array(truth)
    y_pred = np.array(prediction)
    expected = {"ACC": exhaustive_accuracy(truth, prediction), "F1": expected_f1}
    assert label_metrics(y_true, y_pred, ("ACC", "F1")) == pytest.approx(expected)
    assert label_metrics(y_true, y_pred, ("F1", "ACC")) == pytest.approx(expected)
    for metric, value in expected.items():
        assert label_metrics(y_true, y_pred, metric) == pytest.approx({metric: value})


PARTITION_METRICS = {
    "NMI": normalized_mutual_info_score,
    "ARI": adjusted_rand_score,
    "Homo": homogeneity_score,
    "Comp": completeness_score,
}


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        ([0, 0, 0, 0], [7, 7, 7, 0]),
        ([0, 0, 0, 1, 1, 1], [7, 7, 7, 0, 8, 8]),
    ],
)
@pytest.mark.parametrize("alignment_metrics", [(), ("ACC",), ("F1",), ("F1", "ACC")])
def test_partition_scores_are_independent_of_alignment(
    truth, prediction, alignment_metrics
):
    y_true = np.array(truth)
    y_pred = np.array(prediction)
    requested = alignment_metrics + tuple(PARTITION_METRICS)
    scores = label_metrics(y_true, y_pred, requested)
    expected = {
        name: score(y_true, y_pred) for name, score in PARTITION_METRICS.items()
    }
    assert {name: scores[name] for name in expected} == pytest.approx(expected)


@pytest.mark.parametrize(
    "array_factory", [np.array, torch.tensor], ids=["numpy", "cpu"]
)
def test_default_metrics_preserve_inputs(array_factory):
    truth = [0, 0, 0, 0]
    prediction = [7, 7, 7, 0]
    y_true = array_factory(truth)
    y_pred = array_factory(prediction)

    assert label_metrics(y_true, y_pred) == pytest.approx(
        {"NMI": 0.0, "ARI": 0.0, "ACC": 0.75, "F1": 3 / 7}
    )
    np.testing.assert_array_equal(y_true, truth)
    np.testing.assert_array_equal(y_pred, prediction)


def test_equal_count_tied_assignment_is_repeatable():
    truth = np.array([0, 0, 0, 1, 1, 1])
    prediction = np.array([7, 7, 8, 7, 7, 8])
    expected = {"ACC": 0.5, "F1": (4 / 7 + 2 / 5) / 2}
    for _ in range(2):
        assert label_metrics(truth, prediction, ("ACC", "F1")) == pytest.approx(
            expected
        )


def test_unequal_cluster_warning_is_only_emitted_for_alignment():
    truth = np.array([0, 0, 0, 0])
    prediction = np.array([7, 7, 7, 0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        label_metrics(truth, prediction, tuple(PARTITION_METRICS))
    assert not caught

    with pytest.warns(
        UserWarning, match=r"predicted clusters \(2\).*true labels \(1\)"
    ) as caught:
        label_metrics(truth, prediction, ("ACC", "F1"))
    assert len(caught) == 1
