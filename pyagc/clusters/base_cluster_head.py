from abc import abstractmethod, ABC
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn


class BaseClusterHead(nn.Module, ABC):
    r"""
    Base class for clustering heads in neural clustering models.
    """
    def __init__(self):
        super(BaseClusterHead, self).__init__()

    @torch.no_grad()
    @abstractmethod
    def cluster(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""
        Predicts cluster assignments.

        Returns:
            - If soft=False, :obj:`(n_samples,)` tensor of cluster indices.
            - If soft=True, :obj:`(n_samples, n_clusters)` tensor of probabilities.
        """
        pass

    @property
    def predict(self):
        r"""
        Alias for :meth:`cluster`.
        """
        return self.cluster

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        r"""Runs the forward pass of the module."""
        pass
