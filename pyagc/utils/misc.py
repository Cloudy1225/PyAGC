import inspect
import logging
import random
from typing import Callable

import numpy as np
import torch
import yaml
from torch import Tensor


def filter_kwargs(func: Callable, kwargs: dict) -> dict:
    r"""Filter keyword arguments based on function signature.

    This utility function inspects a function's signature and returns only
    the keyword arguments that are valid parameters for that function. This is
    useful for passing flexible kwargs to functions without worrying about
    unsupported parameters.

    Args:
        func (callable): The function whose signature will be inspected.
        kwargs (dict): Dictionary of keyword arguments to filter.

    Returns:
        dict: A filtered dictionary containing only the keys that match
            the function's parameter names.

    Example:
        >>> def my_func(a, b, c=3):
        ...     return a + b + c
        >>> kwargs = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        >>> filtered = filter_kwargs(my_func, kwargs)
        >>> print(filtered)
        {'a': 1, 'b': 2, 'c': 3}
        >>> my_func(**filtered)
        6
    """
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def off_diagonal(x: Tensor) -> Tensor:
    r"""Extract off-diagonal elements from a square matrix.

    Returns a flattened view of all off-diagonal elements of a square matrix.
    This is useful for computing losses or metrics that exclude the diagonal,
    such as off-diagonal regularization in self-supervised learning.

    Args:
        x (Tensor): A square matrix of shape :obj:`(n, n)`.

    Returns:
        Tensor: Flattened tensor of shape :obj:`(n * (n-1),)` containing
            all off-diagonal elements in row-major order.

    Raises:
        AssertionError: If the input is not a square matrix.

    Example:
        >>> x = torch.tensor([[1, 2, 3],
        ...                   [4, 5, 6],
        ...                   [7, 8, 9]])
        >>> off_diagonal(x)
        tensor([2, 3, 4, 6, 7, 8])

    Note:
        This function is memory-efficient as it returns a view rather than
        a copy of the data when possible.
    """
    # Ensure the input is a square matrix
    n, m = x.shape
    assert n == m, f"Input must be square matrix, got shape ({n}, {m})"

    # Flatten the matrix and extract off-diagonal elements
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pairwise_squared_distance(x: Tensor, y: Tensor) -> Tensor:
    r"""Compute pairwise squared Euclidean distances between two sets of vectors.

    Efficiently computes the squared :math:`L_2` distance between all pairs
    of vectors from two sets using the identity:

    .. math::
        \| \mathbf{x}_i - \mathbf{y}_j \|_2^2 = \| \mathbf{x}_i \|_2^2
        - 2 \mathbf{x}_i^\top \mathbf{y}_j + \| \mathbf{y}_j \|_2^2

    where :math:`\mathbf{x}_i` and :math:`\mathbf{y}_j` are the :math:`i`-th
    and :math:`j`-th vectors in sets :math:`x` and :math:`y` respectively.

    This vectorized implementation is more efficient than naive nested loops
    and is commonly used in clustering algorithms (e.g., K-Means) and
    nearest neighbor computations.

    Args:
        x (Tensor): First set of vectors of shape :obj:`(B, D)`, where :math:`B`
            is the number of samples and :math:`D` is the feature dimension.
        y (Tensor): Second set of vectors of shape :obj:`(K, D)`, where :math:`K`
            is the number of reference points (e.g., cluster centers).

    Returns:
        Tensor: Squared distance matrix of shape :obj:`(B, K)`, where element
            :obj:`[i, j]` contains :math:`\| \mathbf{x}_i - \mathbf{y}_j \|_2^2`.

    Example:
        >>> x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 samples
        >>> y = torch.tensor([[0.0, 0.0], [1.0, 1.0]])  # 2 centers
        >>> distances = pairwise_squared_distance(x, y)
        >>> print(distances)
        tensor([[ 5.,  2.],
                [25., 13.]])
    """
    x_norm = (x ** 2).sum(dim=-1, keepdim=True)  # (B, 1)
    y_norm = (y ** 2).sum(dim=-1, keepdim=True).T  # (1, K)
    cross_term = x @ y.T  # (B, K)
    distances = x_norm - 2 * cross_term + y_norm
    return distances


def deep_update_dict(base: dict, overrides: dict) -> dict:
    r"""Recursively update a nested dictionary.

    Performs a deep merge of two dictionaries, where values from :obj:`overrides`
    are recursively merged into :obj:`base`. For nested dictionaries, this function
    recursively updates the nested structure. For non-dict values, the override
    value replaces the base value.

    Args:
        base (dict): The base dictionary to be updated. This dictionary is
            modified in-place.
        overrides (dict): Dictionary containing values to merge into :obj:`base`.
            Nested dictionaries are recursively merged.

    Returns:
        dict: The updated base dictionary (same object as input, modified in-place).

    Example:
        >>> base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> overrides = {'b': {'c': 20, 'e': 4}, 'f': 5}
        >>> result = deep_update_dict(base, overrides)
        >>> print(result)
        {'a': 1, 'b': {'c': 20, 'd': 3, 'e': 4}, 'f': 5}

    Note:
        - This function modifies :obj:`base` in-place
        - For nested dictionaries, only dict values are recursively merged
        - Non-dict values in :obj:`overrides` always replace values in :obj:`base`
        - To preserve the original :obj:`base`, pass :obj:`base.copy()`
    """
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update_dict(base[k], v)
        else:
            base[k] = v
    return base


def get_training_config(dataset: str, config_path: str = 'train.conf.yaml') -> dict:
    r"""Load training configuration from a YAML file with dataset-specific overrides.

    This function loads a hierarchical configuration file where a 'default'
    section provides base configurations and dataset-specific sections override
    these defaults. The merge is performed using deep dictionary updates to
    preserve nested structure.

    The configuration file should follow this structure:

    .. code-block:: yaml

        default:
          learning_rate: 0.001
          hidden_dim: 128
          model:
            num_layers: 2
            dropout: 0.5

        Cora:
          learning_rate: 0.01
          model:
            num_layers: 3

        CiteSeer:
          hidden_dim: 256

    Args:
        dataset (str): Name of the dataset. Should match a top-level key in
            the configuration file (case-sensitive).
        config_path (str, optional): Path to the YAML configuration file.
            (default: :obj:`'train.conf.yaml'`)

    Returns:
        dict: Merged configuration dictionary where dataset-specific values
            override default values. Nested dictionaries are recursively merged.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the configuration file contains invalid YAML syntax.

    Example:
        >>> # Given train.conf.yaml:
        >>> # default:
        >>> #   lr: 0.001
        >>> #   hidden: 128
        >>> # Cora:
        >>> #   lr: 0.01
        >>> config = get_training_config('Cora')
        >>> print(config)
        {'lr': 0.01, 'hidden': 128}

    Note:
        - If the dataset is not found in the config file, only default
          configuration is returned
        - Nested dictionaries are merged recursively via :func:`deep_update`
        - This function does not validate configuration values
    """
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)

    default_config = full_config.get('default', {})
    dataset_config = full_config.get(dataset, {})

    # Deep merge: dataset overrides default
    merged = deep_update_dict(default_config.copy(), dataset_config)
    return merged


def get_logger(filename: str, log_level: int = 1, name: str = None,
               mode: str = 'a') -> logging.Logger:
    r"""Create and configure a logger with both file and console handlers.

    Sets up a logger that writes to both a file and the console (stdout) with
    consistent formatting. The logger can be configured with different verbosity
    levels and can append to or overwrite existing log files.

    Args:
        filename (str): Path to the log file. Parent directories will NOT be
            created automatically.
        log_level (int, optional): Logging verbosity level:

            - :obj:`0`: DEBUG (most verbose)
            - :obj:`1`: INFO (default)
            - :obj:`2`: WARNING (least verbose)

            (default: :obj:`1`)
        name (str, optional): Name for the logger. If :obj:`None`, uses the
            root logger. Use different names to maintain separate loggers.
            (default: :obj:`None`)
        mode (str, optional): File opening mode:

            - :obj:`'a'`: Append to existing file (default)
            - :obj:`'w'`: Overwrite existing file

            (default: :obj:`'a'`)

    Returns:
        logging.Logger: Configured logger instance with both file and console
            handlers attached.
    Note:
        - The log format is: ``'%(asctime)s - %(filename)s - %(levelname)s - %(message)s'``
        - Existing handlers are removed before adding new ones to avoid duplicates
        - Both file and console handlers use the same formatting
        - The logger is returned but also accessible via :obj:`logging.getLogger(name)`
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[log_level])

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    fh = logging.FileHandler(filename, mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def set_seed(seed: int) -> None:
    r"""Set random seeds for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
