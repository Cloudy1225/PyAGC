pyagc.utils
===========

.. contents:: Contents
    :local:

The :mod:`pyagc.utils` package provides utility functions and classes for experiment management,
including checkpoint management for single-stage and multi-stage training, configuration loading,
logging, reproducibility, and common mathematical operations.

.. code-block:: python

    from pyagc.utils import (
        CheckpointManager,
        MultiStageCheckpointManager,
        set_seed,
        get_training_config,
        get_logger,
    )

    # Set random seeds for reproducibility:
    set_seed(42)

    # Load dataset-specific training config from YAML:
    config = get_training_config("Cora", config_path="train.conf.yaml")

    # Create a logger with file and console output:
    logger = get_logger("experiment.log", log_level=1)

    # Manage checkpoints during training:
    ckpt_mgr = CheckpointManager(ckpt_dir="./checkpoints", model_name="dmon")
    ckpt_mgr.save_checkpoint(model, optimizer, epoch=10, loss=0.35, is_best=True)

Checkpoint Management
---------------------

PyAGC provides two checkpoint managers to handle model persistence during training.
:class:`CheckpointManager` supports standard single-stage training workflows,
while :class:`MultiStageCheckpointManager` extends it for multi-stage pipelines
common in decoupled AGC methods (*e.g.*, pre-training followed by fine-tuning).

Both managers automatically track the best model, support intra-epoch saving for
mini-batch training on large graphs, and allow seamless training resumption.

.. code-block:: python

    from pyagc.utils import CheckpointManager, MultiStageCheckpointManager

    # Single-stage checkpoint management:
    ckpt = CheckpointManager("./ckpts", "dmon")
    ckpt.save_checkpoint(model, optimizer, epoch=5, loss=0.42, is_best=True)
    ckpt.load_checkpoint(model, optimizer, load_best=True, device="cuda")

    # Multi-stage checkpoint management (e.g., pretrain + finetune):
    ckpt = MultiStageCheckpointManager(
        "./ckpts", "daegc", stages=["pretrain", "finetune"]
    )
    ckpt.save_checkpoint(model, optimizer, epoch=100, loss=0.5, stage="pretrain", is_best=True)
    ckpt.load_checkpoint(model, stage="pretrain", load_best=True, device="cuda")
    ckpt.save_checkpoint(model, optimizer, epoch=50, loss=0.3, stage="finetune", is_best=True)

.. currentmodule:: pyagc.utils

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   CheckpointManager
   MultiStageCheckpointManager

Configuration & Logging
-----------------------

PyAGC adopts a **configuration-driven** experiment design. All hyperparameters are
specified in YAML files with a hierarchical structure: a ``default`` section provides
base configurations, and dataset-specific sections selectively override these defaults.

.. code-block:: yaml

    # train.conf.yaml
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

.. code-block:: python

    from pyagc.utils import get_training_config, get_logger

    # Load merged configuration (default + dataset-specific overrides):
    config = get_training_config("Cora", config_path="train.conf.yaml")
    # >>> {'learning_rate': 0.01, 'hidden_dim': 128, 'model': {'num_layers': 3, 'dropout': 0.5}}

    # Create a logger with both file and console output:
    logger = get_logger("logs/experiment.log", log_level=1, name="pyagc")
    logger.info("Training started")

.. currentmodule:: pyagc.utils

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   get_training_config
   get_logger
   deep_update_dict

.. autofunction:: get_training_config

.. autofunction:: get_logger

.. autofunction:: deep_update_dict

Reproducibility
---------------

.. currentmodule:: pyagc.utils

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   set_seed

.. autofunction:: set_seed

Mathematical Utilities
----------------------

Common mathematical operations used across the library, including distance computation
and matrix manipulation.

.. code-block:: python

    from pyagc.utils import pairwise_squared_distance, off_diagonal

    # Compute pairwise squared Euclidean distances (e.g., for KMeans):
    x = torch.randn(1000, 128)   # node embeddings
    centers = torch.randn(7, 128)  # cluster centers
    dists = pairwise_squared_distance(x, centers)  # (1000, 7)

    # Extract off-diagonal elements (e.g., for regularization losses):
    corr = torch.randn(128, 128)
    off_diag = off_diagonal(corr)  # (128 * 127,)

.. currentmodule:: pyagc.utils

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   pairwise_squared_distance
   off_diagonal
   filter_kwargs

.. autofunction:: pairwise_squared_distance

.. autofunction:: off_diagonal

.. autofunction:: filter_kwargs
