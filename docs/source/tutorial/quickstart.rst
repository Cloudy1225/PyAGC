Quickstart Tutorial
===================

This tutorial provides a quick introduction to PyAGC's core functionality.

Installation
------------

First, install PyAGC:

.. code-block:: bash

    pip install pyagc

Basic Clustering Example
-------------------------

**Example 1: Training-Free Clustering with SSGC**

SSGC is a non-parametric method that requires no training:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from pyagc.data import get_dataset
    from pyagc.models import SSGC
    from pyagc.clusters import KMeansClusterHead
    from pyagc.metrics import label_metrics

    # Load dataset
    x, edge_index, y = get_dataset('Cora', root='./data')
    data = Data(x=x, edge_index=edge_index)

    # Create SSGC model (training-free)
    model = SSGC(alpha=0.05, K=12, cached=True)

    # Generate embeddings
    z = model.embed(data.x, data.edge_index)

    # Cluster with KMeans
    n_clusters = int(y.max().item()) + 1
    kmeans = KMeansClusterHead(n_clusters=n_clusters, backend='torch')
    pred = kmeans.fit_predict(z)

    # Evaluate
    results = label_metrics(y, pred, metrics=['NMI', 'ARI', 'ACC', 'F1'])
    print(f"ACC: {results['ACC']:.4f}, NMI: {results['NMI']:.4f}")

Output:

.. code-block:: text

    ACC: 0.6538, NMI: 0.5185

**Example 2: Deep Contrastive Clustering with DGI**

DGI uses contrastive learning for representation learning:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from pyagc.data import get_dataset
    from pyagc.models import DGI
    from pyagc.encoders import create_tuned_gnn
    from pyagc.clusters import KMeansClusterHead
    from pyagc.metrics import label_metrics

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, edge_index, y = get_dataset('Cora', root='./data')
    data = Data(x=x, edge_index=edge_index).to(device)

    # Create encoder and model
    encoder = create_tuned_gnn(
        gnn_type='gcn',
        in_channels=data.num_features,
        hidden_channels=512,
        num_layers=1,
        act_last=True
    )
    model = DGI(hidden_channels=512, encoder=encoder).to(device)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, 301):
        loss = model.train_full(data, optimizer, epoch, verbose=(epoch % 50 == 0))

    # Inference and clustering
    model.eval()
    with torch.no_grad():
        z = model.infer_full(data)

    n_clusters = int(y.max().item()) + 1
    kmeans = KMeansClusterHead(n_clusters=n_clusters)
    pred = kmeans.fit_predict(z.cpu())

    # Evaluate
    results = label_metrics(y, pred)
    print(f"ACC: {results['ACC']:.4f}, NMI: {results['NMI']:.4f}")

Configuration-Driven Workflow
------------------------------

For reproducible experiments, PyAGC supports YAML configuration files.

**Create a configuration file (train.conf.yaml):**

.. code-block:: yaml

    default:
      # Training
      lr: 0.001
      wd: 0.0
      epochs: 300
      patience: 50

      # Model architecture
      gnn_type: gcn
      hidden_channels: 512
      num_layers: 1
      dropout: 0.0
      norm: null
      act_last: true

      # Evaluation
      label_metrics: ['NMI', 'ARI', 'ACC', 'F1']
      struct_metrics: ['Mod', 'Cond']
      kmeans_backend: torch
      kmeans_n_init: 10

    # Dataset-specific overrides
    Cora:
      epochs: 300
      hidden_channels: 512
      num_layers: 1

**Run experiment from command line:**

.. code-block:: bash

    cd benchmark/DGI
    python main.py --dataset Cora --device cuda:0 --runs 5

**Output:**

.. code-block:: text

    ============================================================
    Configuration
    ============================================================
      dataset: Cora
      epochs: 300
      hidden_channels: 512
      lr: 0.001
      ...

    ============================================================
    Training Mode: Full-batch
    ============================================================
    Epoch: 001 Loss: 0.6931
    ...
    Epoch: 300 Loss: 0.2134

    ============================================================
    Clustering Stage
    ============================================================
    Run 1/5: NMI=54.78, ARI=45.73, ACC=66.73, F1=65.66
    Run 2/5: NMI=56.45, ARI=54.45, ACC=75.00, F1=72.98
    ...

    ============================================================
    Final Results Summary
    ============================================================
    Clustering Metrics (mean ± std):
      NMI   :  56.22 ± 0.87
      ARI   :  52.03 ± 3.22
      ACC   :  72.22 ± 2.92
      F1    :  70.07 ± 2.56

Comparing Multiple Methods
---------------------------

PyAGC makes it easy to compare different algorithms:

.. code-block:: python

    from pyagc.data import get_dataset
    from pyagc.models import SSGC, DGI, DMoN
    from pyagc.clusters import KMeansClusterHead
    from pyagc.metrics import label_metrics
    import torch

    # Load data
    x, edge_index, y = get_dataset('Cora')
    n_clusters = int(y.max().item()) + 1

    # Method 1: SSGC (training-free)
    ssgc = SSGC(alpha=0.05, K=12)
    z_ssgc = ssgc.embed(x, edge_index)
    pred_ssgc = KMeansClusterHead(n_clusters=n_clusters).fit_predict(z_ssgc)
    results_ssgc = label_metrics(y, pred_ssgc)

    # Method 2: DGI (decoupled)
    # ... train DGI and get embeddings z_dgi ...
    pred_dgi = KMeansClusterHead(n_clusters=n_clusters).fit_predict(z_dgi)
    results_dgi = label_metrics(y, pred_dgi)

    # Method 3: DMoN (joint end-to-end)
    # ... train DMoN which outputs predictions directly ...
    results_dmon = label_metrics(y, pred_dmon)

    # Compare
    print(f"SSGC: ACC={results_ssgc['ACC']:.4f}")
    print(f"DGI:  ACC={results_dgi['ACC']:.4f}")
    print(f"DMoN: ACC={results_dmon['ACC']:.4f}")

Scaling to Large Graphs
------------------------

For large graphs, use mini-batch training:

**Update configuration to enable mini-batch:**

.. code-block:: yaml

    default:
      mini_batch: true
      batch_size: 1024
      fan_out: 10
      num_workers: 4
      infer_batch_size: 256

**Run on large dataset:**

.. code-block:: bash

    python main.py --dataset Products --device cuda:0 --runs 1

PyAGC automatically uses neighbor sampling for training and inference on large graphs like Products (2.4M nodes).

Understanding Evaluation Metrics
---------------------------------

PyAGC provides both label-based and structure-based metrics:

.. code-block:: python

    from pyagc.metrics import label_metrics, structure_metrics

    # Label-based: Compare with ground truth
    label_results = label_metrics(
        y_true, y_pred,
        metrics=['NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp']
    )

    # Structure-based: Evaluate community quality
    struct_results = structure_metrics(
        edge_index, y_pred,
        metrics=['Mod', 'Cond']
    )

**Metrics explanation:**

- **NMI** (Normalized Mutual Information): Measures information shared between clusters and labels
- **ARI** (Adjusted Rand Index): Measures similarity between two clusterings
- **ACC** (Accuracy): Best matching accuracy using Hungarian algorithm
- **F1** (Macro-F1): Harmonic mean of precision and recall
- **Mod** (Modularity): Measures strength of division into communities
- **Cond** (Conductance): Measures cluster separability (lower is better)

Next Steps
----------

- Learn about the :doc:`ECO framework <eco_framework>` to understand PyAGC's design
- Implement :doc:`custom cluster heads <custom_cluster_head>` for novel clustering methods
- Scale to large graphs with :doc:`mini-batch training <scalability>`
