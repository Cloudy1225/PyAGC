pyagc.data
==========

.. contents:: Contents
    :local:

Dataset Loading
---------------

.. autofunction:: pyagc.data.get_dataset

.. autofunction:: pyagc.data.get_tabular_graphland_dataset

Benchmark Datasets
------------------

PyAGC provides a curated collection of 12 benchmark datasets spanning diverse domains, scales, and feature types for comprehensive evaluation of attributed graph clustering algorithms.

Dataset Overview Table
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 8 10 10 12 12 8 8 10 8 8 8

   * - Scale
     - Dataset
     - Domain
     - #Nodes
     - #Edges
     - Avg.Deg
     - #Feat
     - Feat.Type
     - #Clus
     - :math:`\mathcal{H}_e`
     - :math:`\mathcal{H}_n`
   * - Tiny
     - Cora
     - Citation
     - 2,708
     - 10,556
     - 3.9
     - 1,433
     - Textual
     - 7
     - 0.81
     - 0.83
   * - Tiny
     - Photo
     - Co-purchase
     - 7,650
     - 238,162
     - 31.1
     - 745
     - Textual
     - 8
     - 0.83
     - 0.84
   * - Small
     - Physics
     - Co-author
     - 34,493
     - 495,924
     - 14.4
     - 8,415
     - Textual
     - 5
     - 0.93
     - 0.92
   * - Small
     - HM
     - Co-purchase
     - 46,563
     - 21,461,990
     - 460.9
     - 120
     - Tabular
     - 21
     - 0.16
     - 0.35
   * - Small
     - Flickr
     - Social
     - 89,250
     - 899,756
     - 10.1
     - 500
     - Textual
     - 7
     - 0.32
     - 0.32
   * - Medium
     - ArXiv
     - Citation
     - 169,343
     - 1,166,243
     - 6.9
     - 128
     - Textual
     - 40
     - 0.65
     - 0.64
   * - Medium
     - Reddit
     - Social
     - 232,965
     - 23,213,838
     - 99.6
     - 602
     - Textual
     - 41
     - 0.78
     - 0.81
   * - Medium
     - MAG
     - Citation
     - 736,389
     - 10,792,672
     - 14.7
     - 128
     - Textual
     - 349
     - 0.30
     - 0.31
   * - Large
     - Pokec
     - Social
     - 1,632,803
     - 44,603,928
     - 27.3
     - 56
     - Tabular
     - 183
     - 0.43
     - 0.39
   * - Large
     - Products
     - Co-purchase
     - 2,449,029
     - 61,859,140
     - 25.4
     - 100
     - Textual
     - 47
     - 0.81
     - 0.82
   * - Large
     - WebTopic
     - Web
     - 2,890,331
     - 24,754,822
     - 8.6
     - 528
     - Tabular
     - 28
     - 0.22
     - 0.24
   * - Massive
     - Papers100M
     - Citation
     - 111,059,956
     - 1,615,685,872
     - 14.5
     - 128
     - Textual
     - 172
     - 0.57
     - 0.50

.. note::
   - :math:`\mathcal{H}_e`: Edge homophily (proportion of edges connecting same-class nodes)
   - :math:`\mathcal{H}_n`: Node homophily (average neighbor label consistency)
   - **Feat.Type**: Textual (bag-of-words, embeddings) or Tabular (categorical/numerical metadata)
   - For Papers100M, labels are available for a subset of ≈1.5M arXiv papers. The reported homophily metrics are calculated based on the induced subgraph of these labeled nodes

Dataset Details by Scale
~~~~~~~~~~~~~~~~~~~~~~~~~

**Tiny Scale** (:math:`N < 10^4`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These datasets are suitable for rapid prototyping and sanity checking:

- **Cora**: Classic citation network of machine learning papers with sparse bag-of-words features
- **Photo**: Amazon product co-purchase graph with review-based features

**Small Scale** (:math:`10^4 \le N < 10^5`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suitable for comprehensive model development and ablation studies:

- **Physics**: Co-authorship network from Microsoft Academic Graph with keyword features
- **HM**: H&M fashion co-purchase network with **tabular product metadata** (color, weekday statistics)
- **Flickr**: Image-sharing social network with tag-based features

**Medium Scale** (:math:`10^5 \le N < 10^6`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transitional regime requiring efficient implementations:

- **ArXiv**: Computer Science paper citations with title/abstract embeddings
- **Reddit**: Discussion posts connected by common commenters with GloVe features
- **MAG**: Multi-venue academic citations (349 classes) with abstract embeddings

**Large Scale** (:math:`10^6 \le N < 10^8`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Production-scale graphs requiring mini-batch training:

- **Pokec**: Slovak social network with **tabular user profiles** (183 regions, heterophilous)
- **Products**: Amazon co-purchase network with product description features
- **WebTopic**: Web graph with **tabular website metadata** (28 topics, low homophily)

**Massive Scale** (:math:`N > 10^8`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extreme-scale benchmark for testing scalability limits:

- **Papers100M**: 111M paper citation network (172 subjects) — requires neighbor sampling

Example Usage
-------------

Basic Loading
~~~~~~~~~~~~~

.. code-block:: python

    from pyagc.data import get_dataset

    # Load dataset (returns unpacked components)
    x, edge_index, y = get_dataset('Cora', root='./data')

    print(f"Node features shape: {x.shape}")      # [num_nodes, num_features]
    print(f"Edge index shape: {edge_index.shape}") # [2, num_edges]
    print(f"Labels shape: {y.shape}")              # [num_nodes]
    print(f"Number of classes: {y.max().item() + 1}")

Loading with Train/Val/Test Splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load with predefined splits
    x, edge_index, y, train_idx, valid_idx, test_idx = get_dataset(
        'ArXiv',
        root='./data',
        return_splits=True
    )

    print(f"Training nodes: {train_idx.shape[0]}")
    print(f"Validation nodes: {valid_idx.shape[0]}")
    print(f"Test nodes: {test_idx.shape[0]}")

Loading Large-Scale Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Medium/Large datasets work the same way
    x, edge_index, y = get_dataset('Products', root='./data')
    print(f"Large graph: {x.shape[0]:,} nodes, {edge_index.shape[1]:,} edges")

Loading Massive-Scale Datasets (Papers100M)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Papers100M requires special handling due to its size
    # First-time loading will preprocess and cache the undirected graph
    x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph = get_dataset(
        'Papers100M',
        root='./data',
        return_splits=True
    )

    # labeled_subgraph contains structure for computing structural metrics
    # on the labeled subset (≈1.5M nodes)
    print(f"Full graph: {x.shape[0]:,} nodes")
    print(f"Labeled subgraph: {labeled_subgraph['num_nodes']:,} nodes")
    print(f"Subgraph edges: {labeled_subgraph['edge_index'].shape[1]:,}")

.. warning::
   **Papers100M Preprocessing Requirements:**

   - First-time loading requires ~400 GB RAM for preprocessing
   - Preprocessed data is cached to disk for future use
   - Returns additional `labeled_subgraph` dict when `return_splits=True`
   - The labeled subgraph contains only structure (no features) for efficient structural metric computation

Creating PyG Data Object
~~~~~~~~~~~~~~~~~~~~~~~~~

If you need a PyG Data object for compatibility:

.. code-block:: python

    from torch_geometric.data import Data

    x, edge_index, y = get_dataset('Cora', root='./data')

    # Wrap in PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_features}, Classes: {data.y.max().item() + 1}")

Working with Tabular Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GraphLand datasets (HM, Pokec, WebTopic) provide **tabular node features**.

Two usage patterns are supported:

Using Dense Tensor Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Load tabular feature dataset
    x, edge_index, y = get_dataset('HM', root='./data')

    print(f"Tabular features: {x.shape}")  # Mixed categorical/numerical
    print(f"Number of product categories: {y.max().item() + 1}")

    # These features may require preprocessing (normalization, encoding)
    # depending on your clustering algorithm

Using TensorFrame Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pyagc.data import get_tabular_graphland_dataset

    data = get_tabular_graphland_dataset('HM', root='./data')

    x = data.x  # torch_frame.TensorFrame

    print(x)
    print(data.tf_col_stats)  # column-wise statistics

Advantages of TensorFrame:

- Preserves **feature semantics** (categorical vs numerical)
- Avoids lossy preprocessing
- Enables **learnable feature encoders**
- Better suited for **heterogeneous tabular graphs**

.. tip::

   When using TensorFrame inputs, pair with encoders from
   :mod:`pyagc.encoders.tabencoder` or custom torch-frame models.

Loading Other PyG Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets not included in the benchmark suite, you can directly use PyTorch Geometric:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import Planetoid, Amazon, Coauthor
    from torch_geometric.utils import to_undirected

    # Example 1: Load PubMed from Planetoid
    dataset = Planetoid(root='./data', name='PubMed', transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)  # Convert to undirected

    x = data.x
    edge_index = data.edge_index
    y = data.y.squeeze()

    print(f"PubMed: {x.shape[0]} nodes, {edge_index.shape[1]} edges")

    # Example 2: Load Computers from Amazon
    dataset = Amazon(root='./data', name='Computers', transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    # Example 3: Load other Coauthor datasets
    dataset = Coauthor(root='./data', name='CS', transform=T.NormalizeFeatures())
    data = dataset[0]

    # Example 4: Load any OGB dataset
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(root='./data', name='ogbn-proteins')
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)

    # Get splits for OGB datasets
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

.. tip::
   **Working with Custom PyG Datasets:**

   When using PyG datasets directly, remember to:

   - Apply ``to_undirected()`` for clustering tasks (most AGC methods assume undirected graphs)
   - Use ``T.NormalizeFeatures()`` transform for consistent feature scaling
   - Convert labels to 1D tensor: ``y = data.y.squeeze()``
   - Check if the dataset provides train/val/test masks or splits

Loading Custom Datasets
~~~~~~~~~~~~~~~~~~~~~~~~

You can also load your own graph data:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected

    # Create graph from edge list and features
    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 16)  # 3 nodes, 16 features
    y = torch.tensor([0, 1, 1])  # Ground truth labels

    # Ensure undirected
    edge_index = to_undirected(edge_index)

    data = Data(x=x, edge_index=edge_index, y=y)

.. code-block:: python

    # Load from numpy arrays
    import numpy as np
    import torch
    from scipy.sparse import coo_matrix

    # Load adjacency matrix (scipy sparse format)
    adj_matrix = coo_matrix(...)  # Your adjacency matrix
    edge_index = torch.tensor(
        np.vstack([adj_matrix.row, adj_matrix.col]),
        dtype=torch.long
    )

    # Load features and labels
    features = np.load('features.npy')
    labels = np.load('labels.npy')

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Normalize features
    from torch_geometric.transforms import NormalizeFeatures
    data = Data(x=x, edge_index=edge_index, y=y)
    transform = NormalizeFeatures()
    data = transform(data)

Dataset Name Aliases
~~~~~~~~~~~~~~~~~~~~

The following aliases are supported for convenience:

.. code-block:: python

    # Case-insensitive loading
    get_dataset('cora', root='./data')      # ✓
    get_dataset('Cora', root='./data')      # ✓
    get_dataset('CORA', root='./data')      # ✓

    # OGB dataset aliases
    get_dataset('arxiv', root='./data')     # Short form
    get_dataset('ogbn-arxiv', root='./data')  # Full OGB name

    get_dataset('mag', root='./data')        # Short form
    get_dataset('ogbn-mag', root='./data')   # Full OGB name

    get_dataset('products', root='./data')        # Short form
    get_dataset('ogbn-products', root='./data')   # Full OGB name

    get_dataset('papers100m', root='./data')       # Short form
    get_dataset('ogbn-papers100M', root='./data')  # Full OGB name

    # GraphLand aliases
    get_dataset('hm', root='./data')           # Short form
    get_dataset('hm-categories', root='./data')  # Full name

    get_dataset('pokec', root='./data')         # Short form
    get_dataset('pokec-regions', root='./data')  # Full name

    get_dataset('webtopic', root='./data')      # Short form
    get_dataset('web-topics', root='./data')    # Full name

    # Reddit aliases
    get_dataset('reddit', root='./data')   # Either works
    get_dataset('reddit2', root='./data')  # Same dataset

GraphLand Industrial Datasets
------------------------------

PyAGC includes the **GraphLand** benchmark datasets (HM, Pokec, WebTopic) featuring:

- **Tabular node features** (categorical + numerical)
- **Heterophilous structures** (low homophily)
- **Industrial-scale complexity** (millions of nodes)

Two loading paradigms are provided:

Standard Tensor Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using :func:`pyagc.data.get_dataset`:

- Node features are returned as **dense tensors** (:math:`\mathbf{X} \in \mathbb{R}^{N \times F}`)
- Tabular features are **preprocessed (encoded / normalized)** beforehand
- Compatible with traditional GNN pipelines

.. code-block:: python

    x, edge_index, y = get_dataset('HM', root='./data')

TensorFrame-based Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using :func:`pyagc.data.get_tabular_graphland_dataset`:

- Node features are stored as :class:`torch_frame.TensorFrame`
- **Semantic types are preserved** (categorical, numerical, etc.)
- Feature preprocessing is **deferred to model-side encoders**
- Enables advanced tabular learning (e.g., feature-wise embeddings)

.. code-block:: python

    from pyagc.data import get_tabular_graphland_dataset

    data = get_tabular_graphland_dataset('HM', root='./data')

    print(type(data.x))  # torch_frame.TensorFrame
    print(data.edge_index.shape)
    print(data.y.shape)

    # TensorFrame statistics (computed on training nodes)
    print(data.tf_col_stats)

.. note::

   The TensorFrame-based pipeline is designed for integration with
   **torch-frame encoders**, allowing:

   - automatic handling of heterogeneous feature types
   - missing value processing
   - feature-wise embedding learning

Dataset Class
~~~~~~~~~~~~~~

.. autoclass:: pyagc.data.GraphLandTensorFrameDataset
   :members:
   :undoc-members:

See Also
--------

- :doc:`models` - Clustering models compatible with these datasets
- :doc:`metrics` - Evaluation metrics for clustering quality
- `PyG Dataset Documentation <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html>`_
- `OGB Documentation <https://ogb.stanford.edu/docs/nodeprop/>`_
