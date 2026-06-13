import torch
import os
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, Flickr, Reddit2, WikiCS
from torch_geometric.utils import to_undirected, add_remaining_self_loops, subgraph
import psutil

from pyagc.data.graphland import GraphLandDataset


def get_available_ram_gb():
    """Get available RAM in GB."""
    return psutil.virtual_memory().total / (1024 ** 3)


def get_dataset(name: str, root: str, return_splits=False):
    r"""Loads a graph dataset by name and returns its features, edges, and labels.

    This function serves as a unified interface for loading a wide range of
    benchmark datasets used in graph learning, including both classical
    citation networks (e.g., Cora, PubMed) and large-scale Open Graph
    Benchmark (OGB) datasets (e.g., ogbn-arxiv, ogbn-products). It
    automatically normalizes node features, converts the graph to an
    undirected version.

    Optionally, it can also return predefined train/validation/test node
    splits for benchmarking purposes.

    Args:
        name (str): The name of the dataset to load. Supported options include:
            ['cora', 'citeseer', 'pubmed', 'corafull', 'photo', 'computers',
            'cs', 'physics', 'flickr', 'reddit', 'reddit2', 'wikics', 'wiki-cs',
            'wiki_cs', 'ogbn-arxiv', 'arxiv', 'ogbn-mag', 'mag',
            'ogbn-products', 'products', 'ogbn-papers100M', 'papers100m',
            'hm-categories', 'hm', 'pokec-regions', 'pokec',
            'web-topics', 'webtopic'].
        root (str): The root directory where the dataset should be stored.
        return_splits (bool, optional): If True, also returns train/valid/test
            node indices. For WikiCS, the first predefined split is used by default.

    Returns:
        (Tuple): Depending on :attr:`return_splits`:
            - If :obj:`False`, returns ``(x, edge_index, y)``:
                * ``x``: Node feature matrix :obj:`[num_nodes, num_features]`
                * ``edge_index``: Graph connectivity in COO format :obj:`[2, num_edges]`
                * ``y``: Node label vector :obj:`[num_nodes]`
            - If :obj:`True`, returns ``(x, edge_index, y, train_idx, valid_idx, test_idx)``
              with additional index tensors for data splits.

            - For papers100M with return_splits=True, additionally returns:
              ``(x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph)``
              where labeled_subgraph contains only edge_index and original_indices
              for structure metric computation.

    Raises:
        ValueError: If the provided dataset :attr:`name` is not recognized.
    """
    # Normalize dataset name for case-insensitive matching.
    name = name.lower()

    # Special handling for papers100M dataset
    if name in ['ogbn-papers100M', 'papers100m']:
        return _load_papers100m(root, return_splits)

    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['corafull']:
        dataset = CoraFull(f'{root}/{name}', transform=T.NormalizeFeatures())
    elif name in ['photo', 'computers']:
        dataset = Amazon(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['wikics', 'wiki-cs', 'wiki_cs']:
        dataset = WikiCS(
            root=f'{root}/wikics',
            transform=T.NormalizeFeatures(),
            is_undirected=True,
        )
    elif name in ['cs', 'physics']:
        dataset = Coauthor(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['flickr']:
        dataset = Flickr(f'{root}/{name}', transform=T.NormalizeFeatures())
    elif name in ['reddit', 'reddit2']:
        name = 'reddit'
        dataset = Reddit2(root=f'{root}/{name}', transform=T.NormalizeFeatures())
    elif name in ['ogbn-arxiv', 'arxiv']:
        dataset = PygNodePropPredDataset(root=root, name='ogbn-arxiv')
    elif name in ['ogbn-mag', 'mag']:
        dataset = PygNodePropPredDataset(root=root, name='ogbn-mag')
        rel_data = dataset[0]
        # We are only interested in paper <-> paper relations.
        data = Data(
            x=rel_data.x_dict['paper'],
            edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
            y=rel_data.y_dict['paper'])
        dataset._data = data
    elif name in ['ogbn-products', 'products']:
        dataset = PygNodePropPredDataset(root=root, name='ogbn-products')
    elif name in ['hm-categories', 'hm']:
        dataset = GraphLandDataset(root=root, name='hm-categories', split='TH')
    elif name in ['pokec-regions', 'pokec']:
        dataset = GraphLandDataset(root=root, name='pokec-regions', split='TH')
    elif name in ['web-topics', 'webtopic']:
        dataset = GraphLandDataset(root=root, name='web-topics', split='TH')
    else:
        raise ValueError(f'Unknown dataset: {name}')

    # Retrieve data object and apply structural normalization.
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.y = data.y.squeeze()

    # Return with or without split indices.
    if return_splits:
        if isinstance(dataset, PygNodePropPredDataset):
            split_idx = dataset.get_idx_split()
            if name in ['mag']:
                # Handle heterogeneous dataset structure.
                train_idx, valid_idx, test_idx = (
                    split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper'])
            else:
                train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        else:
            # For standard datasets using boolean masks.
            # WikiCS has multiple splits:
            # train_mask / val_mask are usually [num_nodes, num_splits].
            # We directly use the first split by default.
            if data.train_mask.dim() == 2:
                train_mask = data.train_mask[:, 0]
                valid_mask = data.val_mask[:, 0]

                if data.test_mask.dim() == 2:
                    test_mask = data.test_mask[:, 0]
                else:
                    test_mask = data.test_mask
            else:
                train_mask = data.train_mask
                valid_mask = data.val_mask
                test_mask = data.test_mask

            train_idx = train_mask.nonzero(as_tuple=False).view(-1)
            valid_idx = valid_mask.nonzero(as_tuple=False).view(-1)
            test_idx = test_mask.nonzero(as_tuple=False).view(-1)
        return data.x, data.edge_index, data.y, train_idx, valid_idx, test_idx
    else:
        return data.x, data.edge_index, data.y


def _load_papers100m(root: str, return_splits: bool):
    """Special handler for papers100M dataset with preprocessing and caching."""

    # Define paths for preprocessed data
    processed_dir = os.path.join(root, 'ogbn_papers100M', 'processed_undirected')
    os.makedirs(processed_dir, exist_ok=True)

    processed_data_path = os.path.join(processed_dir, 'data.pt')
    processed_splits_path = os.path.join(processed_dir, 'splits.pt')
    processed_subgraph_path = os.path.join(processed_dir, 'labeled_subgraph.pt')

    # Check if preprocessed data exists
    if os.path.exists(processed_data_path):
        print(f"Loading preprocessed papers100M dataset from {processed_data_path}")
        cached_data = torch.load(processed_data_path)
        x = cached_data['x']
        edge_index = cached_data['edge_index']
        y = cached_data['y']

        if return_splits:
            if os.path.exists(processed_splits_path):
                splits = torch.load(processed_splits_path)
                train_idx = splits['train']
                valid_idx = splits['valid']
                test_idx = splits['test']
            else:
                print("Warning: Preprocessed splits not found, loading from dataset...")
                dataset = PygNodePropPredDataset(root=root, name='ogbn-papers100M')
                split_idx = dataset.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
                # Save splits for future use
                torch.save({
                    'train': train_idx,
                    'valid': valid_idx,
                    'test': test_idx
                }, processed_splits_path)

            # Load or create labeled subgraph (only structure, no features)
            if os.path.exists(processed_subgraph_path):
                print(f"Loading preprocessed labeled subgraph from {processed_subgraph_path}")
                labeled_subgraph = torch.load(processed_subgraph_path)
            else:
                print("Warning: Preprocessed labeled subgraph not found.")
                labeled_subgraph = None

            return x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph
        else:
            return x, edge_index, y

    # First time loading - check RAM and preprocess
    print("First time loading papers100M dataset...")
    available_ram = get_available_ram_gb()
    required_ram = 400  # GB

    print(f"Available RAM: {available_ram:.2f} GB")
    print(f"Estimated required RAM: {required_ram} GB")

    if available_ram < required_ram:
        raise MemoryError(
            f"Insufficient RAM for processing papers100M dataset. "
            f"Available: {available_ram:.2f} GB, Required: ~{required_ram} GB. "
            f"Please run this preprocessing step on a machine with sufficient memory."
        )

    print("Loading original dataset...")
    dataset = PygNodePropPredDataset(root=root, name='ogbn-papers100M')
    data = dataset[0]

    print("Converting to undirected graph (this may take a while)...")
    edge_index_undirected = to_undirected(data.edge_index)

    print("Preparing data for saving...")
    y = data.y.squeeze()

    # Save preprocessed data
    print(f"Saving preprocessed data to {processed_data_path}")
    torch.save({
        'x': data.x,
        'edge_index': edge_index_undirected,
        'y': y
    }, processed_data_path)

    # Save splits and create labeled subgraph
    if return_splits:
        print(f"Saving splits to {processed_splits_path}")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
        torch.save({
            'train': train_idx,
            'valid': valid_idx,
            'test': test_idx
        }, processed_splits_path)

        # Create and save labeled subgraph (structure only, no features)
        print("Creating labeled subgraph for structure metrics (this may take a while)...")
        labeled_nodes = torch.cat([train_idx, valid_idx, test_idx])
        print(f"Number of labeled nodes: {labeled_nodes.shape[0]:,}")

        # Extract subgraph edges for labeled nodes
        sub_edge_index = subgraph(
            labeled_nodes,
            edge_index_undirected,
            relabel_nodes=True,
            num_nodes=data.num_nodes
        )[0]

        # Create mapping from original indices to subgraph indices
        node_mapping = torch.full((data.num_nodes,), -1, dtype=torch.long)
        node_mapping[labeled_nodes] = torch.arange(labeled_nodes.shape[0])

        # Create lightweight subgraph data object (no features, no labels)
        labeled_subgraph = {
            'edge_index': sub_edge_index,
            'num_nodes': labeled_nodes.shape[0],
            'original_indices': labeled_nodes  # Keep track of original node indices for mapping predictions
        }

        print(f"Labeled subgraph: {labeled_subgraph['num_nodes']:,} nodes, {sub_edge_index.shape[1]:,} edges")
        print(f"Saving labeled subgraph to {processed_subgraph_path}")
        torch.save(labeled_subgraph, processed_subgraph_path)

    print("Preprocessing complete!")

    if return_splits:
        return data.x, edge_index_undirected, y, train_idx, valid_idx, test_idx, labeled_subgraph
    else:
        return data.x, edge_index_undirected, y
