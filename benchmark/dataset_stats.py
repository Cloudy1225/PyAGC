import argparse
import torch
from torch_geometric.utils import homophily

from pyagc.data import get_dataset
from pyagc.metrics import structure_metrics


def calculate_dataset_statistics(dataset_name, root='../data'):
    """Calculate and print statistics for a single dataset.

    Args:
        dataset_name (str): Name of the dataset
        root (str): Root directory for datasets
    """
    print(f"\n{'=' * 80}")
    print(f"Computing statistics for: {dataset_name}")
    print(f"{'=' * 80}\n")

    # Special handling for Papers100M
    is_papers100m = dataset_name.lower() in ['papers100m', 'ogbn-papers100m']

    if is_papers100m:
        print("Loading Papers100M dataset (this may take a while)...")
        x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph = get_dataset(
            dataset_name, root=root, return_splits=True
        )

        # For Papers100M, work directly with labeled subgraph
        labeled_indices = labeled_subgraph['original_indices']
        num_nodes = x.size(0)  # Full graph nodes
        num_edges = edge_index.size(1)  # Full graph edges
        num_labeled_nodes = labeled_subgraph['num_nodes']
        num_labeled_edges = labeled_subgraph['edge_index'].size(1)

        # Labels for labeled nodes only
        y_labeled = y[labeled_indices]

        # Use labeled subgraph for all computations
        compute_edge_index = labeled_subgraph['edge_index']
        compute_y = y_labeled
        compute_num_nodes = num_labeled_nodes

    else:
        print(f"Loading {dataset_name} dataset...")
        x, edge_index, y = get_dataset(dataset_name, root=root, return_splits=False)

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        # Use full graph for all computations
        compute_edge_index = edge_index
        compute_y = y
        compute_num_nodes = num_nodes

    # Basic statistics
    avg_degree = num_edges / num_nodes
    num_features = x.size(1)

    # Number of clusters (exclude NaN labels)
    valid_mask = ~torch.isnan(compute_y)
    num_clusters = int(compute_y[valid_mask].max().item()) + 1
    num_valid_labels = valid_mask.sum().item()

    print(f"{'Statistic':<25} {'Value':>15}")
    print(f"{'-' * 42}")
    print(f"{'Nodes':<25} {num_nodes:>15,}")
    print(f"{'Edges':<25} {num_edges:>15,}")
    print(f"{'Avg. Degree':<25} {avg_degree:>15.2f}")
    print(f"{'Features':<25} {num_features:>15,}")
    print(f"{'Clusters':<25} {num_clusters:>15}")

    if is_papers100m:
        print(f"\n{'--- Labeled Subgraph ---':<25}")
        print(f"{'Labeled Nodes':<25} {num_labeled_nodes:>15,}")
        print(f"{'Labeled Edges':<25} {num_labeled_edges:>15,}")
        print(f"{'Labeled Avg. Degree':<25} {num_labeled_edges / num_labeled_nodes:>15.2f}")
        print(f"{'Labeled Ratio':<25} {num_labeled_nodes / num_nodes * 100:>14.2f}%")

    # Filter out NaN labels for computations
    if num_valid_labels < compute_num_nodes:
        print(f"{'Valid Labels':<25} {num_valid_labels:>15,}")
        valid_nodes = torch.where(valid_mask)[0]

        # Filter edge_index and labels to valid nodes only
        from torch_geometric.utils import subgraph
        filtered_edge_index, _ = subgraph(
            valid_nodes,
            compute_edge_index,
            relabel_nodes=True,
            num_nodes=compute_num_nodes
        )
        filtered_y = compute_y[valid_nodes].long()
    else:
        filtered_edge_index = compute_edge_index
        filtered_y = compute_y.long()

    # Compute homophily
    print(f"\n{'Computing homophily metrics...'}")

    try:
        edge_homo = homophily(
            filtered_edge_index,
            filtered_y,
            method='edge'
        )
        print(f"{'Edge Homophily (H_e)':<25} {edge_homo:>15.4f}")
    except Exception as e:
        edge_homo = None
        print(f"{'Edge Homophily (H_e)':<25} {'Error':>15}")
        print(f"  Error: {e}")

    try:
        node_homo = homophily(
            filtered_edge_index,
            filtered_y,
            method='node'
        )
        print(f"{'Node Homophily (H_n)':<25} {node_homo:>15.4f}")
    except Exception as e:
        node_homo = None
        print(f"{'Node Homophily (H_n)':<25} {'Error':>15}")
        print(f"  Error: {e}")

    # Compute structure metrics on golden labels
    print(f"\n{'Computing structure metrics on golden labels...'}")

    try:
        # Compute modularity and conductance
        struct_results = structure_metrics(
            filtered_edge_index,
            filtered_y,
            metrics=('Mod', 'Cond')
        )

        modularity = 100 * struct_results['Mod']
        conductance = 100 * struct_results['Cond']

        print(f"{'Modularity':<25} {modularity:>15.2f}")
        print(f"{'Conductance':<25} {conductance:>15.2f}")

    except Exception as e:
        modularity = None
        conductance = None
        print(f"{'Structure Metrics':<25} {'Error':>15}")
        print(f"  Error: {e}")

    print(f"\n{'=' * 80}\n")

    # Print summary in table format for easy copying
    print("Summary (for table):")

    if is_papers100m:
        # For Papers100M, show both full graph and labeled subgraph stats
        print(f"\nFull Graph:")
        print(f"{dataset_name:<12} | {num_nodes:>7,} | {num_edges:>9,} | {avg_degree:>6.1f} | "
              f"{num_features:>6,} | {num_clusters:>3}")

        print(f"\nLabeled Subgraph (for metrics):")
        print(f"{dataset_name:<12} | {num_labeled_nodes:>7,} | {num_labeled_edges:>9,} | "
              f"{num_labeled_edges / num_labeled_nodes:>6.1f} | {num_features:>6,} | {num_clusters:>3} | ", end="")
    else:
        print(f"{dataset_name:<12} | {num_nodes:>7,} | {num_edges:>9,} | {avg_degree:>6.1f} | "
              f"{num_features:>6,} | {num_clusters:>3} | ", end="")

    # Homophily
    if edge_homo is not None and node_homo is not None:
        print(f"{edge_homo:.2f} | {node_homo:.2f} | ", end="")
    else:
        print("N/A | N/A | ", end="")

    # Structure metrics
    if modularity is not None and conductance is not None:
        print(f"{modularity:.2f} | {conductance:.2f}")
    else:
        print("N/A | N/A")


def main():
    parser = argparse.ArgumentParser(description='Calculate dataset statistics')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Photo', 'Physics', 'HM', 'Flickr',
                                 'ArXiv', 'Reddit', 'MAG', 'Pokec', 'Products', 'WebTopic', 'Papers100M'],
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='./data',
                        help='Root directory for datasets')
    args = parser.parse_args()

    calculate_dataset_statistics(args.dataset, args.root)


if __name__ == '__main__':
    main()

# python dataset_stats.py --dataset Cora
# python dataset_stats.py --dataset Papers100M --root /tmp/PyAGC/benchmark/data


# Statistic                           Value
# ------------------------------------------
# Nodes                         111,059,956
# Edges                       3,228,124,712
# Avg. Degree                         29.07
# Features                              128
# Clusters                              172

# --- Labeled Subgraph ---
# Labeled Nodes                   1,546,782
# Labeled Edges                  27,298,702
# Labeled Avg. Degree                 17.65
# Labeled Ratio                       1.39%

# Computing homophily metrics...
# Edge Homophily (H_e)               0.5735
# Node Homophily (H_n)               0.4981

# Computing structure metrics on golden labels...
# Modularity                          51.30
# Conductance                         42.65
