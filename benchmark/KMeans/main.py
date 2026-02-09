import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data

from pyagc.clusters import KMeansClusterHead
from pyagc.data import get_dataset
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.utils import get_training_config, get_logger, set_seed


def clustering_evaluation(z, y, edge_index, args, conf, logger, labeled_subgraph=None, labeled_indices=None):
    """Perform clustering and evaluate metrics.

    Args:
        z: Node embeddings. If labeled_indices is provided, z corresponds to embeddings
           of only those nodes. Otherwise, z contains embeddings for all nodes.
        y: Full label vector (for all nodes)
        edge_index: Full graph edge index
        args: Command line arguments
        conf: Configuration dictionary
        logger: Logger instance
        labeled_subgraph: Optional subgraph information for structure metrics
        labeled_indices: Optional indices of labeled nodes in the original graph
    """
    logger.info("=" * 60)
    logger.info("Clustering Stage")
    logger.info("=" * 60)

    # Normalize embeddings (optional, can be controlled by config)
    if conf.get('normalize_embeddings', True):
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        logger.info("Embeddings normalized using L2 normalization")

    # Determine which nodes we're working with
    if labeled_indices is not None:
        # z corresponds to labeled nodes only
        y = y[labeled_indices]
        logger.info(f"Working with {len(labeled_indices):,} labeled nodes")

    valid_mask = ~torch.isnan(y)
    n_clusters = int(y[valid_mask].max().item()) + 1

    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Valid nodes for evaluation: {valid_mask.sum().item()} / {len(y)}")

    # Get K-Means parameters from config
    kmeans_backend = conf.get('kmeans_backend', 'torch')
    kmeans_n_init = conf.get('kmeans_n_init', 10)
    kmeans_max_iter = conf.get('kmeans_max_iter', 300)

    logger.info(f"K-Means backend: {kmeans_backend}")
    logger.info(f"K-Means n_init: {kmeans_n_init}")
    logger.info(f"K-Means max_iter: {kmeans_max_iter}")

    # Get metrics from config
    label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
    struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))

    logger.info(f"Label metrics: {', '.join(label_metric_names)}")
    logger.info(f"Structure metrics: {', '.join(struct_metric_names)}")

    # Setup for structure metrics
    if labeled_subgraph is not None:
        logger.info(f"Using labeled subgraph for structure metrics:")
        logger.info(f"  Subgraph nodes: {labeled_subgraph['num_nodes']:,}")
        logger.info(f"  Subgraph edges: {labeled_subgraph['edge_index'].shape[1]:,}")
        struct_edge_index = labeled_subgraph['edge_index']
    else:
        struct_edge_index = edge_index

    all_results = []
    clustering_times = []
    metrics_times = []

    start_time = time.time()

    for run in range(args.runs):
        run_start = time.time()

        # K-Means clustering with configurable parameters
        kmeans_start = time.time()
        kmeans = KMeansClusterHead(
            n_clusters=n_clusters,
            backend=kmeans_backend,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=args.seed + 42 * run
        )
        pred = kmeans.fit_predict(z)
        clustering_time = time.time() - kmeans_start
        clustering_times.append(clustering_time)

        # Compute metrics
        metrics_start = time.time()

        # Compute label-based metrics
        label_results = label_metrics(
            y[valid_mask],
            pred[valid_mask],
            metrics=label_metric_names
        )
        label_metrics_end = time.time()

        # Compute structure-based metrics
        struct_results = structure_metrics(
            struct_edge_index,
            pred,
            metrics=struct_metric_names
        )
        struct_metrics_end = time.time()

        label_metrics_time = label_metrics_end - metrics_start
        struct_metrics_time = struct_metrics_end - label_metrics_end
        metrics_time = struct_metrics_end - metrics_start
        metrics_times.append(metrics_time)

        total_run_time = time.time() - run_start

        # Merge results
        run_results = {**label_results, **struct_results}
        all_results.append(run_results)

        # Build log string dynamically
        metric_strs = []
        for name in label_metric_names + struct_metric_names:
            value = run_results[name] * 100
            metric_strs.append(f"{name}={value:.2f}")

        logger.info(
            f'Run {run + 1}/{args.runs}: '
            f'{", ".join(metric_strs)} '
            f'[Cluster: {clustering_time:.2f}s,'
            f' Metrics: {label_metrics_time:.2f}+{struct_metrics_time:.2f}={metrics_time:.2f}s,'
            f' Total: {total_run_time:.2f}s]'
        )

    total_time = time.time() - start_time
    avg_clustering_time = np.mean(clustering_times)
    avg_metrics_time = np.mean(metrics_times)
    avg_total_time = avg_clustering_time + avg_metrics_time

    # Compute statistics across runs
    all_metric_names = label_metric_names + struct_metric_names
    mean_results = {}
    std_results = {}

    for metric_name in all_metric_names:
        values = [result[metric_name] * 100 for result in all_results]
        mean_results[metric_name] = np.mean(values)
        std_results[metric_name] = np.std(values)

    # Log timing summary
    logger.info("=" * 60)
    logger.info("Clustering Timing Summary")
    logger.info("=" * 60)
    logger.info(f"Average clustering time:  {avg_clustering_time:.2f}s")
    logger.info(f"Average metrics time:     {avg_metrics_time:.2f}s")
    logger.info(f"Average total time:       {avg_total_time:.2f}s")
    logger.info(f"Total time ({args.runs} runs): {total_time:.2f}s")

    return mean_results, std_results, avg_clustering_time, avg_metrics_time, total_time


def log_system_info(data, device, logger):
    """Log dataset information."""
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)

    # Dataset info
    logger.info(f"Dataset: {data}")
    logger.info(f"Nodes: {data.num_nodes:,}")
    logger.info(f"Edges: {data.num_edges:,}")
    logger.info(f"Features: {data.num_features}")
    logger.info(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")

    # Device info
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")

    logger.info(f"Method: K-Means baseline (no training)")


def log_final_results(mean_results, std_results, clustering_time, metrics_time, logger):
    """Log final results in a formatted table."""
    logger.info("=" * 60)
    logger.info("Final Results Summary")
    logger.info("=" * 60)

    # Clustering metrics
    logger.info("Clustering Metrics (mean ± std):")
    for metric_name in mean_results.keys():
        mean_val = mean_results[metric_name]
        std_val = std_results[metric_name]
        logger.info(f"  {metric_name:6s}: {mean_val:6.2f} ± {std_val:.2f}")

    # Time statistics
    logger.info("Time Statistics:")
    logger.info(f"  Training time:   {0.0:8.2f}s (no training)")
    logger.info(f"  Inference time:  {0.0:8.2f}s (using raw features)")
    logger.info(f"  Clustering time: {clustering_time:8.2f}s")
    logger.info(f"  Metrics time:    {metrics_time:8.2f}s")
    total_time = clustering_time + metrics_time
    logger.info(f"  Total time:      {total_time:8.2f}s")

    logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='K-Means baseline for node clustering')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--root', type=str, default='../data',
                        help='Root path of dataset')
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'Photo', 'Physics', 'HM', 'Flickr',
                                 'ArXiv', 'Reddit', 'MAG', 'Pokec', 'Products', 'WebTopic', 'Papers100M'],
                        help='Dataset name')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs for stability')
    args = parser.parse_args()

    # Setup device
    if args.device.startswith('cuda'):
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load configuration
    conf = get_training_config(args.dataset, config_path='train.conf.yaml')
    conf = dict(args.__dict__, **conf)

    # Setup logging
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, args.dataset, 'kmeans')
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(os.path.join(log_dir, f'seed{args.seed}_{timestamp}.log'))

    # Log configuration
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    for key, value in sorted(conf.items()):
        logger.info(f"  {key}: {value}")

    """Load Dataset"""
    set_seed(args.seed)
    logger.info(f"Loading dataset: {args.dataset}...")

    is_papers100m = args.dataset.lower() in ['papers100m', 'ogbn-papers100m']
    if is_papers100m:
        # Load with splits and subgraph
        x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph = get_dataset(
            args.dataset, root=args.root, return_splits=True
        )
        labeled_indices = torch.cat([train_idx, valid_idx, test_idx])
        # For Papers100M, only use features of labeled nodes
        x = x[labeled_indices]
    else:
        x, edge_index, y = get_dataset(args.dataset, root=args.root, return_splits=False)
        labeled_subgraph = None
        labeled_indices = None

    data = Data(x=x, edge_index=edge_index)

    # Log system information
    log_system_info(data, device, logger)

    """K-Means Clustering (No Training Required)"""
    logger.info("=" * 60)
    logger.info("K-Means directly uses raw node features")
    logger.info("No training phase required")
    logger.info("=" * 60)

    # Move features to device
    x = x.to(device)

    """Clustering"""
    set_seed(args.seed)
    mean_results, std_results, avg_clustering_time, avg_metrics_time, total_clustering_time = clustering_evaluation(
        x, y, edge_index, args, conf, logger,
        labeled_subgraph=labeled_subgraph,
        labeled_indices=labeled_indices
    )

    """Final Results"""
    # Note: We only count one clustering run for total time
    log_final_results(
        mean_results, std_results,
        avg_clustering_time, avg_metrics_time, logger
    )

    # Additional compact format for easy parsing
    logger.info("Compact Results:")

    # Build metric string dynamically
    metric_strs = []
    for metric_name in mean_results.keys():
        mean_val = mean_results[metric_name]
        std_val = std_results[metric_name]
        metric_strs.append(f"{metric_name}={mean_val:.2f}±{std_val:.2f}")

    logger.info(", ".join(metric_strs))
    logger.info(
        f"Time: Cluster={avg_clustering_time:.2f}s, Metrics={avg_metrics_time:.2f}s, "
        f"Total={avg_clustering_time + avg_metrics_time:.2f}s"
    )


if __name__ == '__main__':
    main()
