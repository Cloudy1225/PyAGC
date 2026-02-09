#!/usr/bin/env python3
"""
Script to evaluate pre-trained Node2Vec embeddings on Papers100M dataset.
Usage: python eval_pretrained_node2vec.py --data_dict_path /path/to/data_dict.pt --device cuda:0
"""

import argparse
import os
import time
import numpy as np
import torch
from pyagc.clusters import KMeansClusterHead
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.utils import get_logger, set_seed


def load_pretrained_data(data_dict_path, device):
    """Load pre-trained embeddings and labels from data_dict.pt"""
    print(f"Loading pre-trained data from {data_dict_path}...")
    data_dict = torch.load(data_dict_path, map_location='cpu')

    embeddings = data_dict['node2vec_embedding']  # [1546782, 128]
    labels = data_dict['label'].squeeze()  # [1546782]
    split_idx = data_dict['split_idx']

    # Move to specified device
    embeddings = embeddings.to(device)
    labels = labels.to(device)

    print(f"✓ Loaded embeddings: {embeddings.shape} on {embeddings.device}")
    print(f"✓ Loaded labels: {labels.shape} on {labels.device}")
    print(f"✓ Train nodes: {len(split_idx['train']):,}")
    print(f"✓ Valid nodes: {len(split_idx['valid']):,}")
    print(f"✓ Test nodes: {len(split_idx['test']):,}")

    return embeddings, labels, split_idx


def load_labeled_subgraph(processed_dir, device):
    """Load labeled subgraph for structure metrics"""
    subgraph_path = os.path.join(processed_dir, 'labeled_subgraph.pt')

    if not os.path.exists(subgraph_path):
        print(f"Warning: Labeled subgraph not found at {subgraph_path}")
        return None

    print(f"Loading labeled subgraph from {subgraph_path}...")
    labeled_subgraph = torch.load(subgraph_path, map_location='cpu')

    # Move edge_index to specified device
    labeled_subgraph['edge_index'] = labeled_subgraph['edge_index'].to(device)

    print(f"✓ Subgraph nodes: {labeled_subgraph['num_nodes']:,}")
    print(f"✓ Subgraph edges: {labeled_subgraph['edge_index'].shape[1]:,}")
    print(f"✓ Edge index on: {labeled_subgraph['edge_index'].device}")

    return labeled_subgraph


def clustering_evaluation(embeddings, labels, labeled_subgraph, args, logger):
    """Perform clustering and evaluate metrics."""
    logger.info("=" * 60)
    logger.info("Clustering Stage")
    logger.info("=" * 60)

    # Ensure embeddings are on the specified device
    if embeddings.device.type != args.device.split(':')[0]:
        logger.info(f"Moving embeddings to {args.device}")
        embeddings = embeddings.to(args.device)

    logger.info(f"Embeddings device: {embeddings.device}")
    logger.info(f"Labels device: {labels.device}")

    # Normalize embeddings
    if args.normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        logger.info("Embeddings normalized using L2 normalization")

    # Check for valid labels
    valid_mask = ~torch.isnan(labels)
    n_clusters = int(labels[valid_mask].max().item()) + 1

    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Valid nodes: {valid_mask.sum().item():,} / {len(labels):,}")

    # Get metrics configuration
    label_metric_names = tuple(args.label_metrics)
    struct_metric_names = tuple(args.struct_metrics)

    logger.info(f"Label metrics: {', '.join(label_metric_names)}")
    logger.info(f"Structure metrics: {', '.join(struct_metric_names)}")

    # Prepare edge_index for structure metrics
    if labeled_subgraph is not None:
        struct_edge_index = labeled_subgraph['edge_index']
        # Ensure edge_index is on the same device
        if struct_edge_index.device != embeddings.device:
            struct_edge_index = struct_edge_index.to(embeddings.device)
        logger.info(f"Using labeled subgraph for structure metrics (device: {struct_edge_index.device})")
    else:
        struct_edge_index = None
        logger.warning("No labeled subgraph available - structure metrics will be skipped")

    all_results = []
    clustering_times = []
    metrics_times = []

    start_time = time.time()

    for run in range(args.runs):
        run_start = time.time()

        # K-Means clustering
        kmeans_start = time.time()
        kmeans = KMeansClusterHead(
            n_clusters=n_clusters,
            backend=args.kmeans_backend,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            random_state=args.seed + 42 * run
        )
        pred = kmeans.fit_predict(embeddings)
        clustering_time = time.time() - kmeans_start
        clustering_times.append(clustering_time)

        # Ensure pred is on the same device
        if pred.device != embeddings.device:
            pred = pred.to(embeddings.device)

        # Compute metrics
        metrics_start = time.time()

        # Label-based metrics
        label_results = label_metrics(
            labels[valid_mask],
            pred[valid_mask],
            metrics=label_metric_names
        )
        label_metrics_end = time.time()

        # Structure-based metrics
        if struct_edge_index is not None:
            struct_results = structure_metrics(
                struct_edge_index,
                pred,
                metrics=struct_metric_names
            )
        else:
            struct_results = {metric: 0.0 for metric in struct_metric_names}

        struct_metrics_end = time.time()

        label_metrics_time = label_metrics_end - metrics_start
        struct_metrics_time = struct_metrics_end - label_metrics_end
        metrics_time = struct_metrics_end - metrics_start
        metrics_times.append(metrics_time)

        total_run_time = time.time() - run_start

        # Merge results
        run_results = {**label_results, **struct_results}
        all_results.append(run_results)

        # Build log string
        metric_strs = []
        for name in label_metric_names + struct_metric_names:
            value = run_results[name] * 100
            metric_strs.append(f"{name}={value:.2f}")

        logger.info(
            f'Run {run + 1}/{args.runs}: '
            f'{", ".join(metric_strs)} '
            f'[Cluster: {clustering_time:.2f}s, '
            f'Metrics: {label_metrics_time:.2f}+{struct_metrics_time:.2f}={metrics_time:.2f}s, '
            f'Total: {total_run_time:.2f}s]'
        )

    total_time = time.time() - start_time
    avg_clustering_time = np.mean(clustering_times)
    avg_metrics_time = np.mean(metrics_times)

    # Compute statistics
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
    logger.info(f"Total time ({args.runs} runs): {total_time:.2f}s")

    return mean_results, std_results, avg_clustering_time, avg_metrics_time


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
    logger.info(f"  Clustering time: {clustering_time:8.2f}s")
    logger.info(f"  Metrics time:    {metrics_time:8.2f}s")
    logger.info(f"  Total time:      {clustering_time + metrics_time:8.2f}s")
    logger.info("=" * 60 + "\n")

    # Compact format
    logger.info("Compact Results:")
    metric_strs = []
    for metric_name in mean_results.keys():
        mean_val = mean_results[metric_name]
        std_val = std_results[metric_name]
        metric_strs.append(f"{metric_name}={mean_val:.2f}±{std_val:.2f}")

    logger.info(", ".join(metric_strs))
    logger.info(
        f"Time: Cluster={clustering_time:.2f}s, Metrics={metrics_time:.2f}s, "
        f"Total={clustering_time + metrics_time:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pre-trained Node2Vec embeddings on Papers100M'
    )
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (e.g., cpu, cuda:0, cuda:1)')
    parser.add_argument('--root', type=str, default='../data',
                        help='Root path of dataset')
    parser.add_argument('--data_dict_path', type=str,
                        default='../data/ogbn_papers100M/data_dict.pt',
                        help='Path to pre-trained data_dict.pt file')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize embeddings before clustering')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                        help='Do not normalize embeddings')

    # K-Means parameters
    parser.add_argument('--kmeans_backend', type=str, default='torch',
                        choices=['torch', 'sklearn'],
                        help='K-Means backend to use')
    parser.add_argument('--kmeans_n_init', type=int, default=10,
                        help='Number of K-Means initializations')
    parser.add_argument('--kmeans_max_iter', type=int, default=300,
                        help='Maximum K-Means iterations')

    # Metrics
    parser.add_argument('--label_metrics', nargs='+',
                        default=['NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp'],
                        help='Label-based metrics to compute')
    parser.add_argument('--struct_metrics', nargs='+',
                        default=['Mod', 'Cond'],
                        help='Structure-based metrics to compute')

    args = parser.parse_args()

    # Validate and setup device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print(f"Warning: CUDA not available, falling back to CPU")
            args.device = 'cpu'
        elif ':' in args.device:
            device_id = int(args.device.split(':')[1])
            if device_id >= torch.cuda.device_count():
                print(f"Warning: CUDA device {device_id} not available, using cuda:0")
                args.device = 'cuda:0'

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup logging
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, 'Papers100M_pretrained')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'seed{args.seed}_{timestamp}.log')
    logger = get_logger(log_path)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Configuration")
    logger.info("=" * 60)
    for key, value in sorted(vars(args).items()):
        logger.info(f"  {key}: {value}")
    logger.info(f"  PyTorch device: {device}")
    if device.type == 'cuda':
        logger.info(f"  CUDA device name: {torch.cuda.get_device_name(device)}")
        logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")

    # Set random seed
    set_seed(args.seed)

    # Load pre-trained data
    embeddings, labels, split_idx = load_pretrained_data(args.data_dict_path, device)

    # Load labeled subgraph
    processed_dir = os.path.join(args.root, 'ogbn_papers100M', 'processed_undirected')
    labeled_subgraph = load_labeled_subgraph(processed_dir, device)

    # Verify alignment
    logger.info("=" * 60)
    logger.info("Data Verification")
    logger.info("=" * 60)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Embeddings device: {embeddings.device}")
    logger.info(f"Labels device: {labels.device}")

    if labeled_subgraph is not None:
        expected_nodes = labeled_subgraph['num_nodes']
        actual_nodes = embeddings.shape[0]

        if expected_nodes == actual_nodes:
            logger.info(f"✓ Data alignment verified: {actual_nodes:,} nodes match")
        else:
            logger.error(
                f"✗ Data alignment mismatch! "
                f"Embeddings: {actual_nodes:,}, Subgraph: {expected_nodes:,}"
            )
            return

    # Perform clustering evaluation
    set_seed(args.seed)
    mean_results, std_results, avg_clustering_time, avg_metrics_time = clustering_evaluation(
        embeddings, labels, labeled_subgraph, args, logger
    )

    # Log final results
    log_final_results(
        mean_results, std_results,
        avg_clustering_time, avg_metrics_time,
        logger
    )

    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()

# wget https://snap.stanford.edu/ogb/data/misc/ogbn_papers100M/data_dict.pt -P /tmp/PyAGC/benchmark/data/ogbn_papers100M
# python eval_pretrained_node2vec.py --data_dict_path /tmp/PyAGC/benchmark/data/ogbn_papers100M/data_dict.pt --root /tmp/PyAGC/benchmark/data  --device cuda:1
