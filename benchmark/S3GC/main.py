import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data

from pyagc.clusters import KMeansClusterHead
from pyagc.data import get_dataset
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.models import S3GC
from pyagc.models.s3gc import precompute_features, CompositeOptimizer
from pyagc.utils import CheckpointManager, get_training_config, get_logger, set_seed


def precompute_and_cache_features(x, edge_index, num_nodes, conf, logger, cache_dir, dataset_name):
    """Precompute AX and SX with parameter-aware caching support for large graphs.

    Args:
        x: Node features
        edge_index: Graph edges
        num_nodes: Number of nodes
        conf: Configuration dictionary
        logger: Logger instance
        cache_dir: Directory to store cached features
        dataset_name: Name of the dataset

    Returns:
        Tuple of (AX, SX) tensors
    """
    # Extract parameters that affect feature computation
    diffusion_method = conf.get('diffusion_method', 'ppr')
    k_hop = conf.get('k_hop', 2)
    alpha = conf.get('alpha', 0.2)
    heat_t = conf.get('heat_t', 5.0)

    # Build descriptive cache filename based on parameters
    if diffusion_method == 'ppr':
        cache_filename = f"{dataset_name}_features_ppr_k{k_hop}_alpha{alpha}.pt"
    elif diffusion_method == 'heat':
        cache_filename = f"{dataset_name}_features_heat_k{k_hop}_t{heat_t}.pt"
    else:
        cache_filename = f"{dataset_name}_features_{diffusion_method}_k{k_hop}.pt"

    cache_path = os.path.join(cache_dir, cache_filename)

    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached['AX'], cached['SX']

    # Precompute features
    logger.info(f"Precomputing features using {diffusion_method} diffusion...")
    logger.info(f"Parameters: k_hop={k_hop}, alpha={alpha}, heat_t={heat_t}")

    AX, SX = precompute_features(
        x, edge_index, num_nodes,
        method=diffusion_method,
        k_hop=k_hop,
        alpha=alpha,
        t=heat_t,
        verbose=True
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Caching features to {cache_path}")
    torch.save({'AX': AX, 'SX': SX}, cache_path)

    return AX, SX


def train_s3gc_with_checkpointing(model, loader, optimizer, conf, logger, device,
                                  ckpt_manager=None, resume_from_ckpt=False):
    """Training loop for S3GC with batch-level checkpoint support."""
    logger.info("=" * 60)
    logger.info("Training Stage")
    logger.info("=" * 60)

    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 20)
    save_every = conf.get('save_every', None)  # None means no periodic epoch saves
    save_every_batch = conf.get('save_every_batch', None)  # Save every N batches
    enable_checkpointing = conf.get('enable_checkpointing', True)  # Global checkpoint enable/disable

    # Log checkpointing status
    if not enable_checkpointing:
        logger.info("⚠️  Checkpointing is DISABLED - no checkpoints will be saved during training")
        logger.info("   This significantly reduces I/O overhead for large models")
    elif ckpt_manager is None:
        logger.info("⚠️  No checkpoint manager provided - checkpoints will not be saved")
    else:
        logger.info("✓ Checkpointing is ENABLED")
        if save_every_batch is not None and save_every_batch > 0:
            logger.info(f"  - Intra-epoch checkpoints: every {save_every_batch} batches")
        if save_every is not None and save_every > 0:
            logger.info(f"  - Periodic checkpoints: every {save_every} epochs")
        logger.info(f"  - Best and last checkpoints will be saved")

    # Resume from checkpoint if requested
    start_epoch = 1
    start_batch = 0
    best_loss = float('inf')
    patience_counter = 0

    # Only attempt to resume if checkpointing is enabled
    if resume_from_ckpt and enable_checkpointing and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint(
            model, optimizer, load_best=False, device=device
        )
        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch_idx', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)

            # If we have intra-epoch checkpoint but save_every_batch is None,
            # we can't resume from mid-epoch, so start from next epoch
            if start_batch > 0 and save_every_batch is None:
                logger.info(
                    f"Warning: Found intra-epoch checkpoint (batch {start_batch}) "
                    f"but save_every_batch is disabled. Starting from next epoch."
                )
                start_epoch += 1
                start_batch = 0

            # If we finished the epoch, move to next epoch
            if start_batch >= len(loader):
                start_epoch += 1
                start_batch = 0

            if start_batch > 0:
                logger.info(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
            else:
                logger.info(f"Resuming training from epoch {start_epoch}")
    elif resume_from_ckpt and not enable_checkpointing:
        logger.warning("Cannot resume training: checkpointing is disabled")

    epoch_times = []
    start_time = time.time()

    # Prepare node mapping on device
    mapping = torch.zeros(model.num_nodes, dtype=torch.long, device=device)

    epoch = start_epoch - 1
    # Choose training strategy based on save_every_batch
    if save_every_batch is None or save_every_batch <= 0:
        # Simple case: Use model's built-in train_epoch
        logger.info("Using standard epoch-level training (no intra-epoch checkpoints)")

        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()

            # Use the model's built-in train_epoch method
            avg_loss = model.train_epoch(
                loader, optimizer, epoch,
                verbose=(epoch == 1 or epoch % 10 == 0)
            )

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # Determine if this is the best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint based on configuration (only if checkpointing is enabled)
            if enable_checkpointing and ckpt_manager is not None:
                # Always save if it's the best or last epoch
                should_save = is_best or (epoch == epochs)

                # Save periodically if save_every is specified
                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss, is_best=is_best,
                        additional_info={'patience_counter': patience_counter}
                    )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    else:
        # Complex case: Save checkpoints at batch level (for very large datasets)
        logger.info(f"Using intra-epoch checkpointing (save every {save_every_batch} batches)")

        for epoch in range(start_epoch, epochs + 1):
            model.train()

            # Determine starting batch for current epoch
            batch_start = start_batch if epoch == start_epoch else 0

            # Setup progress bar for first epoch and every 10 epochs
            if batch_start == 0 and (epoch == 1 or epoch % 10 == 0):
                from tqdm import tqdm
                pbar = tqdm(total=len(loader))
                pbar.set_description(f'Epoch {epoch:03d}')
            else:
                pbar = None

            epoch_start_time = time.time()
            total_loss = 0.0
            total_pos = 0.0
            total_neg = 0.0
            num_batches = 0

            for batch_idx, (pos_rw, neg_rw) in enumerate(loader):
                # Skip batches if resuming from checkpoint
                if batch_idx < batch_start:
                    if pbar is not None:
                        pbar.update(1)
                    continue

                optimizer.zero_grad()

                pos_rw = pos_rw.to(device)
                neg_rw = neg_rw.to(device)

                # Get unique nodes in this batch
                unique = torch.unique(torch.cat((pos_rw, neg_rw), dim=-1))
                mapping.scatter_(0, unique, torch.arange(unique.size(0), device=device))

                # Compute loss
                loss_output = model.loss(pos_rw, neg_rw, mapping)
                loss_output.total.backward()
                optimizer.step()

                total_loss += loss_output.total.item()
                total_pos += loss_output.components['pos']
                total_neg += loss_output.components['neg']
                num_batches += 1

                if pbar is not None:
                    pbar.update(1)

                # Save checkpoint every N batches (only if checkpointing is enabled)
                if enable_checkpointing and ckpt_manager is not None and (batch_idx + 1) % save_every_batch == 0:
                    avg_loss = total_loss / num_batches
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                        patience_counter = 0

                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss,
                        is_best=is_best, batch_idx=batch_idx + 1,
                        additional_info={'patience_counter': patience_counter}
                    )

            if pbar is not None:
                pbar.close()

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # Compute average loss for the epoch
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            avg_pos = total_pos / num_batches if num_batches > 0 else 0.0
            avg_neg = total_neg / num_batches if num_batches > 0 else 0.0

            # Log epoch results
            if epoch == 1 or epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d} Loss: {avg_loss:.4f}, POS: {avg_pos:.4f}, NEG: {avg_neg:.4f}")

            # Determine if this is the best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint at end of epoch (only if checkpointing is enabled)
            if enable_checkpointing and ckpt_manager is not None:
                should_save = is_best or (epoch == epochs)

                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss, is_best=is_best,
                        batch_idx=len(loader),  # Mark as end of epoch
                        additional_info={'patience_counter': patience_counter}
                    )

            # Reset start_batch for subsequent epochs
            if epoch == start_epoch:
                start_batch = 0

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000 if epoch_times else 0.0

    # Log training statistics
    logger.info("=" * 60)
    logger.info("Training Statistics")
    logger.info("=" * 60)
    logger.info(f"Total epochs: {epoch}")
    logger.info(f"Total training time: {total_time:.2f}s")
    logger.info(f"Average epoch time: {avg_epoch_time:.2f}ms")

    if device.type == 'cuda':
        mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logger.info(f"Peak GPU memory reserved: {mem_reserved:.1f} MB")
        logger.info(f"Peak GPU memory allocated: {mem_allocated:.1f} MB")

    return total_time, avg_epoch_time, epoch


@torch.no_grad()
def inference_embeddings(model, data, conf, logger, device, labeled_indices=None):
    """Generate embeddings with support for labeled subset inference.

    For S3GC, since it doesn't use mini-batch sampling during inference and
    embeddings are computed from precomputed features, we always do full inference
    and then optionally select labeled nodes.

    Args:
        model: The trained S3GC model
        data: Full graph data (not used for S3GC, but kept for API consistency)
        conf: Configuration dictionary
        logger: Logger instance
        device: Device to use
        labeled_indices: Optional tensor of node indices to return embeddings for.
                        If provided, only return embeddings for these nodes.

    Returns:
        Tuple of (embeddings, inference_time)
    """
    model.eval()
    logger.info("=" * 60)
    logger.info("Inference Stage")
    logger.info("=" * 60)

    if labeled_indices is not None:
        logger.info(f"Computing embeddings for {len(labeled_indices):,} labeled nodes")
    else:
        logger.info(f"Computing embeddings for all {model.num_nodes:,} nodes")

    start_time = time.time()

    with torch.no_grad():
        model.eval()

        # S3GC inference strategy:
        # Since S3GC uses precomputed features (AX, SX) and learnable embeddings,
        # full inference is very fast (just forward pass through encoder)
        # So we always do full inference and then optionally select nodes

        # For very large graphs, we can optimize by only computing for labeled nodes
        if labeled_indices is not None and model.scale == 'large':
            # Only compute embeddings for labeled nodes (memory efficient)
            logger.info("Using memory-efficient inference for large graph (labeled nodes only)")
            z = model.embed(indices=labeled_indices)
        else:
            # Compute all embeddings
            z = model.embed()

            # Select labeled nodes if specified
            if labeled_indices is not None:
                logger.info("Selecting embeddings for labeled nodes")
                z = z[labeled_indices]

    inference_time = time.time() - start_time

    logger.info(f"✓ Inference completed in {inference_time:.2f}s")

    return z, inference_time


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


def log_system_info(model, data, device, logger, conf):
    """Log dataset and model information."""
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)

    # Dataset info
    logger.info(f"Dataset: {data}")
    logger.info(f"Nodes: {data.num_nodes:,}")
    logger.info(f"Edges: {data.num_edges:,}")
    logger.info(f"Features: {data.num_features}")
    logger.info(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")

    # Model info
    logger.info(f"Model: {model}")
    logger.info(f"Model scale: {model.scale}")
    logger.info(f"Hidden channels: {model.hidden_channels}")
    logger.info(f"Walk length: {model.walk_length + 1}")  # +1 because internally it's walk_length - 1
    logger.info(f"Context size: {model.context_size}")
    logger.info(f"Walks per node: {model.walks_per_node}")

    # S3GC specific info
    logger.info(f"Diffusion method: {conf['diffusion_method']}")
    logger.info(f"Diffusion hops (k): {conf['k_hop']}")
    if conf['diffusion_method'] == 'ppr':
        logger.info(f"PPR alpha: {conf.get('alpha', 0.2)}")
    elif conf['diffusion_method'] == 'heat':
        logger.info(f"Heat diffusion t: {conf.get('heat_t', 5.0)}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params / 1e6:.3f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.3f}M")

    # Device info
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"CUDA version: {torch.version.cuda}")


def log_final_results(mean_results, std_results, train_time, inference_time,
                     clustering_time, metrics_time, logger):
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
    logger.info(f"  Training time:   {train_time:8.2f}s")
    logger.info(f"  Inference time:  {inference_time:8.2f}s")
    logger.info(f"  Clustering time: {clustering_time:8.2f}s")
    logger.info(f"  Metrics time:    {metrics_time:8.2f}s")
    total_eval_time = clustering_time + metrics_time
    logger.info(f"  Total eval time: {total_eval_time:8.2f}s")
    logger.info(f"  Total time:      {train_time + inference_time + total_eval_time:8.2f}s")
    logger.info(f"  Train+Infer+Cluster time: {train_time + inference_time + clustering_time:8.2f}s")

    logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='S3GC for node clustering')
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
    parser.add_argument('--ckpt_dir', type=str, default='ckpts',
                        help='Directory to save checkpoints')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to cache precomputed features')
    parser.add_argument('--load_ckpt', action='store_true',
                        help='Whether to load existing checkpoint for inference only')
    parser.add_argument('--resume', action='store_true',
                        help='Whether to resume training from last checkpoint')
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
    log_dir = os.path.join(args.log_dir, args.dataset)
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
    else:
        x, edge_index, y = get_dataset(args.dataset, root=args.root, return_splits=False)
        labeled_subgraph = None
        labeled_indices = None
    data = Data(x=x, edge_index=edge_index)
    num_nodes = x.size(0)

    """Precompute Features"""
    set_seed(args.seed)
    AX, SX = precompute_and_cache_features(
        x, edge_index, num_nodes, conf, logger,
        args.cache_dir, args.dataset
    )

    """Create Model"""
    set_seed(args.seed)

    # Determine if we should keep embeddings on CPU
    keep_embeddings_on_cpu = is_papers100m  # or use config
    model = S3GC(
        edge_index=edge_index,
        num_nodes=num_nodes,
        in_channels=x.size(1),
        hidden_channels=conf['hidden_channels'],
        walk_length=conf['walk_length'],
        context_size=conf['walk_length'],  # Use same as walk_length
        walks_per_node=conf['walks_per_node'],
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        scale=conf.get('scale', 'auto'),
        keep_embeddings_on_cpu=keep_embeddings_on_cpu
    )

    # Move encoder to GPU, but keep embeddings on CPU if specified
    if keep_embeddings_on_cpu:
        logger.info(f"⚠️  Keeping {model.num_nodes:,} embeddings on CPU to save GPU memory")
        logger.info(f"   Embedding size: {model.num_nodes * model.hidden_channels * 4 / 1e9:.2f} GB")
        # Only move encoder to GPU
        model.encoder = model.encoder.to(device)
    else:
        model = model.to(device)

    # Set precomputed features
    model.set_precomputed_features(
        AX, SX,
        keep_on_cpu=is_papers100m  # Keep features on CPU for Papers100M
    )

    # Log system information
    log_system_info(model, data, device, logger, conf)

    """Setup Checkpoint Manager"""
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
    ckpt_name = f"seed{args.seed}"
    ckpt_manager = CheckpointManager(ckpt_dir, ckpt_name, logger)

    """Setup Checkpoint Manager"""
    enable_checkpointing = conf.get('enable_checkpointing', True)

    if enable_checkpointing:
        ckpt_dir = os.path.join(args.ckpt_dir, args.dataset)
        ckpt_name = f"seed{args.seed}"
        ckpt_manager = CheckpointManager(ckpt_dir, ckpt_name, logger)
        logger.info(f"✓ Checkpoint manager initialized: {ckpt_dir}/{ckpt_name}")
    else:
        ckpt_manager = None
        logger.info("⚠️  Checkpointing disabled - no checkpoints will be saved or loaded")

    """Training"""
    # Check if we should skip training (only if checkpointing is enabled)
    skip_training = (
        enable_checkpointing and
        args.load_ckpt and
        ckpt_manager is not None and
        ckpt_manager.has_checkpoint(load_best=True)
    )

    if skip_training:
        logger.info("=" * 60)
        logger.info("Skipping training - loading best checkpoint")
        logger.info("=" * 60)
        ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)
        train_time = 0.0
        avg_epoch_time = 0.0
        final_epoch = 0
    else:
        # Warning if trying to load checkpoint when checkpointing is disabled
        if args.load_ckpt and not enable_checkpointing:
            logger.warning(
                "⚠️  --load_ckpt flag is set but checkpointing is disabled in config. "
                "Training from scratch."
            )

        set_seed(args.seed)

        # Create data loader
        loader = model.loader(
            batch_size=conf['batch_size'],
            shuffle=True,
            num_workers=0  # S3GC uses custom sampling, set to 0
        )

        # Setup optimizer based on model scale
        if model.scale == 'large':
            # Use CompositeOptimizer for large graphs with sparse embeddings
            logger.info("Using CompositeOptimizer (SparseAdam + Adam) for large-scale model")
            if keep_embeddings_on_cpu:
                # Embeddings are on CPU - no weight decay for sparse optimizer
                sparse_opt = torch.optim.SparseAdam(
                    model.iden_embedding.parameters(),
                    lr=conf['lr']
                )
                logger.info(f"  SparseAdam optimizer on CPU for {model.num_nodes:,} embeddings")
            else:
                # Embeddings are on GPU
                sparse_opt = torch.optim.SparseAdam(
                    model.iden_embedding.parameters(),
                    lr=conf['lr']
                )
                logger.info(f"  SparseAdam optimizer on GPU")

            dense_opt = torch.optim.Adam(
                model.encoder.parameters(),
                lr=conf['lr'],
                weight_decay=conf['wd']
            )
            optimizer = CompositeOptimizer(sparse=sparse_opt, dense=dense_opt)
        else:
            # Use standard Adam for small/medium graphs
            logger.info("Using standard Adam optimizer")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=conf['lr'],
                weight_decay=conf['wd']
            )

        # Train the model
        train_time, avg_epoch_time, final_epoch = train_s3gc_with_checkpointing(
            model, loader, optimizer, conf, logger, device,
            ckpt_manager=ckpt_manager, resume_from_ckpt=args.resume
        )

    """Inference"""
    set_seed(args.seed)

    # Load best checkpoint for inference (only if checkpointing is enabled)
    if enable_checkpointing and ckpt_manager is not None:
        ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)
    else:
        logger.info("Using final training state (no checkpoint loaded)")

    # For Papers100M, only infer embeddings for labeled nodes
    z, inference_time = inference_embeddings(
        model, data, conf, logger, device,
        labeled_indices=labeled_indices
    )

    logger.info(f"Embedding shape: {z.shape}")

    if labeled_indices is not None:
        logger.info(f"Embeddings correspond to {len(labeled_indices):,} labeled nodes")

    """Clustering"""
    set_seed(args.seed)
    mean_results, std_results, avg_clustering_time, avg_metrics_time, total_clustering_time = clustering_evaluation(
        z, y, edge_index, args, conf, logger,
        labeled_subgraph=labeled_subgraph,
        labeled_indices=labeled_indices
    )

    """Final Results"""
    # Note: We only count one clustering run for total time
    log_final_results(
        mean_results, std_results, train_time, inference_time,
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
    total_eval_time = avg_clustering_time + avg_metrics_time
    logger.info(
        f"Time: Train={train_time:.2f}s, Infer={inference_time:.2f}s, "
        f"Cluster={avg_clustering_time:.2f}s, Metrics={avg_metrics_time:.2f}s, "
        f"Total={train_time + inference_time + total_eval_time:.2f}s, "
        f"Train+Infer+Clustering={train_time + inference_time + avg_clustering_time:.2f}s"
    )


if __name__ == '__main__':
    main()
