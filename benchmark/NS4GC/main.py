import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.clusters import KMeansClusterHead
from pyagc.data import get_dataset
from pyagc.encoders import create_tuned_gnn
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.models import NS4GC
from pyagc.transforms import GSSLTransform
from pyagc.utils import CheckpointManager, get_training_config, get_logger, set_seed


def train_full_batch(model, data, optimizer, conf, logger, device, ckpt_manager=None,
                     resume_from_ckpt=False):
    """Full-batch training for small graphs."""
    logger.info("=" * 60)
    logger.info("Training Mode: Full-batch")
    logger.info("=" * 60)

    data = data.to(device)
    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 50)
    save_every = conf.get('save_every', None)  # None means no periodic saves

    # Resume from checkpoint if requested
    start_epoch = 1
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_ckpt and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint(
            model, optimizer, load_best=False, device=device
        )
        if checkpoint is not None:
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)
            logger.info(f"Resuming training from epoch {start_epoch}")

    epoch_times = []
    start_time = time.time()

    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        loss = model.train_full(data, optimizer, epoch, verbose=epoch == 1 or (epoch % 10 == 0))
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # Determine if this is the best model
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint based on configuration
        if ckpt_manager is not None:
            # Always save if it's the best or last epoch
            should_save = is_best or (epoch == epochs)

            # Save periodically if save_every is specified
            if save_every is not None and save_every > 0:
                should_save = should_save or (epoch % save_every == 0)

            if should_save:
                ckpt_manager.save_checkpoint(
                    model, optimizer, epoch, loss, is_best=is_best,
                    additional_info={'patience_counter': patience_counter}
                )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000  # Convert to ms

    return total_time, avg_epoch_time, epoch


def train_mini_batch(model, data, optimizer, conf, logger, device,
                     ckpt_manager=None, resume_from_ckpt=False):
    """Mini-batch training for large graphs."""
    logger.info("=" * 60)
    logger.info("Training Mode: Mini-batch (NeighborLoader)")
    logger.info("=" * 60)

    num_layers = conf.get('num_layers', 1)
    fan_out = conf.get('fan_out', 10)
    batch_size = conf.get('batch_size', 1024)
    num_workers = conf.get('num_workers', 0)

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Fan-out: [{fan_out}] × {num_layers} layers")

    train_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 50)
    save_every = conf.get('save_every', None)  # None means no periodic epoch saves
    save_every_batch = conf.get('save_every_batch', None)  # Save every N batches

    # Resume from checkpoint if requested
    start_epoch = 1
    start_batch = 0
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_ckpt and ckpt_manager is not None:
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
            if start_batch >= len(train_loader):
                start_epoch += 1
                start_batch = 0

            if start_batch > 0:
                logger.info(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
            else:
                logger.info(f"Resuming training from epoch {start_epoch}")

    epoch_times = []
    start_time = time.time()

    epoch = start_epoch - 1
    # Choose training strategy based on save_every_batch
    if save_every_batch is None or save_every_batch <= 0:
        # Simple case: Use model.train_batch() directly
        logger.info("Using standard epoch-level training (no intra-epoch checkpoints)")

        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()

            # Use the model's built-in train_batch method
            avg_loss = model.train_batch(
                train_loader, optimizer, epoch,
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

            # Save checkpoint based on configuration
            if ckpt_manager is not None:
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
                pbar = tqdm(total=len(train_loader) * batch_size)
                pbar.set_description(f'Epoch {epoch:03d}')
            else:
                pbar = None

            epoch_start_time = time.time()
            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Skip batches if resuming from checkpoint
                if batch_idx < batch_start:
                    if pbar is not None:
                        pbar.update(batch.batch_size)
                    continue

                batch = batch.to(device)
                optimizer.zero_grad()

                loss_output = model.loss_batch(batch)

                # Handle both single Tensor and LossOutput returns
                if hasattr(loss_output, 'total'):
                    loss = loss_output.total
                else:
                    loss = loss_output

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if pbar is not None:
                    pbar.update(batch.batch_size)

                # Save checkpoint every N batches
                if (batch_idx + 1) % save_every_batch == 0:
                    avg_loss = total_loss / num_batches
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                        patience_counter = 0

                    if ckpt_manager is not None:
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

            # Log epoch results
            if epoch == 1 or epoch % 10 == 0:
                logger.info(f"Epoch: {epoch:03d} Loss: {avg_loss:.4f}")

            # Determine if this is the best model
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint at end of epoch
            if ckpt_manager is not None:
                # Always save if it's the best or last epoch
                should_save = is_best or (epoch == epochs)

                # Save periodically if save_every is specified
                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, avg_loss, is_best=is_best,
                        batch_idx=len(train_loader),  # Mark as end of epoch
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
    avg_epoch_time = np.mean(epoch_times) * 1000 if epoch_times else 0.0  # Convert to ms

    return total_time, avg_epoch_time, epoch


@torch.no_grad()
def inference_embeddings(model, data, conf, logger, device, labeled_indices=None):
    """Generate embeddings based on mini_batch configuration with automatic fallback.

    Fallback priority:
    1. GPU full-batch
    2. CPU full-batch (if GPU OOM and allowed)
    3. GPU mini-batch

    Args:
        model: The trained model
        data: Full graph data
        conf: Configuration dictionary
        logger: Logger instance
        device: Device to use
        labeled_indices: Optional tensor of node indices to compute embeddings for.
                        If provided, only compute embeddings for these nodes.
    """
    model.eval()
    logger.info("=" * 60)
    logger.info("Inference Stage")
    logger.info("=" * 60)

    # Check configuration
    mini_batch_config = conf.get('mini_batch', True)
    force_full_batch_inference = conf.get('force_full_batch_inference', False)
    allow_cpu_fallback = conf.get('allow_cpu_fallback', True)

    original_device = device
    start_time = time.time()

    if labeled_indices is not None:
        logger.info(f"Computing embeddings only for {len(labeled_indices):,} selected nodes")
    else:
        logger.info(f"Computing embeddings for all nodes")

    # Try full-batch inference first if forced or not configured as mini_batch
    if force_full_batch_inference or not mini_batch_config:
        # Try GPU full-batch first
        logger.info(f"Attempting full-batch inference on {device}...")

        try:
            z = model.infer_full(data.to(device))
            z = z[labeled_indices] if labeled_indices is not None else z

            inference_time = time.time() - start_time
            logger.info(f"✓ Full-batch inference completed on {device} in {inference_time:.2f}s")
            return z.to(original_device), inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                logger.warning(f"Full-batch inference failed on {device} due to OOM")

                # Clear CUDA cache if on GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Try CPU full-batch if allowed
                if allow_cpu_fallback and device.type == 'cuda':
                    logger.info("Attempting full-batch inference on CPU...")

                    try:
                        # Move model to CPU
                        model_device = next(model.parameters()).device
                        model.cpu()

                        cpu_start = time.time()
                        z = model.infer_full(data.to('cpu'))
                        z = z[labeled_indices] if labeled_indices is not None else z
                        cpu_time = time.time() - cpu_start

                        # Move model back to original device
                        model.to(model_device)

                        inference_time = time.time() - start_time
                        logger.info(f"✓ Full-batch inference completed on CPU in {cpu_time:.2f}s "
                                    f"(total with device transfer: {inference_time:.2f}s)")

                        return z.to(original_device), inference_time

                    except RuntimeError as cpu_e:
                        logger.warning(f"CPU full-batch inference also failed: {cpu_e}")
                        # Move model back to original device
                        model.to(model_device)
                        logger.info("Falling back to GPU mini-batch inference...")
                else:
                    logger.info("CPU fallback disabled, falling back to mini-batch inference...")
            else:
                # Re-raise if it's not a memory error
                raise e

    # GPU mini-batch inference (either configured or fallback from full-batch)
    logger.info(f"Using mini-batch inference on {device}...")

    num_layers = conf.get('num_layers', 1)

    # Use separate inference batch size if specified, otherwise use training batch size
    infer_batch_size = conf.get('infer_batch_size', conf.get('batch_size', 1024))

    # If infer_batch_size is still causing OOM, try to automatically reduce it
    original_infer_batch_size = infer_batch_size

    num_workers = conf.get('num_workers', 0)
    infer_fan_out = conf.get('infer_fan_out', -1)  # -1 means sample all neighbors

    logger.info(f"Inference batch size: {infer_batch_size}")
    logger.info(f"Inference fan-out: [{infer_fan_out}] × {num_layers} layers")
    logger.info(f"Num workers: {num_workers}")

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            inference_loader = NeighborLoader(
                data,
                input_nodes=labeled_indices,  # Only sample from labeled nodes
                num_neighbors=[infer_fan_out] * num_layers,
                batch_size=infer_batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            z = model.infer_batch(inference_loader, verbose=True)
            inference_time = time.time() - start_time

            if infer_batch_size < original_infer_batch_size:
                logger.info(f"✓ Mini-batch inference completed with reduced batch size "
                            f"({infer_batch_size}) in {inference_time:.2f}s")
            else:
                logger.info(f"✓ Mini-batch inference completed in {inference_time:.2f}s")

            return z, inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"Mini-batch inference failed after {max_retries} retries")
                    raise e

                # Reduce batch size by half
                infer_batch_size = max(infer_batch_size // 2, 32)

                logger.warning(f"Mini-batch inference OOM (attempt {retry_count}/{max_retries})")
                logger.info(f"Reducing inference batch size to {infer_batch_size} and retrying...")

                # Clear CUDA cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                # Re-raise if it's not a memory error
                raise e

    # Should not reach here
    raise RuntimeError("Inference failed after all retries")


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

    # Training mode info
    mini_batch = conf.get('mini_batch', True)
    gnn_type = conf.get('gnn_type', 'sage')
    logger.info(f"Training mode: {'Mini-batch' if mini_batch else 'Full-batch'}")
    logger.info(f"GNN type: {gnn_type.upper()}")

    # Model info
    logger.info(f"Model: {model}")
    if hasattr(model, 'encoder'):
        logger.info(f"Encoder: {model.encoder}")

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
    parser = argparse.ArgumentParser(description='NS4GC for node clustering')
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

    # Set to 0 for CPU to avoid multiprocessing issues
    if device.type == 'cpu':
        conf['num_workers'] = 0

    # Setup logging
    gnn_type = conf.get('gnn_type', 'sage').lower()
    mini_batch_mode = 'mini' if conf.get('mini_batch', True) else 'full'
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(
        args.log_dir,
        args.dataset,
        f'{gnn_type}_{mini_batch_mode}'
    )
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

    """Create Model"""
    set_seed(args.seed)

    # Get training mode and GNN type from config
    mini_batch_training = conf.get('mini_batch', True)
    gnn_type = conf.get('gnn_type', 'sage').lower()

    # Create data augmentation transforms
    transform1 = GSSLTransform(
        p_feat_mask=conf.get('p_fm1', 0.1),
        p_edge_drop=conf.get('p_ed1', 0.1)
    )
    transform2 = GSSLTransform(
        p_feat_mask=conf.get('p_fm2', 0.1),
        p_edge_drop=conf.get('p_ed2', 0.1)
    )

    # Create encoder
    encoder = create_tuned_gnn(
        gnn_type=gnn_type,
        in_channels=data.num_features,
        hidden_channels=conf.get('hidden_channels', 512),
        num_layers=conf.get('num_layers', 2),
        out_channels=None,
        dropout=conf.get('dropout', 0.0),
        act=conf.get('act', 'relu'),
        act_first=conf.get('act_first', False),
        act_last=conf.get('act_last', False),
        norm=conf.get('norm', None) if conf.get('norm') != 'none' else None,
        residual=conf.get('residual', False),
        pre_linear=conf.get('pre_linear', False),
        jk=conf.get('jk', None),
        add_self_loops=conf.get('add_self_loops', None),
        normalize=conf.get('normalize', True),
        improved=conf.get('improved', False),
        cached=conf.get('cached', True),
        aggr=conf.get('aggr', 'mean'),
        project=conf.get('project', False),
        heads=conf.get('heads', 1),
        concat=conf.get('concat', True),
        negative_slope=conf.get('negative_slope', 0.2),
        train_eps=conf.get('train_eps', False),
        bias=conf.get('bias', True),
    )

    # Create model
    model = NS4GC(
        encoder=encoder,
        transform1=transform1,
        transform2=transform2,
        s=conf.get('s', 0.6),
        tau=conf.get('tau', 0.1),
        lam=conf.get('lam', 1.0),
        gam=conf.get('gam', 1.0)
    ).to(device)

    model.set_logger(logger)

    # Log system information
    log_system_info(model, data, device, logger, conf)

    """Setup Checkpoint Manager"""
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, f'{gnn_type}_{mini_batch_mode}')
    ckpt_name = f"seed{args.seed}"
    ckpt_manager = CheckpointManager(ckpt_dir, ckpt_name, logger)

    """Training"""
    # Check if we should skip training
    skip_training = args.load_ckpt and ckpt_manager.has_checkpoint(load_best=True)

    if skip_training:
        logger.info("=" * 60)
        logger.info("Skipping training - loading best checkpoint")
        logger.info("=" * 60)
        ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)
        train_time = 0.0
        avg_epoch_time = 0.0
        final_epoch = 0
    else:
        set_seed(args.seed)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.get('lr', 0.001),
            weight_decay=conf.get('wd', 0.0)
        )

        # Choose training mode based on configuration
        if mini_batch_training:
            train_time, avg_epoch_time, final_epoch = train_mini_batch(
                model, data, optimizer, conf, logger, device,
                ckpt_manager=ckpt_manager, resume_from_ckpt=args.resume
            )
        else:
            train_time, avg_epoch_time, final_epoch = train_full_batch(
                model, data, optimizer, conf, logger, device,
                ckpt_manager=ckpt_manager, resume_from_ckpt=args.resume
            )

        # Log training statistics
        logger.info("=" * 60)
        logger.info("Training Statistics")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {final_epoch}")
        logger.info(f"Total training time: {train_time:.2f}s")
        logger.info(f"Average epoch time: {avg_epoch_time:.2f}ms")

        if device.type == 'cuda':
            mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(f"Peak GPU memory reserved: {mem_reserved:.1f} MB")
            logger.info(f"Peak GPU memory allocated: {mem_allocated:.1f} MB")

    """Inference"""
    set_seed(args.seed)

    # Load best checkpoint for inference
    ckpt_manager.load_checkpoint(model, optimizer=None, load_best=True, device=device)

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
