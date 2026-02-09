import argparse
import os
import time

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from pyagc.data import get_dataset
from pyagc.encoders import create_tuned_gnn
from pyagc.metrics import label_metrics, structure_metrics
from pyagc.models import Neuromap
from pyagc.utils import CheckpointManager, get_training_config, get_logger, set_seed


def train_full_batch(model, data, optimizer, conf, logger, device, run_id=0, verbose=True):
    """Full-batch training for small graphs.

    Args:
        model: The model to train
        data: Training data
        optimizer: Optimizer
        conf: Configuration dictionary
        logger: Logger instance
        device: Device to use
        run_id: Current run ID (for logging)
        verbose: Whether to print verbose logs

    Returns:
        Tuple of (total_time, avg_epoch_time, final_epoch, best_loss)
    """
    if verbose:
        logger.info("=" * 60)
        logger.info(f"Training Run {run_id + 1}")
        logger.info("=" * 60)

    data = data.to(device)
    epochs = conf.get('epochs', 200)
    patience = conf.get('patience', 50)

    best_loss = float('inf')
    patience_counter = 0

    epoch_times = []
    start_time = time.time()

    epoch = 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = model.train_full(data, optimizer, epoch, verbose=verbose and (epoch == 1 or epoch % 10 == 0))
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # Determine if this is the best model
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000  # Convert to ms

    return total_time, avg_epoch_time, epoch, best_loss


@torch.no_grad()
def inference_predictions(model, data, conf, logger, device, labeled_indices=None, verbose=True):
    """Generate cluster predictions with automatic fallback strategy.

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
        labeled_indices: Optional tensor of node indices to compute predictions for.
                        If provided, only compute predictions for these nodes.
        verbose: Whether to print verbose logs
    """
    model.eval()
    if verbose:
        logger.info("=" * 60)
        logger.info("Inference Stage - Predictions")
        logger.info("=" * 60)

    # Check configuration
    mini_batch_config = conf.get('mini_batch', False)
    force_full_batch_inference = conf.get('force_full_batch_inference', False)
    allow_cpu_fallback = conf.get('allow_cpu_fallback', True)

    original_device = device
    start_time = time.time()

    if verbose:
        if labeled_indices is not None:
            logger.info(f"Computing predictions only for {len(labeled_indices):,} selected nodes")
        else:
            logger.info(f"Computing predictions for all nodes")

    # Try full-batch inference first if forced or not configured as mini_batch
    if force_full_batch_inference or not mini_batch_config:
        # Try GPU full-batch first
        if verbose:
            logger.info(f"Attempting full-batch inference on {device}...")

        try:
            pred = model.infer_full(data.to(device))
            pred = pred[labeled_indices] if labeled_indices is not None else pred

            inference_time = time.time() - start_time
            if verbose:
                logger.info(f"✓ Full-batch inference completed on {device} in {inference_time:.2f}s")
            return pred.to(original_device), inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                if verbose:
                    logger.warning(f"Full-batch inference failed on {device} due to OOM")

                # Clear CUDA cache if on GPU
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Try CPU full-batch if allowed
                if allow_cpu_fallback and device.type == 'cuda':
                    if verbose:
                        logger.info("Attempting full-batch inference on CPU...")

                    try:
                        # Move model to CPU
                        model_device = next(model.parameters()).device
                        model.cpu()

                        cpu_start = time.time()
                        pred = model.infer_full(data.to('cpu'))
                        pred = pred[labeled_indices] if labeled_indices is not None else pred
                        cpu_time = time.time() - cpu_start

                        # Move model back to original device
                        model.to(model_device)

                        inference_time = time.time() - start_time
                        if verbose:
                            logger.info(f"✓ Full-batch inference completed on CPU in {cpu_time:.2f}s "
                                        f"(total with device transfer: {inference_time:.2f}s)")

                        return pred.to(original_device), inference_time

                    except RuntimeError as cpu_e:
                        if verbose:
                            logger.warning(f"CPU full-batch inference also failed: {cpu_e}")
                        # Move model back to original device
                        model.to(model_device)
                        if verbose:
                            logger.info("Falling back to GPU mini-batch inference...")
                else:
                    if verbose:
                        logger.info("CPU fallback disabled, falling back to mini-batch inference...")
            else:
                # Re-raise if it's not a memory error
                raise e

    # GPU mini-batch inference (either configured or fallback from full-batch)
    if verbose:
        logger.info(f"Using mini-batch inference on {device}...")

    num_layers = conf.get('num_layers', 1)

    # Use separate inference batch size if specified, otherwise use training batch size
    infer_batch_size = conf.get('infer_batch_size', conf.get('batch_size', 1024))

    # If infer_batch_size is still causing OOM, try to automatically reduce it
    original_infer_batch_size = infer_batch_size

    num_workers = conf.get('num_workers', 0)
    infer_fan_out = conf.get('infer_fan_out', -1)  # -1 means sample all neighbors

    if verbose:
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

            pred = model.infer_batch(inference_loader, verbose=verbose)
            inference_time = time.time() - start_time

            if verbose:
                if infer_batch_size < original_infer_batch_size:
                    logger.info(f"✓ Mini-batch inference completed with reduced batch size "
                                f"({infer_batch_size}) in {inference_time:.2f}s")
                else:
                    logger.info(f"✓ Mini-batch inference completed in {inference_time:.2f}s")

            return pred, inference_time

        except RuntimeError as e:
            if 'out of memory' in str(e).lower() or 'oom' in str(e).lower():
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error(f"Mini-batch inference failed after {max_retries} retries")
                    raise e

                # Reduce batch size by half
                infer_batch_size = max(infer_batch_size // 2, 32)

                if verbose:
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


def evaluation(pred, y, edge_index, conf, logger, labeled_subgraph=None, labeled_indices=None, verbose=True):
    """Perform evaluation on cluster predictions.

    Args:
        pred: Cluster predictions
        y: Full label vector (for all nodes)
        edge_index: Full graph edge index
        conf: Configuration dictionary
        logger: Logger instance
        labeled_subgraph: Optional subgraph information for structure metrics
        labeled_indices: Optional indices of labeled nodes in the original graph
        verbose: Whether to print verbose logs

    Returns:
        Tuple of (results, metrics_time) where results is a dictionary of metric results
        and metrics_time is the total time spent computing metrics
    """
    # Determine which nodes we're working with
    if labeled_indices is not None:
        y = y[labeled_indices]

    valid_mask = ~torch.isnan(y)

    # Get metrics from config
    label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
    struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))

    # Setup for structure metrics
    if labeled_subgraph is not None:
        struct_edge_index = labeled_subgraph['edge_index']
    else:
        struct_edge_index = edge_index

    # Start timing metrics computation
    metrics_start = time.time()

    # Compute label-based metrics
    label_results = label_metrics(
        y[valid_mask],
        pred[valid_mask],
        metrics=label_metric_names
    )

    # Compute structure-based metrics
    struct_results = structure_metrics(
        struct_edge_index,
        pred,
        metrics=struct_metric_names
    )

    # Calculate total metrics time
    metrics_time = time.time() - metrics_start

    # Merge results
    results = {**label_results, **struct_results}

    return results, metrics_time


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
    mini_batch = conf.get('mini_batch', False)
    gnn_type = conf.get('gnn_type', 'gcn')
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


def log_final_results(mean_results, std_results, train_time, inference_time, metrics_time, logger):
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
    logger.info(f"  Avg training time:   {train_time:8.2f}s")
    logger.info(f"  Avg inference time:  {inference_time:8.2f}s")
    logger.info(f"  Avg metrics time:    {metrics_time:8.2f}s")
    logger.info(f"  Avg total time:      {train_time + inference_time + metrics_time:8.2f}s")

    logger.info("=" * 60 + "\n")


def create_model(data, conf, device, n_clusters):
    """Create a new model instance.

    Args:
        data: Graph data
        conf: Configuration dictionary
        device: Device to use
        n_clusters: Number of clusters

    Returns:
        Model instance
    """
    gnn_type = conf.get('gnn_type', 'gcn').lower()

    # Create encoder
    encoder = create_tuned_gnn(
        gnn_type=gnn_type,
        in_channels=data.num_features,
        hidden_channels=conf.get('hidden_channels', 512),
        num_layers=conf.get('num_layers', 2),
        out_channels=None,
        dropout=conf.get('dropout', 0.0),
        act=conf.get('act', 'selu'),
        act_first=conf.get('act_first', False),
        act_last=conf.get('act_last', True),
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
    model = Neuromap(
        encoder=encoder,
        n_features=conf.get('hidden_channels', 64),
        n_clusters=n_clusters,
        lam=conf.get('lam', 1.0),
        alpha=conf.get('alpha', 0.15),
        n_iters=conf.get('n_iters', 100)
    ).to(device)

    return model


def main():
    parser = argparse.ArgumentParser(description='Neuromap for node clustering')
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
    gnn_type = conf.get('gnn_type', 'gcn').lower()
    mini_batch_mode = 'mini' if conf.get('mini_batch', False) else 'full'
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

    # Neuromap does not support mini-batch training
    if conf.get('mini_batch', False):
        logger.warning("Neuromap does not support mini-batch training. Setting mini_batch=False.")
        conf['mini_batch'] = False

    # Determine number of clusters
    valid_mask = ~torch.isnan(y)
    n_clusters = int(y[valid_mask].max().item()) + 1

    # Create a dummy model for logging system info
    dummy_model = create_model(data, conf, device, n_clusters)
    log_system_info(dummy_model, data, device, logger, conf)
    del dummy_model  # Free memory

    """Setup Checkpoint Manager"""
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, f'{gnn_type}_{mini_batch_mode}')
    os.makedirs(ckpt_dir, exist_ok=True)

    # Store results across runs
    all_results = []
    all_train_times = []
    all_inference_times = []
    all_metrics_times = []

    """Multi-run Training and Evaluation"""
    logger.info("=" * 60)
    logger.info(f"Starting {args.runs} independent runs")
    logger.info("=" * 60)

    for run in range(args.runs):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Run {run + 1}/{args.runs}")
        logger.info(f"{'=' * 60}")

        # Set seed for this run
        run_seed = args.seed + run
        set_seed(run_seed)

        """Create Model for this run"""
        model = create_model(data, conf, device, n_clusters)
        model.set_logger(logger)

        """Create Optimizer"""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.get('lr', 0.001),
            weight_decay=conf.get('wd', 0.00001)
        )

        """Training"""
        train_start = time.time()
        train_time, avg_epoch_time, final_epoch, best_loss = train_full_batch(
            model, data, optimizer, conf, logger, device,
            run_id=run, verbose=(run == 0)  # Only verbose for first run
        )
        train_time = time.time() - train_start

        all_train_times.append(train_time)

        # Log training statistics (verbose only for first run)
        if run == 0:
            logger.info("=" * 60)
            logger.info("Training Statistics")
            logger.info("=" * 60)
            logger.info(f"Total epochs: {final_epoch}")
            logger.info(f"Total training time: {train_time:.2f}s")
            logger.info(f"Average epoch time: {avg_epoch_time:.2f}ms")
            logger.info(f"Best loss: {best_loss:.4f}")

            if device.type == 'cuda':
                mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                logger.info(f"Peak GPU memory reserved: {mem_reserved:.1f} MB")
                logger.info(f"Peak GPU memory allocated: {mem_allocated:.1f} MB")

        """Inference"""
        set_seed(run_seed)

        pred, inference_time = inference_predictions(
            model, data, conf, logger, device,
            labeled_indices=labeled_indices,
            verbose=(run == 0)  # Only verbose for first run
        )

        all_inference_times.append(inference_time)

        if run == 0:
            logger.info(f"Prediction shape: {pred.shape}")
            if labeled_indices is not None:
                logger.info(f"Predictions correspond to {len(labeled_indices):,} labeled nodes")

        """Evaluation"""
        set_seed(run_seed)

        results, metrics_time = evaluation(
            pred, y, edge_index, conf, logger,
            labeled_subgraph=labeled_subgraph,
            labeled_indices=labeled_indices,
            verbose=False  # No verbose logging during evaluation
        )

        all_results.append(results)
        all_metrics_times.append(metrics_time)

        # Build log string dynamically
        label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
        struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))
        all_metric_names = label_metric_names + struct_metric_names

        metric_strs = []
        for name in all_metric_names:
            value = results[name] * 100
            metric_strs.append(f"{name}={value:.2f}")

        logger.info(
            f'Run {run + 1}/{args.runs}: '
            f'{", ".join(metric_strs)} '
            f'[Train: {train_time:.2f}s, Infer: {inference_time:.2f}s, Metrics: {metrics_time:.2f}s]'
        )

        # Save checkpoint for this run (optional)
        if conf.get('save_runs', False):
            ckpt_name = f"seed{args.seed}_run{run}"
            ckpt_manager = CheckpointManager(ckpt_dir, ckpt_name, logger)
            ckpt_manager.save_checkpoint(
                model, optimizer, final_epoch, best_loss, is_best=True,
                additional_info={'run': run, 'run_seed': run_seed}
            )

        # Free memory
        del model, optimizer, pred
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    """Compute Statistics Across Runs"""
    mean_results = {}
    std_results = {}

    label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
    struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))
    all_metric_names = label_metric_names + struct_metric_names

    for metric_name in all_metric_names:
        values = [result[metric_name] * 100 for result in all_results]
        mean_results[metric_name] = np.mean(values)
        std_results[metric_name] = np.std(values)

    avg_train_time = np.mean(all_train_times)
    avg_inference_time = np.mean(all_inference_times)
    avg_metrics_time = np.mean(all_metrics_times)

    """Final Results"""
    log_final_results(
        mean_results, std_results, avg_train_time, avg_inference_time, avg_metrics_time, logger
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
        f"Time: Train={avg_train_time:.2f}s, Infer={avg_inference_time:.2f}s, "
        f"Metrics={avg_metrics_time:.2f}s, "
        f"Total={avg_train_time + avg_inference_time + avg_metrics_time:.2f}s"
    )

    # Log per-run statistics
    logger.info("\n" + "=" * 60)
    logger.info("Per-Run Statistics")
    logger.info("=" * 60)
    for run in range(args.runs):
        metric_strs = []
        for name in all_metric_names:
            value = all_results[run][name] * 100
            metric_strs.append(f"{name}={value:.2f}")
        logger.info(
            f"Run {run + 1}: {', '.join(metric_strs)} "
            f"[Train: {all_train_times[run]:.2f}s, Infer: {all_inference_times[run]:.2f}s, "
            f"Metrics: {all_metrics_times[run]:.2f}s]"
        )


if __name__ == '__main__':
    main()
