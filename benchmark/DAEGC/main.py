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
from pyagc.models import DAEGC
from pyagc.utils import MultiStageCheckpointManager, get_training_config, get_logger, set_seed


def train_stage(model, data, optimizer, conf, logger, device, stage='pretrain',
               ckpt_manager=None, resume_from_ckpt=False):
    """Unified training function for both pretrain and finetune stages.

    Args:
        model: DAEGC model
        data: Graph data or data loader
        optimizer: Optimizer
        conf: Configuration dictionary
        logger: Logger instance
        device: Device to use
        stage (str): 'pretrain' or 'finetune'
        ckpt_manager: CheckpointManager instance
        resume_from_ckpt (bool): Whether to resume from checkpoint

    Returns:
        Tuple of (total_time, avg_epoch_time, final_epoch)
    """
    logger.info("=" * 60)
    logger.info(f"Training Stage: {stage.upper()}")
    logger.info("=" * 60)

    # Get stage-specific configuration
    if stage == 'pretrain':
        epochs = conf.get('pretrain_epochs', 200)
        save_every = conf.get('pretrain_save_every', None)
        save_every_batch = conf.get('pretrain_save_every_batch', None)
    else:  # finetune
        epochs = conf.get('finetune_epochs', 200)
        save_every = conf.get('finetune_save_every', None)
        save_every_batch = conf.get('finetune_save_every_batch', None)

    patience = conf.get('patience', 50)
    mini_batch = conf.get('mini_batch', False)

    # Resume from checkpoint if requested
    start_epoch = 1
    start_batch = 0
    best_loss = float('inf')
    patience_counter = 0

    if resume_from_ckpt and ckpt_manager is not None:
        checkpoint = ckpt_manager.load_checkpoint(
            model, optimizer, stage=stage, load_best=False, device=device
        )
        if checkpoint is not None:
            start_epoch = checkpoint['epoch']
            start_batch = checkpoint.get('batch_idx', 0)
            best_loss = checkpoint.get('best_loss', float('inf'))
            patience_counter = checkpoint.get('patience_counter', 0)

            if start_batch > 0:
                logger.info(f"Resuming from epoch {start_epoch}, batch {start_batch}")
            else:
                logger.info(f"Resuming from epoch {start_epoch}")

    epoch_times = []
    start_time = time.time()

    epoch = start_epoch - 1

    if mini_batch:
        # Mini-batch training
        num_layers = conf.get('num_layers', 2)
        fan_out = conf.get('fan_out', 10)
        batch_size = conf.get('batch_size', 1024)
        num_workers = conf.get('num_workers', 0)

        logger.info(f"Training mode: Mini-batch")
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

        # Mini-batch training loop
        for epoch in range(start_epoch, epochs + 1):
            epoch_start_time = time.time()
            model.train()

            batch_start = start_batch if epoch == start_epoch else 0

            if batch_start == 0 and (epoch == 1 or epoch % 10 == 0):
                from tqdm import tqdm
                pbar = tqdm(total=len(train_loader) * batch_size)
                pbar.set_description(f'{stage.capitalize()} Epoch {epoch:03d}')
            else:
                pbar = None

            total_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if batch_idx < batch_start:
                    if pbar is not None:
                        pbar.update(batch.batch_size)
                    continue

                batch = batch.to(device)
                optimizer.zero_grad()

                # Pass pretrain flag to loss computation
                loss_output = model.loss_batch(batch, pretrain=(stage == 'pretrain'))

                loss = loss_output.total
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if pbar is not None:
                    pbar.update(batch.batch_size)

                # Save checkpoint every N batches if configured
                if save_every_batch is not None and (batch_idx + 1) % save_every_batch == 0:
                    avg_loss = total_loss / num_batches
                    is_best = avg_loss < best_loss
                    if is_best:
                        best_loss = avg_loss
                        patience_counter = 0

                    if ckpt_manager is not None:
                        ckpt_manager.save_checkpoint(
                            model, optimizer, epoch, avg_loss, stage=stage,
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
                        model, optimizer, epoch, avg_loss, stage=stage,
                        is_best=is_best, batch_idx=len(train_loader),
                        additional_info={'patience_counter': patience_counter}
                    )

            # Reset start_batch for subsequent epochs
            if epoch == start_epoch:
                start_batch = 0

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    else:
        # Full-batch training
        logger.info(f"Training mode: Full-batch")
        data = data.to(device)

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            loss = model.train_full(
                data, optimizer, epoch,
                verbose=(epoch == 1 or epoch % 10 == 0),
                pretrain=(stage == 'pretrain')
            )
            epoch_time = time.time() - t0
            epoch_times.append(epoch_time)

            # Determine if this is the best model
            is_best = loss < best_loss
            if is_best:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            if ckpt_manager is not None:
                should_save = is_best or (epoch == epochs)
                if save_every is not None and save_every > 0:
                    should_save = should_save or (epoch % save_every == 0)

                if should_save:
                    ckpt_manager.save_checkpoint(
                        model, optimizer, epoch, loss, stage=stage,
                        is_best=is_best,
                        additional_info={'patience_counter': patience_counter}
                    )

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    avg_epoch_time = np.mean(epoch_times) * 1000 if epoch_times else 0.0

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
    logger.info("Inference Stage - Embeddings")
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
            z = model.embed(data.x.to(device), data.edge_index.to(device))
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
                        z = model.embed(data.x.to('cpu'), data.edge_index.to('cpu'))
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

            all_z = []
            for batch in inference_loader:
                batch = batch.to(device)
                z_batch = model.embed(batch.x, batch.edge_index)
                all_z.append(z_batch[:batch.batch_size].cpu())
            z = torch.cat(all_z, dim=0).to(device)

            # z = model.infer_batch(inference_loader, verbose=True)
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

@torch.no_grad()
def inference_predictions(model, data, conf, logger, device, labeled_indices=None):
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
    """
    model.eval()
    logger.info("=" * 60)
    logger.info("Inference Stage - Predictions")
    logger.info("=" * 60)

    mini_batch_config = conf.get('mini_batch', True)
    force_full_batch_inference = conf.get('force_full_batch_inference', False)
    allow_cpu_fallback = conf.get('allow_cpu_fallback', True)

    original_device = device
    start_time = time.time()

    if labeled_indices is not None:
        logger.info(f"Computing predictions only for {len(labeled_indices):,} selected nodes")
    else:
        logger.info(f"Computing predictions for all nodes")

    # Try full-batch inference first if forced or not configured as mini_batch
    if force_full_batch_inference or not mini_batch_config:
        logger.info(f"Attempting full-batch inference on {device}...")

        try:
            pred = model.infer_full(data.to(device))
            pred = pred[labeled_indices] if labeled_indices is not None else pred

            inference_time = time.time() - start_time
            logger.info(f"✓ Full-batch inference completed on {device} in {inference_time:.2f}s")
            return pred.to(original_device), inference_time

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
                        model_device = next(model.parameters()).device
                        model.cpu()

                        cpu_start = time.time()
                        pred = model.infer_full(data.to('cpu'))
                        pred = pred[labeled_indices] if labeled_indices is not None else pred
                        cpu_time = time.time() - cpu_start

                        # Move model back to original device
                        model.to(model_device)

                        inference_time = time.time() - start_time
                        logger.info(f"✓ Full-batch inference completed on CPU in {cpu_time:.2f}s "
                                    f"(total with device transfer: {inference_time:.2f}s)")

                        return pred.to(original_device), inference_time

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

            pred = model.infer_batch(inference_loader, verbose=True)
            inference_time = time.time() - start_time

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


def evaluate_pretrained_embeddings(model, data, y, edge_index, conf, logger, device, seed,
                                   labeled_indices=None):
    """Evaluate clustering performance of pretrained embeddings using K-Means."""
    logger.info("=" * 60)
    logger.info("Evaluating Pretrained Embeddings")
    logger.info("=" * 60)
    logger.info("Method: K-Means clustering on learned embeddings")

    eval_start_time = time.time()

    # Generate embeddings
    z, _ = inference_embeddings(model, data, conf, logger, device, labeled_indices)

    # Normalize embeddings
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    logger.info("Embeddings normalized using L2 normalization")

    # Determine which labels to use
    if labeled_indices is not None:
        y = y[labeled_indices]
        logger.info(f"Evaluating on {len(labeled_indices):,} labeled nodes")

    valid_mask = ~torch.isnan(y)
    n_clusters = int(y[valid_mask].max().item()) + 1

    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Valid nodes: {valid_mask.sum().item()} / {len(y)}")
    logger.info(f"Running K-Means on all {len(z)} nodes")

    # Get metrics from config
    label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
    struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))

    # Fit K-Means on all nodes
    kmeans_backend = conf.get('kmeans_backend', 'torch')
    kmeans_n_init = conf.get('kmeans_n_init', 10)
    kmeans_max_iter = conf.get('kmeans_max_iter', 300)
    kmeans = KMeansClusterHead(
        n_clusters=n_clusters,
        backend=kmeans_backend,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=seed
    )
    pred = kmeans.fit_predict(z)

    # Compute metrics
    label_results = label_metrics(y[valid_mask], pred[valid_mask], metrics=label_metric_names)
    struct_results = structure_metrics(edge_index, pred, metrics=struct_metric_names)

    eval_time = time.time() - eval_start_time

    # Log results
    metric_strs = []
    for name in label_metric_names:
        value = label_results[name] * 100
        metric_strs.append(f"{name}={value:.2f}")
    for name in struct_metric_names:
        value = struct_results[name] * 100
        metric_strs.append(f"{name}={value:.2f}")

    logger.info(f"Results: {', '.join(metric_strs)}")
    logger.info(f"Pretrain evaluation completed in {eval_time:.2f}s")

    return eval_time


def initialize_cluster_centers_from_inference(model, data, y, conf, logger, device, seed, labeled_indices=None):
    """Initialize cluster centers using the inference_embeddings function."""
    logger.info("=" * 60)
    logger.info("Initializing Cluster Centers")
    logger.info("=" * 60)
    logger.info("Method: K-Means on pretrained embeddings")

    # Generate embeddings using the same function as evaluation
    z, inference_time = inference_embeddings(model, data, conf, logger, device, labeled_indices)

    # Normalize embeddings
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    logger.info("Embeddings normalized using L2 normalization")

    # Determine number of clusters
    if labeled_indices is not None:
        y_subset = y[labeled_indices]
    else:
        y_subset = y

    valid_mask = ~torch.isnan(y_subset)
    n_clusters = int(y_subset[valid_mask].max().item()) + 1

    logger.info(f"Number of clusters: {n_clusters}")
    logger.info(f"Running K-Means on {len(z)} embeddings...")

    # Fit K-Means
    kmeans_backend = conf.get('kmeans_backend', 'torch')
    kmeans_n_init = conf.get('kmeans_n_init', 10)
    kmeans_max_iter = conf.get('kmeans_max_iter', 300)
    kmeans = KMeansClusterHead(
        n_clusters=n_clusters,
        backend=kmeans_backend,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
        random_state=seed
    )
    kmeans.fit_predict(z)

    # Set cluster centers in the model
    model.cluster_head.reset_cluster_centers(
        kmeans.cluster_centers.detach().to(device)
    )

    logger.info(f"✓ Cluster centers initialized: shape={model.cluster_head.cluster_centers.shape}")
    logger.info(f"Initialization completed in {inference_time:.2f}s")

    return inference_time


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


def log_final_results_multiple_runs(mean_results, std_results,
                                    pretrain_time, eval_pretrain_time,
                                    avg_init_time, std_init_time,
                                    avg_finetune_time, std_finetune_time,
                                    avg_inference_time, std_inference_time,
                                    avg_metrics_time, std_metrics_time,
                                    num_runs, logger):
    """Log final results for multiple independent runs."""
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
    logger.info("\nTime Statistics:")
    logger.info(f"  Pretraining (1 run):")
    logger.info(f"    Training time:        {pretrain_time:8.2f}s")
    if eval_pretrain_time > 0:
        logger.info(f"    Evaluation time:      {eval_pretrain_time:8.2f}s")

    logger.info(f"\n  Per Run ({num_runs} runs):")
    logger.info(f"    Init time:            {avg_init_time:8.2f} ± {std_init_time:.2f}s")
    logger.info(f"    Finetune time:        {avg_finetune_time:8.2f} ± {std_finetune_time:.2f}s")
    logger.info(f"    Inference time:       {avg_inference_time:8.2f} ± {std_inference_time:.2f}s")
    logger.info(f"    Metrics time:         {avg_metrics_time:8.2f} ± {std_metrics_time:.2f}s")

    avg_run_time = avg_init_time + avg_finetune_time + avg_inference_time + avg_metrics_time
    logger.info(f"    Avg run time:         {avg_run_time:8.2f}s")

    logger.info(f"\n  Total:")
    total_init_time = avg_init_time * num_runs
    total_finetune_time = avg_finetune_time * num_runs
    total_inference_time = avg_inference_time * num_runs
    total_metrics_time = avg_metrics_time * num_runs
    logger.info(f"    All init time:        {total_init_time:8.2f}s")
    logger.info(f"    All finetune time:    {total_finetune_time:8.2f}s")
    logger.info(f"    All inference time:   {total_inference_time:8.2f}s")
    logger.info(f"    All metrics time:     {total_metrics_time:8.2f}s")

    total_time = pretrain_time + total_init_time + total_finetune_time + total_inference_time + total_metrics_time
    logger.info(f"    Grand total time:     {total_time:8.2f}s")

    logger.info("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='DAEGC for node clustering')
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
        # Load with splits and subgraph for Papers100M
        x, edge_index, y, train_idx, valid_idx, test_idx, labeled_subgraph = get_dataset(
            args.dataset, root=args.root, return_splits=True
        )
        labeled_indices = torch.cat([train_idx, valid_idx, test_idx])
        logger.info(f"Papers100M: Using {len(labeled_indices):,} labeled nodes for inference")
    else:
        # Load normally for other datasets
        x, edge_index, y = get_dataset(args.dataset, root=args.root, return_splits=False)
        labeled_subgraph = None
        labeled_indices = None
    data = Data(x=x, edge_index=edge_index)

    """Create Model"""
    set_seed(args.seed)

    # Get training mode and GNN type from config
    mini_batch_training = conf.get('mini_batch', True)
    gnn_type = conf.get('gnn_type', 'sage').lower()

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

    # Get number of clusters from labels
    n_clusters = int(y[~torch.isnan(y)].max().item()) + 1

    # Create model
    model = DAEGC(
        encoder=encoder,
        n_clusters=n_clusters,
        hidden_channels=conf.get('hidden_channels', 256),
        gamma=conf.get('gamma', 10.0),
        update_interval=conf.get('update_interval', 5)
    ).to(device)

    model.set_logger(logger)

    # Log system information
    log_system_info(model, data, device, logger, conf)

    """Setup Checkpoint Manager"""
    ckpt_dir = os.path.join(args.ckpt_dir, args.dataset, f'{gnn_type}_{mini_batch_mode}')
    ckpt_name = f"seed{args.seed}"
    ckpt_manager = MultiStageCheckpointManager(ckpt_dir, ckpt_name,
                                               stages=['pretrain', 'finetune'], logger=logger)

    """Stage 1: Pretraining (Once)"""
    pretrain_time = 0.0
    eval_pretrain_time = 0.0

    # Check if we can skip pretraining
    skip_pretrain = args.load_ckpt and ckpt_manager.has_checkpoint(stage='pretrain', load_best=True)

    if skip_pretrain:
        logger.info("=" * 60)
        logger.info("Skipping pretraining - loading pretrain checkpoint")
        logger.info("=" * 60)
        ckpt_manager.load_checkpoint(model, optimizer=None, stage='pretrain',
                                     load_best=True, device=device)
    else:
        set_seed(args.seed)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.get('pretrain_lr', 0.01),
            weight_decay=conf.get('wd', 0.0)
        )

        pretrain_time, avg_pretrain_epoch_time, final_pretrain_epoch = train_stage(
            model, data, optimizer, conf, logger, device,
            stage='pretrain',
            ckpt_manager=ckpt_manager,
            resume_from_ckpt=args.resume
        )

        # Log pretraining statistics
        logger.info("=" * 60)
        logger.info("Pretraining Statistics")
        logger.info("=" * 60)
        logger.info(f"Total epochs: {final_pretrain_epoch}")
        logger.info(f"Total pretraining time: {pretrain_time:.2f}s")
        logger.info(f"Average epoch time: {avg_pretrain_epoch_time:.2f}ms")

        if device.type == 'cuda':
            mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(f"Peak GPU memory reserved: {mem_reserved:.1f} MB")
            logger.info(f"Peak GPU memory allocated: {mem_allocated:.1f} MB")

        # Load best pretrained model
        ckpt_manager.load_checkpoint(model, optimizer=None, stage='pretrain',
                                     load_best=True, device=device)

    # ============ Evaluate Pretrained Embeddings (Optional) ============
    if conf.get('eval_pretrain', True) and not skip_pretrain:
        set_seed(args.seed)
        if labeled_subgraph is not None:
            struct_edge_index = labeled_subgraph['edge_index']
        else:
            struct_edge_index = edge_index
        eval_pretrain_time = evaluate_pretrained_embeddings(
            model, data, y, struct_edge_index, conf, logger, device,
            args.seed, labeled_indices
        )
        # eval_pretrain_time = evaluate_pretrained_embeddings(
        #     model, data, y, edge_index, conf, logger, device,
        #     args.seed, labeled_indices
        # )

    """Stage 2: Multiple Independent Runs (Initialization + Finetuning + Evaluation)"""
    logger.info("\n" + "=" * 60)
    logger.info(f"Starting {args.runs} Independent Runs")
    logger.info("=" * 60)
    logger.info("Each run: Initialize cluster centers → Finetune → Evaluate")

    all_results = []
    all_init_times = []  # Track initialization time
    all_finetune_times = []
    all_inference_times = []
    all_metrics_times = []

    for run in range(args.runs):
        logger.info("\n" + "=" * 60)
        logger.info(f"Run {run + 1}/{args.runs}")
        logger.info("=" * 60)

        # Set different seed for each run
        run_seed = args.seed + run * 42
        set_seed(run_seed)

        # Reload pretrained model for each run
        ckpt_manager.load_checkpoint(model, optimizer=None, stage='pretrain',
                                     load_best=True, device=device)

        # ============ Initialize Cluster Centers ============
        init_time = initialize_cluster_centers_from_inference(
            model, data, y, conf, logger, device, run_seed, labeled_indices
        )
        all_init_times.append(init_time)

        # Reset GPU memory tracking for finetuning
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # ============ Finetuning ============
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=conf.get('finetune_lr', 0.001),
            weight_decay=conf.get('wd', 0.0)
        )

        # Create run-specific checkpoint manager
        run_ckpt_name = f"seed{args.seed}_run{run + 1}"
        run_ckpt_manager = MultiStageCheckpointManager(ckpt_dir, run_ckpt_name,
                                                       stages=['pretrain', 'finetune'], logger=logger)

        finetune_time, avg_finetune_epoch_time, final_finetune_epoch = train_stage(
            model, data, optimizer, conf, logger, device,
            stage='finetune',
            ckpt_manager=run_ckpt_manager,
            resume_from_ckpt=False
        )

        all_finetune_times.append(finetune_time)

        # Log finetuning statistics
        logger.info(f"Finetuning completed in {finetune_time:.2f}s "
                    f"({final_finetune_epoch} epochs, avg {avg_finetune_epoch_time:.2f}ms/epoch)")

        if device.type == 'cuda':
            mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            logger.info(f"Peak GPU memory: {mem_reserved:.1f} MB")

        # Load best finetuned model for this run
        run_ckpt_manager.load_checkpoint(model, optimizer=None, stage='finetune',
                                         load_best=True, device=device)

        # ============ Inference ============
        pred, inference_time = inference_predictions(
            model, data, conf, logger, device, labeled_indices
        )
        all_inference_times.append(inference_time)

        # ============ Evaluation ============
        # Determine which nodes we're working with
        eval_y = y[labeled_indices] if labeled_indices is not None else y
        valid_mask = ~torch.isnan(eval_y)

        # Get metrics from config
        label_metric_names = tuple(conf.get('label_metrics', ['NMI', 'ARI', 'ACC', 'F1']))
        struct_metric_names = tuple(conf.get('struct_metrics', ['Mod', 'Cond']))

        # Setup for structure metrics
        if labeled_subgraph is not None:
            struct_edge_index = labeled_subgraph['edge_index']
        else:
            struct_edge_index = edge_index

        # Compute metrics
        start_time = time.time()
        label_results = label_metrics(
            eval_y[valid_mask],
            pred[valid_mask],
            metrics=label_metric_names
        )
        struct_results = structure_metrics(
            struct_edge_index,
            pred,
            metrics=struct_metric_names
        )

        metrics_time = time.time() - start_time
        all_metrics_times.append(metrics_time)

        # Merge results
        run_results = {**label_results, **struct_results}
        all_results.append(run_results)

        # Build log string
        metric_strs = []
        for name in label_metric_names + struct_metric_names:
            value = run_results[name] * 100
            metric_strs.append(f"{name}={value:.2f}")

        logger.info(f"Results: {', '.join(metric_strs)}")
        logger.info(f"Metrics computation time: {metrics_time:.2f}s")

    """Aggregate Results Across Runs"""
    logger.info("\n" + "=" * 60)
    logger.info("Aggregating Results Across Runs")
    logger.info("=" * 60)

    # Compute statistics across runs
    all_metric_names = label_metric_names + struct_metric_names
    mean_results = {}
    std_results = {}

    for metric_name in all_metric_names:
        values = [result[metric_name] * 100 for result in all_results]
        mean_results[metric_name] = np.mean(values)
        std_results[metric_name] = np.std(values)

    # Compute average times
    avg_init_time = np.mean(all_init_times)
    std_init_time = np.std(all_init_times)
    avg_finetune_time = np.mean(all_finetune_times)
    std_finetune_time = np.std(all_finetune_times)
    avg_inference_time = np.mean(all_inference_times)
    std_inference_time = np.std(all_inference_times)
    avg_metrics_time = np.mean(all_metrics_times)
    std_metrics_time = np.std(all_metrics_times)

    # Log individual run results
    logger.info(f"\nIndividual Run Results:")
    for run in range(args.runs):
        metric_strs = []
        for name in all_metric_names:
            value = all_results[run][name] * 100
            metric_strs.append(f"{name}={value:.2f}")
        logger.info(f"  Run {run + 1}: {', '.join(metric_strs)}")

    """Final Results Summary"""
    log_final_results_multiple_runs(
        mean_results, std_results,
        pretrain_time, eval_pretrain_time,
        avg_init_time, std_init_time,
        avg_finetune_time, std_finetune_time,
        avg_inference_time, std_inference_time,
        avg_metrics_time, std_metrics_time,
        args.runs, logger
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

    total_time = pretrain_time + avg_finetune_time + avg_inference_time + avg_metrics_time
    logger.info(
        f"Time: Pretrain={pretrain_time:.2f}s, "
        f"Finetune={avg_finetune_time:.2f}±{std_finetune_time:.2f}s, "
        f"Infer={avg_inference_time:.2f}±{std_inference_time:.2f}s, "
        f"Metrics={avg_metrics_time:.2f}±{std_metrics_time:.2f}s, "
        f"Avg Total={total_time:.2f}s"
    )


if __name__ == '__main__':
    main()
