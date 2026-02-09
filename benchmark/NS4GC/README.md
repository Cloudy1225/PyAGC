# NS4GC Benchmark

This directory contains the benchmark implementation for **Neighbor-aware Structural Similarity Learning for Graph Clustering (NS4GC)**.

## Overview

NS4GC is an unsupervised graph clustering method that learns node representations through:
- **Alignment Loss**: Ensures consistency between different augmented views
- **Neighbor Consistency Loss**: Preserves structural similarity for neighbors
- **Sparsity Loss**: Encourages sparsity of similarity values between non-neighbor pairs

## Quick Start

### Basic Usage

```bash
# Run on Cora dataset
python main.py --dataset Cora --device cuda:0 --seed 0

# Run on large dataset with mini-batch training
python main.py --dataset Flickr --device cuda:0 --seed 0

# Run on Papers100M (requires preprocessing)
python main.py --dataset Papers100M --device cuda:0 --seed 0
```

## Command Line Arguments

- `--seed`: Random seed for reproducibility (default: 0)
- `--device`: Device to use (cuda:0, cuda:1, cpu, etc.)
- `--root`: Root path of dataset (default: ../data)
- `--dataset`: Dataset name (Cora, Photo, Physics, HM, Flickr, ArXiv, Reddit, MAG, Pokec, Products, WebTopic, Papers100M)
- `--log_dir`: Directory to save logs (default: logs)
- `--ckpt_dir`: Directory to save checkpoints (default: ckpts)
- `--load_ckpt`: Load existing checkpoint for inference only
- `--resume`: Resume training from last checkpoint
- `--runs`: Number of evaluation runs for stability (default: 5)

### Examples

```bash
# Resume training from checkpoint
python main.py --dataset Cora --resume

# Load checkpoint and evaluate only (skip training)
python main.py --dataset Cora --load_ckpt

# Run multiple times with different seeds
for seed in 0 1 2 3 4; do
    python main.py --dataset Cora --seed $seed
done

# Run on large dataset with checkpoint recovery
python main.py --dataset Papers100M --device cuda:0 --resume
```

## Configuration

Model hyperparameters are defined in `train.conf.yaml`. The configuration supports:

### General Settings
- `lr`, `wd`: Learning rate and weight decay
- `epochs`, `patience`: Training epochs and early stopping patience
- `hidden_channels`, `num_layers`: Model architecture
- `dropout`, `act`, `norm`: Regularization and activations

### NS4GC-Specific Settings
- `p_fm1`, `p_ed1`: Feature masking and edge dropping probability for view 1
- `p_fm2`, `p_ed2`: Feature masking and edge dropping probability for view 2
- `s`: Similarity threshold for neighbor consistency (default: 0.6)
- `tau`: Temperature parameter for contrastive loss (default: 0.1)
- `lam`: Weight for neighbor consistency loss (default: 1.0)
- `gam`: Weight for sparsity loss (default: 1.0)

### Training Mode
- `mini_batch`: Whether to use mini-batch training (default: false for small graphs)
- `batch_size`, `fan_out`: Mini-batch parameters
- `save_every`, `save_every_batch`: Checkpoint saving frequency

### Evaluation
- `label_metrics`: Metrics for label-based evaluation (NMI, ARI, ACC, F1, Homo, Comp)
- `struct_metrics`: Metrics for structure-based evaluation (Mod, Cond)
- `kmeans_backend`: K-Means implementation (torch, triton, sklearn)
- `normalize_embeddings`: Whether to normalize embeddings before clustering

## Features

### Checkpoint Management
- Automatic checkpoint saving (best and last)
- Resume training from interruption
- Support for intra-epoch checkpoints (Papers100M)
- Load checkpoint for inference only

### Training Modes
- **Full-batch**: For small graphs (Cora, Photo, Physics, HM)
- **Mini-batch**: For large graphs (Flickr, ArXiv, Reddit, MAG, Pokec, Products, WebTopic, Papers100M)
- Automatic mode selection based on dataset

### Evaluation
- Multiple runs for stability (mean ± std)
- Comprehensive metrics (label-based and structure-based)
- Detailed timing statistics
- Support for Papers100M labeled subgraph

## Dataset Support

### Small Datasets (Full-batch)
- Cora (2,708 nodes)
- Photo (7,650 nodes)
- Physics (34,493 nodes)
- HM (94,405 nodes)

### Large Datasets (Mini-batch)
- Flickr (89,250 nodes)
- ArXiv (169,343 nodes)
- Reddit (232,965 nodes)
- MAG (1,134,649 nodes)
- Pokec (1,632,803 nodes)
- Products (2,449,029 nodes)
- WebTopic (4,911,150 nodes)
- Papers100M (111,059,956 nodes)

## Output

The benchmark outputs:
1. Training logs with loss components (ali, nei, spa)
2. Inference timing
3. Clustering metrics (NMI, ARI, ACC, F1, Homo, Comp, Mod, Cond)
4. Mean and standard deviation over multiple runs
5. Detailed timing breakdown

Example output:
```
Epoch: 001 Loss: 2.5430, ALI: 1.8234, NEI: 0.3456, SPA: 0.3740
...
Run 1/5: NMI=65.43, ARI=58.21, ACC=70.12, F1=68.90, Homo=63.45, Comp=67.89, Mod=52.30, Cond=45.67
...
Final Results Summary
====================
Clustering Metrics (mean ± std):
  NMI   :  65.23 ± 0.45
  ARI   :  58.12 ± 0.38
  ...
Time Statistics:
  Training time:      120.45s
  Inference time:       2.34s
  Clustering time:      5.67s
  Metrics time:         1.23s
```

## Tips for Large Datasets

1. **Papers100M**: 
   - First run will preprocess and cache the data
   - Use `save_every_batch: 500` for checkpoint recovery
   - Consider using `infer_fan_out: 10` to limit memory usage

2. **Memory Issues**:
   - Reduce `batch_size` or `infer_batch_size`
   - Limit `infer_fan_out` (use 10 instead of -1)
   - Use gradient accumulation if needed

3. **Training Speed**:
   - Increase `batch_size` for better GPU utilization
   - Use multiple `num_workers` for data loading
   - Consider reducing `num_layers` if applicable
