# K-Means Baseline for Attributed Graph Clustering

This directory contains the implementation and evaluation scripts for the K-Means baseline method on attributed graph clustering tasks.

## Overview

K-Means is a simple yet effective baseline that directly clusters nodes based on their raw feature vectors without utilizing graph structure information. This serves as an important baseline to understand the contribution of graph structure in attributed graph clustering.

## Method Description

K-Means clustering:
1. Takes raw node features as input: $\boldsymbol{X} \in \mathbb{R}^{n \times f}$
2. Applies K-Means algorithm to partition nodes into $K$ clusters
3. No training phase required
4. Can optionally normalize features using L2 normalization

## Quick Start

### Run on a single dataset

```bash
python main.py --dataset Cora --device cuda:0 --seed 0 --runs 5
```

### Run on all datasets

```bash
bash run.sh
```

### Custom configuration

```bash
bash run.sh --seed 42 --device cuda:1 --runs 10
```

## Arguments

- `--seed`: Random seed for reproducibility (default: 0)
- `--device`: Device to use (cuda:0, cuda:1, cpu, etc.) (default: cuda:0)
- `--root`: Root path of dataset (default: ../data)
- `--dataset`: Dataset name (default: Cora)
  - Small: Cora, Photo, Physics
  - Medium: HM, Flickr, ArXiv
  - Large: Reddit, MAG, Pokec, Products, WebTopic
  - Extra Large: Papers100M
- `--log_dir`: Directory to save logs (default: logs)
- `--runs`: Number of evaluation runs for stability (default: 5)

## Configuration

The `train.conf.yaml` file contains dataset-specific configurations:

```yaml
default:
  # Evaluation metrics
  label_metrics: ['NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp']
  struct_metrics: ['Mod', 'Cond']
  
  # Feature preprocessing
  normalize_embeddings: true
  
  # K-Means parameters
  kmeans_backend: torch  # Options: torch, triton, sklearn
  kmeans_n_init: 10
  kmeans_max_iter: 300
```

### Key Parameters

- **normalize_embeddings**: Whether to L2-normalize embeddings before clustering
- **kmeans_backend**: Backend implementation for K-Means
  - `torch`: PyTorch implementation (GPU-accelerated)
  - `triton`: Triton kernel implementation (faster on GPU)
  - `sklearn`: scikit-learn implementation (CPU-only)
- **kmeans_n_init**: Number of random initializations
- **kmeans_max_iter**: Maximum number of iterations

## Output

The evaluation produces:

1. **Log files**: Saved in `logs/{dataset}/kmeans/`
   - Detailed training progress
   - Metric values for each run
   - Timing statistics

2. **Metrics reported**:
   - Label-based: NMI, ARI, ACC, F1, Homogeneity, Completeness
   - Structure-based: Modularity, Conductance

3. **Timing information**:
   - Clustering time
   - Metrics computation time
   - Total time

## Example Output

```
======================================================================
Final Results Summary
======================================================================
Clustering Metrics (mean ± std):
  NMI   :  45.32 ± 0.85
  ARI   :  32.18 ± 1.23
  ACC   :  58.76 ± 0.92
  F1    :  51.23 ± 1.05
  Homo  :  43.89 ± 0.78
  Comp  :  46.92 ± 0.81
  Mod   :  0.421 ± 0.012
  Cond  :  0.385 ± 0.015

Time Statistics:
  Training time:         0.00s (no training)
  Inference time:        0.00s (using raw features)
  Clustering time:       2.34s
  Metrics time:          1.12s
  Total time:            3.46s
======================================================================

Compact Results:
NMI=45.32±0.85, ARI=32.18±1.23, ACC=58.76±0.92, F1=51.23±1.05, Homo=43.89±0.78, Comp=46.92±0.81, Mod=0.42±0.01, Cond=0.39±0.02
Time: Cluster=2.34s, Metrics=1.12s, Total=3.46s
```

## Notes

### Memory Considerations

For very large graphs (e.g., Papers100M):
- K-Means is memory-efficient as it only uses node features
- No need to load the entire graph structure into GPU memory
- Only labeled nodes are used for evaluation to reduce memory footprint

### Backend Selection

- **torch**: Best for small to medium datasets on GPU
- **triton**: Best for large datasets on GPU (requires triton installation)
- **sklearn**: Best for CPU-only environments or very small datasets

### Comparison with Graph Methods

K-Means serves as a baseline to understand:
1. How much performance gain comes from using graph structure
2. Whether node features alone are sufficient for clustering
3. The trade-off between computational cost and performance

Expected results:
- K-Means typically performs worse than graph-based methods
- Performance gap increases on datasets with strong graph structure
- Competitive on datasets with weak graph signals
