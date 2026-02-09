# DAEGC Benchmark

This directory contains the PyAGC implementation of **DAEGC** (Deep Attentional Embedded Graph Clustering), proposed in the paper:

> **Attributed Graph Clustering: A Deep Attentional Embedding Approach**  
> Chun Wang, Shirui Pan, Ruiqi Hu, Guodong Long, Jing Jiang, Chengqi Zhang  
> *IJCAI 2019*  
> [[Paper]](https://arxiv.org/abs/1906.06532)

## Overview

DAEGC is a two-stage deep clustering method that jointly optimizes graph embedding and clustering objectives:

### Stage 1: Pretraining (Graph Attentional Autoencoder)
- Learns meaningful node representations using a GAT-based encoder
- Reconstructs graph structure via inner product decoder
- Loss function: Binary cross-entropy reconstruction loss

### Stage 2: Finetuning (Joint Optimization)
- Initializes cluster centers using K-Means on pretrained embeddings
- Jointly optimizes reconstruction and clustering objectives
- Uses self-training with confident assignments as soft labels
- Loss function: `L = L_recon + γ * L_cluster` where `L_cluster = KL(P||Q)`

## Key Features

- **Two-stage training**: Pretrain → Initialize centers → Finetune
- **Attention mechanism**: Uses GAT encoder for learning node representations
- **Self-training**: Iteratively refines clustering using target distribution
- **Flexible evaluation**: Optional K-Means evaluation after pretraining
- **Scalable**: Supports both full-batch and mini-batch training modes

## Requirements

```bash
pip install torch torch_geometric
pip install pyagc  # or install from source
```

## Quick Start

### Basic Usage

```bash
# Train on Cora dataset
python main.py --dataset Cora --device cuda:0 --seed 0

# Train on large dataset with mini-batch mode
python main.py --dataset ArXiv --device cuda:0 --seed 0

# Load existing checkpoint and evaluate
python main.py --dataset Cora --load_ckpt

# Resume training from checkpoint
python main.py --dataset Cora --resume
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seed` | int | 0 | Random seed for reproducibility |
| `--device` | str | cuda:0 | Device to use (cuda:0, cuda:1, cpu, etc.) |
| `--root` | str | ../data | Root directory for datasets |
| `--dataset` | str | Cora | Dataset name (see supported datasets below) |
| `--log_dir` | str | logs | Directory to save logs |
| `--ckpt_dir` | str | ckpts | Directory to save checkpoints |
| `--load_ckpt` | flag | False | Load checkpoint for inference only |
| `--resume` | flag | False | Resume training from last checkpoint |
| `--runs` | int | 5 | Number of evaluation runs |

## Configuration

All hyperparameters can be configured via `train.conf.yaml`. The configuration file follows a hierarchical structure with dataset-specific overrides.

### Key Configuration Parameters

#### Training Hyperparameters

```yaml
# Pretraining stage
pretrain_lr: 0.01           # Learning rate for pretraining
pretrain_epochs: 200        # Number of pretraining epochs

# Finetuning stage
finetune_lr: 0.001          # Learning rate for finetuning
finetune_epochs: 200        # Number of finetuning epochs

# Common parameters
wd: 0.0                     # Weight decay
patience: 50                # Early stopping patience
```

#### Model Architecture

```yaml
hidden_channels: 256        # Hidden dimension size
num_layers: 2               # Number of GAT layers
dropout: 0.0                # Dropout probability
act: elu                    # Activation function
heads: 1                    # Number of attention heads (GAT)
```

#### DAEGC-Specific Parameters

```yaml
gamma: 10.0                 # Weight for clustering loss
update_interval: 5          # Update target distribution every N iterations
```

#### Training Mode

```yaml
mini_batch: false           # Use mini-batch training for large graphs
batch_size: 1024            # Batch size for mini-batch training
fan_out: 10                 # Neighborhood sampling size
num_workers: 8              # Number of data loading workers
```

#### Inference Configuration

```yaml
force_full_batch_inference: false   # Try full-batch inference first
allow_cpu_fallback: true            # Fallback to CPU if GPU OOM
infer_batch_size: 65536             # Batch size for inference
infer_fan_out: -1                   # -1 = sample all neighbors
```

#### Initialization Strategy

```yaml
init_use_train_only: false          # Use only training nodes for initialization
kmeans_use_train_split: false       # Use train-test split for pretrain evaluation
```

#### Evaluation Settings

```yaml
eval_pretrain: true                 # Evaluate clustering after pretraining
label_metrics: ['NMI', 'ARI', 'ACC', 'F1', 'Homo', 'Comp']
struct_metrics: ['Mod', 'Cond']
```

#### Checkpoint Settings

```yaml
pretrain_save_every: null           # Save checkpoint every N epochs
pretrain_save_every_batch: null     # Save checkpoint every N batches
finetune_save_every: null
finetune_save_every_batch: null
```

### Dataset-Specific Configuration Example

```yaml
Cora:
  mini_batch: false
  pretrain_lr: 0.01
  finetune_lr: 0.001
  pretrain_epochs: 200
  finetune_epochs: 200
  hidden_channels: 256
  num_layers: 2
  gamma: 10.0
  eval_pretrain: true
  init_use_train_only: false
  kmeans_use_train_split: false

Reddit:
  mini_batch: true
  pretrain_lr: 0.001
  finetune_lr: 0.0001
  pretrain_epochs: 10
  finetune_epochs: 10
  batch_size: 10240
  hidden_channels: 256
  gamma: 10.0
  eval_pretrain: true
  init_use_train_only: true        # Use training nodes only for large graphs
  kmeans_use_train_split: true
```

## Training Pipeline

### 1. Pretraining Stage

```python
# Stage 1: Learn node embeddings via reconstruction
for epoch in range(pretrain_epochs):
    z = encoder(x, edge_index)
    loss_recon = reconstruction_loss(z, edge_index)
    loss_recon.backward()
    optimizer.step()
```

**Output**: Pretrained node embeddings that capture graph structure

### 2. Cluster Initialization

```python
# Initialize cluster centers using K-Means
z = encoder(x, edge_index)
z = normalize(z)
kmeans = KMeans(n_clusters=K)
kmeans.fit(z)
cluster_centers = kmeans.cluster_centers_
```

### 3. Finetuning Stage

```python
# Stage 2: Joint optimization of reconstruction and clustering
for epoch in range(finetune_epochs):
    z = encoder(x, edge_index)
    loss_recon = reconstruction_loss(z, edge_index)
    loss_cluster = clustering_loss(z, cluster_centers)
    loss = loss_recon + gamma * loss_cluster
    loss.backward()
    optimizer.step()
    
    # Update target distribution every N iterations
    if iteration % update_interval == 0:
        update_target_distribution(z)
```

**Output**: Final cluster assignments directly from the model

## Evaluation Metrics

### Label-based Metrics
- **NMI** (Normalized Mutual Information): Measures mutual information between predicted and ground-truth clusters
- **ARI** (Adjusted Rand Index): Measures similarity between two clusterings
- **ACC** (Clustering Accuracy): Measures accuracy with optimal label assignment
- **F1** (F1 Score): Harmonic mean of precision and recall
- **Homo** (Homogeneity): Measures if clusters contain only members of a single class
- **Comp** (Completeness): Measures if all members of a class are in the same cluster

### Structure-based Metrics
- **Mod** (Modularity): Measures the strength of community structure
- **Cond** (Conductance): Measures the quality of cluster boundaries (lower is better)

## Output Format

### Training Log Example

```
============================================================
Configuration
============================================================
  dataset: Cora
  seed: 0
  device: cuda:0
  pretrain_epochs: 200
  finetune_epochs: 200
  ...

============================================================
System Information
============================================================
Dataset: Data(x=[2708, 1433], edge_index=[2, 10556])
Nodes: 2,708
Edges: 10,556
Features: 1,433
Training mode: Full-batch
Model: DAEGC(n_clusters=7, gamma=10.0)
Total parameters: 2.145M
Device: cuda:0

============================================================
Training Stage: PRETRAIN
============================================================
Epoch: 001 Loss: 0.7234, Recon: 0.7234
...
Epoch: 200 Loss: 0.3124, Recon: 0.3124
✓ Best pretrain model saved at epoch 178 with loss 0.3089

============================================================
Pretraining Statistics
============================================================
Total epochs: 200
Total pretraining time: 45.23s
Average epoch time: 226.15ms
Peak GPU memory reserved: 1234.5 MB

============================================================
Evaluating Pretrained Embeddings
============================================================
Method: K-Means clustering on learned embeddings
Using full clustering evaluation (from config):
Running K-Means on all 2708 nodes
Results: NMI=45.23, ARI=32.45, ACC=56.78, F1=48.90, Homo=43.21, Comp=47.65, Mod=34.56, Cond=12.34
Pretrain evaluation completed in 2.34s

============================================================
Initializing Cluster Centers
============================================================
Method: K-Means on pretrained embeddings
Using all-node initialization (from config)
Initializing with all 2708 nodes
✓ Cluster centers initialized: shape=torch.Size([7, 256])

============================================================
Training Stage: FINETUNE
============================================================
Epoch: 001 Loss: 0.4567, Recon: 0.3124, Cluster: 0.1443
...
Epoch: 200 Loss: 0.2345, Recon: 0.2012, Cluster: 0.0333
✓ Best finetune model saved at epoch 187 with loss 0.2289

============================================================
Finetuning Statistics
============================================================
Total epochs: 200
Total finetuning time: 52.67s
Average epoch time: 263.35ms
Peak GPU memory reserved: 1345.2 MB

============================================================
Final Inference
============================================================
Note: DAEGC directly outputs cluster assignments from the model
✓ Inference completed in 0.45s
Prediction shape: torch.Size([2708])

============================================================
Clustering Evaluation
============================================================
Note: DAEGC produces deterministic cluster assignments after training.
Multiple runs will yield identical results.

Run 1/5: NMI=52.34, ARI=41.23, ACC=67.89, F1=62.45, Homo=50.12, Comp=54.78, Mod=42.31, Cond=9.87
Run 2/5: NMI=52.34, ARI=41.23, ACC=67.89, F1=62.45, Homo=50.12, Comp=54.78, Mod=42.31, Cond=9.87
Run 3/5: NMI=52.34, ARI=41.23, ACC=67.89, F1=62.45, Homo=50.12, Comp=54.78, Mod=42.31, Cond=9.87
Run 4/5: NMI=52.34, ARI=41.23, ACC=67.89, F1=62.45, Homo=50.12, Comp=54.78, Mod=42.31, Cond=9.87
Run 5/5: NMI=52.34, ARI=41.23, ACC=67.89, F1=62.45, Homo=50.12, Comp=54.78, Mod=42.31, Cond=9.87

============================================================
Clustering Timing Summary
============================================================
Metrics computation time: 0.12s
Total time (5 runs): 0.12s

============================================================
Final Results Summary
============================================================
Clustering Metrics (mean ± std):
  NMI   :  52.34 ± 0.00
  ARI   :  41.23 ± 0.00
  ACC   :  67.89 ± 0.00
  F1    :  62.45 ± 0.00
  Homo  :  50.12 ± 0.00
  Comp  :  54.78 ± 0.00
  Mod   :  42.31 ± 0.00
  Cond  :   9.87 ± 0.00
Time Statistics:
  Pretrain time:            45.23s
  Pretrain eval time:        2.34s
  Finetune time:            52.67s
  Inference time:            0.45s
  Metrics time:              0.12s
  Total train time:         97.90s
  Total time:               98.47s
============================================================

Compact Results:
NMI=52.34±0.00, ARI=41.23±0.00, ACC=67.89±0.00, F1=62.45±0.00, Homo=50.12±0.00, Comp=54.78±0.00, Mod=42.31±0.00, Cond=9.87±0.00
Time: Pretrain=45.23s, Finetune=52.67s, Infer=0.45s, Metrics=0.12s, Total=98.47s
```


