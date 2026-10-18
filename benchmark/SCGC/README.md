# SCGC Benchmark

This directory contains the benchmark implementation for **Simple Contrastive Graph Clustering (SCGC)** (Liu et al., IEEE TNNLS 2023).

## Overview

PyAGC implements SCGC in the same decoupled contrastive setting as `benchmark/NS4GC/` (shared GNN encoder, two `GSSLTransform` views, post-hoc K-Means). Only the representation objective differs: the **neighbor-oriented structural contrastive loss** from Liu et al. (TNNLS 2023):

$$
\mathcal{L} = \frac{1}{N^2} \sum_{i,j} \left( S_{ij} - \hat{A}_{ij} \right)^2,
\quad S = Z^{(1)} (Z^{(2)})^\top,\ \hat{A} = A + I.
$$

Training logs report loss components **`pos`** (edge / self-loop terms) and **`neg`** (non-edge terms).

## Quick Start

```bash
# Run on Cora dataset
python main.py --dataset Cora --device cuda:0 --seed 0

# Run on large dataset with mini-batch training
python main.py --dataset Flickr --device cuda:0 --seed 0
```

## Command Line Arguments

`--seed`, `--device`, `--root`, `--dataset`, `--log_dir`, `--ckpt_dir`, `--load_ckpt`, `--resume`, `--runs` (same CLI as other PyAGC benchmarks).

## Configuration

Model hyperparameters are defined in `train.conf.yaml`.

### SCGC view augmentation

- `p_fm1`, `p_ed1`: Feature masking and edge dropping for view 1
- `p_fm2`, `p_ed2`: Feature masking and edge dropping for view 2

GNN architecture, mini-batch training, K-Means, and metrics use the same YAML layout as `benchmark/NS4GC/train.conf.yaml` (without NS4GC-only loss weights).

## Output

Example training log:

```
Epoch: 001 Loss: 0.4521, POS: 0.1203, NEG: 0.3318
```

Final output includes clustering metrics (mean ± std) and timing breakdown, consistent with other PyAGC benchmarks.
