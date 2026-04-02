# Lab02 - Experiment Running Guide

This document explains how to train and evaluate the 4 implemented recommenders:
- ConvNCF
- LightGCN
- NGCF
- SimpleX

Supported datasets:
- movielens
- amazon-book

## 1. Environment setup

```bash
pip install torch numpy pandas scipy matplotlib
```

If you do not need loss plots, you can skip matplotlib.

## 2. Important folder structure

```text
src/
|- main.py
|- dataset.py
|- evaluator.py
|- models/
|- logs/
`- results/
```

## 3. General command

```bash
python src/main.py \
  --model <ConvNCF|LightGCN|NGCF|SimpleX|all> \
  --dataset <movielens|amazon-book|all> \
  --device <cpu|cuda> \
  --epochs <num_epochs> \
  --batch_size <batch_size> \
  --emb_dim <embedding_dim> \
  --lr <learning_rate> \
  --decay <l2_weight> \
  --n_layers <num_gcn_layers> \
  --seed <seed>
```

## 4. Main arguments

| Argument | Default | Description |
|---|---:|---|
| --model | all | Run one model or all models |
| --dataset | all | Run one dataset or all datasets |
| --device | cpu | Use cuda if GPU is available |
| --epochs | 50 | Number of training epochs |
| --batch_size | 2048 | Batch size |
| --emb_dim | 64 | Embedding dimension |
| --lr | 1e-3 | Learning rate |
| --decay | 1e-4 | L2 regularization |
| --n_layers | 3 | Used by LightGCN and NGCF |
| --seed | 42 | Random seed |
| --max_eval_users | None | Limit evaluated users for faster testing |
| --eval_only | false | Skip training and evaluate a checkpoint |
| --checkpoint | None | Checkpoint path for eval_only mode |

## 5. Recommended commands

### 5.1 Quick pipeline sanity check

```bash
python src/main.py --model ConvNCF --dataset movielens --epochs 1 --batch_size 512 --max_eval_users 200
```

### 5.2 Run one model on one dataset

```bash
python src/main.py --model LightGCN --dataset movielens --device cpu --epochs 50 --batch_size 2048 --lr 1e-3 --decay 1e-4 --n_layers 3
```

### 5.3 Run all models on one dataset

```bash
python src/main.py --model all --dataset movielens --device cpu --epochs 50 --batch_size 2048
```

### 5.4 Run all models on all datasets

```bash
python src/main.py --model all --dataset all --device cpu --epochs 50 --batch_size 2048
```

### 5.5 Run with GPU (Kaggle/Colab)

```bash
python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

### 5.6 Evaluate from checkpoint only (no training)

```bash
python src/main.py --model LightGCN --dataset movielens --eval_only --checkpoint src/results/checkpoints/LightGCN_movielens_emb64_lr0.001_bs2048_ep50.pt
```

## 6. Suggested batch sizes by resource

| Model | CPU | GPU T4 |
|---|---:|---:|
| ConvNCF | 512 | 2048 |
| LightGCN | 2048 | 4096 |
| NGCF | 1024 | 2048 |
| SimpleX | 512 | 1024 |

If VRAM is not enough, reduce batch_size before reducing epochs.

## 7. Where outputs are saved

Each run creates:
- src/logs/<timestamp>.log
- src/results/<model_dataset_...>.json
- src/results/checkpoints/<model_dataset_...>.pt
- src/results/summary_<timestamp>.json

If matplotlib is installed and training is performed, a loss plot is also saved:
- src/results/<model_dataset_...>_loss.png

## 8. Quick result check

```bash
tail -n 40 src/logs/*.log
```

```bash
python -c "import glob, json; f=sorted(glob.glob('src/results/summary_*.json'))[-1]; print(json.dumps(json.load(open(f)), indent=2))"
```

The second command automatically loads the newest summary file.

## 9. Build a comparison table with Python

```python
import glob
import json
import pandas as pd

files = glob.glob('src/results/*.json')
files = [f for f in files if '/summary_' not in f]
records = [json.load(open(f)) for f in files]

df = pd.json_normalize(records)
cols = [
    'model', 'dataset',
    'metrics.Recall@10', 'metrics.NDCG@10',
    'metrics.Recall@20', 'metrics.NDCG@20',
    'train_time_s', 'eval_time_s', 'gpu_peak_mb'
]
print(df[cols].sort_values(['dataset', 'model']).to_string(index=False))
```

## 10. Run on Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02')

!pip install torch numpy pandas scipy matplotlib
!python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

## 11. Run on Kaggle

1. Set Accelerator to GPU.
2. Install packages:

```bash
pip install torch numpy pandas scipy matplotlib
```

3. Run:

```bash
python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

## 12. Common issues

- CUDA not available:
  - Switch --device to cpu.
- Out of memory:
  - Reduce --batch_size.
- Checkpoint not found in eval_only mode:
  - Verify the checkpoint filename in src/results/checkpoints.

