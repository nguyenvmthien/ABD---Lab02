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
python main.py \
  --model <MODEL> \
  --dataset <DATASET> \
  --device <DEVICE> \
  --epochs <N> \
  --batch_size <N> \
  --emb_dim <N> \
  --lr <FLOAT> \
  --decay <FLOAT> \
  --n_layers <N> \
  --seed <N> \
  --max_eval_users <N>
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
# ConvNCF trên MovieLens
python main.py --model ConvNCF --dataset movielens --epochs 1 --batch_size 512 --max_eval_users 50

# LightGCN trên Amazon-Book
python main.py --model LightGCN --dataset amazon-book --epochs 1 --batch_size 2048 --max_eval_users 50
```

### 5.2 Run one model on one dataset

```bash
# ConvNCF
python main.py --model ConvNCF --dataset movielens --epochs 50 --batch_size 512 --lr 0.05

# LightGCN
python main.py --model LightGCN --dataset movielens --epochs 100 --batch_size 2048 --lr 1e-3 --decay 1e-4 --n_layers 3

# NGCF
python main.py --model NGCF --dataset movielens --epochs 50 --batch_size 1024 --lr 1e-4 --decay 1e-5 --n_layers 3

# SimpleX
python main.py --model SimpleX --dataset movielens --epochs 50 --batch_size 512 --lr 1e-3
```

### 5.3 Run all models on one dataset

```bash
python main.py --model all --dataset movielens --epochs 50 --batch_size 2048
```

### 5.4 Run all models on all datasets

```bash
python main.py --model all --dataset all --epochs 50 --batch_size 2048
```

### 5.5 Run with GPU (Kaggle/Colab)

```bash
# Tất cả model, tất cả dataset, GPU
python main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096

# Chỉ LightGCN trên GPU
python main.py --model LightGCN --dataset movielens --device cuda --epochs 100 --batch_size 4096 --n_layers 3

# SimpleX trên GPU (giảm batch vì num_negs=50 tốn VRAM)
python main.py --model SimpleX --dataset amazon-book --device cuda --epochs 50 --batch_size 1024
```

### 5.6 Evaluate from checkpoint only (no training)

```bash
# Grid search LightGCN
for lr in 1e-4 5e-4 1e-3; do
  for decay in 1e-5 1e-4 1e-3; do
    for layers in 2 3 4; do
      python main.py \
        --model LightGCN \
        --dataset movielens \
        --device cuda \
        --lr $lr \
        --decay $decay \
        --n_layers $layers \
        --epochs 50 \
        --batch_size 4096
    done
  done
done

# Grid search NGCF
for lr in 1e-5 1e-4 5e-4; do
  for decay in 1e-6 1e-5 1e-4; do
    python main.py \
      --model NGCF \
      --dataset movielens \
      --device cuda \
      --lr $lr \
      --decay $decay \
      --epochs 50 \
      --batch_size 2048
  done
done

# Grid search SimpleX
for lr in 5e-4 1e-3 5e-3; do
  for emb in 32 64 128; do
    python main.py \
      --model SimpleX \
      --dataset movielens \
      --device cuda \
      --lr $lr \
      --emb_dim $emb \
      --epochs 50 \
      --batch_size 1024
  done
done
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
# Xem log
cat logs/*.log | tail -30

# Xem kết quả JSON
python -c "import json; print(json.dumps(json.load(open('results/LightGCN_movielens.json')), indent=2))"
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

files = glob.glob("results/*.json")
files = [f for f in files if "summary" not in f]
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

# 3. Cài đặt dependencies
!pip install torch numpy pandas scipy

# 4. Chạy thí nghiệm
!python main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

## 11. Run on Kaggle

```python
# 1. Upload data hoặc thêm dataset
# 2. Chọn Accelerator: GPU T4 x2
# 3. Chạy trong notebook cell:
!pip install torch numpy pandas scipy
!python main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
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

