# Configuration & Hyperparameter Documentation

This document describes the configuration, training parameters, and hyperparameters for all 4 recommendation models implemented in `src/models/`.

---

## 1. ConvNCF (Convolutional Neural Collaborative Filtering)

**Paper**: He et al., "Outer Product-based Neural Collaborative Filtering", IJCAI 2018

### Architecture
- User/item embeddings → outer-product interaction map → stacked 2×2 stride-2 convolutions → linear score

### Hyperparameters (`ConvNCFConfig`)

| Parameter    | Default          | Description                                          |
|-------------|------------------|------------------------------------------------------|
| `emb_dim`   | 64               | Embedding dimensionality                             |
| `channels`  | (32,32,32,32,32,32) | Number of output channels per conv layer           |
| `dropout`   | 0.0              | Dropout rate after conv stack                        |
| `lr_embed`  | 0.05             | Learning rate for embedding parameters (Adagrad)     |
| `lr_net`    | 0.05             | Learning rate for conv/linear parameters (Adagrad)   |
| `reg_embed` | 1e-2             | L2 regularisation weight for embeddings              |
| `reg_net`   | 1e-2             | L2 regularisation weight for conv/linear             |
| `epochs`    | 3                | Number of training epochs                            |
| `batch_size`| 512              | Mini-batch size                                      |
| `device`    | "cpu"            | `"cpu"` or `"cuda"`                                  |

### Loss
BPR (Bayesian Personalised Ranking) with separate L2 penalties for embeddings and network weights.

### Optimizer
Adagrad (separate optimiser for embeddings vs. network).

---

## 2. LightGCN (Light Graph Convolution Network)

**Paper**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation", SIGIR 2020

### Architecture
- User/item embeddings → K rounds of neighbourhood aggregation on the bipartite graph (no feature transform, no activation) → mean pooling over all layers → dot-product scoring

### Hyperparameters (`LightGCNConfig`)

| Parameter    | Default | Description                                       |
|-------------|---------|---------------------------------------------------|
| `emb_dim`   | 64      | Embedding dimensionality                          |
| `n_layers`  | 3       | Number of GCN propagation layers                  |
| `lr`        | 1e-3    | Learning rate (Adam)                              |
| `decay`     | 1e-4    | L2 regularisation weight on initial embeddings    |
| `epochs`    | 50      | Number of training epochs                         |
| `batch_size`| 2048    | Mini-batch size                                   |
| `device`    | "cpu"   | `"cpu"` or `"cuda"`                               |

### Loss
BPR loss + L2 regularisation on the **initial** (0-th layer) embeddings only.

### Optimizer
Adam

---

## 3. NGCF (Neural Graph Collaborative Filtering)

**Paper**: Wang et al., "Neural Graph Collaborative Filtering", SIGIR 2019

### Architecture
- User/item embeddings → K layers of message-passing with **learnable W_gc, W_bi** matrices → element-wise product interaction → LeakyReLU → L2-normalisation → concatenation of all layers → dot-product scoring

### Hyperparameters (`NGCFConfig`)

| Parameter      | Default     | Description                                         |
|---------------|-------------|-----------------------------------------------------|
| `emb_dim`     | 64          | Embedding dimensionality                            |
| `layer_sizes` | (64,64,64)  | Output dimension per GCN layer                      |
| `mess_dropout`| 0.1         | Message dropout rate                                |
| `node_dropout`| 0.0         | Node dropout rate (disabled by default)             |
| `lr`          | 1e-4        | Learning rate (Adam)                                |
| `decay`       | 1e-5        | L2 regularisation weight                            |
| `epochs`      | 50          | Number of training epochs                           |
| `batch_size`  | 1024        | Mini-batch size                                     |
| `device`      | "cpu"       | `"cpu"` or `"cuda"`                                 |

### Loss
BPR loss + L2 regularisation on initial embeddings.

### Optimizer
Adam

---

## 4. SimpleX

**Paper**: Mao et al., "SimpleX: A Simple and Strong Baseline for Collaborative Filtering", CIKM 2021

### Architecture
- User representation = `γ * user_emb + (1 − γ) * aggregated_history`
- Aggregation modes: `mean`, `user_attention`, `self_attention`
- Item representation = L2-normalised item embedding
- Scoring via cosine similarity

### Hyperparameters (`SimpleXConfig`)

| Parameter      | Default | Description                                          |
|---------------|---------|------------------------------------------------------|
| `emb_dim`     | 64      | Embedding dimensionality                             |
| `gamma`       | 0.5     | Weight between user ID embedding and history          |
| `aggregator`  | "mean"  | Aggregation type: `mean` / `user_attention` / `self_attention` |
| `history_len` | 50      | Max number of historical items to use                 |
| `dropout`     | 0.0     | Dropout on user vector                               |
| `num_negs`    | 50      | Negative samples per positive                        |
| `margin`      | 0.9     | CCL margin threshold                                 |
| `neg_weight`  | 150.0   | Weight on negative loss term                         |
| `lr`          | 1e-3    | Learning rate (Adam)                                 |
| `weight_decay`| 1e-6    | Adam weight decay                                    |
| `epochs`      | 50      | Number of training epochs                            |
| `batch_size`  | 512     | Mini-batch size                                      |
| `device`      | "cpu"   | `"cpu"` or `"cuda"`                                  |

### Loss
Cosine Contrastive Loss (CCL):
```
L = ReLU(1 − cos(u, pos)) + (w / n_neg) * Σ ReLU(cos(u, neg_i) − margin)
```

### Optimizer
Adam with weight decay

---

## Training Guide – Chọn Epochs và Chiến lược Huấn luyện

### Số Epochs khuyến nghị theo từng mô hình và dataset

| Model    | MovieLens (1M) | Amazon-Book | Ghi chú                                                        |
|----------|---------------|-------------|----------------------------------------------------------------|
| ConvNCF  | 30–50         | 50–80       | Hội tụ chậm do outer product; cần nhiều epoch hơn              |
| LightGCN | 50–100        | 100–200     | Nhẹ nhất, mỗi epoch nhanh; thường cần nhiều epoch để hội tụ    |
| NGCF     | 30–50         | 50–100      | Nặng hơn LightGCN do weight matrices; hội tụ sớm hơn          |
| SimpleX  | 30–50         | 50–80       | CCL loss ổn định; thường hội tụ sau ~40 epoch                   |

### Cách nhận biết mô hình đã hội tụ
1. **Loss không giảm**: nếu loss trung bình không giảm trong 5–10 epoch liên tiếp → dừng
2. **Recall@20 trên validation**: nếu không cải thiện sau 10 epoch → dừng (early stopping)
3. **Overfitting**: loss giảm nhưng Recall@20 giảm → đã overfit, giảm epochs hoặc tăng regularisation

### Lời khuyên thực tế

- **Chạy nhanh trước**: dùng `--epochs 3 --max_eval_users 200` để kiểm tra pipeline chạy đúng
- **Sau đó chạy đầy đủ**: `--epochs 50` trên GPU T4 với batch lớn
- **Amazon-Book** lớn hơn MovieLens ~3x → cần thêm thời gian và epoch

---

## Hyperparameter Tuning

### Các hyperparameter quan trọng nhất (theo thứ tự ưu tiên)

| Thứ tự | Parameter   | Ảnh hưởng                                    | Phạm vi thử                      |
|--------|-------------|----------------------------------------------|-----------------------------------|
| 1      | `lr`        | Quá lớn → diverge, quá nhỏ → chậm hội tụ    | {1e-4, 5e-4, 1e-3, 5e-3}         |
| 2      | `emb_dim`   | Khả năng biểu diễn vs. overfitting           | {32, 64, 128}                     |
| 3      | `decay`     | Regularisation, chống overfitting             | {1e-6, 1e-5, 1e-4, 1e-3}         |
| 4      | `n_layers`  | Chỉ LightGCN/NGCF – over-smoothing nếu quá lớn | {1, 2, 3, 4}                     |
| 5      | `batch_size` | Ổn định gradient vs. tốc độ                  | {512, 1024, 2048, 4096}          |

### Hyperparameter riêng từng model

**ConvNCF:**
- `channels`: thử (32,32,32,32,32,32) vs. (64,64,64,64)
- `lr_embed` / `lr_net`: thử riêng biệt {0.01, 0.05, 0.1}

**LightGCN:**
- `n_layers`: quan trọng nhất, 2–4 là tối ưu (>4 gây over-smoothing)

**NGCF:**
- `mess_dropout`: {0.0, 0.1, 0.2} – giúp chống overfitting
- `layer_sizes`: thử (64,64,64) vs. (128,64,32)

**SimpleX:**
- `gamma`: {0.3, 0.5, 0.7} – cân bằng user ID vs. history
- `num_negs`: {20, 50, 100} – nhiều neg → tốt hơn nhưng chậm hơn
- `margin`: {0.8, 0.9, 0.95}
- `aggregator`: thử cả 3: `mean`, `user_attention`, `self_attention`

### Grid Search Script

Tạo file `src/tune.py` hoặc chạy trực tiếp bằng bash:

```bash
# Grid search LightGCN trên movielens
for lr in 1e-4 5e-4 1e-3; do
  for decay in 1e-5 1e-4 1e-3; do
    for layers in 2 3 4; do
      python src/main.py \
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
```

Kết quả sẽ được ghi vào `src/results/` và `src/logs/` với timestamp, dễ so sánh.

### So sánh kết quả tuning

Sau khi chạy xong, dùng Python để tổng hợp:

```python
import json, glob, pandas as pd

files = glob.glob("src/results/LightGCN_movielens*.json")
records = [json.load(open(f)) for f in files]
df = pd.DataFrame(records)
print(df[["lr", "emb_dim", "metrics"]].to_string())
```

---

## Running Experiments

### Quick Start (CPU)
```bash
# Single model, single dataset
python src/main.py --model LightGCN --dataset movielens --epochs 5 --batch_size 2048

# All models, all datasets
python src/main.py --model all --dataset all --epochs 50 --batch_size 2048
```

### GPU T4 (Kaggle / Google Colab)

#### Setup on Kaggle
1. Upload the project folder or clone from GitHub
2. Select **GPU T4 x2** in Accelerator settings
3. Install dependencies:
```bash
pip install torch numpy pandas scipy
```

4. Run experiments with GPU:
```bash
python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

#### Setup on Google Colab
1. Go to **Runtime → Change runtime type → T4 GPU**
2. Mount Google Drive or upload data:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Install and run:
```bash
pip install torch numpy pandas scipy
python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

#### GPU T4 Recommended Hyperparameters

Với GPU T4 (16 GB VRAM), có thể tăng `batch_size` để tận dụng GPU:

| Model    | batch_size | emb_dim | epochs | Ghi chú                        |
|----------|-----------|---------|--------|-------------------------------|
| ConvNCF  | 2048      | 64      | 50     | Outer product tốn VRAM nhanh  |
| LightGCN | 4096      | 64      | 100    | Nhẹ nhất, tăng batch thoải mái |
| NGCF     | 2048      | 64      | 50     | Nhiều weight matrices          |
| SimpleX  | 1024      | 64      | 50     | num_negs=50 tốn bộ nhớ        |

#### Monitoring GPU Usage
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"Memory cached:    {torch.cuda.memory_reserved()/1024**2:.1f} MB")
```

### CLI Arguments

| Argument            | Default | Description                           |
|--------------------|---------|---------------------------------------|
| `--model`          | all     | ConvNCF, LightGCN, NGCF, SimpleX, all|
| `--dataset`        | all     | movielens, amazon-book, all           |
| `--device`         | cpu     | cpu or cuda                           |
| `--epochs`         | 50      | Training epochs                       |
| `--batch_size`     | 2048    | Batch size                            |
| `--emb_dim`        | 64      | Embedding dimension                   |
| `--lr`             | 1e-3    | Learning rate                         |
| `--decay`          | 1e-4    | L2 regularisation                     |
| `--n_layers`       | 3       | GCN layers (LightGCN, NGCF)          |
| `--seed`           | 42      | Random seed                           |
| `--max_eval_users` | None    | Limit eval users for speed            |

### Output Structure
```
src/
├── logs/          # Timestamped .log files
└── results/
    ├── checkpoints/                    # Model .pt files
    ├── ConvNCF_movielens.json          # Per-experiment results
    ├── LightGCN_amazon-book.json
    └── summary_20260331_120000.json    # Combined summary
```

### Evaluation Metrics
- **Recall@K**: fraction of relevant items in top-K
- **NDCG@K**: normalised discounted cumulative gain
- **Precision@K**: fraction of top-K that are relevant
- **HR@K**: hit rate (binary: ≥1 relevant in top-K)

Default K values: **10, 20**
