# Hướng dẫn các lệnh chạy thực nghiệm

## Cài đặt môi trường

```bash
pip install torch numpy pandas scipy
```

---

## Cấu trúc lệnh chung

```bash
python src/main.py \
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

---

## Tham số dòng lệnh

| Tham số              | Giá trị mặc định | Mô tả                                               |
|---------------------|-------------------|------------------------------------------------------|
| `--model`           | `all`             | Tên model: `ConvNCF`, `LightGCN`, `NGCF`, `SimpleX`, hoặc `all` |
| `--dataset`         | `all`             | Tên dataset: `movielens`, `amazon-book`, hoặc `all`  |
| `--device`          | `cpu`             | Thiết bị: `cpu` hoặc `cuda`                          |
| `--epochs`          | `50`              | Số epoch huấn luyện                                  |
| `--batch_size`      | `2048`            | Kích thước batch                                     |
| `--emb_dim`         | `64`              | Chiều embedding                                      |
| `--lr`              | `1e-3`            | Learning rate                                        |
| `--decay`           | `1e-4`            | Hệ số L2 regularisation                              |
| `--n_layers`        | `3`               | Số layer GCN (LightGCN, NGCF)                        |
| `--seed`            | `42`              | Random seed                                          |
| `--max_eval_users`  | `None`            | Giới hạn số user đánh giá (để chạy nhanh)            |

---

## Các lệnh cụ thể

### 1. Kiểm tra nhanh (sanity check)

Chạy 1 epoch, chỉ đánh giá 50 user để kiểm tra pipeline:

```bash
# ConvNCF trên MovieLens
python src/main.py --model ConvNCF --dataset movielens --epochs 1 --batch_size 512 --max_eval_users 50

# LightGCN trên Amazon-Book
python src/main.py --model LightGCN --dataset amazon-book --epochs 1 --batch_size 2048 --max_eval_users 50
```

### 2. Chạy từng model riêng (CPU)

```bash
# ConvNCF
python src/main.py --model ConvNCF --dataset movielens --epochs 50 --batch_size 512 --lr 0.05

# LightGCN
python src/main.py --model LightGCN --dataset movielens --epochs 100 --batch_size 2048 --lr 1e-3 --decay 1e-4 --n_layers 3

# NGCF
python src/main.py --model NGCF --dataset movielens --epochs 50 --batch_size 1024 --lr 1e-4 --decay 1e-5 --n_layers 3

# SimpleX
python src/main.py --model SimpleX --dataset movielens --epochs 50 --batch_size 512 --lr 1e-3
```

### 3. Chạy tất cả model trên 1 dataset (CPU)

```bash
python src/main.py --model all --dataset movielens --epochs 50 --batch_size 2048
```

### 4. Chạy tất cả model trên tất cả dataset (CPU)

```bash
python src/main.py --model all --dataset all --epochs 50 --batch_size 2048
```

### 5. Chạy trên GPU T4 (Kaggle / Colab)

```bash
# Tất cả model, tất cả dataset, GPU
python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096

# Chỉ LightGCN trên GPU
python src/main.py --model LightGCN --dataset movielens --device cuda --epochs 100 --batch_size 4096 --n_layers 3

# SimpleX trên GPU (giảm batch vì num_negs=50 tốn VRAM)
python src/main.py --model SimpleX --dataset amazon-book --device cuda --epochs 50 --batch_size 1024
```

### 6. Hyperparameter tuning (Grid search)

```bash
# Grid search LightGCN
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

# Grid search NGCF
for lr in 1e-5 1e-4 5e-4; do
  for decay in 1e-6 1e-5 1e-4; do
    python src/main.py \
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
    python src/main.py \
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

---

## Batch size khuyến nghị theo GPU T4

| Model    | CPU       | GPU T4    | Ghi chú                       |
|----------|-----------|-----------|-------------------------------|
| ConvNCF  | 512       | 2048      | Outer product tốn VRAM        |
| LightGCN | 2048      | 4096      | Nhẹ nhất, có thể tăng cao     |
| NGCF     | 1024      | 2048      | Nhiều weight matrices          |
| SimpleX  | 512       | 1024      | num_negs=50 nhân bộ nhớ       |

---

## Kết quả đầu ra

Sau khi chạy, kết quả được lưu tại:

```
src/
├── logs/
│   └── 20260331_150000.log          # Log chi tiết (train loss, metrics, time, memory)
└── results/
    ├── checkpoints/
    │   ├── ConvNCF_movielens.pt      # Model checkpoint
    │   ├── LightGCN_movielens.pt
    │   └── ...
    ├── ConvNCF_movielens.json        # Kết quả từng thí nghiệm
    ├── LightGCN_amazon-book.json
    └── summary_20260331_150000.json  # Tổng hợp tất cả thí nghiệm
```

### Xem nhanh kết quả

```bash
# Xem log
cat src/logs/*.log | tail -30

# Xem kết quả JSON
python -c "import json; print(json.dumps(json.load(open('src/results/LightGCN_movielens.json')), indent=2))"
```

### Tổng hợp kết quả bằng Python

```python
import json, glob, pandas as pd

files = glob.glob("src/results/*.json")
files = [f for f in files if "summary" not in f]
records = [json.load(open(f)) for f in files]
df = pd.json_normalize(records)
print(df[["model", "dataset", "metrics.Recall@10", "metrics.NDCG@10",
          "train_time_s", "gpu_peak_mb"]].to_string(index=False))
```

---

## Setup Google Colab

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Di chuyển đến thư mục dự án
import os
os.chdir('/content/drive/MyDrive/A+SCHOOL-HK11/applied-big-data/Lab/Lab02')

# 3. Cài đặt dependencies
!pip install torch numpy pandas scipy

# 4. Chạy thí nghiệm
!python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```

## Setup Kaggle

```python
# 1. Upload data hoặc thêm dataset
# 2. Chọn Accelerator: GPU T4 x2
# 3. Chạy trong notebook cell:
!pip install torch numpy pandas scipy
!python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096
```
