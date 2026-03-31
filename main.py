"""
main.py – Unified experiment runner for 4 recommendation models.

Usage examples
--------------
# Run one model on one dataset (CPU):
  python src/main.py --model LightGCN --dataset movielens --epochs 5 --batch_size 2048

# Run all models on all datasets (GPU T4 on Colab/Kaggle):
  python src/main.py --model all --dataset all --device cuda --epochs 50 --batch_size 4096

# Quick sanity check (1 epoch, small eval set):
  python src/main.py --model ConvNCF --dataset movielens --epochs 1 --batch_size 512 --max_eval_users 200

# Evaluate only from checkpoint (no training):
  python src/main.py --model LightGCN --dataset movielens --eval_only --batch_size 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so relative imports work
# when invoked as  `python src/main.py`
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataset import load_dataset  # noqa: E402
from evaluator import evaluate_model  # noqa: E402

# Model imports
from models.ConvNCF import ConvNCFConfig, ConvNCFTrainer  # noqa: E402
from models.LightGCN import LightGCNConfig, LightGCNTrainer  # noqa: E402
from models.NGCF import NGCFConfig, NGCFTrainer  # noqa: E402
from models.SimpleX import SimpleXConfig, SimpleXTrainer  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_MODELS = ["ConvNCF", "LightGCN", "NGCF", "SimpleX"]
ALL_DATASETS = ["movielens", "amazon-book"]
TOP_KS = [10, 20]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path, experiment_tag: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_tag}.log"

    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def create_trainer(model_name: str, dataset, device: str, args):
    """Instantiate the appropriate Trainer for *model_name*."""
    if model_name == "ConvNCF":
        cfg = ConvNCFConfig(
            emb_dim=args.emb_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        return ConvNCFTrainer(dataset.n_users, dataset.n_items, cfg)

    elif model_name == "LightGCN":
        cfg = LightGCNConfig(
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            lr=args.lr,
            decay=args.decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        return LightGCNTrainer(
            dataset.n_users, dataset.n_items, dataset.norm_adj, cfg
        )

    elif model_name == "NGCF":
        layer_sizes = tuple([args.emb_dim] * args.n_layers)
        cfg = NGCFConfig(
            emb_dim=args.emb_dim,
            layer_sizes=layer_sizes,
            lr=args.lr,
            decay=args.decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        return NGCFTrainer(
            dataset.n_users, dataset.n_items, dataset.norm_adj, cfg
        )

    elif model_name == "SimpleX":
        cfg = SimpleXConfig(
            emb_dim=args.emb_dim,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        return SimpleXTrainer(dataset.n_users, dataset.n_items, cfg)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def run_experiment(
    model_name: str,
    dataset_name: str,
    device: str,
    args,
    logger: logging.Logger,
    results_dir: Path,
):
    logger.info("=" * 70)
    logger.info(f"MODEL={model_name}  DATASET={dataset_name}  DEVICE={device}")
    logger.info("=" * 70)

    # ---- Load data --------------------------------------------------------
    t0 = time.time()
    ds = load_dataset(dataset_name)
    load_time = time.time() - t0
    logger.info(
        f"Dataset loaded in {load_time:.2f}s  "
        f"(users={ds.n_users}, items={ds.n_items}, "
        f"train={ds.n_train}, test={ds.n_test})"
    )

    # ---- Memory tracking --------------------------------------------------
    tracemalloc.start()
    gpu_mem_before = 0
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated()

    # ---- Create trainer ---------------------------------------------------
    trainer = create_trainer(model_name, ds, device, args)

    # ---- Load checkpoint or train -----------------------------------------
    train_time = 0.0
    ckpt_dir = results_dir / "checkpoints"
    ckpt_tag = (
        f"{model_name}_{dataset_name}"
        f"_emb{args.emb_dim}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    )
    ckpt_path = ckpt_dir / f"{ckpt_tag}.pt"

    if args.eval_only:
        # Load from checkpoint
        if args.checkpoint:
            ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            logger.error(
                f"Checkpoint not found: {ckpt_path}. "
                f"Run training first (without --eval_only)."
            )
            return None
        trainer.load_checkpoint(ckpt_path)
        logger.info(f"Loaded checkpoint from {ckpt_path} (skipping training)")
        loss_history = []
    else:
        t_train_start = time.time()
        loss_history = trainer.fit(ds.train, seed=args.seed)
        train_time = time.time() - t_train_start
        logger.info(f"Training finished in {train_time:.2f}s ({len(loss_history)} epochs)")

    # ---- Evaluate ---------------------------------------------------------
    t_eval_start = time.time()
    metrics = evaluate_model(
        score_fn=trainer.score_items,
        test_dict=ds.test,
        train_dict=ds.train,
        n_items=ds.n_items,
        ks=TOP_KS,
        max_users=args.max_eval_users,
    )
    eval_time = time.time() - t_eval_start
    logger.info(f"Evaluation finished in {eval_time:.2f}s")

    # ---- Memory stats -----------------------------------------------------
    cpu_current, cpu_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_peak = 0
    if device == "cuda" and torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated()

    # ---- Log results ------------------------------------------------------
    logger.info("--- Results ---")
    for k, v in metrics.items():
        if k != "n_eval_users":
            logger.info(f"  {k}: {v:.6f}")
    logger.info(f"  Evaluated users: {int(metrics['n_eval_users'])}")
    logger.info(f"  Train time     : {train_time:.2f}s")
    logger.info(f"  Eval  time     : {eval_time:.2f}s")
    logger.info(f"  CPU mem peak   : {cpu_peak / 1024**2:.1f} MB")
    if gpu_peak > 0:
        logger.info(f"  GPU mem peak   : {gpu_peak / 1024**2:.1f} MB")

    # ---- Save JSON --------------------------------------------------------
    record = {
        "model": model_name,
        "dataset": dataset_name,
        "device": device,
        "epochs": args.epochs,
        "emb_dim": args.emb_dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "train_time_s": round(train_time, 2),
        "eval_time_s": round(eval_time, 2),
        "cpu_peak_mb": round(cpu_peak / 1024**2, 1),
        "gpu_peak_mb": round(gpu_peak / 1024**2, 1),
        "loss_history": [round(l, 6) for l in loss_history],
        "actual_epochs": len(loss_history),
        "timestamp": datetime.now().isoformat(),
    }

    json_path = results_dir / f"{ckpt_tag}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # ---- Plot loss curve ---------------------------------------------------
    if loss_history and HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = list(range(1, len(loss_history) + 1))
        ax.plot(epochs_range, loss_history, marker="o", markersize=3, linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Avg Loss", fontsize=12)
        ax.set_title(f"{model_name} on {dataset_name} — Training Loss", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = results_dir / f"{ckpt_tag}_loss.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Loss plot saved to {plot_path}")

    # ---- Save checkpoint (only if we trained) ------------------------------
    if not args.eval_only:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_save_path = ckpt_dir / f"{ckpt_tag}.pt"
        trainer.save_checkpoint(ckpt_save_path)
        logger.info(f"Checkpoint saved to {ckpt_save_path}")

    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run recommendation experiments"
    )
    p.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name (ConvNCF|LightGCN|NGCF|SimpleX|all)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset name (movielens|amazon-book|all)",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decay", type=float, default=1e-4)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_eval_users",
        type=int,
        default=None,
        help="Limit eval to N random test users (for speed)",
    )
    p.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
        help="Skip training, load checkpoint and evaluate only",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (used with --eval_only)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Resolve device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARN: CUDA requested but not available, falling back to CPU")
        device = "cpu"

    # Determine which models / datasets to run
    models = ALL_MODELS if args.model == "all" else [args.model]
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    # Directories
    results_dir = _SCRIPT_DIR / "results"
    log_dir = _SCRIPT_DIR / "logs"
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(log_dir, tag)

    logger.info(f"Experiment tag : {tag}")
    logger.info(f"Models         : {models}")
    logger.info(f"Datasets       : {datasets}")
    logger.info(f"Device         : {device}")
    logger.info(f"Epochs         : {args.epochs}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info(f"Embedding dim  : {args.emb_dim}")
    logger.info(f"Learning rate  : {args.lr}")
    logger.info(f"Seed           : {args.seed}")

    if device == "cuda" and torch.cuda.is_available():
        logger.info(f"GPU name       : {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU memory     : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    all_records = []
    for ds_name in datasets:
        for m_name in models:
            try:
                rec = run_experiment(
                    m_name, ds_name, device, args, logger, results_dir
                )
                all_records.append(rec)
            except Exception as e:
                logger.error(f"FAILED: {m_name} on {ds_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # ---- Summary table ----------------------------------------------------
    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY")
    logger.info("=" * 90)
    header = f"{'Model':<12} {'Dataset':<14} " + " ".join(
        [f"{'R@'+str(k):<10} {'N@'+str(k):<10}" for k in TOP_KS]
    ) + f" {'Train(s)':<10} {'Eval(s)':<10} {'CPU(MB)':<10} {'GPU(MB)':<10}"
    logger.info(header)
    logger.info("-" * 90)

    for r in all_records:
        m = r["metrics"]
        row = f"{r['model']:<12} {r['dataset']:<14} "
        for k in TOP_KS:
            row += f"{m.get(f'Recall@{k}', 0):<10.4f} {m.get(f'NDCG@{k}', 0):<10.4f} "
        row += (
            f"{r['train_time_s']:<10.1f} "
            f"{r['eval_time_s']:<10.1f} "
            f"{r['cpu_peak_mb']:<10.1f} "
            f"{r['gpu_peak_mb']:<10.1f}"
        )
        logger.info(row)

    # Save summary JSON
    summary_path = results_dir / f"summary_{tag}.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()
