"""
Evaluation metrics for top-K recommendation.

Supports:
  - Recall@K
  - NDCG@K
  - Precision@K
  - Hit Rate@K (HR@K)
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


def recall_at_k(ranked_list: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """Recall@K = |{recommended} ∩ {relevant}| / |{relevant}|."""
    if not ground_truth:
        return 0.0
    top_k = set(ranked_list[:k].tolist())
    return len(top_k & ground_truth) / len(ground_truth)


def precision_at_k(ranked_list: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """Precision@K = |{recommended} ∩ {relevant}| / K."""
    if not ground_truth:
        return 0.0
    top_k = set(ranked_list[:k].tolist())
    return len(top_k & ground_truth) / k


def ndcg_at_k(ranked_list: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain @ K."""
    if not ground_truth:
        return 0.0
    top_k = ranked_list[:k]
    dcg = sum(
        1.0 / np.log2(idx + 2)
        for idx, item in enumerate(top_k)
        if item in ground_truth
    )
    # Ideal DCG
    n_rel = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(ranked_list: np.ndarray, ground_truth: Set[int], k: int) -> float:
    """Hit Rate@K: 1 if at least one relevant item in top-K, else 0."""
    top_k = set(ranked_list[:k].tolist())
    return 1.0 if top_k & ground_truth else 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------
def evaluate_model(
    score_fn,
    test_dict: Dict[int, Set[int]],
    train_dict: Dict[int, Set[int]],
    n_items: int,
    ks: List[int] = (10, 20),
    max_users: int | None = None,
) -> Dict[str, float]:
    """Evaluate a model on the test set.

    Parameters
    ----------
    score_fn : callable(user_id, item_array) -> score_array
        A function that scores candidate items for a given user.
    test_dict : dict
        Ground-truth test interactions.
    train_dict : dict
        Training interactions (to exclude from scoring).
    n_items : int
        Total number of items.
    ks : list of int
        Cut-off values.
    max_users : int or None
        If set, only evaluate on a random subset of test users (for speed).

    Returns
    -------
    dict : metric_name -> value  (averaged over users)
    """
    all_items = np.arange(n_items, dtype=np.int64)
    users = list(test_dict.keys())

    if max_users is not None and max_users < len(users):
        rng = np.random.default_rng(0)
        users = rng.choice(users, size=max_users, replace=False).tolist()

    results = {f"Recall@{k}": 0.0 for k in ks}
    results.update({f"NDCG@{k}": 0.0 for k in ks})
    results.update({f"Precision@{k}": 0.0 for k in ks})
    results.update({f"HR@{k}": 0.0 for k in ks})

    n_eval = 0
    for uid in users:
        gt = test_dict.get(uid, set())
        if not gt:
            continue

        # Items the user has NOT interacted with in training (+ test ground truth)
        train_items = train_dict.get(uid, set())

        # Score all items, mask training items with -inf
        scores = score_fn(uid, all_items)
        scores_copy = scores.copy()
        for ti in train_items:
            if ti < len(scores_copy):
                scores_copy[ti] = -np.inf

        # Rank by score (descending)
        max_k = max(ks)
        top_k_idx = np.argpartition(scores_copy, -max_k)[-max_k:]
        top_k_idx = top_k_idx[np.argsort(scores_copy[top_k_idx])[::-1]]

        for k in ks:
            results[f"Recall@{k}"] += recall_at_k(top_k_idx, gt, k)
            results[f"NDCG@{k}"] += ndcg_at_k(top_k_idx, gt, k)
            results[f"Precision@{k}"] += precision_at_k(top_k_idx, gt, k)
            results[f"HR@{k}"] += hit_at_k(top_k_idx, gt, k)

        n_eval += 1

    if n_eval > 0:
        for key in results:
            results[key] /= n_eval

    results["n_eval_users"] = float(n_eval)
    return results
