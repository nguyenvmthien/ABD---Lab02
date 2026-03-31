"""
Dataset loader for recommendation experiments.

Reads preprocessed CSV files (user_id, item_id) and produces:
  - train dict:  {user_id: set(item_ids)}
  - test  dict:  {user_id: set(item_ids)}
  - n_users, n_items
  - (optional) sparse adjacency matrix for graph-based models
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_ROOT = Path(__file__).resolve().parent / "data"

DATASET_PATHS = {
    "movielens": _DATA_ROOT / "movielens" / "preprocessed",
    "amazon-book": _DATA_ROOT / "amazon-book" / "preprocessed",
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class RecDataset:
    """Immutable container returned by ``load_dataset``."""

    name: str
    n_users: int
    n_items: int
    n_train: int
    n_test: int
    train: Dict[int, Set[int]] = field(repr=False)
    test: Dict[int, Set[int]] = field(repr=False)

    # ----- helpers used by graph models -----
    _norm_adj: sp.csr_matrix | None = field(default=None, repr=False)

    @property
    def norm_adj(self) -> sp.csr_matrix:
        """Lazily build the symmetric normalised adj matrix."""
        if self._norm_adj is None:
            self._norm_adj = _build_norm_adj(
                self.n_users, self.n_items, self.train
            )
        return self._norm_adj

    # ---- user interaction history as a list (for SimpleX) ----
    _user_histories: Dict[int, np.ndarray] | None = field(
        default=None, repr=False
    )

    def user_history(self, max_len: int = 50) -> Dict[int, np.ndarray]:
        """Return padded user interaction history arrays."""
        if self._user_histories is None:
            self._user_histories = {}
            for u, items in self.train.items():
                arr = np.array(sorted(items), dtype=np.int64)
                if len(arr) > max_len:
                    arr = arr[-max_len:]  # keep most recent (last)
                pad = np.zeros(max_len, dtype=np.int64)
                pad[: len(arr)] = arr
                self._user_histories[u] = pad
        return self._user_histories


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_dataset(name: str) -> RecDataset:
    """Load a dataset by name (``movielens`` or ``amazon-book``)."""
    if name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from {list(DATASET_PATHS)}"
        )

    data_dir = DATASET_PATHS[name]
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    n_users = max(train_df["user_id"].max(), test_df["user_id"].max()) + 1
    n_items = max(train_df["item_id"].max(), test_df["item_id"].max()) + 1

    # Build dicts -------------------------------------------------------
    train_dict: Dict[int, Set[int]] = {}
    for u, i in zip(train_df["user_id"], train_df["item_id"]):
        train_dict.setdefault(int(u), set()).add(int(i))

    test_dict: Dict[int, Set[int]] = {}
    for u, i in zip(test_df["user_id"], test_df["item_id"]):
        test_dict.setdefault(int(u), set()).add(int(i))

    return RecDataset(
        name=name,
        n_users=int(n_users),
        n_items=int(n_items),
        n_train=len(train_df),
        n_test=len(test_df),
        train=train_dict,
        test=test_dict,
    )


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------
def _build_norm_adj(
    n_users: int, n_items: int, train: Dict[int, Set[int]]
) -> sp.csr_matrix:
    """Build the symmetric normalised adjacency matrix (D^{-1/2} A D^{-1/2}).

    The adjacency matrix is a bipartite graph of shape
    ``(n_users + n_items) x (n_users + n_items)``.
    """
    rows, cols = [], []
    for u, items in train.items():
        for i in items:
            rows.append(u)
            cols.append(n_users + i)  # items offset by n_users
    rows, cols = np.array(rows), np.array(cols)

    # Symmetric entries
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones(len(all_rows), dtype=np.float32)

    n_nodes = n_users + n_items
    adj = sp.coo_matrix((data, (all_rows, all_cols)), shape=(n_nodes, n_nodes))
    adj = adj.tocsr()

    # D^{-1/2} A D^{-1/2}
    degree = np.array(adj.sum(axis=1)).flatten()
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    return norm_adj.tocsr()
