"""
LightGCN – Light Graph Convolution Network for Collaborative Filtering.

Reference:
    He, X. et al. "LightGCN: Simplifying and Powering Graph Convolution
    Network for Recommendation." SIGIR 2020.

Key idea: simplify GCN by removing feature transformation and nonlinear
activation, keeping only neighbourhood aggregation.  The final embedding
is the mean of all layer embeddings (including the 0-th / initial one).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Network
# =====================================================================
class LightGCNNet(nn.Module):
    """Pure LightGCN backbone.

    Parameters
    ----------
    n_users, n_items : int
        Number of users / items.
    emb_dim : int
        Embedding dimensionality.
    n_layers : int
        Number of graph convolution layers.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

        # Adjacency will be registered as a sparse buffer later
        self._adj: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def set_adj(self, norm_adj: sp.spmatrix, device: torch.device) -> None:
        """Convert a scipy sparse matrix to a torch sparse tensor and store."""
        coo = norm_adj.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        self._adj = torch.sparse_coo_tensor(indices, values, shape).to(device)

    # ------------------------------------------------------------------
    def forward_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute all user/item embeddings via LightGCN propagation."""
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embs = [ego]

        x = ego
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self._adj, x)
            all_embs.append(x)

        # Mean pooling over layers
        all_embs = torch.stack(all_embs, dim=1)
        out = all_embs.mean(dim=1)

        users_emb, items_emb = torch.split(
            out, [self.n_users, self.n_items], dim=0
        )
        return users_emb, items_emb

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return embeddings for BPR loss computation."""
        all_users, all_items = self.forward_all()

        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        # Initial embeddings for regularisation
        u_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        return u_emb, pos_emb, neg_emb, u_emb_0, pos_emb_0, neg_emb_0


# =====================================================================
# Config
# =====================================================================
@dataclass
class LightGCNConfig:
    emb_dim: int = 64
    n_layers: int = 3
    lr: float = 1e-3
    decay: float = 1e-4          # L2 regularisation weight
    epochs: int = 50
    batch_size: int = 2048
    device: str = "cpu"
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4


# =====================================================================
# Trainer
# =====================================================================
class LightGCNTrainer:
    """BPR trainer for LightGCN.

    Expected train format: ``train[user] = set(item_ids)``
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        norm_adj: sp.spmatrix,
        cfg: LightGCNConfig | None = None,
    ) -> None:
        self.cfg = cfg or LightGCNConfig()
        self.n_users = n_users
        self.n_items = n_items
        self.device = torch.device(self.cfg.device)

        self.model = LightGCNNet(
            n_users=n_users,
            n_items=n_items,
            emb_dim=self.cfg.emb_dim,
            n_layers=self.cfg.n_layers,
        ).to(self.device)

        self.model.set_adj(norm_adj, self.device)

    # ------------------------------------------------------------------
    def _sample_negative(
        self, user: int, train: Dict[int, Set[int]], rng: np.random.Generator
    ) -> int | None:
        seen = train.get(user, set())
        if len(seen) >= self.n_items:
            return None
        for _ in range(30):
            neg = int(rng.integers(0, self.n_items))
            if neg not in seen:
                return neg
        for neg in range(self.n_items):
            if neg not in seen:
                return neg
        return None

    def _build_triplets(
        self, train: Dict[int, Set[int]], rng: np.random.Generator
    ) -> Tuple[List[int], List[int], List[int]]:
        users, pos_items, neg_items = [], [], []
        interactions = [(u, i) for u, items in train.items() for i in items]
        rng.shuffle(interactions)
        for u, i in interactions:
            j = self._sample_negative(u, train, rng)
            if j is None:
                continue
            users.append(u)
            pos_items.append(i)
            neg_items.append(j)
        return users, pos_items, neg_items

    # ------------------------------------------------------------------
    def fit(self, train: Dict[int, Set[int]], seed: int = 42) -> List[float]:
        rng = np.random.default_rng(seed)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr
        )
        self.model.train()
        best_loss = float("inf")
        no_improve = 0
        loss_history: List[float] = []

        for epoch in range(self.cfg.epochs):
            users, pos_items, neg_items = self._build_triplets(train, rng)
            if not users:
                print(f"[LightGCN] Epoch {epoch+1}/{self.cfg.epochs}: no triplets")
                continue

            epoch_loss = 0.0
            n_batch = 0
            for start in range(0, len(users), self.cfg.batch_size):
                end = start + self.cfg.batch_size
                b_u = torch.tensor(users[start:end], dtype=torch.long, device=self.device)
                b_p = torch.tensor(pos_items[start:end], dtype=torch.long, device=self.device)
                b_n = torch.tensor(neg_items[start:end], dtype=torch.long, device=self.device)

                u_emb, pos_emb, neg_emb, u0, p0, n0 = self.model(b_u, b_p, b_n)

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)
                bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

                reg_loss = (
                    u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()
                ) / b_u.shape[0]

                loss = bpr_loss + self.cfg.decay * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            avg_loss = epoch_loss / max(n_batch, 1)
            loss_history.append(avg_loss)
            print(
                f"[LightGCN] Epoch {epoch+1}/{self.cfg.epochs}: "
                f"{len(users)} triplets, avg_loss={avg_loss:.6f}"
            )

            # Early stopping
            if avg_loss < best_loss - self.cfg.min_delta:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    print(
                        f"[LightGCN] Early stopping at epoch {epoch+1} "
                        f"(no improvement for {self.cfg.patience} epochs, best_loss={best_loss:.6f})"
                    )
                    break

        return loss_history

    # ------------------------------------------------------------------
    def score_items(self, user: int, items: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            all_users, all_items = self.model.forward_all()
            u_vec = all_users[user]
            scores = (u_vec.unsqueeze(0) @ all_items.T).squeeze(0)
            return scores.cpu().numpy().astype(np.float64)[items]

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.cfg.__dict__,
                "n_users": self.n_users,
                "n_items": self.n_items,
            },
            p,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
