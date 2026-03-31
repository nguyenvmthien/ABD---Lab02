"""
NGCF – Neural Graph Collaborative Filtering.

Reference:
    Wang, X. et al. "Neural Graph Collaborative Filtering."
    SIGIR 2019.

Key idea: message-passing on the user–item bipartite graph with learnable
weight matrices for both the aggregated neighbour signal and the
element-wise product interaction signal, followed by L2-normalisation.
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
class NGCFNet(nn.Module):
    """NGCF backbone with learnable W_gc and W_bi per layer."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        layer_sizes: Tuple[int, ...] = (64, 64, 64),
        mess_dropout: float = 0.1,
        node_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = len(layer_sizes)
        self.mess_dropout_rate = mess_dropout
        self.node_dropout_rate = node_dropout

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Build per-layer weight matrices
        sizes = [emb_dim] + list(layer_sizes)
        self.W_gc = nn.ModuleList()
        self.b_gc = nn.ParameterList()
        self.W_bi = nn.ModuleList()
        self.b_bi = nn.ParameterList()

        for k in range(self.n_layers):
            self.W_gc.append(nn.Linear(sizes[k], sizes[k + 1], bias=False))
            self.b_gc.append(nn.Parameter(torch.zeros(1, sizes[k + 1])))
            self.W_bi.append(nn.Linear(sizes[k], sizes[k + 1], bias=False))
            self.b_bi.append(nn.Parameter(torch.zeros(1, sizes[k + 1])))

        self._adj: torch.Tensor | None = None

    def set_adj(self, norm_adj: sp.spmatrix, device: torch.device) -> None:
        coo = norm_adj.tocoo().astype(np.float32)
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        self._adj = torch.sparse_coo_tensor(indices, values, shape).to(device)

    # ------------------------------------------------------------------
    def forward_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ego = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embs = [ego]
        x = ego

        for k in range(self.n_layers):
            side = torch.sparse.mm(self._adj, x)

            # W_gc * (A * e) + b
            sum_emb = F.leaky_relu(self.W_gc[k](side) + self.b_gc[k])

            # Element-wise product interaction
            bi_emb = torch.mul(x, side)
            bi_emb = F.leaky_relu(self.W_bi[k](bi_emb) + self.b_bi[k])

            x = sum_emb + bi_emb

            # Message dropout
            if self.training and self.mess_dropout_rate > 0:
                x = F.dropout(x, p=self.mess_dropout_rate, training=True)

            # L2 normalisation
            x = F.normalize(x, p=2, dim=1)
            all_embs.append(x)

        out = torch.cat(all_embs, dim=1)
        users_emb, items_emb = torch.split(
            out, [self.n_users, self.n_items], dim=0
        )
        return users_emb, items_emb

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        all_users, all_items = self.forward_all()

        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        u_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)

        return u_emb, pos_emb, neg_emb, u_emb_0, pos_emb_0, neg_emb_0


# =====================================================================
# Config
# =====================================================================
@dataclass
class NGCFConfig:
    emb_dim: int = 64
    layer_sizes: Tuple[int, ...] = (64, 64, 64)
    mess_dropout: float = 0.1
    node_dropout: float = 0.0
    lr: float = 1e-4
    decay: float = 1e-5
    epochs: int = 50
    batch_size: int = 1024
    device: str = "cpu"
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4


# =====================================================================
# Trainer
# =====================================================================
class NGCFTrainer:
    """BPR trainer for NGCF.

    Expected train format: ``train[user] = set(item_ids)``
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        norm_adj: sp.spmatrix,
        cfg: NGCFConfig | None = None,
    ) -> None:
        self.cfg = cfg or NGCFConfig()
        self.n_users = n_users
        self.n_items = n_items
        self.device = torch.device(self.cfg.device)

        self.model = NGCFNet(
            n_users=n_users,
            n_items=n_items,
            emb_dim=self.cfg.emb_dim,
            layer_sizes=self.cfg.layer_sizes,
            mess_dropout=self.cfg.mess_dropout,
            node_dropout=self.cfg.node_dropout,
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
                print(f"[NGCF] Epoch {epoch+1}/{self.cfg.epochs}: no triplets")
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
                f"[NGCF] Epoch {epoch+1}/{self.cfg.epochs}: "
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
                        f"[NGCF] Early stopping at epoch {epoch+1} "
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
