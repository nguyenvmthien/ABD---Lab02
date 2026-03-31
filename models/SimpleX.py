"""
SimpleX – A Simple and Strong Baseline for Collaborative Filtering.

Reference:
    Mao, K. et al. "SimpleX: A Simple and Strong Baseline for
    Collaborative Filtering." CIKM 2021.

Key idea: user representation = γ * user_emb + (1-γ) * aggregated_history.
Uses Cosine Contrastive Loss (CCL) instead of BPR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# Behaviour aggregator
# =====================================================================
class BehaviorAggregator(nn.Module):
    """Aggregate user interaction history via mean / attention pooling."""

    def __init__(
        self,
        emb_dim: int,
        gamma: float = 0.5,
        aggregator: str = "mean",
    ) -> None:
        super().__init__()
        self.aggregator = aggregator
        self.gamma = gamma
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)

        if aggregator == "user_attention":
            self.W_k = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.Tanh()
            )
        elif aggregator == "self_attention":
            self.W_k = nn.Sequential(
                nn.Linear(emb_dim, emb_dim), nn.Tanh()
            )
            self.W_q = nn.Parameter(torch.Tensor(emb_dim, 1))
            nn.init.xavier_normal_(self.W_q)

    def forward(
        self, uid_emb: torch.Tensor, history_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        uid_emb:     (B, D)
        history_emb: (B, S, D)   where S = max_history_len
        """
        if self.aggregator == "mean":
            agg = self._mean_pool(history_emb)
        elif self.aggregator == "user_attention":
            agg = self._user_attention(uid_emb, history_emb)
        elif self.aggregator == "self_attention":
            agg = self._self_attention(history_emb)
        else:
            agg = self._mean_pool(history_emb)

        return self.gamma * uid_emb + (1 - self.gamma) * agg

    # ----- pooling variants -----
    def _mean_pool(self, seq: torch.Tensor) -> torch.Tensor:
        mask = seq.sum(dim=-1) != 0  # (B, S)
        denom = mask.float().sum(dim=-1, keepdim=True).clamp(min=1e-9)
        mean = seq.sum(dim=1) / denom
        return self.W_v(mean)

    def _user_attention(
        self, uid_emb: torch.Tensor, seq: torch.Tensor
    ) -> torch.Tensor:
        key = self.W_k(seq)  # (B, S, D)
        mask = seq.sum(dim=-1) == 0  # (B, S)
        attn = torch.bmm(key, uid_emb.unsqueeze(-1)).squeeze(-1)  # (B, S)
        attn = attn.masked_fill(mask, 0.0)
        e = torch.exp(attn)
        attn_w = e / (e.sum(dim=1, keepdim=True) + 1e-9)
        out = torch.bmm(attn_w.unsqueeze(1), seq).squeeze(1)
        return self.W_v(out)

    def _self_attention(self, seq: torch.Tensor) -> torch.Tensor:
        key = self.W_k(seq)
        mask = seq.sum(dim=-1) == 0
        attn = torch.matmul(key, self.W_q).squeeze(-1)
        attn = attn.masked_fill(mask, 0.0)
        e = torch.exp(attn)
        attn_w = e / (e.sum(dim=1, keepdim=True) + 1e-9)
        out = torch.bmm(attn_w.unsqueeze(1), seq).squeeze(1)
        return self.W_v(out)


# =====================================================================
# Network
# =====================================================================
class SimpleXNet(nn.Module):
    """SimpleX backbone."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        gamma: float = 0.5,
        aggregator: str = "mean",
        history_len: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.history_len = history_len

        self.user_embedding = nn.Embedding(n_users, emb_dim)
        # +1 for padding index 0 (we'll remap items by +1 internally)
        self.item_embedding = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=1e-4)
        nn.init.normal_(self.item_embedding.weight, std=1e-4)

        self.aggregator = BehaviorAggregator(emb_dim, gamma=gamma, aggregator=aggregator)
        self.dropout = nn.Dropout(dropout)

    def get_user_vec(
        self, users: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        """
        users:   (B,) user indices
        history: (B, S) item indices (0 = padding)
        """
        uid_emb = self.user_embedding(users)  # (B, D)
        hist_emb = self.item_embedding(history)  # (B, S, D)
        user_vec = self.aggregator(uid_emb, hist_emb)
        user_vec = self.dropout(user_vec)
        user_vec = F.normalize(user_vec, p=2, dim=1)
        return user_vec

    def get_item_vec(self, items: torch.Tensor) -> torch.Tensor:
        """items: (B,) or (B, K) item indices (already shifted +1)."""
        item_emb = self.item_embedding(items)
        item_emb = F.normalize(item_emb, p=2, dim=-1)
        return item_emb


# =====================================================================
# Config
# =====================================================================
@dataclass
class SimpleXConfig:
    emb_dim: int = 64
    gamma: float = 0.5
    aggregator: str = "mean"           # mean | user_attention | self_attention
    history_len: int = 50
    dropout: float = 0.0
    num_negs: int = 50                 # negative samples per positive
    margin: float = 0.9                # CCL margin
    neg_weight: float = 150.0          # negative weight in CCL
    lr: float = 1e-3
    weight_decay: float = 1e-6
    epochs: int = 50
    batch_size: int = 512
    device: str = "cpu"
    # Early stopping
    patience: int = 5
    min_delta: float = 1e-4


# =====================================================================
# Trainer
# =====================================================================
class SimpleXTrainer:
    """Cosine Contrastive Loss (CCL) trainer for SimpleX.

    Expected train format: ``train[user] = set(item_ids)``
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        cfg: SimpleXConfig | None = None,
    ) -> None:
        self.cfg = cfg or SimpleXConfig()
        self.n_users = n_users
        self.n_items = n_items
        self.device = torch.device(self.cfg.device)

        self.model = SimpleXNet(
            n_users=n_users,
            n_items=n_items,
            emb_dim=self.cfg.emb_dim,
            gamma=self.cfg.gamma,
            aggregator=self.cfg.aggregator,
            history_len=self.cfg.history_len,
            dropout=self.cfg.dropout,
        ).to(self.device)

    # ------------------------------------------------------------------
    def _build_user_histories(
        self, train: Dict[int, Set[int]]
    ) -> Dict[int, np.ndarray]:
        """Pad and shift user histories (+1 so 0 = padding)."""
        histories: Dict[int, np.ndarray] = {}
        ml = self.cfg.history_len
        for u, items in train.items():
            arr = np.array(sorted(items), dtype=np.int64) + 1  # shift
            if len(arr) > ml:
                arr = arr[-ml:]
            pad = np.zeros(ml, dtype=np.int64)
            pad[: len(arr)] = arr
            histories[u] = pad
        return histories

    # ------------------------------------------------------------------
    def fit(self, train: Dict[int, Set[int]], seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        histories = self._build_user_histories(train)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # Build training pairs
        pairs = [(u, i) for u, items in train.items() for i in items]

        self.model.train()
        for epoch in range(self.cfg.epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            n_batch = 0

            for start in range(0, len(pairs), self.cfg.batch_size):
                batch = pairs[start : start + self.cfg.batch_size]
                b_users = []
                b_hist = []
                b_pos = []
                b_neg = []

                for u, pos_i in batch:
                    b_users.append(u)
                    b_hist.append(histories.get(u, np.zeros(self.cfg.history_len, dtype=np.int64)))
                    b_pos.append(pos_i + 1)  # shift for padding

                    # Sample negatives
                    seen = train.get(u, set())
                    negs = []
                    for _ in range(self.cfg.num_negs):
                        for __ in range(30):
                            neg = int(rng.integers(0, self.n_items))
                            if neg not in seen:
                                negs.append(neg + 1)  # shift
                                break
                        else:
                            negs.append(int(rng.integers(1, self.n_items + 1)))
                    b_neg.append(negs)

                t_users = torch.tensor(b_users, dtype=torch.long, device=self.device)
                t_hist = torch.tensor(np.array(b_hist), dtype=torch.long, device=self.device)
                t_pos = torch.tensor(b_pos, dtype=torch.long, device=self.device)
                t_neg = torch.tensor(b_neg, dtype=torch.long, device=self.device)  # (B, num_negs)

                user_vec = self.model.get_user_vec(t_users, t_hist)  # (B, D)
                pos_vec = self.model.get_item_vec(t_pos)  # (B, D)
                neg_vecs = self.model.get_item_vec(t_neg)  # (B, num_negs, D)

                # Cosine similarities (already L2-normalised)
                pos_score = (user_vec * pos_vec).sum(dim=1)  # (B,)
                neg_score = torch.bmm(
                    neg_vecs, user_vec.unsqueeze(-1)
                ).squeeze(-1)  # (B, num_negs)

                # CCL loss
                pos_loss = F.relu(1.0 - pos_score)
                neg_loss = F.relu(neg_score - self.cfg.margin)
                loss = (
                    pos_loss.mean()
                    + self.cfg.neg_weight / self.cfg.num_negs * neg_loss.sum(dim=1).mean()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            avg = epoch_loss / max(n_batch, 1)
            print(
                f"[SimpleX] Epoch {epoch+1}/{self.cfg.epochs}: "
                f"{len(pairs)} pairs, avg loss = {avg:.6f}"
            )

    # ------------------------------------------------------------------
    def score_items(self, user: int, items: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            # Build single-user batch
            hist = np.zeros(self.cfg.history_len, dtype=np.int64)
            # We need to reconstruct history; store it during fit
            if hasattr(self, "_histories") and user in self._histories:
                hist = self._histories[user]

            t_user = torch.tensor([user], dtype=torch.long, device=self.device)
            t_hist = torch.tensor([hist], dtype=torch.long, device=self.device)
            user_vec = self.model.get_user_vec(t_user, t_hist)  # (1, D)

            # Score all items (shifted)
            all_idx = torch.arange(1, self.n_items + 1, dtype=torch.long, device=self.device)
            all_item_vec = self.model.get_item_vec(all_idx)  # (n_items, D)
            scores = (user_vec @ all_item_vec.T).squeeze(0)  # (n_items,)
            return scores.cpu().numpy().astype(np.float64)[items]

    # Override fit to keep histories around for scoring
    _fit_orig = None

    def fit(self, train: Dict[int, Set[int]], seed: int = 42) -> None:
        self._histories = self._build_user_histories(train)

        rng = np.random.default_rng(seed)
        histories = self._histories

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        pairs = [(u, i) for u, items in train.items() for i in items]

        self.model.train()
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.cfg.epochs):
            rng.shuffle(pairs)
            epoch_loss = 0.0
            n_batch = 0

            for start in range(0, len(pairs), self.cfg.batch_size):
                batch = pairs[start : start + self.cfg.batch_size]
                b_users = []
                b_hist = []
                b_pos = []
                b_neg = []

                for u, pos_i in batch:
                    b_users.append(u)
                    b_hist.append(histories.get(u, np.zeros(self.cfg.history_len, dtype=np.int64)))
                    b_pos.append(pos_i + 1)

                    seen = train.get(u, set())
                    negs = []
                    for _ in range(self.cfg.num_negs):
                        for __ in range(30):
                            neg = int(rng.integers(0, self.n_items))
                            if neg not in seen:
                                negs.append(neg + 1)
                                break
                        else:
                            negs.append(int(rng.integers(1, self.n_items + 1)))
                    b_neg.append(negs)

                t_users = torch.tensor(b_users, dtype=torch.long, device=self.device)
                t_hist = torch.tensor(np.array(b_hist), dtype=torch.long, device=self.device)
                t_pos = torch.tensor(b_pos, dtype=torch.long, device=self.device)
                t_neg = torch.tensor(b_neg, dtype=torch.long, device=self.device)

                user_vec = self.model.get_user_vec(t_users, t_hist)
                pos_vec = self.model.get_item_vec(t_pos)
                neg_vecs = self.model.get_item_vec(t_neg)

                pos_score = (user_vec * pos_vec).sum(dim=1)
                neg_score = torch.bmm(
                    neg_vecs, user_vec.unsqueeze(-1)
                ).squeeze(-1)

                pos_loss = F.relu(1.0 - pos_score)
                neg_loss = F.relu(neg_score - self.cfg.margin)
                loss = (
                    pos_loss.mean()
                    + self.cfg.neg_weight / self.cfg.num_negs * neg_loss.sum(dim=1).mean()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batch += 1

            avg = epoch_loss / max(n_batch, 1)
            print(
                f"[SimpleX] Epoch {epoch+1}/{self.cfg.epochs}: "
                f"{len(pairs)} pairs, avg loss = {avg:.6f}"
            )

            # Early stopping
            if avg < best_loss - self.cfg.min_delta:
                best_loss = avg
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    print(
                        f"[SimpleX] Early stopping at epoch {epoch+1} "
                        f"(no improvement for {self.cfg.patience} epochs, best_loss={best_loss:.6f})"
                    )
                    break

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
