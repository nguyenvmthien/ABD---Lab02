from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNCFNet(nn.Module):
	"""Outer-product based ConvNCF backbone.

	This mirrors the original idea from ConvNCF:
	1) user/item embeddings
	2) outer product interaction map
	3) stacked 2x2 stride-2 convolutions
	4) final linear score
	"""

	def __init__(
		self,
		n_users: int,
		n_items: int,
		emb_dim: int = 64,
		channels: Sequence[int] = (32, 32, 32, 32, 32, 32),
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.user_embedding = nn.Embedding(n_users, emb_dim)
		self.item_embedding = nn.Embedding(n_items, emb_dim)
		nn.init.normal_(self.user_embedding.weight, std=0.01)
		nn.init.normal_(self.item_embedding.weight, std=0.01)

		conv_blocks: List[nn.Module] = []
		in_channels = 1
		for out_channels in channels:
			conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2))
			conv_blocks.append(nn.ReLU())
			in_channels = out_channels
		self.conv_stack = nn.Sequential(*conv_blocks)

		side = emb_dim
		for _ in channels:
			side = max(1, side // 2)
		self.flat_dim = side * side * channels[-1]

		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(self.flat_dim, 1)

	def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
		user_vec = self.user_embedding(users)
		item_vec = self.item_embedding(items)

		# Relation map: outer product between user/item embeddings.
		relation = torch.bmm(user_vec.unsqueeze(2), item_vec.unsqueeze(1))
		x = relation.unsqueeze(1)
		x = self.conv_stack(x)
		x = self.dropout(x)
		x = x.flatten(start_dim=1)
		return self.out(x).squeeze(-1)


@dataclass
class ConvNCFConfig:
	emb_dim: int = 64
	channels: Tuple[int, ...] = (32, 32, 32, 32, 32, 32)
	dropout: float = 0.0
	lr_embed: float = 0.05
	lr_net: float = 0.05
	reg_embed: float = 1e-2
	reg_net: float = 1e-2
	epochs: int = 3
	batch_size: int = 512
	device: str = "cpu"
	# Early stopping
	patience: int = 5
	min_delta: float = 1e-4


class ConvNCFTrainer:
	"""Minimal BPR trainer for implicit feedback data.

	Expected train format: train[user] = set(item_ids)
	"""

	def __init__(self, n_users: int, n_items: int, cfg: ConvNCFConfig | None = None) -> None:
		self.cfg = cfg or ConvNCFConfig()
		self.n_users = n_users
		self.n_items = n_items
		self.device = torch.device(self.cfg.device)

		self.model = ConvNCFNet(
			n_users=n_users,
			n_items=n_items,
			emb_dim=self.cfg.emb_dim,
			channels=self.cfg.channels,
			dropout=self.cfg.dropout,
		).to(self.device)

	def _sample_negative(self, user: int, user_train: Dict[int, Set[int]], rng: np.random.Generator) -> int | None:
		seen = user_train.get(user, set())
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
		self,
		train: Dict[int, Set[int]],
		rng: np.random.Generator,
	) -> Tuple[List[int], List[int], List[int]]:
		users: List[int] = []
		pos_items: List[int] = []
		neg_items: List[int] = []

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

	def fit(self, train: Dict[int, Set[int]], seed: int = 42) -> List[float]:
		rng = np.random.default_rng(seed)

		emb_params = list(self.model.user_embedding.parameters()) + list(self.model.item_embedding.parameters())
		net_params = [p for name, p in self.model.named_parameters() if "embedding" not in name]

		opt_embed = torch.optim.Adagrad(emb_params, lr=self.cfg.lr_embed)
		opt_net = torch.optim.Adagrad(net_params, lr=self.cfg.lr_net)

		self.model.train()
		best_loss = float("inf")
		no_improve = 0
		loss_history: List[float] = []

		for epoch_idx in range(self.cfg.epochs):
			users, pos_items, neg_items = self._build_triplets(train, rng)
			if not users:
				print(f"[ConvNCF] Epoch {epoch_idx + 1}/{self.cfg.epochs}: no training triplets")
				continue

			epoch_loss = 0.0
			n_batches = 0

			for start in range(0, len(users), self.cfg.batch_size):
				end = start + self.cfg.batch_size

				b_users = torch.tensor(users[start:end], dtype=torch.long, device=self.device)
				b_pos = torch.tensor(pos_items[start:end], dtype=torch.long, device=self.device)
				b_neg = torch.tensor(neg_items[start:end], dtype=torch.long, device=self.device)

				pos_scores = self.model(b_users, b_pos)
				neg_scores = self.model(b_users, b_neg)

				bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

				reg_embed = torch.tensor(0.0, device=self.device)
				for p in emb_params:
					reg_embed = reg_embed + p.pow(2).sum()

				reg_net = torch.tensor(0.0, device=self.device)
				for p in net_params:
					reg_net = reg_net + p.pow(2).sum()

				loss = (
					bpr_loss
					+ self.cfg.reg_embed * reg_embed / b_users.shape[0]
					+ self.cfg.reg_net * reg_net / b_users.shape[0]
				)

				opt_embed.zero_grad()
				opt_net.zero_grad()
				loss.backward()
				opt_embed.step()
				opt_net.step()

				epoch_loss += loss.item()
				n_batches += 1

			avg_loss = epoch_loss / max(n_batches, 1)
			loss_history.append(avg_loss)
			print(
				f"[ConvNCF] Epoch {epoch_idx + 1}/{self.cfg.epochs}: "
				f"{len(users)} triplets, avg_loss={avg_loss:.6f}"
			)

			# Early stopping check
			if avg_loss < best_loss - self.cfg.min_delta:
				best_loss = avg_loss
				no_improve = 0
			else:
				no_improve += 1
				if no_improve >= self.cfg.patience:
					print(
						f"[ConvNCF] Early stopping at epoch {epoch_idx + 1} "
						f"(no improvement for {self.cfg.patience} epochs, best_loss={best_loss:.6f})"
					)
					break

		return loss_history

	def score_items(self, user: int, items: np.ndarray) -> np.ndarray:
		self.model.eval()
		with torch.no_grad():
			users = torch.tensor([user] * len(items), dtype=torch.long, device=self.device)
			item_tensor = torch.tensor(items.tolist(), dtype=torch.long, device=self.device)
			scores = self.model(users, item_tensor).detach().cpu().numpy()
		return scores.astype(np.float64)

	def save_checkpoint(self, path: str | Path) -> None:
		ckpt_path = Path(path)
		ckpt_path.parent.mkdir(parents=True, exist_ok=True)
		torch.save(
			{
				"model_state_dict": self.model.state_dict(),
				"config": {
					"emb_dim": self.cfg.emb_dim,
					"channels": self.cfg.channels,
					"dropout": self.cfg.dropout,
					"lr_embed": self.cfg.lr_embed,
					"lr_net": self.cfg.lr_net,
					"reg_embed": self.cfg.reg_embed,
					"reg_net": self.cfg.reg_net,
					"epochs": self.cfg.epochs,
					"batch_size": self.cfg.batch_size,
					"device": self.cfg.device,
				},
				"n_users": self.n_users,
				"n_items": self.n_items,
			},
			ckpt_path,
		)

	def load_checkpoint(self, path: str | Path) -> None:
		ckpt = torch.load(Path(path), map_location=self.device)
		self.model.load_state_dict(ckpt["model_state_dict"])
		self.model.to(self.device)
