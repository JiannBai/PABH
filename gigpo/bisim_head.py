# Copyright 2026 - BiPACE improvement experiments.
#
# Bisimulation-metric projection head (实验 B 核心):
#   冻结 actor backbone，hidden state 外接一个低成本、只训 head 的 value-等价投影。
#
#   冷启动：proj 用正交初始化（随机正交投影近似保距离，Johnson-Lindenstrauss），
#   因此训练一开始 head 输出的聚类结果 ≈ 裸 hidden state 聚类，不会掉点。
#   训练目标（MICo 版）：让 ‖z_i - z_j‖ 拟合 |r_i - r_j| + γ·‖z'_i - z'_j‖，
#   其中 r 是规则 PRM 的每步即时 process_reward（无时间趋势）。
#   防坍缩：目标项用慢更新的 target head（EMA），并 detach。
#
# 用法（在 verl/trainer/ppo/ray_trainer.py 集成）：
#   trainer = BisimHeadTrainer(in_dim=H, out_dim=D, device=...)
#   z = trainer.project(hidden_np)               # 聚类用
#   loss = trainer.train_step(h_cur, h_next, r, pairs)   # 训练 head

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BisimHead(nn.Module):
    """Linear value-equivalence projection: H -> D, orthogonal init, L2 out."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        # 正交初始化：随机正交投影近似保距离（JL），冷启动 warm-start。
        nn.init.orthogonal_(self.proj.weight)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        z = F.normalize(z, dim=-1)
        return z


class BisimHeadTrainer:
    """Wraps head + EMA target head + optimizer + MICo bisim loss.

    variance_reg_coef: optional anti-collapse regularization. When > 0, a
    negative variance term encourages the head embeddings to stay spread out
    (diverse), preventing them from collapsing to a narrow region. Default 0
    preserves the original (实验 B) bisim_head behaviour exactly.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        gamma: float = 0.9,
        lr: float = 1e-3,
        tau: float = 0.005,
        device: str = "cpu",
        variance_reg_coef: float = 0.0,
    ):
        self.head = BisimHead(in_dim, out_dim).to(device)
        self.head_bar = BisimHead(in_dim, out_dim).to(device)
        self.head_bar.load_state_dict(self.head.state_dict())
        for p in self.head_bar.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.variance_reg_coef = variance_reg_coef

    def state_dict(self) -> dict:
        """Return a checkpointable dict for the head, EMA target, and optimizer.

        All tensors are moved to CPU so the file is device-independent and can
        be reloaded after a container change or onto a different accelerator.
        """
        return {
            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
            "head_bar": {k: v.detach().cpu() for k, v in self.head_bar.state_dict().items()},
            "optimizer": {
                "state": {
                    g: {kk: vv.detach().cpu() if torch.is_tensor(vv) else vv
                        for kk, vv in s.items()}
                    for g, s in self.optimizer.state_dict()["state"].items()
                },
                "param_groups": self.optimizer.state_dict()["param_groups"],
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore head + EMA target + optimizer from a state_dict() payload."""
        self.head.load_state_dict(state["head"])
        self.head_bar.load_state_dict(state["head_bar"])
        # Load optimizer state back onto the head's device. PyTorch accepts
        # CPU tensors here and relocates them against the current param tensors.
        self.optimizer.load_state_dict(state["optimizer"])

    def project(self, hidden_np: np.ndarray) -> np.ndarray:
        """Map (N, H) frozen hidden states -> (N, D) head embeddings."""
        x = torch.from_numpy(np.asarray(hidden_np, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            z = self.head(x)
        return z.cpu().numpy()

    def train_step(
        self,
        h_cur: torch.Tensor,
        h_next: torch.Tensor,
        proc_reward: torch.Tensor,
        pairs: torch.Tensor,
        d_g: torch.Tensor = None,
    ) -> float:
        """One bisim loss step. pairs: (M, 2) long tensor of step-index pairs.
        d_g: optional (M,) tensor of Jaccard progress-set distances for the
        sampled pairs. When provided, replaces the sparse |r_i - r_j| term
        as the reward anchor (bisim_anchor_mode='d_g')."""
        i = pairs[:, 0].to(self.device)
        j = pairs[:, 1].to(self.device)
        h_cur = h_cur.to(self.device)
        h_next = h_next.to(self.device)
        proc_reward = proc_reward.to(self.device)

        z_i = self.head(h_cur[i])
        z_j = self.head(h_cur[j])
        zn_i = self.head_bar(h_next[i])
        zn_j = self.head_bar(h_next[j])

        pred = (z_i - z_j).norm(dim=-1)
        if d_g is not None:
            reward_term = d_g.to(self.device)
        else:
            reward_term = (proc_reward[i] - proc_reward[j]).abs()
        target = reward_term + self.gamma * (zn_i - zn_j).norm(dim=-1)
        loss = ((pred - target.detach()) ** 2).mean()

        # Anti-collapse variance regularization (only when enabled):
        # encourage the L2-normalized head embeddings of the sampled steps to
        # stay spread out, so clusters keep discriminating power.
        if self.variance_reg_coef > 0:
            all_idx = torch.unique(torch.cat([i, j]))
            z_all = self.head(h_cur[all_idx])
            var = z_all.var(dim=0).mean()
            loss = loss - self.variance_reg_coef * var

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for p, pb in zip(self.head.parameters(), self.head_bar.parameters()):
                pb.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
        return float(loss.item())


def align_next(hidden_np: np.ndarray, traj_uids: np.ndarray):
    """For each step, return the next step's hidden within the same trajectory.

    Assumes gather_rollout_data ordering: steps of the same traj are contiguous
    and in increasing step order. Returns (next_hidden, valid_mask).
    - next_hidden[i] = hidden of the next step if it exists (same traj), else 0.
    - valid_mask[i] = True iff step i has a next step.
    """
    n = hidden_np.shape[0]
    next_hidden = np.zeros_like(hidden_np)
    valid = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if traj_uids[i] == traj_uids[i + 1]:
            next_hidden[i] = hidden_np[i + 1]
            valid[i] = True
    return next_hidden, valid


def compute_d_g(g_sets: np.ndarray, pairs: torch.Tensor) -> torch.Tensor:
    """Jaccard distance between progress sets for sampled pairs.

    d_g(i,j) = |g_i XOR g_j| / |g_i UNION g_j|, in [0,1].
    Dense, ground-truth, and distinguishes combinatorial progress states
    (e.g. {torch,stick} vs {ingot,stick} -> 2/3), unlike scalar progress
    or sparse PRM rewards.

    Args:
        g_sets: (N,) object array of frozenset (completed closure nodes).
        pairs: (M, 2) long tensor of step-index pairs.
    Returns:
        (M,) float tensor of d_g values.
    """
    d = torch.zeros(pairs.shape[0], dtype=torch.float32)
    for k in range(pairs.shape[0]):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        si, sj = g_sets[i], g_sets[j]
        union = si | sj
        if len(union) == 0:
            d[k] = 0.0
        else:
            d[k] = len(si ^ sj) / len(union)
    return d


def sample_pairs(
    valid: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    min_dt: int = 0,
    traj_uids: np.ndarray = None,
) -> np.ndarray:
    """Sample (M, 2) step-index pairs, both ends must have a valid next state.

    min_dt: minimum |t_i - t_j| (within-trajectory step distance) required for
    a pair to be kept. 0 = off (original behaviour). Only filters pairs whose
    two ends belong to the same trajectory; cross-trajectory pairs are always
    kept (they are the contrastive signal the bisim metric needs — MICo only
    has non-trivial targets for states at different progress).
    """
    idx = np.flatnonzero(valid)
    if len(idx) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    i = rng.choice(idx, size=n_pairs, replace=True)
    j = rng.choice(idx, size=n_pairs, replace=True)
    if min_dt > 0 and traj_uids is not None:
        # Vectorized dt-filter: keep cross-traj pairs and same-traj pairs with
        # |t_i - t_j| >= min_dt. Same-traj index distance == step distance
        # because steps of one trajectory are contiguous.
        same_traj = traj_uids[i] == traj_uids[j]
        dt = np.abs(i - j)
        keep = (~same_traj) | (dt >= min_dt)
        if not keep.any():
            return np.zeros((0, 2), dtype=np.int64)
        i, j = i[keep], j[keep]
    return np.stack([i, j], axis=1).astype(np.int64)
