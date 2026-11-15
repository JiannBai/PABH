# Copyright 2026 BiPACE contributors.
"""
BiGPO diagnostics logger.

Item (3) of the v2 brainstorm: dump rich per-step cluster diagnostics so we
can produce the GiGPO-style histogram figure and analyze
match-length / match-content. Strictly opt-in via env var
BIGPO_LOG_DIR — when unset this module is a no-op.

Outputs (all under $BIGPO_LOG_DIR/):
  cluster_hist.csv       step, n_clusters, singleton_frac, bin_1, bin_2, ..., bin_9, bin_10_50, bin_gt50, avg, med, p90, max
  match_distance.csv     step, mean_dt, p90_dt, max_dt, n_pairs    (Δt within multi-traj clusters)
  cluster_signal.csv     step, mean_within_var, p10_within_var, p90_within_var, mean_size_weighted_var
  dpi_calibration.csv    step, d_cos, abs_dv, n_a, n_b   (V5: exact-state pair calibration, see below)
  qualitative_<step>.txt sample of 3 random clusters with member obs strings

dpi_calibration.csv records the calibration analysis for
embedding-quality appendix: within each prompt group, repeated visits
of the *exact same* anchor observation give a Monte-Carlo value
estimate V_hat(s) = mean return-to-go over visits. For pairs of
distinct exact-states (a, b) we log the actor-hidden cosine distance
between their embedding centroids against |V_hat(a) - V_hat(b)|. The
value-difference bound |V(s)-V(s')| <= L * d_cos holds iff these pairs
sit below a stable linear envelope; scripts/plot_dpi_calibration.py
fits that envelope and renders the appendix figure. Requires the
actor_hidden (precomputed-embeddings) path.

The match-distance and per-cluster signal CSVs require step indices and
returns to be passed in (optional kwargs). When unavailable they are
skipped silently.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_BIN_EDGES = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 50), (51, 10**9)]
_BIN_NAMES = ["bin_1", "bin_2", "bin_3", "bin_4", "bin_5", "bin_6", "bin_7", "bin_8", "bin_9", "bin_10_50", "bin_gt50"]


def _bin_counts(sizes: np.ndarray) -> List[int]:
    out = []
    for lo, hi in _BIN_EDGES:
        out.append(int(np.sum((sizes >= lo) & (sizes <= hi))))
    return out


class BiGPOLogger:
    """Lazy file-handles; instantiated once per training run."""

    def __init__(self, log_dir: Optional[str] = None):
        self.dir: Optional[Path] = None
        if log_dir is None:
            log_dir = os.environ.get("BIGPO_LOG_DIR", None)
        if not log_dir:
            return
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _append(self, name: str, header: List[str], row: List):
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)

    def _append_many(self, name: str, header: List[str], rows: List[List]):
        if not rows:
            return
        path = self.dir / name
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerows(rows)

    def log_clusters(
        self,
        cluster_members: Sequence[Sequence[int]],
        *,
        traj_index: Optional[np.ndarray] = None,
        step_within_traj: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        anchor_obs: Optional[np.ndarray] = None,
        embeddings: Optional[np.ndarray] = None,
        group_index: Optional[np.ndarray] = None,
        eps: float = -1.0,
        embedder_tag: str = "actor_hidden",
    ) -> None:
        """One call per training step. Writes one row to each CSV."""
        if not self.enabled:
            return
        self._step += 1
        sizes = np.asarray([len(c) for c in cluster_members], dtype=np.int64)
        if sizes.size == 0:
            return

        # ---- cluster_hist.csv -----------------------------------------
        bins = _bin_counts(sizes)
        self._append(
            "cluster_hist.csv",
            ["step", "embedder", "eps", "n_clusters", "singleton_frac", *_BIN_NAMES, "avg", "med", "p90", "max"],
            [self._step, embedder_tag, f"{eps:.3f}", int(sizes.size), float(np.mean(sizes == 1)), *bins, float(sizes.mean()), float(np.median(sizes)), float(np.percentile(sizes, 90)), int(sizes.max())],
        )

        # ---- match_distance.csv ---------------------------------------
        # Within each multi-member, multi-trajectory cluster, compute
        # pairwise |t_i - t_j| over all member pairs that come from
        # different trajectories. This tells us whether matches are
        # at "the same physical phase" (small Δt, expected) vs spurious
        # (large Δt across the whole horizon).
        if traj_index is not None and step_within_traj is not None:
            dts: List[int] = []
            for members in cluster_members:
                if len(members) < 2:
                    continue
                ts = step_within_traj[list(members)]
                trs = traj_index[list(members)]
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        if trs[a] != trs[b]:
                            dts.append(int(abs(ts[a] - ts[b])))
            if dts:
                arr = np.asarray(dts)
                self._append(
                    "match_distance.csv",
                    ["step", "n_pairs", "mean_dt", "p50_dt", "p90_dt", "max_dt"],
                    [self._step, len(arr), float(arr.mean()), float(np.median(arr)), float(np.percentile(arr, 90)), int(arr.max())],
                )

        # ---- cluster_signal.csv --------------------------------------
        # Within-cluster variance of returns is the signal-to-noise the
        # step-level baseline can extract: Var=0 → useless cluster.
        if returns is not None:
            within_vars: List[float] = []
            sizes_for_var: List[int] = []
            for members in cluster_members:
                if len(members) < 2:
                    continue
                rs = np.asarray(returns)[list(members)]
                within_vars.append(float(np.var(rs)))
                sizes_for_var.append(len(members))
            if within_vars:
                wv = np.asarray(within_vars)
                sw = np.asarray(sizes_for_var)
                self._append(
                    "cluster_signal.csv",
                    ["step", "n_clusters_used", "mean_var", "p10_var", "p90_var", "size_weighted_mean_var"],
                    [self._step, len(wv), float(wv.mean()), float(np.percentile(wv, 10)), float(np.percentile(wv, 90)), float(np.average(wv, weights=sw))],
                )

        # ---- dpi_calibration.csv (V5) --------------------------------
        # Exact-state value estimates vs embedding distance. Pairs are
        # formed within prompt groups so they share the comparison
        # universe of the step-level estimator. Capped per training
        # step to keep file size negligible.
        if embeddings is not None and returns is not None and group_index is not None and anchor_obs is not None:
            try:
                rows = self._dpi_calibration_rows(
                    np.asarray(embeddings, dtype=np.float32),
                    np.asarray(returns, dtype=np.float64),
                    np.asarray(group_index),
                    anchor_obs,
                )
                self._append_many(
                    "dpi_calibration.csv",
                    ["step", "d_cos", "abs_dv", "n_a", "n_b"],
                    [[self._step, f"{d:.5f}", f"{dv:.5f}", na, nb] for d, dv, na, nb in rows],
                )
            except Exception as e:  # never crash training
                print(f"[BiGPOLogger dpi_calibration warning] {e}")

        # ---- qualitative dump every 25 steps -------------------------
        if anchor_obs is not None and self._step % 25 == 0:
            self._qualitative_dump(cluster_members, anchor_obs)

    _DPI_MIN_VISITS = 2  # exact-state visits needed for a V_hat estimate
    _DPI_MAX_PAIRS = 512  # cap on logged pairs per training step

    def _dpi_calibration_rows(
        self,
        embeddings: np.ndarray,
        returns: np.ndarray,
        group_index: np.ndarray,
        anchor_obs: Sequence,
    ) -> List[tuple]:
        """Exact-state pair calibration rows: (d_cos, |dV_hat|, n_a, n_b)."""
        # Defensive re-normalization; actor-hidden embeddings arrive unit-norm.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

        rows: List[tuple] = []
        for g in np.unique(group_index):
            locs = np.where(group_index == g)[0]
            # Group repeated visits of the exact same observation string.
            visits: dict = {}
            for i in locs:
                visits.setdefault(str(anchor_obs[i]), []).append(int(i))
            states = [(np.asarray(idx), len(idx)) for idx in visits.values() if len(idx) >= self._DPI_MIN_VISITS]
            if len(states) < 2:
                continue
            v_hat = [float(returns[idx].mean()) for idx, _ in states]
            cents = []
            for idx, _ in states:
                c = embeddings[idx].mean(axis=0)
                n = np.linalg.norm(c)
                cents.append(c / n if n > 0 else c)
            for a in range(len(states)):
                for b in range(a + 1, len(states)):
                    d_cos = float(1.0 - np.dot(cents[a], cents[b]))
                    rows.append(
                        (
                            max(d_cos, 0.0),
                            abs(v_hat[a] - v_hat[b]),
                            states[a][1],
                            states[b][1],
                        )
                    )
        if len(rows) > self._DPI_MAX_PAIRS:
            rows = random.sample(rows, self._DPI_MAX_PAIRS)
        return rows

    def _qualitative_dump(self, cluster_members, anchor_obs) -> None:
        big = [c for c in cluster_members if len(c) >= 2]
        if not big:
            return
        pick = random.sample(big, k=min(3, len(big)))
        path = self.dir / f"qualitative_{self._step:05d}.txt"
        with open(path, "w") as f:
            for k, members in enumerate(pick):
                f.write(f"=== cluster {k}  size={len(members)} ===\n")
                for m in members[:8]:  # cap each cluster at 8 lines
                    f.write(f"  [{m}] {str(anchor_obs[m])[:240]}\n")
                f.write("\n")
