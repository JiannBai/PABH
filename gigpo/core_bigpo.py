# Copyright 2026 BiPACE contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
BiGPO — Bisimulation Group Policy Optimization.

Single-mechanism extension of GiGPO. Replaces observation-equality clustering
in build_step_group() with clustering under a policy-induced behavioral metric
(MICo-style fingerprint distance). GiGPO with eps=0 and the identity embedder
is the degenerate special case.

Theoretical anchor:
    Castro et al., MICo (NeurIPS'21):  |V^pi(s) - V^pi(s')| <= d_pi(s, s').
    Therefore clustering with cosine distance < eps in a policy-induced
    embedding space yields a step-baseline whose bias is bounded by ~eps.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------- #
# ----------------------- Embedders ------------------------ #
# ---------------------------------------------------------- #

# An Embedder is any callable: List[obs] -> np.ndarray (B, D).
Embedder = Callable[[Sequence[Any]], np.ndarray]


def _to_text(obs: Any) -> str:
    """Best-effort coerce an observation to a string for text-based embedders."""
    if isinstance(obs, str):
        return obs
    if isinstance(obs, (bytes, bytearray)):
        return obs.decode("utf-8", errors="replace")
    if isinstance(obs, dict):
        # Stable serialization across runs.
        items = sorted((str(k), _to_text(v)) for k, v in obs.items())
        return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"
    if isinstance(obs, (list, tuple)):
        return "[" + ",".join(_to_text(e) for e in obs) + "]"
    if isinstance(obs, np.ndarray):
        # Multimodal placeholder: hash array bytes; same array -> same text.
        return "ndarray:" + hashlib.blake2b(obs.tobytes(), digest_size=16).hexdigest()
    return str(obs)


class HashNgramEmbedder:
    """
    Zero-dependency feature-hashing embedder over character n-grams.

    Used as the default v1 backend so the algorithm is testable and the trainer
    has no new mandatory dependency. Empirically captures lexical similarity
    well enough to validate the grouping mechanism end-to-end; for the
    theoretically-correct policy-induced metric, swap in ActorHiddenEmbedder.
    """

    def __init__(self, dim: int = 256, ngram: int = 4):
        self.dim = dim
        self.ngram = ngram

    def __call__(self, obs_list: Sequence[Any]) -> np.ndarray:
        out = np.zeros((len(obs_list), self.dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            text = _to_text(obs)
            if not text:
                continue
            n = self.ngram
            # Pad short strings so we still get at least one ngram.
            if len(text) < n:
                text = text + " " * (n - len(text))
            for j in range(len(text) - n + 1):
                gram = text[j : j + n]
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest(),
                    "little",
                )
                # Signed hash trick to keep mean-zero on random inputs.
                sign = 1.0 if (h & 1) else -1.0
                out[i, h % self.dim] += sign
            # L2 normalize (cosine-ready).
            norm = np.linalg.norm(out[i])
            if norm > 0.0:
                out[i] /= norm
        return out


class IdentityHashEmbedder:
    """
    Degenerate embedder used to recover GiGPO exactly (regression test).

    Hashes the entire observation to a one-hot column. Two observations
    cluster together iff they are bit-identical. Combined with eps=0,
    BiGPO becomes numerically identical to GiGPO's exact-match path.
    """

    def __init__(self, dim: int = 65536):
        self.dim = dim

    def __call__(self, obs_list: Sequence[Any]) -> np.ndarray:
        out = np.zeros((len(obs_list), self.dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            text = _to_text(obs)
            h = int.from_bytes(
                hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
                "little",
            )
            out[i, h % self.dim] = 1.0
        return out


class ActorHiddenEmbedder:
    """
    Policy-induced behavioral fingerprint via the actor's last-layer hidden state.

    This is the theoretically-correct embedder for BiGPO: the cluster metric
    is computed in the same representation space the policy itself uses to
    decide actions, making the resulting partition a sample-based MICo-style
    approximation of policy bisimulation (Castro et al., NeurIPS'21).

    This class is a *driver-side passthrough*. The embedding tensor is
    actually produced inside the actor worker's `compute_anchor_embeddings`
    method (see verl/workers/fsdp_workers.py) and handed back; this class
    just exposes it to build_step_group_bisim with a stable interface.
    """

    def __init__(self, embeddings: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(f"ActorHiddenEmbedder expects 2D, got {embeddings.shape}")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        # L2-normalize so cosine clustering is stable.
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.where(norms > 0, norms, 1.0)
        self._cursor = 0

    def __call__(self, obs_list: Sequence[Any]) -> np.ndarray:
        # Not used in the precomputed path (build_step_group_bisim consumes
        # `embeddings=` directly). Provided for API completeness.
        n = len(obs_list)
        out = self.embeddings[self._cursor : self._cursor + n]
        self._cursor += n
        return out


def get_default_embedder(name: str = "hash_ngram", **kwargs) -> Embedder:
    """Factory for the v1 embedder backends."""
    name = name.lower()
    if name in ("hash_ngram", "hash", "default"):
        return HashNgramEmbedder(**kwargs)
    if name in ("identity", "identity_hash", "exact"):
        return IdentityHashEmbedder(**kwargs)
    if name in ("frozen_st", "sentence_transformers", "st"):
        # Lazy import — only required if the user opts in.
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_name = kwargs.pop("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)

        def _embed(obs_list: Sequence[Any]) -> np.ndarray:
            texts = [_to_text(o) for o in obs_list]
            v = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return v.astype(np.float32)

        return _embed
    raise ValueError(f"Unknown embedder backend: {name}")


# ---------------------------------------------------------- #
# ----------------------- Clustering ----------------------- #
# ---------------------------------------------------------- #


def _greedy_cosine_cluster(vecs: np.ndarray, eps: float) -> List[List[int]]:
    """
    Online greedy clustering under cosine distance.

    A new point joins the existing cluster whose centroid has the smallest
    cosine distance, provided that distance is < eps. Otherwise it opens a
    new cluster. Centroids are running L2-normalized means.

    eps is a *distance* threshold in [0, 2]; eps=0 forces exact-direction
    matching (i.e., behaves like equality on the embedding).
    """
    n = vecs.shape[0]
    if n == 0:
        return []

    # Pre-normalize once — embedders should already have normalized, but be safe.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    v = vecs / norms

    centroids: List[np.ndarray] = []  # each L2-normalized
    members: List[List[int]] = []

    for i in range(n):
        x = v[i]
        if not centroids:
            centroids.append(x.copy())
            members.append([i])
            continue
        sims = np.stack(centroids) @ x  # (K,)
        k = int(np.argmax(sims))
        best_dist = 1.0 - float(sims[k])
        if best_dist <= eps:
            members[k].append(i)
            # Update centroid: running mean, then renormalize.
            m = len(members[k])
            centroids[k] = centroids[k] + (x - centroids[k]) / m
            cn = np.linalg.norm(centroids[k])
            if cn > 0:
                centroids[k] = centroids[k] / cn
        else:
            centroids.append(x.copy())
            members.append([i])

    return members


# ---------------------------------------------------------- #
# ------------------- Step-group Builder ------------------- #
# ---------------------------------------------------------- #


_LOGGER_SINGLETON = None


def _get_logger():
    """Lazy global BiGPOLogger; respects $BIGPO_LOG_DIR env var."""
    global _LOGGER_SINGLETON
    if _LOGGER_SINGLETON is None:
        from gigpo.bigpo_logger import BiGPOLogger

        _LOGGER_SINGLETON = BiGPOLogger()
    return _LOGGER_SINGLETON


def build_step_group_bisim(
    anchor_obs: np.ndarray,
    index: np.ndarray,
    embedder: Optional[Embedder] = None,
    eps: float = 0.10,
    summarize: bool = False,
    embeddings: Optional[np.ndarray] = None,
    traj_index: Optional[np.ndarray] = None,
    step_within_traj: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Drop-in replacement for gigpo.core_gigpo.build_step_group() that clusters
    by behavioral fingerprint instead of observation hash.

    Args:
        anchor_obs: array of per-step observations (any python object).
        index: array of episode-group ids (uid). Same shape as anchor_obs.
        embedder: callable mapping a list of obs to an (N, D) np.ndarray.
                  Defaults to HashNgramEmbedder() if `embeddings` is None.
        eps: cosine-distance threshold for joining a cluster. eps=0 with
             IdentityHashEmbedder reproduces GiGPO exact-match grouping.
        summarize: if True, print group-size histogram.
        embeddings: optional precomputed (N, D) np.ndarray parallel to
            anchor_obs (e.g. produced by ActorHiddenEmbedder on the workers).
            If provided, the `embedder` argument is ignored.

    Returns:
        np.ndarray of dtype object: one UUID per step, identical UUIDs for
        steps in the same cluster. Shape == anchor_obs.shape.
    """
    use_precomputed = embeddings is not None
    if use_precomputed:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(anchor_obs):
            raise ValueError(f"Precomputed embeddings have {embeddings.shape[0]} rows but anchor_obs has {len(anchor_obs)}.")
    elif embedder is None:
        embedder = HashNgramEmbedder()

    n = len(anchor_obs)
    step_group_uids = np.empty(n, dtype=object)
    group_size: List[int] = []
    # Member list in original-array indices, used by the optional logger.
    cluster_members_global: List[List[int]] = []

    unique_indices = np.unique(index)
    for idx in unique_indices:
        locs = np.where(index == idx)[0]
        if len(locs) == 0:
            continue

        if use_precomputed:
            vecs = embeddings[locs]
        else:
            sub_obs = [anchor_obs[loc] for loc in locs]
            vecs = embedder(sub_obs)
            if vecs.shape[0] != len(locs):
                raise ValueError(f"Embedder returned {vecs.shape[0]} rows for {len(locs)} obs.")
        clusters = _greedy_cosine_cluster(vecs, eps=eps)

        for cluster_local_idxs in clusters:
            cluster_uid = str(uuid.uuid4())
            group_size.append(len(cluster_local_idxs))
            members_global = [int(locs[li]) for li in cluster_local_idxs]
            cluster_members_global.append(members_global)
            for li in cluster_local_idxs:
                step_group_uids[locs[li]] = cluster_uid

    # Sanity: every step assigned.
    if any(u is None for u in step_group_uids):
        missing = [i for i, u in enumerate(step_group_uids) if u is None]
        raise ValueError(f"BiGPO: failed to assign UIDs at indices {missing}")

    if summarize:
        from collections import Counter

        c = Counter(group_size)
        total = sum(c.values()) or 1
        print("BiGPO step-group size distribution:")
        print(" size | count | proportion")
        for s in sorted(c):
            print(f" {s:>4} | {c[s]:>5} | {c[s] / total:>9.2%}")
    if group_size:
        gs = np.asarray(group_size)
        singleton_frac = float(np.mean(gs == 1))
        mode_tag = "actor_hidden" if use_precomputed else type(embedder).__name__
        print(f"BiGPO[{mode_tag}] eps={eps:.3f} | clusters={len(gs)} avg={gs.mean():.2f} med={np.median(gs):.0f} p90={np.percentile(gs, 90):.0f} max={gs.max()} singleton={singleton_frac:.1%}")
        # Opt-in CSV diagnostics (no-op if BIGPO_LOG_DIR is unset).
        try:
            _logger = _get_logger()
            _logger.log_clusters(
                cluster_members_global,
                traj_index=traj_index,
                step_within_traj=step_within_traj,
                returns=returns,
                anchor_obs=anchor_obs,
                embeddings=embeddings if use_precomputed else None,
                group_index=index,
                eps=float(eps),
                embedder_tag=mode_tag,
            )
        except Exception as e:
            # Logger must never crash training.
            print(f"[BiGPOLogger warning] {e}")

    return step_group_uids
