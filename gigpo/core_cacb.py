# Copyright 2026 BiPACE contributors.
"""
CACB — Counterfactual Action-Conditional Baseline.

Drop-in replacement for gigpo.core_gigpo.step_norm_reward that asks the
RIGHT question for agentic RL: "did your action at this state beat what
peers did at this state?" instead of GiGPO/BiGPO's "did your trajectory
beat peers in the same cluster?"

Within each step-cluster C, partition members by action identity (first-K
response token n-gram hash). For member i with action a_i:

    b(s_i, a_i) = mean{ R_j : j ∈ C, a_j ≠ a_i }
    A^step(i)   = (R_i − b(s_i, a_i)) / (σ_C + ε)         # std from cluster

If all members of C took the same action, fall back to leave-one-out:

    b(s_i, a_i) = mean{ R_j : j ∈ C, j ≠ i }              # RLOO

This closes BiGPO/GiGPO's "same-action collapse" failure mode (cluster
where everyone took the same action gives A ≈ 0 by construction) and
properly disentangles "good state" from "good action."

The std normalizer is kept at the cluster level (not action-conditional)
so the magnitude of the advantage stays comparable across clusters.

Composes with BiGPO's bisimulation grouping: just replace
``step_norm_reward`` with ``step_norm_reward_cacb`` and pass an action
key per row.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import defaultdict

import numpy as np
import torch

# Regex matching the agent's <action>...</action> block. Uses [\s\S]+? so it
# spans newlines but stays non-greedy (some responses have multiple tags;
# we always take the first one, which is the action that was actually executed).
_ACTION_TAG_RE = re.compile(r"<action>([\s\S]+?)</action>", re.IGNORECASE)
_TAG_RE_CACHE = {"action": _ACTION_TAG_RE}


def _get_tag_re(tag_name: str):
    """Memoized compile of the per-tag regex. tag_name='action' (default)
    matches <action>...</action>; tag_name='code' matches <code>...</code>
    (AppWorld), etc."""
    r = _TAG_RE_CACHE.get(tag_name)
    if r is None:
        r = re.compile(rf"<{re.escape(tag_name)}>([\s\S]+?)</{re.escape(tag_name)}>", re.IGNORECASE)
        _TAG_RE_CACHE[tag_name] = r
    return r


def compute_action_keys(responses: torch.Tensor, attention_mask: torch.Tensor, n_first_tokens: int = 8, mode: str = "first_n", tokenizer=None, fallback_to_first_n: bool = True, action_tag_name: str = "action") -> np.ndarray:
    """
    Build per-row action-identity hashes for CACB partitioning.

    Two modes:
      * ``"first_n"`` (default, original behavior): hash the first
        ``n_first_tokens`` non-pad token ids. Cheap, tokenizer-free, but on
        prompts that emit ``<think>...</think>`` before ``<action>...``,
        the leading tokens are the think prefix — i.e. action identity ends
        up dominated by think text rather than the action.
      * ``"action_tag"``: decode the response, extract the content between
        the first ``<action>`` / ``</action>`` tags, and hash that text.
        Bypasses the think-prefix problem and gives a faithful action key.
        Requires ``tokenizer``. Falls back to ``"first_n"`` for any row
        whose response has no parseable action tag (model format failure).

    Args:
        responses : (bsz, response_length) token ids
        attention_mask : (bsz, response_length) 1=valid, 0=pad. If your
            pipeline keeps a separate response-only mask, pass that.
        n_first_tokens : how many leading non-pad tokens to keep in
            ``first_n`` (and in the fallback path of ``action_tag``).
        mode : ``"first_n"`` or ``"action_tag"``.
        tokenizer : HuggingFace tokenizer, required when
            ``mode == "action_tag"``.
        fallback_to_first_n : if True (default), rows where the
            ``<action>...</action>`` regex does not match get the legacy
            first-N hash so they still receive a valid key (and CACB does
            not crash on a malformed rollout).

    Returns:
        np.ndarray of dtype object, shape (bsz,), with hex-string keys.
    """
    if responses.dim() != 2:
        raise ValueError(f"responses must be 2D, got {responses.shape}")
    if mode not in ("first_n", "action_tag", "tag_verb"):
        raise ValueError(f"unknown mode={mode!r}; expected 'first_n', 'action_tag', or 'tag_verb'")
    if mode in ("action_tag", "tag_verb") and tokenizer is None:
        raise ValueError(f"mode={mode!r} requires a tokenizer.")

    # tag_verb: action_tag_name carries a comma-separated list of candidate
    # tag names (e.g. "search,answer" for Search-R1). The action identity is
    # the NAME of whichever tag fires first in the decoded response, not the
    # tag body. Two-or-three-cluster partition for envs with verb-typed
    # actions and free-form bodies (HopQA / multi-hop QA).
    tag_verb_names = None
    if mode == "tag_verb":
        tag_verb_names = [t.strip() for t in action_tag_name.split(",") if t.strip()]
        if not tag_verb_names:
            raise ValueError("tag_verb mode requires action_tag_name as a non-empty comma-separated list of tag names")

    bsz = responses.shape[0]
    out = np.empty(bsz, dtype=object)
    resp = responses.detach().cpu().numpy()
    mask = attention_mask.detach().cpu().numpy() if attention_mask is not None else None

    n_action_match = 0  # diagnostic: how many rows parsed an <action> tag

    for i in range(bsz):
        valid = resp[i][mask[i].astype(bool)] if mask is not None else resp[i]

        if mode == "tag_verb":
            text = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
            chosen = None
            chosen_pos = None
            for name in tag_verb_names:
                pos = text.find(f"<{name}>")
                if pos >= 0 and (chosen_pos is None or pos < chosen_pos):
                    chosen = name
                    chosen_pos = pos
            if chosen is not None:
                n_action_match += 1
                key_bytes = ("VERB::" + chosen).encode("utf-8")
                out[i] = hashlib.blake2b(key_bytes, digest_size=8).hexdigest()
            else:
                # No recognized tag in this response — give it a "none" bucket
                # so all malformed rows still cluster together (not singleton).
                out[i] = hashlib.blake2b(b"VERB::__none__", digest_size=8).hexdigest()
            continue

        if mode == "action_tag":
            # Decode the response and try to pull out the first <action>...</action>.
            # skip_special_tokens=True is fine: <action> is not a special token,
            # and stripping pads/eos here only affects the surrounding text.
            text = tokenizer.decode(valid.tolist(), skip_special_tokens=True)
            m = _get_tag_re(action_tag_name).search(text)
            if m is not None:
                body = m.group(1).strip()
                n_action_match += 1
                key_bytes = ("ACT::" + body).encode("utf-8")
                out[i] = hashlib.blake2b(key_bytes, digest_size=8).hexdigest()
                continue
            if not fallback_to_first_n:
                # Stable but distinct key per malformed row → each lives in its
                # own action bucket, falling out of CACB peer set.
                out[i] = hashlib.blake2b(f"NOACT::{i}".encode(), digest_size=8).hexdigest()
                continue
            # else: fall through to first_n branch below

        head = valid[:n_first_tokens].tobytes()
        out[i] = hashlib.blake2b(head, digest_size=8).hexdigest()

    if mode in ("action_tag", "tag_verb"):
        compute_action_keys.last_action_match_rate = n_action_match / max(bsz, 1)
    return out


def step_norm_reward_cacb(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    cluster_index: np.ndarray,
    action_keys: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
    estimator: str = "counterfactual",
) -> torch.Tensor:
    """
    Compute step-level advantage with action-conditional baseline (CACB / PACE).

    Two estimator variants:
      * ``"counterfactual"`` (diff-peer, original): for row i in cluster C,
        baseline = mean of cluster members whose action_key DIFFERS from i.
        Falls back to RLOO if all peers share i's action; A=0 for size-1
        clusters.
      * ``"q_style"`` (default for BiPACE): for row i with action a in cluster C,
            Q_hat(s, a) = mean of same-action peers' returns
            V_hat(s)    = mean of ALL members' returns
            A_i         = Q_hat(s, a) - V_hat(s)
        When everyone in C took the same action, A_i=0 (no in-cluster action
        contrast → no step-level signal; gradient flows from steps where
        actions actually differed).

    Args:
        step_rewards : (bsz,) discounted return-to-go per step.
        response_mask : (bsz, response_length) used to broadcast the scalar
            advantage onto response tokens.
        cluster_index : (bsz,) cluster UUID per row (from BiGPO or GiGPO).
        action_keys : (bsz,) action identity per row (from compute_action_keys).
        epsilon : numerical floor on σ.
        remove_std : if True (mean_norm mode), do not divide by cluster σ.
        estimator : ``"counterfactual"`` or ``"q_style"``.

    Returns:
        advantages : (bsz, response_length) tensor on step_rewards.device.
    """
    if estimator not in ("counterfactual", "q_style"):
        raise ValueError(f"unknown estimator={estimator!r}; expected 'counterfactual' or 'q_style'")

    bsz = step_rewards.shape[0]
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    # Build per-cluster member lists.
    cluster_to_idxs: dict = defaultdict(list)
    for i in range(bsz):
        cluster_to_idxs[cluster_index[i]].append(i)

    new_scores = torch.zeros_like(scores)
    n_singleton_rows = 0
    n_uniform_action = 0
    n_cacb = 0
    n_clusters_total = 0
    n_clusters_multi_action = 0
    sum_unique_actions_per_cluster = 0

    with torch.no_grad():
        for cid, members in cluster_to_idxs.items():
            if not members:
                continue
            R = scores[members]
            if len(members) == 1:
                sigma = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                sigma = R.float().std(unbiased=False).to(scores.dtype)

            keys = action_keys[members]
            unique_keys = set(keys.tolist() if hasattr(keys, "tolist") else list(keys))
            n_clusters_total += 1
            sum_unique_actions_per_cluster += len(unique_keys)
            if len(unique_keys) >= 2:
                n_clusters_multi_action += 1

            if estimator == "q_style" and len(members) > 1:
                key_to_idxs: dict = defaultdict(list)
                for j, m in enumerate(members):
                    key_to_idxs[keys[j]].append(m)
                key_to_R = {k: scores[idxs].float().mean().to(scores.dtype) for k, idxs in key_to_idxs.items()}
                V_C = R.float().mean().to(scores.dtype)
                for li, gi in enumerate(members):
                    if len(unique_keys) == 1:
                        delta = torch.zeros((), device=scores.device, dtype=scores.dtype)
                        n_uniform_action += 1
                    else:
                        delta = key_to_R[keys[li]] - V_C
                        n_cacb += 1
                    new_scores[gi] = delta if remove_std else delta / (sigma + epsilon)
                continue

            # counterfactual (diff-peer) path
            for li, gi in enumerate(members):
                same = [members[j] for j, k in enumerate(keys) if k == keys[li]]
                diff = [members[j] for j, k in enumerate(keys) if k != keys[li]]
                if len(diff) > 0:
                    b = scores[diff].float().mean().to(scores.dtype)
                    n_cacb += 1
                elif len(same) > 1:
                    others = [m for m in same if m != gi]
                    b = scores[others].float().mean().to(scores.dtype)
                    n_uniform_action += 1
                else:
                    b = scores[gi]
                    n_singleton_rows += 1
                new_scores[gi] = (scores[gi] - b) if remove_std else (scores[gi] - b) / (sigma + epsilon)

        advantages = new_scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    if os.environ.get("CACB_VERBOSE", "").strip() not in ("", "0", "false", "False"):
        n_total = bsz
        nc = max(n_clusters_total, 1)
        avg_unique = sum_unique_actions_per_cluster / nc
        multi_pct = n_clusters_multi_action / nc
        print(
            f"CACB | est={estimator} rows={n_total} cacb={n_cacb} ({n_cacb / max(n_total, 1):.1%}) "
            f"rloo_or_zero={n_uniform_action} ({n_uniform_action / max(n_total, 1):.1%}) "
            f"singleton_rows={n_singleton_rows} | "
            f"clusters={n_clusters_total} multi_action={n_clusters_multi_action} ({multi_pct:.1%}) "
            f"avg_unique_actions/cluster={avg_unique:.2f}"
        )

    step_norm_reward_cacb.last_stats = {
        "estimator": estimator,
        "rows": bsz,
        "rows_cacb": n_cacb,
        "rows_rloo_or_zero": n_uniform_action,
        "rows_singleton": n_singleton_rows,
        "n_clusters": n_clusters_total,
        "n_multi_action_clusters": n_clusters_multi_action,
        "avg_unique_actions_per_cluster": sum_unique_actions_per_cluster / max(n_clusters_total, 1),
        "multi_action_frac": n_clusters_multi_action / max(n_clusters_total, 1),
    }

    return advantages
