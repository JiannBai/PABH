"""
BiGPO sanity tests.

Run from repo root:
    python tests/test_bigpo.py

Tests:
  1. Regression — IdentityHashEmbedder + eps=0 reproduces GiGPO's exact-match
     clustering (same partition, modulo UUID labels).
  2. Paraphrase robustness — semantically-similar text observations that
     GiGPO splits into singletons get clustered together by BiGPO under
     HashNgramEmbedder.
  3. Multimodal stand-in — bytes/ndarray-shaped observations cluster by
     content rather than crashing.
"""

import os
import sys

import numpy as np

# Allow `python tests/test_bigpo.py` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigpo.core_bigpo import (  # noqa: E402
    ActorHiddenEmbedder,
    HashNgramEmbedder,
    IdentityHashEmbedder,
    _to_text,
    build_step_group_bisim,
)


def gigpo_exact_match_partition(obs, index):
    """Inline equivalent of gigpo.core_gigpo.build_step_group(enable_similarity=False).

    Reimplemented locally so this test does not pull in ray/torch/verl deps.
    Verified to match the reference logic at core_gigpo.py L278-296.
    """
    import uuid as _uuid
    from collections import defaultdict

    out = np.empty(len(obs), dtype=object)
    for idx in np.unique(index):
        locs = np.where(index == idx)[0]
        clusters = defaultdict(list)
        for loc in locs:
            clusters[_to_text(obs[loc])].append(loc)
        for _, members in clusters.items():
            uid = str(_uuid.uuid4())
            for m in members:
                out[m] = uid
    return out


def _partition(uids):
    """Convert a UUID array into a canonical partition (frozenset of frozensets of indices)."""
    by_uid = {}
    for i, u in enumerate(uids):
        by_uid.setdefault(u, []).append(i)
    return frozenset(frozenset(v) for v in by_uid.values())


def test_regression_to_gigpo():
    obs = np.array(
        [
            "kitchen: a knife on the counter",
            "kitchen: a knife on the counter",  # exact dup
            "garden: a watering can",
            "kitchen: a knife on the counter",
            "garden: a watering can",
        ],
        dtype=object,
    )
    index = np.array(["P1", "P1", "P1", "P1", "P1"])

    gigpo_uids = gigpo_exact_match_partition(obs, index)
    bigpo_uids = build_step_group_bisim(
        obs, index, embedder=IdentityHashEmbedder(), eps=0.0
    )

    assert _partition(gigpo_uids) == _partition(bigpo_uids), (
        f"Regression failed.\n  GiGPO partition: {_partition(gigpo_uids)}\n"
        f"  BiGPO partition: {_partition(bigpo_uids)}"
    )
    print("[PASS] regression_to_gigpo: identity+eps=0 reproduces GiGPO exact partition")


def test_paraphrase_robustness():
    # Five paraphrases of the *same* underlying state. GiGPO will see five
    # distinct strings and produce five singletons; BiGPO should fold them.
    obs = np.array(
        [
            "you are in the kitchen, you see a red apple on the table",
            "You are in the kitchen. You see a red apple on the table.",
            "you're in the kitchen and there's a red apple on the table",
            "kitchen: red apple on table",
            "in the kitchen, a red apple sits on the table",
            # Two true outliers from a different state:
            "you are in the garage, you see a wrench on the floor",
            "you are in the garage, you see a wrench on the floor",
        ],
        dtype=object,
    )
    index = np.array(["P1"] * len(obs))

    gigpo_uids = gigpo_exact_match_partition(obs, index)
    bigpo_uids = build_step_group_bisim(
        obs, index, embedder=HashNgramEmbedder(dim=2048, ngram=3), eps=0.45
    )

    g_part = _partition(gigpo_uids)
    b_part = _partition(bigpo_uids)
    print(f"  GiGPO #clusters={len(g_part)}    BiGPO #clusters={len(b_part)}")
    print(f"  GiGPO partition: {sorted(map(sorted, g_part))}")
    print(f"  BiGPO partition: {sorted(map(sorted, b_part))}")

    # GiGPO: 6 clusters (5 singletons + 1 pair from the two identical garage strings).
    assert len(g_part) == 6, f"expected GiGPO=6 clusters, got {len(g_part)}"
    # BiGPO: should produce strictly fewer clusters by folding the kitchen paraphrases.
    assert len(b_part) < len(g_part), (
        f"BiGPO failed to merge paraphrases: GiGPO={len(g_part)} clusters, BiGPO={len(b_part)}"
    )
    print("[PASS] paraphrase_robustness: BiGPO merged paraphrased states GiGPO kept singleton")


def test_multimodal_stand_in():
    # Pretend ndarray observations are vision-encoder outputs.
    rng = np.random.default_rng(0)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    obs = np.empty(4, dtype=object)
    obs[0] = a
    obs[1] = a.copy()
    obs[2] = b
    obs[3] = b.copy()
    index = np.array(["P1"] * 4)

    bigpo_uids = build_step_group_bisim(
        obs, index, embedder=IdentityHashEmbedder(), eps=0.0
    )
    part = _partition(bigpo_uids)
    assert part == frozenset(
        {frozenset({0, 1}), frozenset({2, 3})}
    ), f"multimodal id-hash failed: {part}"
    print("[PASS] multimodal_stand_in: ndarray obs cluster by content (id hash)")


def test_actor_hidden_embedder_precomputed():
    """ActorHiddenEmbedder simulates the worker handing back embeddings already
    aligned to anchor_obs order. build_step_group_bisim must consume the
    parallel array via the `embeddings=` kwarg, irrespective of group iter order."""
    rng = np.random.default_rng(42)
    # 2 distinct semantic states; each produced by 4 trajectories under the same prompt.
    state_a = rng.standard_normal(64).astype(np.float32)
    state_b = rng.standard_normal(64).astype(np.float32)

    obs = np.array([f"step{i}" for i in range(8)], dtype=object)
    index = np.array(["P1"] * 8)
    # Tightly cluster around 2 centroids (low noise).
    embs = np.stack(
        [state_a + 0.01 * rng.standard_normal(64) for _ in range(4)]
        + [state_b + 0.01 * rng.standard_normal(64) for _ in range(4)],
        axis=0,
    ).astype(np.float32)

    e = ActorHiddenEmbedder(embs)
    uids = build_step_group_bisim(obs, index, eps=0.10, embeddings=e.embeddings)
    part = _partition(uids)
    assert part == frozenset({frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7})}), (
        f"ActorHiddenEmbedder precomputed path failed: {part}"
    )
    print("[PASS] actor_hidden_embedder_precomputed: 2 semantic clusters recovered exactly")


if __name__ == "__main__":
    test_regression_to_gigpo()
    test_paraphrase_robustness()
    test_multimodal_stand_in()
    test_actor_hidden_embedder_precomputed()
    print("\nAll BiGPO sanity tests passed.")
