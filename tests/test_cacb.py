"""
CACB sanity tests — counterfactual action-conditional baseline.

Three scenarios:
  1. Mixed-action cluster: action A returns 10, action B returns 0
     → A's CACB baseline = mean(B) = 0  → A advantage = +10
     → B's CACB baseline = mean(A) = 10 → B advantage = -10
     (vs vanilla GiGPO mean baseline = 5 → A=+5, B=-5; CACB doubles the gap)
  2. All-same-action cluster: leave-one-out RLOO fallback.
  3. Singleton cluster: zero advantage (preserved from GiGPO).

Run:
    python tests/test_cacb.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigpo.core_cacb import compute_action_keys, step_norm_reward_cacb


def test_mixed_action_cluster():
    # 6 rows, single cluster, 3 took action A (returns 10), 3 took action B (returns 0).
    bsz, T = 6, 4
    step_rewards = torch.tensor([10., 10., 10., 0., 0., 0.])
    response_mask = torch.ones(bsz, T)
    cluster_index = np.array(['C0'] * 6, dtype=object)
    action_keys = np.array(['A', 'A', 'A', 'B', 'B', 'B'], dtype=object)

    # mean_norm (no σ divide).
    A = step_norm_reward_cacb(step_rewards, response_mask, cluster_index,
                              action_keys, remove_std=True)
    # Each row's per-token advantage.
    per_row = A[:, 0].tolist()
    expected = [10. - 0., 10. - 0., 10. - 0., 0. - 10., 0. - 10., 0. - 10.]
    assert all(abs(a - e) < 1e-5 for a, e in zip(per_row, expected)), (
        f"mixed-action: got {per_row}, expected {expected}")
    print(f"[PASS] mixed_action_cluster: A→+10, B→-10 (vanilla GiGPO would give ±5)")


def test_all_same_action_rloo_fallback():
    # 4 rows, single cluster, all action A, returns [4, 6, 8, 2].
    # leave-one-out baselines: row 0 → mean(6,8,2)=16/3, advantage 4 - 16/3 = -4/3
    # row 1 → mean(4,8,2)=14/3, advantage 6 - 14/3 = +4/3
    # row 2 → mean(4,6,2)=12/3=4,  advantage 8 - 4 = +4
    # row 3 → mean(4,6,8)=18/3=6,  advantage 2 - 6 = -4
    bsz, T = 4, 4
    step_rewards = torch.tensor([4., 6., 8., 2.])
    response_mask = torch.ones(bsz, T)
    cluster_index = np.array(['C0'] * 4, dtype=object)
    action_keys = np.array(['A'] * 4, dtype=object)

    A = step_norm_reward_cacb(step_rewards, response_mask, cluster_index,
                              action_keys, remove_std=True)
    per_row = A[:, 0].tolist()
    expected = [4. - 16./3, 6. - 14./3, 8. - 4., 2. - 6.]
    assert all(abs(a - e) < 1e-5 for a, e in zip(per_row, expected)), (
        f"rloo: got {per_row}, expected {expected}")
    print(f"[PASS] all_same_action_rloo_fallback: leave-one-out baseline kicks in")


def test_singleton_zero_advantage():
    bsz, T = 1, 4
    step_rewards = torch.tensor([7.])
    response_mask = torch.ones(bsz, T)
    cluster_index = np.array(['C0'], dtype=object)
    action_keys = np.array(['solo'], dtype=object)

    A = step_norm_reward_cacb(step_rewards, response_mask, cluster_index,
                              action_keys, remove_std=True)
    assert torch.allclose(A, torch.zeros_like(A)), f"singleton: got {A}"
    print(f"[PASS] singleton_zero_advantage: preserves GiGPO's null behavior")


def test_compute_action_keys_dedup():
    # Two responses with identical first 4 tokens should hash identically.
    responses = torch.tensor([
        [101, 202, 303, 404, 999, 999],
        [101, 202, 303, 404, 555, 555],
        [101, 202, 303, 999, 999, 999],   # different at pos 3
    ], dtype=torch.long)
    mask = torch.tensor([
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0],
    ], dtype=torch.long)
    keys = compute_action_keys(responses, mask, n_first_tokens=4)
    assert keys[0] == keys[1], f"first 4 tokens identical → keys should match: {keys}"
    assert keys[0] != keys[2], f"different at pos 3 → keys should differ: {keys}"
    print(f"[PASS] compute_action_keys_dedup: same prefix → same key, different prefix → different key")


def test_mixed_action_with_std():
    # remove_std=False: divide by σ_C.  Returns [10, 10, 0, 0]; σ=5; σ unbiased=False
    # actions: A, A, B, B. CACB: A→10-0=10, B→0-10=-10. Divide by 5 → ±2.
    bsz, T = 4, 2
    step_rewards = torch.tensor([10., 10., 0., 0.])
    response_mask = torch.ones(bsz, T)
    cluster_index = np.array(['C0'] * 4, dtype=object)
    action_keys = np.array(['A', 'A', 'B', 'B'], dtype=object)

    A = step_norm_reward_cacb(step_rewards, response_mask, cluster_index,
                              action_keys, remove_std=False)
    per_row = A[:, 0].tolist()
    expected = [+2., +2., -2., -2.]
    assert all(abs(a - e) < 1e-4 for a, e in zip(per_row, expected)), (
        f"with_std: got {per_row}, expected {expected}")
    print(f"[PASS] mixed_action_with_std: σ divisor applies correctly")


if __name__ == "__main__":
    test_mixed_action_cluster()
    test_all_same_action_rloo_fallback()
    test_singleton_zero_advantage()
    test_compute_action_keys_dedup()
    test_mixed_action_with_std()
    print("\nAll CACB sanity tests passed.")
