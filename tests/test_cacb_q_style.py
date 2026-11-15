"""
Unit + property tests for the new q_style CACB estimator.

Run:
  python3 tests/test_cacb_q_style.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gigpo.core_cacb import step_norm_reward_cacb


def _broadcast_advantage_per_row(adv_full):
    # advantages tensor is (bsz, response_length); same value broadcast.
    return adv_full[:, 0]


def test_q_style_uniform_action_zero():
    """Cluster with all-same action under q_style → A=0 for every row."""
    R = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = torch.ones((4, 1))
    cidx = np.array(["c0"] * 4, dtype=object)
    keys = np.array(["a"] * 4, dtype=object)
    out = step_norm_reward_cacb(R, mask, cidx, keys,
                                remove_std=True, estimator="q_style")
    adv = _broadcast_advantage_per_row(out)
    assert torch.allclose(adv, torch.zeros(4)), (
        f"uniform-action cluster should give zero advantages, got {adv}")
    print("[OK] q_style uniform-action → zero")


def test_q_style_two_actions_balanced():
    """[a,a,b,b] with R=[1,3,5,7] → R_a=2, R_b=6, V=4
       a-rows get -2, b-rows get +2."""
    R = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mask = torch.ones((4, 1))
    cidx = np.array(["c0"] * 4, dtype=object)
    keys = np.array(["a", "a", "b", "b"], dtype=object)
    out = step_norm_reward_cacb(R, mask, cidx, keys,
                                remove_std=True, estimator="q_style")
    adv = _broadcast_advantage_per_row(out)
    expected = torch.tensor([-2.0, -2.0, 2.0, 2.0])
    assert torch.allclose(adv, expected), f"got {adv}, expected {expected}"
    print(f"[OK] q_style balanced 2-action → A_a={-2}, A_b={2}")


def test_q_style_solo_action_falls_back_to_bigpo():
    """[a,a,a,b] with R=[2,4,6,8] → R_a=4, R_b=8, V=5
       a-rows: 4-5 = -1, b-row: 8-5 = +3 (note the b-row's R_a=R_b=R_i)."""
    R = torch.tensor([2.0, 4.0, 6.0, 8.0])
    mask = torch.ones((4, 1))
    cidx = np.array(["c0"] * 4, dtype=object)
    keys = np.array(["a", "a", "a", "b"], dtype=object)
    out = step_norm_reward_cacb(R, mask, cidx, keys,
                                remove_std=True, estimator="q_style")
    adv = _broadcast_advantage_per_row(out)
    expected = torch.tensor([-1.0, -1.0, -1.0, 3.0])
    assert torch.allclose(adv, expected), f"got {adv}, expected {expected}"
    print(f"[OK] q_style solo-action → A_a=-1, A_b=+3")


def test_q_style_singleton_cluster_zero():
    """Singleton cluster: A=0 (no peers, no signal)."""
    R = torch.tensor([5.0])
    mask = torch.ones((1, 1))
    cidx = np.array(["c0"], dtype=object)
    keys = np.array(["a"], dtype=object)
    out = step_norm_reward_cacb(R, mask, cidx, keys,
                                remove_std=True, estimator="q_style")
    assert out[0, 0].item() == 0.0
    print("[OK] q_style singleton → 0")


def test_counterfactual_unchanged():
    """Legacy estimator must be byte-identical on the same input."""
    R = torch.tensor([1.0, 3.0, 5.0, 7.0])
    mask = torch.ones((4, 1))
    cidx = np.array(["c0"] * 4, dtype=object)
    keys = np.array(["a", "a", "b", "b"], dtype=object)
    out = step_norm_reward_cacb(R, mask, cidx, keys,
                                remove_std=True, estimator="counterfactual")
    adv = _broadcast_advantage_per_row(out)
    # legacy: each row's A = R_i - mean(R of different-action peers)
    # row 0 (a): 1 - mean(5,7)=6 → -5
    # row 1 (a): 3 - 6 → -3
    # row 2 (b): 5 - mean(1,3)=2 → 3
    # row 3 (b): 7 - 2 → 5
    expected = torch.tensor([-5.0, -3.0, 3.0, 5.0])
    assert torch.allclose(adv, expected), f"got {adv}, expected {expected}"
    print(f"[OK] counterfactual estimator preserved")


def test_diagnostics_recorded():
    """The diagnostic dict on the function must reflect the cluster shape."""
    R = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.ones((5, 1))
    cidx = np.array(["c0", "c0", "c0", "c1", "c1"], dtype=object)
    keys = np.array(["a", "a", "b", "x", "y"], dtype=object)
    step_norm_reward_cacb(R, mask, cidx, keys,
                          remove_std=True, estimator="q_style")
    s = step_norm_reward_cacb.last_stats
    assert s["n_clusters"] == 2, s
    assert s["n_multi_action_clusters"] == 2, s
    assert abs(s["avg_unique_actions_per_cluster"] - 2.0) < 1e-6, s
    print(f"[OK] diagnostics: {s}")


if __name__ == "__main__":
    test_q_style_uniform_action_zero()
    test_q_style_two_actions_balanced()
    test_q_style_solo_action_falls_back_to_bigpo()
    test_q_style_singleton_cluster_zero()
    test_counterfactual_unchanged()
    test_diagnostics_recorded()
    print("\nALL Q-STYLE TESTS PASSED")
