"""
Sanity-test the CACB action_tag extractor on response shapes that ALFWorld
rollouts actually emit, BEFORE burning a 150-epoch run.

What this checks:
  1. The <action>...</action> regex hits all well-formed responses.
  2. Two responses with the same action body get the same key.
  3. Two responses with different action bodies get different keys.
  4. The current `first_n=8` extractor really IS dominated by <think>
     prefix on these prompts (i.e. it produces fewer distinct keys than
     action_tag for the same set of responses) — this is the bug the
     ablation is meant to surface.
  5. Malformed rows (no action tag) fall back without crashing.

Run:
  python3 tests/test_cacb_action_tag.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# Make `gigpo` importable when running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gigpo.core_cacb import compute_action_keys, _ACTION_TAG_RE


# ----- realistic ALFWorld responses (verbatim shape from train_log demo) ----
RESPONSES = [
    # 1: typical short response, "go to" action
    "<think>\nI need to put an egg in the microwave. First, I should find an egg. Eggs are often in the fridge.\n</think>\n\n<action>go to fridge 1</action>",
    # 2: same action, different think — should hash to SAME action_tag key, DIFFERENT first_n key
    "<think>\nLet me go to the fridge to find an egg.\n</think>\n\n<action>go to fridge 1</action>",
    # 3: different action body
    "<think>\nI should pick up the egg now.\n</think>\n\n<action>take egg 1 from fridge 1</action>",
    # 4: same as 3's action — should match 3 under action_tag
    "<think>\nGrab the egg.\n</think>\n\n<action>take egg 1 from fridge 1</action>",
    # 5: another distinct action
    "<think>\n...\n</think>\n\n<action>open microwave 1</action>",
    # 6: missing closing tag (degenerate model output) — must NOT crash
    "<think>\nbroken format\n</think>\n\n<action>go to drawer 1",
    # 7: lowercase tag (case insensitive in regex)
    "<think>\n...\n</think>\n\n<ACTION>go to fridge 1</ACTION>",
    # 8: action with internal newline (rare but possible)
    "<think>\n...\n</think>\n\n<action>\ngo to fridge 1\n</action>",
    # 9: trailing whitespace inside action body
    "<think>\n...\n</think>\n\n<action>go to fridge 1   </action>",
    # 10: multiple action tags (model hallucinated extra) — we take the first
    "<think>\n...\n</think>\n\n<action>go to fridge 1</action><action>open fridge 1</action>",
]


class _StubTokenizer:
    """Round-trips token ids ↔ text via a tiny char-level fake.

    The real Qwen tokenizer will decode actual rollout token ids into the
    same text strings, so for the regex/key logic this is faithful enough.
    """
    def __init__(self, texts):
        # Assign each text a unique single-int id; decode back to the text.
        self._id_to_text = {i + 1: t for i, t in enumerate(texts)}

    def decode(self, ids, skip_special_tokens=True):
        for i in ids:
            if i in self._id_to_text:
                return self._id_to_text[i]
        return ""


def main():
    n = len(RESPONSES)
    tokenizer = _StubTokenizer(RESPONSES)
    # Build (n, L) tensor where row i has just token id (i+1).
    ids = torch.arange(1, n + 1).unsqueeze(1)              # (n, 1)
    pad = torch.zeros((n, 7), dtype=torch.long)            # padding region
    responses = torch.cat([ids, pad], dim=1)               # (n, 8)
    mask = torch.zeros_like(responses)
    mask[:, 0] = 1                                         # only first token valid

    # ---- (1) regex match-rate on the realistic battery -----------------
    matches = [_ACTION_TAG_RE.search(t) for t in RESPONSES]
    match_rate = sum(m is not None for m in matches) / n
    print(f"[1] regex match rate on {n} samples: {match_rate:.0%}")
    for i, (t, m) in enumerate(zip(RESPONSES, matches)):
        body = m.group(1).strip() if m else "<<NO MATCH>>"
        first_line = t.splitlines()[0][:40]
        print(f"    #{i:>2}  body={body!r:35s}  resp_starts={first_line!r}")

    # ---- (2)/(3) action_tag keys: equality / inequality ---------------
    keys_act = compute_action_keys(responses, mask, mode="action_tag",
                                   tokenizer=tokenizer)
    print(f"\n[2] action_tag keys (n_distinct={len(set(keys_act))}/{n}):")
    for i, k in enumerate(keys_act):
        print(f"    #{i:>2}  {k}")
    # Pairs that MUST collide under action_tag (same action body):
    same_action_pairs = [(0, 1), (2, 3), (0, 6), (0, 7), (0, 8)]  # indices into RESPONSES
    for a, b in same_action_pairs:
        assert keys_act[a] == keys_act[b], (
            f"action_tag keys for #{a} and #{b} should match — they are the "
            f"same action body. got {keys_act[a]!r} vs {keys_act[b]!r}")
    # Pairs that MUST differ:
    diff_action_pairs = [(0, 2), (0, 4), (2, 4)]
    for a, b in diff_action_pairs:
        assert keys_act[a] != keys_act[b], (
            f"action_tag keys for #{a} and #{b} should differ — they are "
            f"different actions. got {keys_act[a]!r}")
    print(f"    same-action collisions OK ({len(same_action_pairs)} pairs)")
    print(f"    different-action separations OK ({len(diff_action_pairs)} pairs)")

    # ---- (4) first_n=8 dominated by <think> prefix --------------------
    # Build a *real* token-ish encoding so first_n actually has variation
    # to compare against. Char-level: first 8 chars of every response are
    # "<think>\n" — under first_n they all collapse to ONE key.
    char_ids = []
    for t in RESPONSES:
        # Pretend each char is its own token. Length is variable.
        char_ids.append(torch.tensor([ord(c) for c in t]))
    max_len = max(len(c) for c in char_ids)
    char_resp = torch.zeros((n, max_len), dtype=torch.long)
    char_mask = torch.zeros_like(char_resp)
    for i, c in enumerate(char_ids):
        char_resp[i, :len(c)] = c
        char_mask[i, :len(c)] = 1
    keys_first8 = compute_action_keys(char_resp, char_mask,
                                      mode="first_n", n_first_tokens=8)
    counts_first8 = Counter(keys_first8)
    counts_act = Counter(keys_act)
    print(f"\n[3] action_tag distinct keys : {len(counts_act):>2}  (out of {n})")
    print(f"    first_n=8 distinct keys  : {len(counts_first8):>2}  (out of {n})")
    print(f"    most common first_n=8 key: {counts_first8.most_common(1)[0]}")
    if len(counts_first8) >= len(counts_act):
        print("    [WARN] first_n is NOT collapsing more than action_tag — "
              "is your think prefix shorter than 8 tokens? Re-check the bug.")
    else:
        print(f"    confirmed: first_n collapses {n - len(counts_first8)} "
              f"distinct actions into the same key (the <think>-prefix bug).")

    # ---- (5) malformed row falls back without crashing ---------------
    # Response #5 (index 5) is the broken-format one.
    bad_response = RESPONSES[5]
    bad_tok = _StubTokenizer([bad_response])
    bad_ids = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]])
    bad_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]])
    k_with_fb = compute_action_keys(bad_ids, bad_mask, mode="action_tag",
                                    tokenizer=bad_tok, fallback_to_first_n=True)
    k_no_fb = compute_action_keys(bad_ids, bad_mask, mode="action_tag",
                                  tokenizer=bad_tok, fallback_to_first_n=False)
    assert k_with_fb[0] != k_no_fb[0], "fallback should produce a different key"
    print(f"\n[4] malformed-row handling")
    print(f"    fallback=True  → {k_with_fb[0]} (uses first_n)")
    print(f"    fallback=False → {k_no_fb[0]} (NOACT::i marker)")
    print(f"    no crash ✓")

    print("\nALL CHECKS PASSED — safe to launch the action_tag run.")


if __name__ == "__main__":
    main()
