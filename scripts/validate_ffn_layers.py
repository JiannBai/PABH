"""
P0 (4) — FFN-layer validation for BiGPO.

Standalone, single-GPU, no Ray / no FSDP. Loads the actor model in eval
mode, forwards a synthetic ALFWorld audit set covering all 6 task
families, captures hidden states from every layer at multiple pooling
positions, and reports three things the v2 brainstorm marked as P0:

  V1. Clustering quality per (layer, pool, eps): n_clusters,
      singleton_frac, mean intra-cluster cohesion.
  V2. Pooling ablation: last_token vs mean_prompt vs attention_weighted.
  V3. Probing: linear-probe task-id accuracy from frozen hidden states
      (5-fold CV). Higher = the layer encodes task-relevant structure.

V4 (drift) and V5 (d_pi calibration) need training-time data — those
will land in a follow-up. This single offline run answers the central
question: *is last-FFN-layer hidden state a valid fingerprint at all,
and is it the best layer to use?*

Usage:
    python scripts/validate_ffn_layers.py \
        --model-path /path/to/Qwen2.5-7B-Instruct \
        --output-dir ./val_outputs/ffn_layers \
        --layers 0 4 8 12 16 20 24 28 \
        --pools last_token mean_prompt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------- #
# 1. Synthetic audit set: 6 ALFWorld task families × N each. #
# ---------------------------------------------------------- #
_TASK_TEMPLATES = {
    "pick_and_place": [
        "You are in the kitchen. You see a {obj} on the {src}. Your task is to: put the {obj} on the {dst}.",
        "You are in the bedroom. There is a {obj} on the {src}. Your task is to: put a {obj} in/on {dst}.",
        "You see a {obj} on the {src}. Your task is to: place the {obj} on the {dst}.",
    ],
    "pick_clean_then_place_in_recep": [
        "You are in the kitchen. You see a {obj} on the {src}. Your task is to: put a clean {obj} in {dst}.",
        "There is a {obj} in front of you. Your task is to: clean the {obj} and place it on the {dst}.",
        "You see a dirty {obj} on the {src}. Your task is to: wash the {obj} and put it on the {dst}.",
    ],
    "pick_heat_then_place_in_recep": [
        "You are in the kitchen. You see a {obj} on the {src}. Your task is to: put a hot {obj} in {dst}.",
        "There is a cold {obj} on the {src}. Your task is to: heat the {obj} and place it on the {dst}.",
        "You see a {obj} on the {src}. Your task is to: warm up the {obj} and put it in {dst}.",
    ],
    "pick_cool_then_place_in_recep": [
        "You are in the kitchen. You see a {obj} on the {src}. Your task is to: put a cool {obj} in {dst}.",
        "You see a warm {obj} on the {src}. Your task is to: cool the {obj} and place it on the {dst}.",
        "There is a {obj} on the {src}. Your task is to: chill the {obj} and put it in {dst}.",
    ],
    "look_at_obj_in_light": [
        "You are in the bedroom. You see a {obj} on the {src}. Your task is to: examine the {obj} with the desklamp.",
        "There is a {obj} in front of you. Your task is to: look at the {obj} under the floorlamp.",
        "You see a {obj} on the {src}. Your task is to: inspect the {obj} using the lamp.",
    ],
    "pick_two_obj_and_place": [
        "You are in the kitchen. You see two {obj}s. Your task is to: put two {obj}s on the {dst}.",
        "There are two {obj} items. Your task is to: place both {obj}s on the {dst}.",
        "You see two {obj} on the {src}. Your task is to: move both {obj}s to the {dst}.",
    ],
}

_OBJECTS = ["apple", "book", "mug", "knife", "pillow", "pen", "tomato",
            "bowl", "spoon", "lettuce", "potato", "egg", "watch", "soap"]
_SRCS = ["table", "counter", "shelf", "drawer", "cabinet", "sofa",
         "bed", "desk", "fridge", "microwave"]
_DSTS = ["table", "counter", "shelf", "garbagecan", "sidetable",
         "diningtable", "bed", "fridge", "microwave"]


def build_audit_set(n_per_class: int = 40, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    texts: List[str] = []
    labels: List[str] = []
    for cls, templates in _TASK_TEMPLATES.items():
        for _ in range(n_per_class):
            tmpl = templates[rng.integers(len(templates))]
            obj = _OBJECTS[rng.integers(len(_OBJECTS))]
            src = _SRCS[rng.integers(len(_SRCS))]
            dst = _DSTS[rng.integers(len(_DSTS))]
            texts.append(tmpl.format(obj=obj, src=src, dst=dst))
            labels.append(cls)
    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm], [labels[i] for i in perm]


# ---------------------------------------------------------- #
# Real ALFWorld obs loader (from qualitative_*.txt dumps).   #
# ---------------------------------------------------------- #
_TASK_PATTERNS = [
    (re.compile(r"clean|\bwash\b", re.I),                          "pick_clean_then_place_in_recep"),
    (re.compile(r"\bhot\b|\bheat\b|warm", re.I),                   "pick_heat_then_place_in_recep"),
    (re.compile(r"\bcool\b|chill|\bcold\b", re.I),                 "pick_cool_then_place_in_recep"),
    (re.compile(r"examine|look at|inspect|desklamp|floorlamp", re.I), "look_at_obj_in_light"),
    (re.compile(r"\btwo\b", re.I),                                 "pick_two_obj_and_place"),
    (re.compile(r"put|place|move", re.I),                          "pick_and_place"),
]


def _label_from_obs(text: str) -> str:
    for pat, cls in _TASK_PATTERNS:
        if pat.search(text):
            return cls
    return "unknown"


def load_qualitative_obs(qual_dir: str, max_obs: int = 240
                         ) -> Tuple[List[str], List[str]]:
    """Load real anchor_obs strings dumped by gigpo.bigpo_logger.

    Each qualitative_<step>.txt file contains lines of the form
    ``  [<idx>] <obs string>`` interleaved with cluster headers we ignore.
    Task labels are inferred heuristically by regex over the obs text.
    """
    qual = Path(qual_dir)
    if qual.is_file():
        files = [qual]
    else:
        files = sorted(qual.glob("qualitative_*.txt"))
    if not files:
        raise FileNotFoundError(f"No qualitative_*.txt under {qual_dir}")

    line_re = re.compile(r"^\s*\[\d+\]\s+(.*)$")
    texts: List[str] = []
    seen: set = set()
    for f in files:
        for ln in f.read_text(errors="replace").splitlines():
            m = line_re.match(ln)
            if not m:
                continue
            t = m.group(1).strip()
            if t and t not in seen:
                seen.add(t)
                texts.append(t)
                if len(texts) >= max_obs:
                    break
        if len(texts) >= max_obs:
            break
    labels = [_label_from_obs(t) for t in texts]
    return texts, labels


# ---------------------------------------------------------- #
# 2. Embedding extraction (multi-layer, multi-pool).         #
# ---------------------------------------------------------- #
@dataclass
class EmbedConfig:
    layers: List[int]
    pools: List[str]                 # last_token / mean_prompt / attn_weighted
    max_len: int = 512
    micro_bsz: int = 8


def _pool(hidden: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """hidden: (B, T, D), mask: (B, T) → (B, D).

    Implementation note: model runs in bf16 on H100, so we cast everything
    that participates in arithmetic (mask, intermediates) to ``hidden.dtype``
    to keep einsum / matmul happy. The final L2-normalize happens in fp32
    upstream.
    """
    dt = hidden.dtype
    if mode == "last_token":
        T = mask.shape[1]
        rev = mask.flip(dims=[1]).to(torch.int64)
        last_from_right = rev.argmax(dim=1)
        last_pos = (T - 1 - last_from_right).clamp(min=0)
        idx = last_pos.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        return hidden.gather(1, idx).squeeze(1)
    if mode == "mean_prompt":
        m = mask.unsqueeze(-1).to(dt)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
    if mode == "attn_weighted":
        # Self-similarity attention: each token's weight ∝ its similarity to
        # the cls=mean(hidden). Cheap, no extra params.
        m = mask.unsqueeze(-1).to(dt)
        cls = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)        # (B, D), dt
        sims = torch.einsum("btd,bd->bt", hidden, cls)             # all dt
        sims = sims.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(sims.float(), dim=1).to(dt).unsqueeze(-1)
        return (hidden * w).sum(1)
    raise ValueError(f"Unknown pool: {mode}")


def extract_embeddings(model, tokenizer, texts: List[str], cfg: EmbedConfig,
                       device: torch.device) -> dict:
    """Returns dict[(layer, pool)] -> (N, D) numpy float32 (L2-normalized)."""
    n_layers_total = model.config.num_hidden_layers
    layers = [n_layers_total + l if l < 0 else l for l in cfg.layers]
    out: dict = {(l, p): [] for l in layers for p in cfg.pools}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), cfg.micro_bsz):
            chunk = texts[start: start + cfg.micro_bsz]
            enc = tokenizer(chunk, padding="max_length", truncation=True,
                            max_length=cfg.max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            res = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        output_hidden_states=True, use_cache=False)
            # res.hidden_states is tuple of (n_layers + 1) tensors:
            # [embedding, layer1_out, ..., layerN_out].
            for li in layers:
                h = res.hidden_states[li + 1]      # +1 to skip embedding
                for pool in cfg.pools:
                    v = _pool(h, enc["attention_mask"], pool).float()
                    v = F.normalize(v, dim=-1)
                    out[(li, pool)].append(v.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ---------------------------------------------------------- #
# 3. Analyses.                                               #
# ---------------------------------------------------------- #
def greedy_cluster(vecs: np.ndarray, eps: float) -> List[List[int]]:
    n = vecs.shape[0]
    if n == 0:
        return []
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    v = vecs / np.where(norms > 0, norms, 1.0)
    cents: List[np.ndarray] = []
    members: List[List[int]] = []
    for i in range(n):
        x = v[i]
        if not cents:
            cents.append(x.copy()); members.append([i]); continue
        sims = np.stack(cents) @ x
        k = int(sims.argmax())
        if 1.0 - float(sims[k]) <= eps:
            members[k].append(i)
            m = len(members[k])
            cents[k] = cents[k] + (x - cents[k]) / m
            cn = np.linalg.norm(cents[k])
            if cn > 0:
                cents[k] /= cn
        else:
            cents.append(x.copy()); members.append([i])
    return members


def cluster_metrics(vecs: np.ndarray, members: List[List[int]]) -> dict:
    """Singleton fraction, mean intra-cluster cohesion (avg pairwise cosine
    within cluster), purity if labels supplied externally (not here)."""
    sizes = np.array([len(m) for m in members])
    cohesions = []
    for m in members:
        if len(m) < 2:
            continue
        sub = vecs[m]
        sub /= np.linalg.norm(sub, axis=1, keepdims=True).clip(min=1e-9)
        sim = sub @ sub.T
        n = sub.shape[0]
        # mean pairwise (excluding diagonal)
        cohesions.append(float((sim.sum() - n) / (n * (n - 1))))
    return {
        "n_clusters": int(sizes.size),
        "singleton_frac": float(np.mean(sizes == 1)) if sizes.size else float("nan"),
        "median_size": float(np.median(sizes)) if sizes.size else float("nan"),
        "mean_cohesion": float(np.mean(cohesions)) if cohesions else float("nan"),
    }


def linear_probe_acc(X: np.ndarray, y: List[str], n_folds: int = 5,
                     seed: int = 0) -> Tuple[float, float]:
    """5-fold CV mean / std accuracy of a multinomial logreg probe."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as e:
        raise RuntimeError(
            "scikit-learn is required for V3 probe analysis. "
            "Install it with: pip install scikit-learn"
        ) from e

    le = {c: i for i, c in enumerate(sorted(set(y)))}
    yi = np.array([le[c] for c in y])
    # Drop classes with too few members for stratified k-fold.
    from collections import Counter
    cnt = Counter(yi.tolist())
    keep = np.array([c for c in yi for _ in [0] if cnt[c] >= n_folds])
    if len(keep) < len(yi):
        keep_mask = np.array([cnt[c] >= n_folds for c in yi])
        X = X[keep_mask]
        yi = yi[keep_mask]
        if len(set(yi)) < 2:
            raise RuntimeError(
                f"Probe needs ≥2 classes with ≥{n_folds} members each; "
                f"after filtering have {len(set(yi))} classes."
            )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in skf.split(X, yi):
            clf = LogisticRegression(max_iter=2000, solver="lbfgs",
                                     random_state=seed, C=1.0)
            clf.fit(X[tr], yi[tr])
            accs.append(clf.score(X[te], yi[te]))
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------- #
# 4. Driver.                                                 #
# ---------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", default="./val_outputs/ffn_layers")
    p.add_argument("--layers", nargs="+", type=int,
                   default=[0, 4, 8, 12, 16, 20, 24, 27])
    p.add_argument("--pools", nargs="+",
                   default=["last_token", "mean_prompt", "attn_weighted"])
    p.add_argument("--eps-grid", nargs="+", type=float,
                   default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--n-per-class", type=int, default=40)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--micro-bsz", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--obs-source", default="synthetic",
                   help="'synthetic' (default), or path to a directory of "
                        "qualitative_*.txt dumps or a single such file. "
                        "Real-obs path infers task labels heuristically.")
    p.add_argument("--max-obs", type=int, default=240,
                   help="Cap on number of obs loaded from qualitative dumps.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] outputs → {out_dir}")

    print(f"[info] loading model from {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    n_layers = model.config.num_hidden_layers
    layers = [n_layers + l if l < 0 else l for l in args.layers]
    print(f"[info] model has {n_layers} layers; analyzing {layers}")

    if args.obs_source == "synthetic":
        texts, labels = build_audit_set(n_per_class=args.n_per_class)
        src_tag = "synthetic"
    else:
        texts, labels = load_qualitative_obs(args.obs_source,
                                             max_obs=args.max_obs)
        src_tag = f"qualitative({args.obs_source})"
    print(f"[info] audit set ({src_tag}): {len(texts)} obs across "
          f"{len(set(labels))} task families: {sorted(set(labels))}")

    cfg = EmbedConfig(layers=args.layers, pools=args.pools,
                      max_len=args.max_len, micro_bsz=args.micro_bsz)
    device = next(model.parameters()).device
    embs = extract_embeddings(model, tokenizer, texts, cfg, device)

    rows = []
    for (layer, pool), X in embs.items():
        print(f"\n[layer={layer:>3}  pool={pool}]")
        try:
            probe_mean, probe_std = linear_probe_acc(X, labels)
            print(f"  V3 probe acc: {probe_mean*100:.2f}% ± {probe_std*100:.2f}%")
        except Exception as e:
            probe_mean, probe_std = float("nan"), float("nan")
            print(f"  V3 probe failed: {e}")
        for eps in args.eps_grid:
            members = greedy_cluster(X, eps)
            m = cluster_metrics(X, members)
            print(f"  V1 eps={eps:.2f}: n={m['n_clusters']:>3} "
                  f"singleton={m['singleton_frac']*100:5.1f}% "
                  f"med_size={m['median_size']:.0f} "
                  f"cohesion={m['mean_cohesion']:.3f}")
            rows.append({
                "layer": layer, "pool": pool, "eps": eps,
                "n_clusters": m["n_clusters"],
                "singleton_frac": m["singleton_frac"],
                "median_size": m["median_size"],
                "mean_cohesion": m["mean_cohesion"],
                "probe_acc_mean": probe_mean,
                "probe_acc_std": probe_std,
            })

    csv_path = out_dir / "ffn_layer_validation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] CSV → {csv_path}")

    # Summary plain-text table by layer (eps=0.10, attn_weighted as default).
    sel = [r for r in rows
           if abs(r["eps"] - 0.10) < 1e-9 and r["pool"] == args.pools[0]]
    if sel:
        print("\nLayer summary (eps=0.10, pool=" + args.pools[0] + "):")
        print(f"  {'layer':>5} {'singleton%':>10} {'med_size':>9} "
              f"{'cohesion':>9} {'probe_acc%':>11}")
        for r in sorted(sel, key=lambda x: x["layer"]):
            print(f"  {r['layer']:>5} {r['singleton_frac']*100:>10.1f} "
                  f"{r['median_size']:>9.0f} {r['mean_cohesion']:>9.3f} "
                  f"{r['probe_acc_mean']*100:>11.2f}")
    print()


if __name__ == "__main__":
    main()
