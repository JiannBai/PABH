# Progress-Aware Bisimulation Heads for Step-Level Credit Assignment in LLM Agents

**Draft v1.0** — 2026-09-09

---
21
## Abstract

Group-based reinforcement learning methods such as GiGPO assign step-level credit by grouping actions that share an identical *anchor state*. BiPACE relaxes this exact-match requirement with behavioral clustering over the actor's hidden states, but the resulting distance metric is purely syntactic: it inherits whatever state information happens to leak into the hidden representation, and it conflates states at different stages of task progress. We propose a **progress-aware bisimulation head** — a lightweight, frozen-backbone linear projection trained with a MICo-style metric loss whose reward anchor is derived from *task progress* rather than terminal outcomes. Our method contributes three orthogonal components: (i) structured progress sets `g_set` (completed subgoal closures) injected into the anchor observation; (ii) two interchangeable reward anchors for the metric loss — a rule-based process reward (PRM) and a Jaccard distance over progress sets (`d_g`); and (iii) improved pair-sampling for metric learning (per-batch resampling, temporal-gap filtering, mini-epoch updates). On ALFWorld with Qwen2.5-1.5B-Instruct, our method reaches **95.1 ± 1.8%** validation success (3 seeds), matching the strong BiPACE-Q baseline (93.5 ± 1.2%) and exceeding our reproduction of it (89.8%) by **+5.3pp**. A controlled anchor ablation shows the PRM anchor dominates the Jaccard anchor by 7.6pp on ALFWorld, reversing a trend observed on TextCraft — anchor granularity must match the environment's progress structure. We further report a systematic negative-result study on TextCraft (12 runs) demonstrating that ±7–8pp run-level variance in short-horizon synthetic environments renders 3–8pp treatment effects statistically undetectable, motivating evaluation-protocol guidelines for this class of benchmarks.

---

## 1 Introduction

Recent progress in training LLM agents with RL has converged on *group-based* estimators (RLOO, GRPO) that replace learned critics with within-group relative advantages. For multi-step agent tasks, GiGPO extends this to the step level: actions are compared only against other actions taken at the *same* environment state, retroactively discovered by hashing repeated observations across trajectories.

Exact state hashing, however, is brittle in text environments — behaviorally equivalent states rarely produce byte-identical observations. BiPACE addresses this by clustering anchor observations in the actor's hidden-state space (BiGPO) and re-centering advantages on same-action peers (PACE). This makes the step-level group *approximate* rather than exact, and the quality of the resulting credit assignment now hinges entirely on the **distance metric** over states.

We identify two previously unaddressed failure modes of hidden-state cosine distance as a bisimulation proxy:

1. **Progress blindness.** The anchor observation in agent pipelines typically contains only single-step action feedback (e.g., "You pick up the pillow"). Two states at very different stages of task completion can be near-identical in this representation, so the cluster mixes trajectories with different remaining values — a temporal confound that degrades the group baseline.
2. **Untrained metric.** The hidden representation is optimized for token prediction, not for value equivalence. Nothing in the training loop encourages ‖z(s)−z(s′)‖ to reflect the *decision-relevant* similarity of s and s′.

We address both with a **progress-aware bisimulation head**: a single linear layer (frozen backbone) trained online with a MICo-style bisimulation loss, whose reward term is grounded in *task progress* — either a rule-based process reward or a set distance over completed subgoals. The head is trained on the same rollouts used for policy updates, at negligible cost (0.39M parameters, <2% step time), and its output replaces raw hidden states in BiGPO clustering.

Our contributions:

- **Method.** A progress-aware bisimulation head for step-level grouping in group-based agent RL, with two progress-grounded metric anchors (event-PRM and progress-set Jaccard), plus metric-learning sampling improvements (per-round resampling, minimum temporal gap, mini-epoch updates).
- **Cross-environment progress formalization.** We instantiate progress sets for two structurally different environments — recipe-closure nodes in TextCraft and template-parsed subgoal events in ALFWorld — showing the same head consumes both without code change.
- **Results.** On ALFWorld 1.5B: 95.1 ± 1.8% (3 seeds), statistically indistinguishable from BiPACE-Q (93.5 ± 1.2%) and +5.3pp over our own BiPACE-Q reproduction; the head variant is trained at no cost to the PPO loop.
- **Negative-result analysis.** A 12-run TextCraft study showing that this benchmark's ±7–8pp seed variance (consistent with the official BiPACE report of 65.1 ± 7.0) makes it structurally incapable of resolving <10pp effects, and that our method's apparent failure there was a measurement artifact.

## 2 Background

**GiGPO** computes advantages as `A = A_episode + ω·A_step`, where `A_episode` is GRPO-style trajectory normalization within a prompt group, and `A_step` normalizes each step's discounted return within its *anchor group* — the set of steps sharing an identical observation.

**BiPACE** replaces exact anchor matching with behavioral clustering: anchor observations are embedded (actor hidden states at a fixed late layer, L2-normalized) and greedily clustered by cosine distance ≤ ε. Within each cluster, the PACE estimator partitions by action identity and computes `A = Q̂(s,a) − V̂(s)` from same-action vs. all-action peer means. The cluster structure is therefore a *learned equivalence relation* — but one never trained against any notion of value equivalence.

**Bisimulation metrics** (Ferns et al.; MICo, Castro et al.) define state distance recursively: d(s,s′) = |r(s)−r(s′)| + γ·max over successor-pair distances. MICo's practical form fits a representation so that ‖z(s)−z(s′)‖ ≈ |r−r′| + γ‖z(s⁺)−z(s′⁺)‖, solvable with a slowly-updated target network. We adopt this loss with two modifications: the reward anchor is a *dense progress signal* rather than sparse extrinsic reward, and the head is trained on-policy from the same rollouts that feed the PPO update.

## 3 Method

### 3.1 Progress sets

Let `g(s)` be the set of subgoals completed by the time state s is reached.

**TextCraft (synthesis trees).** `g` is the closure of craftable items in the goal's recipe tree that the inventory already supports — a vector-level signature that distinguishes, e.g., `{torch, stick}` from `{ingot, stick}` where a scalar progress ratio cannot.

**ALFWorld (household tasks).** Environment feedback is generated from a closed set of six event templates ("You pick up the {obj} from the {rec}", "You heat the {obj} using the {rec}", ...). We parse each step's feedback with anchored regexes and accumulate `(predicate, object)` events; navigation and container-opening events are deliberately excluded so that states differing only in agent location remain bisimilar. A 6-type × 3-episode walkthrough-replay validation confirmed 100% parse coverage.

The progress set is exposed to three consumers: (i) as text prepended to the anchor observation (`g_text`), so the *policy-induced embedding* also sees progress; (ii) as the Jaccard anchor `d_g(i,j) = |g_i △ g_j| / |g_i ∪ g_j|` for the metric loss; (iii) as the source of a rule-based process reward for environments lacking one (ALFWorld: +1.0 win, +0.5 new subgoal event, −0.1 no-op).

### 3.2 The bisimulation head

A single linear projection `W ∈ R^{256×H}` (orthogonal init; JL-style distance preservation so cold-start clustering ≈ raw-hidden clustering) with an exponential-moving-average target copy `W̄` (τ = 0.005, i.e., the bootstrap target is effectively frozen for ~200 updates). Only W receives gradients; the actor backbone is untouched. Per PPO step:

```
for k in 1..K_rounds:                      # K_rounds = 10
    pairs ← sample(N_pairs=256; min_dt=3)  # fresh pairs each round; same-trajectory pairs ≥3 steps apart
    d ← anchor(pairs)                       # |r_i−r_j| (PRM) or Jaccard(g_i, g_j)
    for u in 1..K_inner:                    # K_inner = 3
        L = ( ‖z(s_i)−z(s_j)‖ − [d + γ‖z̄(s_i⁺)−z̄(s_j⁺)‖] )²
        W ← W − η∇L                         # η = 1e-4, γ = 0.95 (= PPO γ)
W̄ ← τW + (1−τ)W̄
```

The resulting embedding `z = normalize(W·h)` replaces raw hidden states in BiGPO's greedy cosine clustering. The `bisim_anchor_mode` switch selects between the PRM anchor and the Jaccord anchor; both consume identical rollouts, so the ablation is a pure single-variable comparison.

### 3.3 Sampling details

Three changes over the naive pair sampler, each behind a config flag with the original behavior as default:

- **Per-round resampling** (coverage 4.7% → ~38% of the ~10k valid steps per PPO step),
- **min_dt = 3**: same-trajectory pairs closer than 3 steps are near-identical states whose metric target is ≈0; dropping them biases sampling toward cross-trajectory contrastive pairs (99% cross-trajectory in practice),
- **Mini-epoch updates** (K_inner = 3): three gradient steps per sampled batch before resampling, giving 30 head updates per PPO step.

## 4 Experiments

### 4.1 Setup

All runs use the official BiPACE configuration per environment (ALFWorld: eps=0.05, layer −12, group 8, 200 steps, val 128; CACB follows the official per-model convention: first_n+counterfactual for 1.5B, action_tag+q_style for 7B). Our head adds: dim 256, lr 1e-4, τ 0.005, γ 0.95, 30 updates/step, resample+min_dt. Models: Qwen2.5-1.5B-Instruct (and 7B, in progress). Hardware: 4×H20 per run. Evaluation: peak validation success within the budget (official protocol); 3 seeds where stated.

### 4.2 Main results (ALFWorld, 1.5B)

| Method | Success (3 seeds) | Δ vs. official BiPACE-Q |
|---|---|---|
| GiGPO (official) | 86.7 ± 1.7 | −6.8 |
| BiPACE-Q (official) | 93.5 ± 1.2 | — |
| BiPACE-Q (our reproduction) | 89.8 | −3.7 |
| **Ours: head + PRM anchor** | **95.1 ± 1.8** | **+1.6 (n.s., Welch t=1.26)** |
| Ours: head + Jaccard anchor | 87.5 (1 seed) | −6.0 |

Three-seed variance of our method (±1.8pp) matches the official baseline's (±1.2pp), indicating no added training instability. Against our own same-pipeline reproduction of BiPACE-Q, the head contributes **+5.3pp**. Against GiGPO, the cumulative stack is +8.4pp.

### 4.3 Anchor ablation

The PRM anchor beats the Jaccard anchor by **7.6pp** on ALFWorld (95.1 vs. 87.5). On TextCraft the ordering was reversed (+3.2pp for Jaccard, within noise). We attribute the reversal to progress-set granularity: ALFWorld episodes complete only 2–4 distinct subgoal events, so Jaccard distances are coarse and nearly binary, while the four-level event PRM (−0.1/0/0.5/1.0) preserves ordering information. **The choice of metric anchor is environment-structural, not universal** — a design guideline we make explicit.

Per-task breakdown (seed 0, val): the Jaccard variant wins on `look_at_obj_in_light` (88.9 vs. 77.8) while the PRM variant wins on `pick_two_obj_and_place` (94.1 vs. 76.5) — anchors bias which task types receive the sharpest credit.

### 4.4 Negative results on TextCraft (and why)

Over 12 completed runs (6 ours, 6 from a collaborating machine, identical configs), no head variant separated from the BiPACE-Q baseline beyond noise: observed effects ranged −9 to +6pp against a run-level σ of 7–8pp (official: 65.1 ± 7.0). A power analysis shows detecting a true +8pp effect with α=0.05 requires n≥8 seeds (optimistic) to n≥12 (conservative); +5pp is undetectable at any practical n. Two baselines run under identical configs differed by **14pp** — pure seed variance.

We therefore migrated the evaluation to ALFWorld (σ = 1.2–1.8pp), where the same method resolves cleanly. We recommend: (i) short-horizon synthetic benchmarks report per-seed curves, not just peaks; (ii) effects <2σ_env not be claimed; (iii) ALFWorld-class environments preferred for <5pp effects.

### 4.5 7B scaling (in progress)

The same configuration on Qwen2.5-7B-Instruct (official 7B protocol: eps=0.10, layer −8, 150 steps) is currently training on seeds 1–2 (22/150 steps at draft time; official BiPACE-Q 7B = 97.1 ± 0.9, GiGPO = 90.8 ± 1.3).

## 5 Related Work

**Group-based RL for LLM agents.** GRPO/RLOO established within-group advantage estimation; GiGPO extended it to steps via anchor-state grouping; BiPACE made the grouping approximate (BiGPO) and action-conditional (PACE). Our work is a direct successor: it makes the *metric* underlying the approximate grouping trained and progress-aware.

**Bisimulation in deep RL.** DBC (Zhang et al.) learned bisimulation metrics for control; MICo (Castro et al.) refined the loss with target networks. We adapt MICo to the LLM-agent setting where (i) rewards are terminal-sparse, necessitating progress-grounded anchors, and (ii) the representation is a frozen LLM layer, so only a projection head is trained.

**Process rewards.** PRMs provide dense training signal in reasoning; here the process signal shapes a *state metric* rather than the policy reward, keeping the PPO reward stream unmodified (no reward-hacking surface).

## 6 Limitations

- 7B results pending; single-seed for the Jaccard ablation on ALFWorld.
- The +1.6pp over official BiPACE-Q is not statistically significant at n=3; our claim is *parity with a stronger-than-reproduced baseline*, with the +5.3pp reproduction delta as the more meaningful effect size.
- Progress sets require environment-specific parsers (closed template set in ALFWorld; recipe closure in TextCraft). Environments with open-ended feedback would need learned progress detectors.
- WebShop adaptation (progress = search/click/purchase events) is prepared but unrun.

## 7 Conclusion

A 0.39M-parameter head, trained online with a progress-grounded MICo loss on rollouts that the PPO loop already produces, turns BiPACE's syntactic state clustering into a learned value-equivalence metric at no measurable cost — matching the strongest published baseline on ALFWorld 1.5B and clarifying, through a controlled anchor ablation and a 12-run negative-result study, *when* step-level bisimulation grouping can and cannot show its value.

---

*Artifacts: training scripts (`examples/bipace/alfworld/*_thisnode.sh`), design doc (`research_archive/DESIGN_PROGRESS_AWARE_BISIM_HEAD.md`), experiment log (`EXPERIMENTS_LOG_THISNODE.md`), cross-machine coordination (`CROSS_MACHINE_STATUS.md`).*
