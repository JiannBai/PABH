# CVPR'27 实验补全规划（子维度指标 + 缺种子）

> 更新：2026-10-29。目标：把 `tables.tex` 里所有 `TBD` 填满，
> 重点是 ALFWorld 6 个子任务（Pick/Look/Clean/Heat/Cool/Pick2）和
> WebShop 2 个子维度（Score/Success）的“每个方法 × 每个规模”指标。
> 指标口径：val/success_rate 峰值（peak within budget），从训练日志取 max，
> 不用 checkpoint 的 best_checkpoint.txt（resume 会漏写峰值）。

## 0. 缺口清单（来自 tables.tex）

### 表1（ALFWorld + WebShop 主表）

| 行 | 缺什么 | 数据源 |
|---|---|---|
| ALF 1.5B BiPACE-Q | 6 个 per-task 格子 TBD | 日志里已有（离线抽取） |
| ALF 7B BiPACE-Q | 6 个 per-task 格子 TBD | 日志里已有（离线抽取） |
| ALF 7B GRPO | 6 per-task + All + WebShop Score/Succ 全 TBD | **缺实验**（GRPO 7B ×3 ×2 数据集） |
| WebShop 7B GiGPO | Score + Success TBD | seed0/seed1 已完（0.7227/0.7227），seed2 需补 1 run |

### 表2（ScienceWorld）

| 行 | 缺什么 | 数据源 |
|---|---|---|
| 1.5B GRPO | TBD | 缺实验（×3） |
| 1.5B 方法 | TBD | 已有 4 seeds（0.895），离线回填 |
| 7B GRPO | TBD | 缺实验（×3，需确认 7B 是否要 GRPO 行） |
| 7B GiGPO | TBD | 缺实验（×3） |

### 表4（ScienceWorld 1.5B head 控制组）

| 臂 | 缺什么 |
|---|---|
| Frozen orthogonal head（MICo off） | TBD（MODE=bisimhead，1 run） |
| Trained head（adopted） | TBD（MODE=bisimhead_prm，已有 seed0 可回填 + 视情况补 seeds） |

## 1. Phase 0：离线抽取（0 GPU，立即做，优先）

写 `scripts/extract_subtask_metrics.py`：

- ALFWorld：从每个 run 的最终日志段，取 `val/success_rate` 达到峰值的那个 step，
  同一步读 6 个子任务 `val/pick_and_place_success_rate`、`val/look_at_obj_in_light_success_rate`、
  `val/pick_clean_then_place_in_recep_success_rate`、`val/pick_heat_then_place_in_recep_success_rate`、
  `val/pick_cool_then_place_in_recep_success_rate`、`val/pick_two_obj_and_place_success_rate`；
  All 用 count-weighted peak，口径与表注释一致。
- WebShop：Score=`val/text/test_score`、Success=`val/success_rate`，各自在预算内独立取 max。
- 输出 JSON/CSV 供 `tables.tex` 直接填数。

这一步能立刻填掉：ALF BiPACE-Q 的 12 个 per-task 格子、ScienceWorld 1.5B 方法格、
WebShop 7B GiGPO 的 Score/Succ（seed0/seed1），并复核已填格子的口径。

## 2. Phase 1：跑完进行中的 ALF 补种子（在跑）

- GPU 0-3：ALF 1.5B BiPACE-Q seed2（200 步）
- GPU 4-7：ALF 7B BiPACE-Q seed2（150 步）→ 自动接 ALF 7B PABH seed2（150 步）

完成后 ALF 1.5B/7B 的 BiPACE-Q 与 7B PABH 都到 3 seeds，per-task 指标重新按 3-seed 口径抽取。

## 3. Phase 2：需要新 GPU 的实验（按优先级）

| # | 实验 | seeds | 环境 | 机器 | 填什么 |
|---|---|---|---|---|---|
| 1 | WebShop 7B GiGPO seed2 | 1 | bipace-webshop | 他机 | 表1 7B GiGPO Score/Succ |
| 2 | ALF 7B GRPO | 3 | bipace | 本机 | 表1 7B GRPO per-task+All |
| 3 | WebShop 7B GRPO | 3 | bipace-webshop | 他机 | 表1 7B GRPO Score/Succ |
| 4 | ScienceWorld 1.5B GRPO | 3 | bipace-webshop | 他机 | 表2 1.5B GRPO |
| 5 | ScienceWorld 7B GiGPO | 3 | bipace-webshop | 他机 | 表2 7B GiGPO |
| 6 | ScienceWorld 1.5B bisimhead（frozen） | 1-3 | bipace-webshop | 他机 | 表4 Frozen head |
| 7 | ScienceWorld 7B GRPO | 3 | bipace-webshop | 他机 | 表2 7B GRPO |

**决策（2026-10-29 用户拍板：全部都要）**：7B 阶梯保留 GRPO 行，ScienceWorld 7B 也做 GRPO。
即 #2/#3/#7 全部必跑：GRPO 7B ×3 ×3 个数据集（ALF/WebShop/ScienceWorld），共 9 个 run。

## 4. 机器分工

- **本机 ai-1845503（bipace py3.12）**：ALFWorld 相关（BiPACE-Q/PABH 补种子、ALF 7B GRPO ×3）。
- **他机 ai-1851164（bipace-webshop py3.10）**：WebShop + ScienceWorld 相关
  （WebShop 7B GiGPO seed2、WebShop 7B GRPO ×3、ScienceWorld GRPO/GiGPO/head 控制组）。
- 若他机产能不足，本机需重建 bipace-webshop 才能接 ScienceWorld。

## 5. 完成判据

- 每个 (数据集 × 方法 × 规模) 都 ≥3 seeds，主表方法为 GRPO/GiGPO/BiPACE-Q/PABH，
  四个数据集（ALFWorld/WebShop/ScienceWorld/AppWorld）两种规模都覆盖；
- 每个 run 的 6 个子任务 + WebShop Score/Succ 都从“整体峰值同一步”抽出并写回 `tables.tex`；
- `tables.tex` 里不再有 `TBD`（或只剩明确声明不做/不可得的行）。

## 6. 消融/归因臂也要 3 seeds（2026-10-31 用户定稿）

**Jaccard（dg）锚消融已废弃，论文不写、也不跑**（用户拍板）。

归因三层栈（tab:attrib，ALFWorld）需补：

| 臂 | 1.5B | 7B | 需补 |
|---|---|---|---|
| Raw（BiPACE-Q） | 重跑正确配置中 → 3s | 旧 2s 日志丢失 + 新 seed2 | 7B 补 seed0/seed1 |
| g_text only（midarm） | 2s | 旧 2s 日志丢失 | 1.5B +seed2；7B 重跑 ×3 |
| g_text + head（PABH） | 5s ✅ | seed1 + seed2 跑中 | 7B +seed0 |

ScienceWorld 1.5B head 控制组（tab:head_controls）：

| 臂 | 现状 | 需补 |
|---|---|---|
| raw BiPACE-Q | 3s ✅ | — |
| Frozen head（MODE=bisimhead，MICo off） | 1s | 重跑 ×3 |
| Trained head | 4s ✅ | — |

待补消融 run（不含 Jaccard）：
1. ALF 1.5B midarm seed2（`train_midarm_g_actorhidden_thisnode.sh`）
2. ALF 7B midarm ×3（重跑，`train_official_bipaceq_thisnode.sh` + 显式 gtext_inject）
3. ALF 7B PABH seed0（`train_bisimhead_g_prmanchor_thisnode.sh`）
4. ALF 7B Raw seed0/seed1（`train_official_bipaceq_rawanchor_thisnode.sh`）
5. SW 1.5B Frozen head ×3（`train_sciworld_1.5b_node1613826.sh` MODE=bisimhead）
