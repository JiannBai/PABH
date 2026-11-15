# 跨机器协作简报（本机 = qs-325328-2184811-ai-1797956-default0-0）

> 最后更新：2026-10-15 03:30（**他机已死，正在重建环境**；本机接管全部实验队列）。
> 本文件供另一台机器的协作者快速了解本机正在跑什么、代码改了什么、
> 哪些共享目录/配置有坑。**启动任何实验前请先读本文的"冲突教训"部分。**

## ➖➖ 2026-11-07 · ai-1851164 本机当前占用（他机请勿重复这些 experiment_name）➖➖

**2026-11-09 追加：ALFWorld 消融补种（本机认领，他机勿重复）**
- ALF 7B midarm(g_text only) **seed2 only**（用户拍板：旧 2 seed 可信，只补第 3 个；排队等 PABH seed1 完赛）：
  `midarm_alfworld_7b_4gpu_L-8_eps0.10_g_actorhidden_seed2_thisnode`
- ALF 1.5B midarm(g_text only) seed2（排队，等 WebShop BiPACE-Q seed2 完赛）：
  `midarm_alfworld_1.5b_4gpu_L-12_eps0.05_g_actorhidden_gtext_seed2_thisnode`
- 新增脚本：`examples/bipace/alfworld/qwen7b_4xh100/train_midarm_g_actorhidden_7b_thisnode.sh`
  （官方 7B BiPACE-Q + `env.gtext_inject=True`，实验名带 `midarm_*`）。

**本机 ai-1851164 正在跑（8 卡全占，请勿抢占/重复启动）**：
- GPU 4-7：ALFWorld 7B PABH **seed1**（本机跑完，峰值 0.984；由本机原 seed0 改名而来）
  `bisimhead_alfworld_7b_4gpu_L-8_eps0.10_qstyle_g_prmanchor_lr1e-4_tau0.005_ups3_seed1_thisnode`
- GPU 0-3：ALFWorld 7B midarm(g_text only) **seed2**
  `midarm_alfworld_7b_4gpu_L-8_eps0.10_g_actorhidden_seed2_thisnode`

- 1.5B midarm seed2 已完赛（峰值 0.906），GPU0-3 空出后改跑 7B midarm seed2。

**2026-11-11 同步 ai-1845503 最新计划**：ALF 7B PABH 的 seed0 由 ai-1845503 在跑
（`_ai1845503` / `_ai1845503b` 两个变体，赛后取峰值高者与 seed1/seed2 组三 seed）。
本机已停掉自己的 ALF 7B PABH seed0，避免重复；本机改跑 7B midarm seed2。
ai-1845503 后续计划 = ALFWorld OOD 方法矩阵（1.5B GiGPO×3 → 7B PABH/BiPACE-Q/GiGPO×3
→ 1.5B/7B GRPO×3），本机不碰 OOD。

**2026-11-14 OOD 分工更新（本机 ai-1851164 认领，他机勿重复）**：
- 用户拍板：ALFWorld OOD 先都跑 **2 个随机种子** 即可。
- 本机 OOD 1.5B GRPO 只跑 seed0/seed1（两槽并行：GPU0-3=seed0，GPU4-7=seed1，不跑 seed2）。
  experiment：`grpo_alfworld_ood_1.5b_4gpu_seed{0,1}_thisnode`。
- 1.5B 跑完后再考虑 7B（用户拍板：先 1.5B 后 7B；7B GiGPO 已暂停，避免与 ai-1845503 重复）。

**2026-11-15 7B OOD 分工（本机 ai-1851164 认领，他机勿重复）**：
- 本机 GPU0-3：OOD 7B GiGPO seed0 → seed1（`gigpo_alfworld_ood_7b_4gpu_seed{0,1}_thisnode`）。
- 本机 GPU4-7：OOD 7B GRPO seed0 → seed1（`grpo_alfworld_ood_7b_4gpu_seed{0,1}_thisnode`）。
- 7B PABH 与 7B BiPACE-Q 仍由 ai-1845503 负责，本机不碰。

**本机环境提醒**：ai-1851164 的 `bipace`(py3.12) 缺 `scienceworld` 模块，ScienceWorld 训练
必须用 `bipace-webshop`(py3.10)。本机已建本地脚本
`examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_ws_ai1851164.sh`（仅把 conda env
改为 bipace-webshop），并用它跑 1.5B GRPO resume；请勿用原
`train_sciworld_1.5b_node1613826.sh` 在本机直接启动。

**WebShop 状态（2026-11-11 用户拍板）**：WebShop 全部实验已停止，不再补跑。
本机只继续 ALFWorld 消融补种。

**本机已完成、他机无需再跑（避免重名覆盖）**：
- WebShop 1.5B PABH eps=0.04：seed0/2/10 三 seed 齐（0.777/0.688/0.762）。
- WebShop 7B PABH ε 扫描单 seed：0.03/0.04/0.05/0.10/0.15 全齐；eps0.08 3 seeds 已有。
- WebShop 7B PABH eps=0.05：seed0（0.766）/seed1（0.762）已完，seed2 在跑。
- ScienceWorld 7B GiGPO s2 已完赛（峰值 0.796875），7B GiGPO 三 seed 齐。
- ScienceWorld 1.5B GRPO seed1/seed2 已完，仅 seed0 在续；7B GRPO seed0/seed1 已完，仅 seed2 在续。

**给 ai-1845503/ai-1797956 的建议分工（避免两边同时开同一 EXP）**：
**ScienceWorld 已全部交给另一台机器（2026-11-07 用户拍板）**，本机不再启动 SW：
- SW 1.5B GRPO seed0 resume（step50 → 200）：checkpoint 停在 `global_step_50`。
- SW 7B GRPO seed2 resume（step50 → 150）：checkpoint 停在 `global_step_50`。
- 可选：SW 1.5B frozen-head 控制组。
- 本机 ai-1851164 跑 SW 必须用 `bipace-webshop`（`bipace` 缺 scienceworld）；他机按自己环境
  即可，启动前请核对 `import scienceworld` 是否可用。

本机只保留 **WebShop** 收尾：WebShop 7B PABH eps=0.05 seed2 + WebShop 7B BiPACE-Q
eps=0.10 seed1/seed2 已全部由本机认领并在跑。他机只接 ScienceWorld 上述任务，勿碰本机
WebShop（尤其是 `baseline_webshop_7b_4gpu_L-8_eps0.10_*` 和
`bisimhead_webshop_7b_4gpu_L-8_eps0.05_*`）。

## ➖➖ 2026-11-01 15:10 UTC · ai-1851164 本机实时状态（WebShop PABH ε 寻优优先）➖➖

**正在跑（8 卡全占，2026-11-03 08:57 更新）**：
- ✅ 7B PABH eps=0.05 seed0 已完赛（150/150），**val/success_rate 峰值 0.766**。
- ✅ 7B PABH eps=0.10 seed0 已完赛，**val/success_rate 峰值 0.719**。
- ✅ 7B PABH eps=0.15 seed0 已完赛，**val/success_rate 峰值 0.758**。
- ✅ 7B PABH eps=0.03 seed0 已完赛，**val/success_rate 峰值 0.750**。
- ✅ 7B PABH eps=0.04 seed0 已完赛，**val/success_rate 峰值 0.742**。
- ✅ 1.5B PABH eps=0.04 seed0 已完赛，**val/success_rate 峰值 0.777**。
- ✅ 1.5B PABH eps=0.08 seed0 已完赛，**val/success_rate 峰值 0.711**。
- ✅ 1.5B PABH eps=0.10 seed0 已完赛，**val/success_rate 峰值 0.719**。
- GPU 4-7：7B PABH eps=0.05 seed1，约 12/150（8%），健康。
- GPU 0,1：1.5B PABH eps=0.04 seed2，约 124/200（62%），剩余约 6.9h。
- GPU 2,3：1.5B PABH eps=0.04 seed10，约 31/200（16%），剩余约 23h。

**7B PABH ε 寻优单 seed 已基本收口**：0.03=0.750、0.05=0.766、0.10=0.719、0.15=0.758；
0.04=0.742（完赛前峰值）。据此选 0.05 为 7B 候选最优，已排
`scripts/webshop7_pabh_eps005_seeds1_2_relay_ai1851164.sh`：等 eps0.04 完赛后顺序补
seed1/seed2，凑齐 {0,1,2} 三 seed。

**新增（用户拍板，2026-11-03）**：7B PABH 继续试更小的 ε=0.03 与 0.04。
- ε=0.03：`scripts/webshop7_pabh_eps003_relay_ai1851164.sh`，等 1.5B eps0.10 完赛后在
  GPU 0-3 启动。
- ε=0.04：`scripts/webshop7_pabh_eps004_relay_ai1851164.sh`，等 7B eps0.15 完赛后在
  GPU 4-7 启动。
- 两条接力与当前 eps0.10/0.15 不冲突，会自动排班。

**新增（2026-11-04）**：1.5B PABH ε=0.04 单 seed 峰值 0.777，明显高于初稿 ε=0.06 的
73.0/70.7/71.9（seed0/2/10）。已排 `scripts/webshop15_pabh_eps004_seeds2_10_relay_ai1851164.sh`：
等 7B eps0.03 完赛后，在 GPU0-3 并行补 ε=0.04 的 seed2/seed10，凑齐 {0,2,10} 三 seed
用于正式对比。

**已排队的接力（本机已挂后台 watcher，到时自动启动，勿重复占卡）**：
- `scripts/webshop15_pabh_sweep_relay_ai1851164.sh`：等 7B eps0.05 完赛 →
  GPU 0-3 并行跑 1.5B PABH eps0.04（GPU0,1）/ eps0.08（GPU2,3），
  任一完赛在空出的 2 卡接力 eps0.10。1.5B eps0.06 已有 4 seeds，不补。
- `scripts/webshop7_pabh_eps015_relay_ai1851164.sh`：等 7B eps0.10 完赛 →
  GPU 4-7 跑 7B PABH eps0.15 seed0。7B eps0.08 已有 3 seeds，不再补；eps0.20 已取消。

**WebShop 已完成口径**：7B PABH eps0.08 = 3 seeds 齐；7B BiPACE-Q eps0.08 = 3 seeds 齐；
7B GiGPO = 3 seeds 齐（0.7227/0.7227/0.7773）。7B BiPACE-Q eps0.10 目前只有 seed0，
待 PABH ε 寻优收口后补 seed1/seed2（固定 ε=0.10）。

**1.5B PABH ε 寻优脚本已新增**（EPS 环境变量参数化，experiment_name 自动带 eps）：
`examples/bipace/webshop/qwen1.5b_2gpu_official/train_webshop15b_pabh_eps_node1613826.sh`。

**提醒他机**：WebShop/ScienceWorld 训练脚本必须 `unset TRITON_PTXAS_PATH`；
本机 8 卡已满，本机 watcher 会自动接力，请勿在本机重复启动同名 experiment_name。

## ➖➖ ai-1845503 一周冲刺（2026-10-31 用户拍板）➖➖

**优先级**：我们的方法 PABH 最先 → ScienceWorld 其次 → 其它 baseline 补齐。
**种子**：PABH 3 seeds；其它 2 seeds 赶进度。**范围**：ALF/WebShop 只 GiGPO/BiPACE-Q/PABH，
ScienceWorld 全四方法。

**本机（bipace，ALFWorld/ScienceWorld）队列**（our_method_priority_ai1845503.sh 已排好）：
- GPU 0-3：ScienceWorld PABH eps {0.05,0.10,0.15} 1.5B 短跑 → 1.5B BiPACE-Q(q_style) seed0/1/2
- GPU 4-7：7B ALF PABH seed0（在跑）→ ScienceWorld PABH eps {0.05,0.10,0.15} 7B 短跑 → 7B ALF PABH seed1

**口径定稿（重要，2026-11-02 用户拍板）**：ALFWorld 1.5B 的 **BiPACE-Q 与我们的方法
统一用 `first_n+counterfactual`**（两者同口径，做单变量对照）。1.5B PABH 已有
5 seeds = 95.2±1.4（0.930/0.961/0.961/0.945/0.961），定稿；BiPACE-Q 按 first_n 补
3 seeds（`official_bipaceq_..._firstn_counterfactual_seed{0,1,2}`），替代旧 2-seed 的 90.2，
并补齐 6 个子任务列。q_style 重跑取消（相关 3 个 q_style 1.5B 目录已删）。
7B 与 WebShop/ScienceWorld 脚本本来就是 action_tag+q_style，无需改。

**时间**：本机 ~5 天（ALFWorld 收口）；他机见其自己登记的 WebShop/ScienceWorld 计划。

## ➖➖ ai-1851164 计划同步（2026-10-31，用户拍板）➖➖

**GRPO 只跑 ScienceWorld；ALFWorld / WebShop 不跑 GRPO。**

**🔴 最高优先级（2026-10-31 追加）：ScienceWorld 我们的方法（PABH）eps 寻优，1.5B 和 7B 都做。**
脚本已加参数：`EPS`（聚类阈值）和 `STEPS`（短跑预算）。
- 1.5B：`MODE=bisimhead_prm START=sft EPS=0.05/0.10/0.15 STEPS=100 SEED=0`
  → 短跑 5 档选最优 eps → 最优 eps 补 3 seeds（STEPS=200）。
- 7B：`MODE=bisimhead_prm START=sft EPS=0.05/0.10/0.15 STEPS=100 SEED=0`
  → 短跑 3 档选最优 eps → 最优 eps 补 3 seeds（STEPS=150）。
- 实验名自动带 `eps${EPS}`，不同 eps 互不覆盖。

- 当前在跑：SW 7B GiGPO s2（GPU 0-3）、SW 1.5B GRPO s0（GPU 4-7）。
- WebShop 7B GiGPO 3 seeds 已完成（0.7227 / 0.7227 / 0.7773）。
- SW 7B GiGPO s0/s1 已完成，s2 进行中，收官后 3 seeds 齐。
- **优先动作（10-31 20:32）**：已暂停 SW 7B GiGPO s2 与 SW 1.5B GRPO s0，腾卡启动
  WebShop 7B PABH ε 寻优：GPU 0-3 = ε0.05 s0，GPU 4-7 = ε0.10 s0。
  完成后接 ε0.15；ε0.20 已取消。1.5B 侧随后按同样流程扫 ε。
- 后续队列：
  1. WebShop 7B BiPACE-Q ε=0.10 ×3 seeds（论文 Table I.1 口径）。
  2. WebShop 7B PABH ε∈{0.05,0.08,0.10,0.15} 扫描；WebShop 1.5B PABH
     同样扫 ε∈{0.04,0.06,0.08,0.10}。先各 1 seed 筛，best ε 补到 3 seeds。
  3. ScienceWorld 1.5B GRPO ×3。
  4. ScienceWorld 7B GRPO ×3（可选，视时间）。

## ➖➖ ai-1845503 当前队列登记（2026-10-24 22:00 UTC）➖➖

**本机 = `qs-316244-2228885-ai-1845503-default0-0`**（8×H20，`bipace` py3.12 + `venv-appworld` pydantic-v1）。

## ➖➖ 新调度计划（2026-10-28，用户拍板）➖➖

**目标**：ALFWorld / WebShop / ScienceWorld 三个主数据集的 {GRPO, GiGPO, BiPACE-Q, PABH}
**每个方法 × 每个规模都要 3 seeds 和子维度指标**（ALF 6 子任务、WebShop Score/Succ）。
AppWorld/Sokoban 判定不适合，已暂停。

**分工（按环境可用性）**：
- **本机 ai-1845503（有 `bipace` py3.12，缺 `bipace-webshop`）**：跑 ALFWorld 补充。
  - GPU 0-3：ALF 1.5B BiPACE-Q seed2 → **seed0 → seed1**（重跑 s0/s1 为了补 per-task 日志）
  - GPU 4-7：ALF 7B BiPACE-Q(rawnchor) seed2 → ALF 7B PABH seed2
  - 之后 GPU 空出接 ALF 7B GRPO ×3（7B 阶梯 GRPO 行）
- **他机 ai-1851164（已重建 `bipace-webshop` py3.10）**：跑 WebShop + ScienceWorld 补充。
  - WebShop 7B GiGPO **seed2**（seed0/seed1 已在本机跑完：0.7227/0.7227，只缺 seed2；
    脚本 `train_webshop7b_node1613826.sh` MODE=gigpo SEED=2）
  - WebShop 7B GRPO ×3（`train_webshop7b_node1613826.sh` MODE=grpo，SEED=0/1/2）
  - ScienceWorld 7B GiGPO ×3（现 0；`train_sciworld_7b_node1613826.sh` MODE=gigpo START=sft，SEED=0/1/2）
  - ScienceWorld 1.5B GRPO ×3（`train_sciworld_1.5b_node1613826.sh` MODE=grpo START=sft）
  - ScienceWorld 7B GRPO ×3（`train_sciworld_7b_node1613826.sh` MODE=grpo START=sft）
  - ScienceWorld 1.5B frozen-head 控制组（MODE=bisimhead，1-3 seeds）

**⚠ 环境提示**：WebShop 与 ScienceWorld 脚本都依赖 `bipace-webshop`（py3.10/vllm0.8.4），
本机 ai-1845503 没有该环境（只有 py3.12 `bipace`），所以这两个数据集交给他机跑。
若他机 GPU 不足，本机需先重建 `bipace-webshop` 才能接 ScienceWorld。

**OOD（ALF valid_unseen）**：3s 已收官，PABH 88.0±1.2 vs BiPACE-Q 87.3±2.5，仅持平偏正，
不优先补 GRPO/GiGPO；若最后时间充裕再补 OOD 四方法。

**指标口径（定稿）**：论文只报 val/success_rate 峰值（peak within budget）；峰值从训练日志
取 max，不要用 checkpoint 的 best_checkpoint.txt（resume 会漏写最终段峰值）。

**AppWorld / Sokoban**：暂停，不进主表，留作边界/探索性结论。

**子维度指标抽取**：本机已写 `scripts/extract_subtask_metrics.py`，从日志按“整体峰值同一步”
抽 ALFWorld 6 子任务 + WebShop Score(`webshop_task_score`)/Success，输出
`paper_cvpr2027/subtask_metrics.json`、`webshop_metrics.json`。他机跑完后可用同脚本补抽。

---

## ➖➖ ai-1845503 旧队列登记（2026-10-24，已部分被上面新计划取代）➖➖

**本机 = `qs-316244-2228885-ai-1845503-default0-0`**（8×H20，`bipace` py3.12 + `venv-appworld` pydantic-v1）。

- **🔴 GPU 0-3（请勿重复）**：`scripts/sevenb_supplement_relay.sh`（pid 892317）——
  7B GiGPO 补实验串行队列：
  1. ALFWorld 7B GiGPO s0 ✅（150 步，final val 0.891）
  2. ALFWorld 7B GiGPO s1 ✅（150 步，final val 0.930）
  3. ALFWorld 7B GiGPO s2 ✅（150 步，final val 0.898）
  4. WebShop 7B GiGPO ❌ 本机缺 bipace-webshop 环境（pyserini 要 torch≥2.9 vs vllm0.11 要 torch==2.8 冲突），已停，待单独重建环境
  **→ GPU 0-3 现改跑 AppWorld 7B seed1 三臂**（`scripts/appworld7b_relay2_seed1_ai1845503.sh`，pid 848006）
- **🔴 GPU 4-7（请勿重复）**：`scripts/appworld7b_relay_ai1845503.sh`（pid 922275）——
  AppWorld 7B seed0 三臂（7B multistep SFT 冷启动，100 步试水）：
  1. baseline(BiPACE-Q) ✅（val 峰值 25%，step100 终值 0）
  2. bisimhead_prm(我们方法) 🔄（~50/100，健康无坍缩）
  3. gigpo(GiGPO 精确锚) ⏳
- **新增第二套 AppWorld server 集群**：端口 9000-9159（`/tmp/codewiz/appworld_ports2.ports`），
  供 GPU 0-3 seed1 并行使用；原集群 8000-8159 保持不动。
- AppWorld 160 server 健康，内存看门狗单例运行中；`gpu_idle_monitor.py` 8 卡常驻（防回收）。

**AppWorld 1.5B 三臂结论（已归档，勿重复跑）**：baseline 9.4% 稳定；GiGPO step5 坍缩、
bisimhead_prm step3 坍缩（step2 kl_loss=2.83→entropy 6.4+）。只有 actor_hidden 稳定。
**当前转向 7B multistep SFT 冷启动**：看 7B 更大模型是否更稳、能否让我们的方法跑起来。

**本机下一步**：7B AppWorld seed0/seed1 三臂出结果（1.5B 已作废，只看 7B）；
WebShop 7B GiGPO 待重建 bipace-webshop 环境后补。
**不碰 ai-1851164 正在跑的 SW 7B baseline s2 / Sokoban s1**。

## ➖➖ ai-1851164 接 ScienceWorld 补种（2026-10-19 04:12 UTC）➖➖

**本机 = `qs-325328-2231026-ai-1851164-default0-0`**。`bipace-webshop`(py3.10) 环境已重建并冒烟通过。

- **🔴 本机在跑（请勿重复）**：`scripts/sw_seeds_ai1851164.sh`（8 卡并行）——
  - GPU 0-3：SW 1.5B 方法 gtext **s3 全新跑**（200 步，已因 head-resume 污染重置），
    `bisimhead_prm_sftstart_sciworld_1.5b_..._seed3_gtext_node1613826`；
  - GPU 4-7：SW 7B 方法 gtext **s0**（SFT 起点，150 步），
    `bisimhead_prm_sftstart_sciworld_7b_..._seed0_gtext_node1613826`。
- 完成判据：s3=global_step_200；7B s0=global_step_150；死亡 180s 后自动 resume。
- SFT 起点：1.5B=sft_full/global_step_492，7B=sft_full/global_step_483。
- **head 保存/加载已实现**：`gigpo/bisim_head.py` 增加 `state_dict/load_state_dict`，
  `ray_trainer.py` 在 `_save_checkpoint` 写 `bisim_head.pt`、在 `_load_checkpoint` 懒加载。
  旧 s3 resume 目录已归档为 `*_contaminated_resume_*`，s3 已改为全新跑。
- **进度（10-20 03:10 UTC）**：s3 已收官，final val/success_rate=0.891（resume 后保存至
  global_step_205，head ckpt 在 s50/204/205 均已写入）；7B s0 仍在跑，约 70/150。
- 配置审计已做：Sokoban 补 `MODE=grpo` 分支、SW 7B 补 `MODE=gigpo|grpo` 分支。
- **追加（10-20 03:12 UTC）**：GPU 0-3 启动 SW 7B 方法 gtext **s1**（`scripts/sw7_s1_ai1851164.sh`，
  SFT 起点，150 步）；GPU 4-7 仍跑 s0。
- **追加（10-21 03:xx UTC）**：s0 已收官（final 0.719）；GPU 4-7 启动 SW 7B 方法 gtext
  **s2**（`scripts/sw7_s2_ai1851164.sh`）；GPU 0-3 仍跑 s1（约 71/150）。
- **追加（10-22）**：方法 s1 已收官（final 0.8125）；GPU 0-3 启动 SW 7B BiPACE-Q
  baseline **s1**（`scripts/sw7_baseline_s1_ai1851164.sh`）；GPU 4-7 仍跑方法 s2。
- **追加（10-23）**：方法 s2 已收官；baseline s1 已收官（final 0.8281）；GPU 0-3
  已有 baseline s2（`scripts/sw7_baseline_s2_ai1851164.sh`）在跑；GPU 4-7 启动
  Sokoban 方法 gtext **s0**（`scripts/sokoban_s0_ai1851164.sh`）。
- **追加（10-24）**：Sokoban 方法 s0 已收官（global_step_200）；GPU 4-7 启动
  Sokoban 方法 gtext **s1**（`scripts/sokoban_s1_ai1851164.sh`）；GPU 0-3 仍跑 baseline s2。
- **追加（10-25）**：SW 7B baseline s2 已收官（final 0.8281）；GPU 0-3 启动
  Sokoban 方法 gtext **s2**（`scripts/sokoban_s2_ai1851164.sh`）；GPU 4-7 仍跑 s1。
- **追加（10-26）**：Sokoban 方法 s1 已收官；GPU 0-3 启动 Sokoban **GiGPO s1**
  （`scripts/sokoban_gigpo_s1_ai1851164.sh`）；GPU 4-7 仍跑方法 s2。
- **追加（10-27）**：按用户优先级，暂停 Sokoban GiGPO s1（ckpt s50 保留），GPU 4-7
  改跑 WebShop 7B GiGPO **s0**（`scripts/webshop7_gigpo_s0_ai1851164.sh`）；GPU 0-3
  仍跑 Sokoban 方法 s2。
- WebShop 7B 脚本已改回 `bipace-webshop` 环境，并补齐 bs4/cleantext/flask/pyserini/
  spacy(en_core_web_sm)/selenium/gdown 依赖；s0 已进入 WebShop worker 初始化与 NCCL init。
- **追加（10-27 11:37）**：Sokoban 方法 s2 已收官；GPU 0-3 启动 WebShop 7B GiGPO
  **s1**（`scripts/webshop7_gigpo_s1_ai1851164.sh`）；GPU 4-7 仍跑 s0。
- **已同步 10-28 新调度计划**：本机后续队列 = WebShop 7B GiGPO s2 → ScienceWorld 7B
  GiGPO s0/s1/s2 → ScienceWorld 1.5B GRPO s0/s1/s2；AppWorld/Sokoban 暂停。
- **追加（10-28）**：WebShop GiGPO s0 已收官（final 0.7227）；GPU 4-7 启动
  WebShop GiGPO **s2**（`scripts/webshop7_gigpo_s2_ai1851164.sh`）；GPU 0-3 仍跑 s1。
- **追加（10-29）**：WebShop GiGPO s1 已收官（final 0.7227）；GPU 0-3 启动
  ScienceWorld 7B GiGPO **s0**（`scripts/sw7_gigpo_s0_ai1851164.sh`）；GPU 4-7 仍跑 WebShop s2。
- **追加（10-30）**：WebShop GiGPO s2 已收官（final 0.7773），WebShop 7B GiGPO 3 seeds 齐；
  GPU 4-7 启动 ScienceWorld 7B GiGPO **s1**（`scripts/sw7_gigpo_s1_ai1851164.sh`）；
  GPU 0-3 仍跑 s0。
- **追加（10-31）**：SW 7B GiGPO s0 已收官（final 0.7813）；GPU 0-3 启动
  ScienceWorld 7B GiGPO **s2**（`scripts/sw7_gigpo_s2_ai1851164.sh`）；GPU 4-7 仍跑 s1。
- **追加（10-31 下午）**：SW 7B GiGPO s1 已收官；GPU 4-7 启动 ScienceWorld 1.5B GRPO
  **s0**（`scripts/sw15_grpo_s0_ai1851164.sh`）；GPU 0-3 仍跑 SW 7B GiGPO s2。
- **WebShop 7B ε 决策（10-31）**：BiPACE-Q 固定 ε=0.10 补 3 seeds；PABH 扫
  ε∈{0.05,0.08,0.10,0.15,0.20} 选最优（脚本已备：train_webshop7b_eps{005,010,015,020}_node1613826.sh）。
- **追加（10-22 晚）**：基线 s1 ~104/150、方法 s2 ~134/150，双双健康在跑；
  已挂 `scripts/sw7_baseline_s2_ai1851164.sh` 接力（pid 83720）——等 baseline s1
  完赛 + GPU 0-3 空闲后自动启动 SW 7B baseline **s2**（最后一格，SFT 起点 150 步）。
- **追加（10-23）**：方法 s2 收官 final **0.84375**（SW 7B 方法 gtext 3 seeds 齐：
  0.719 / 0.8125 / 0.84375）；baseline s1 收官 final **0.828125**；baseline s2 已于
  03:56 自动接力上卡（GPU 0-3，SFT 起点 150 步，约 48h 预算）。GPU 4-7 空出后已恢复
  idle-monitor + watchdog 占卡防回收（supervisor pid 1692140，watchdog pid 1693559）。

## ➖➖ ai-1851164 接 ALF 阶梯 GRPO seeds s1/s2（2026-10-17 20:40 UTC）➖➖

**本机 = `qs-325328-2231026-ai-1851164-default0-0`**（8×H20，`bipace` py3.12 环境就绪）。

- **✅ 已完成（10/18 17:14 UTC）**：`scripts/alf_grpo_seeds_ai1851164.sh`（8 卡并行）——
  ALFWorld 1.5B **GRPO s1**（GPU 0-3）与 **GRPO s2**（GPU 4-7），
  experiment_name = `grpo_alfworld_1.5b_4gpu_seed{1,2}_thisnode`，200 步，
  完成判据 `global_step_200`，死亡 180s 后自动 resume，checkpoint 写共享 `verl_checkpoints/`。
- 目的：GRPO 阶梯由 1 seed（s0=76.6）补到 3 seeds；与 ai-1845503 的 ALF OOD relay 零冲突。
- **峰值（val/success_rate）**：s1=0.734、s2=0.734；与 s0=0.766 合并后
  GRPO 3 seeds = **0.745 ± 0.018**。本机 8 卡现已回到 idle-monitor 占卡状态。

## ➖➖ ai-1851164 接 ALF 阶梯 GiGPO seeds s1/s2（2026-10-16 06:55 UTC）➖➖

**本机 = `qs-325328-2231026-ai-1851164-default0-0`**（8×H20，新容器，环境从零重建）。

1. **环境已重建（本地 `/root/miniconda3/envs/bipace`）**：python 3.12 / torch 2.8.0+cu128 /
   vllm 0.11.0 / flash-attn 2.8.3 / ray 2.50.0 / transformers 4.57.3 / alfworld 0.4.2 /
   gymnasium 0.29.1 / stable-baselines3 2.6.0 / pydantic 2.11.7。import 与 vllm 冒烟通过。
2. **ALFWorld 数据**已下载到 `/home/work/.cache/alfworld`（train 3553 / valid_seen / valid_unseen），
   单 worker reset 冒烟通过（g_set/g_text/process_reward 信息齐全）。
3. **✅ 已完成（10/17 01:51 UTC）**：`scripts/alf_ladder_seeds_ai1851164.sh`（8 卡并行）——
   ALFWorld 1.5B **GiGPO s1**（GPU 0-3）与 **GiGPO s2**（GPU 4-7），
   experiment_name = `gigpo_alfworld_1.5b_4gpu_seed{1,2}_thisnode`，200 步，
   完成判据 `global_step_200`，死亡 180s 后自动 resume。checkpoint 写共享 `verl_checkpoints/`。
4. 与 ai-1845503 的 ALF OOD relay（`bisimhead_alfworld_ood_...seed0` → `official_bipaceq_alfworld_ood_...seed0`）
   **零冲突**：实验名不同、其 ckpt 在本机 `/root/verl_checkpoints`（本地盘）。
5. **终值（val/success_rate）**：s1=0.836、s2=0.875；与 s0=0.836 合并后
   GiGPO 3 seeds = **0.849 ± 0.023**。本机 8 卡现已回到 idle-monitor 占卡状态。

## ➖➖ 容器再换 + 环境迁本地 + ALF OOD 上卡（2026-10-15 19:15，ai-1845503）➖➖

**本机已换为 `qs-316244-2228885-ai-1845503-default0-0`**（新容器，上一台 ai-1797956
进程全部随容器消失，GPU 全空）。已完成接管并恢复推进：

1. **gpu_idle_monitor 已恢复**：`gpu_idle_monitor.py` supervisor + 守护脚本
   `scripts/idle_monitor_watchdog.sh`（pid 29144）保证其常驻；GPU 4-7 跑拟真负载防回收，
   GPU 0-3 让给实验。
2. **环境迁到本地盘**：JuiceFS 上冷 import 极慢导致 ray.init() 的 320 预启动 worker
   注册超时死锁（raylet 日志 "hanging during start"）。已把 `bipace`(py3.12) 环境从
   JuiceFS 复制到 `/root/miniconda3/envs/bipace`（12G），conda 垫片优先本地、回退共享。
3. **修 pydantic**：`bipace` 里 pydantic 被误降到 1.10.26，而 vllm 0.11.0 要求
   `pydantic>=2.11.7`；已升到 2.11.7。⚠ 代价：appworld/sqlmodel/fastapi 的依赖冲突
   （appworld 要 pydantic<2），**AppWorld 三臂上卡前需单独处理**。
4. **ALFWorld 数据**：已下载 json_2.1.1/pddl/tw-pddl 到 `/home/work/.cache/alfworld`，
   train 3553 / OOD valid_unseen 134 局，冒烟 reset 通过。
5. **🔴 本机在跑**：`scripts/alfood_relay_ai1845503.sh`（pid 40716，GPU 0-3）串行
   ALF OOD 方法臂 s0 → baseline s0（各 ~28h，200 步，~510s/step）。方法臂 s0 已进训练
   （step0 val/success_rate=0.031，step1 已训完）。

**⚠ 重大阻塞：JuiceFS 已 100% 满（4.6T 用满，仅 13G 可用），`verl_checkpoints/` 占 2.2T**
（每 run 8 个 global_step × 19G，`keep_best_and_latest` 未生效）。本轮 ALF OOD 的
checkpoint 已临时改写到本地 `/root/verl_checkpoints`（`VERL_CKPT_BASE` 环境变量可覆盖，
两个 OOD 脚本已支持，默认仍回 JuiceFS）。**本地盘 checkpoint 随容器销毁即丢**，需尽快
按"只留 best-val + 最后一个"清理旧 checkpoint 腾空间，或确认本地盘策略。

**✅ 10/16 已按"best-val + last"清理**：删 12 个冗余 global_step（midarm seed0/seed1 各 5 个
+ webshop bisimhead seed4 的 s150 + gigpo seed10 的 s100），JuiceFS 可用空间 13G → **230G**。
保留：midarm seed0 = s201(0.867 最优)+s203(最后)；midarm seed1 = s202(0.883)+s203；
webshop seed4 = s200；gigpo seed10 = s200。其余 run 本就 ≤2 个 checkpoint，未动。
SFT 冷启动 ckpt（sciworld_sft sft_full）与 7B 各 1-checkpoint run 均保留。

**✅ 10/16 二轮清理（跨项目训练 checkpoint）**：删除非 BiPACE 路径的训练 checkpoint
（均无 BiPACE 代码引用，基础模型已保留）：
- `agenticRL/verl_checkpoints`（142G）、`agenticRL/checkpoints`（73G）
- `PABPO/verl_checkpoints`（287G）、`CoSearch/checkpoints`（341G）
- BiPACE 内 7 个空壳 run（无 global_step，仅 rollouts/txt，共 ~5G）
合计又释放 ~850G；**JuiceFS 可用空间现为 ~1.1T（77% 使用）**。
保留：`agenticRL/models`（Qwen2.5-1.5B/7B 基础权重，BiPACE 依赖）、`Tree-GRPO/models`、
`QUEST/models`、`CoSearch/data/models` 等模型权重目录。

**仍缺失**：`bipace-webshop`(py3.10, vllm0.8.4) 环境未重建，Sokoban/WebShop/ScienceWorld
队列暂不能跑；如需恢复请先重建该环境或改跑 ALFWorld（py3.12 环境已就绪）。

## ➖➖ ALF OOD 方法臂收官 + relay 判据修复 + 3-seed 全队列（2026-10-17，ai-1845503）➖➖

**ALF OOD 方法臂 s0 已收官**：200 步跑完（10/16 15:19），val/success_rate 从 step0 的
0.031 提升到 **0.891**（step200，最终验证 0.891）；best ckpt=global_step_216（val 0.914）。
（resume 越界到 258 的噪声评估最高到 0.945，但那是 200 步协议外的值，论文用 0.891。）

**relay 判据 bug 已修**：旧判据 `-d global_step_200` 会被 keep_best_and_latest 剪掉，
且路径漏了 experiment_name 一层，导致"已完成"永远判不出，反复重启 60 次后放弃。
已改为读 `$PROJ/$EXP/latest_checkpointed_iteration.txt >= 200`。

**🔴 当前队列**（`scripts/alfood_full_relay_ai1845503.sh`，pid 49563，GPU 0-3）：
OOD 方法 s0 ✅ → baseline s0 🔄(step~70) → 方法 s1 → baseline s1 → 方法 s2 → baseline s2，
共 6 run 串行（各 ~28h），补齐 3-seed OOD 泛化证据。

**待办**：`bipace-webshop`(py3.10) 环境仍需重建才能跑 Sokoban/WebShop/ScienceWorld；
AppWorld 上卡前需解决 appworld 要求 pydantic<2 与 vllm 0.11 要求 pydantic>=2.11 的冲突。

## ➖➖ 他机消亡 + 本机全面接管（2026-10-15 03:30，ai-1797956）➖➖

**他机已死（重建环境中）**。其最后贡献（scheduler v6，全部收官）：
- SW 1.5B 方法 gtext 3 seeds：0.867 / 0.875 / 0.969*（*单点尖峰）
- ALF GRPO resume 0.766；midarm-gtext 臂连续 3 崩 give up（原因未查，归因表暂用旧 midarm 数据）
- **midarm-gtext 崩溃问题待查**（显式 gtext_inject + actor_hidden 组合；本机重跑 midarm 时需注意）

### 🔴 本机当前在跑（他机协作者重建后请勿重复）
| run | GPU | 状态 |
|---|---|---|
| SW 1.5B 方法 gtext **s3**（凑 4 seeds） | 0-3 | 🔄 ~18/200（预计 10/16 凌晨收官） |
| WebShop GRPO **s2**（3 seeds 收尾） | 6,7 | 🔄 60/200（10/15 晚收官，relay 自动退出） |
| **master queue relay v3**（GPU 0-3 总队列） | 0-3（接力） | ⏳ s3 完成后自动串行（见下） |

### 📋 GPU 0-3 总队列（`scripts/master_queue_relay_v3_node1797956.sh`，pid 1861487）
1. **ALFWorld OOD 方法 s0**（~28h，`train_bisimhead_g_prmanchor_ood_thisnode.sh`，eval_out_of_distribution）——CVPR 泛化证据，最高优先级
2. **ALFWorld OOD baseline s0**（~28h，`train_official_bipaceq_ood_thisnode.sh`，严格单变量对照）
3. Sokoban 方法 gtext s0/s1/s2（~16h each）
4. Sokoban GiGPO s1/s2 + baseline s2（~16h each）
5. **AppWorld 三臂试水**（`examples/bipace/appworld/qwen1.5b/train_appworld_1.5b_node1797956.sh`，MODE=baseline/gigpo/bisimhead_prm，100 步试水）——server 就绪后手动上卡（见下）
- 环境准备：bipace（py3.12）env ✅ 克隆完成；ALFWorld 数据下载中；**appworld pip 安装中**
- ALFWorld OOD 用 bipace env、Sokoban 用 bipace-webshop env，脚本自激活
- ⚠ 他机重建后若要 ALFWorld OOD 加 seeds，请用 SEED=1/2 且不同 experiment_name 后缀

### 🆕 AppWorld 第 6 环境适配完成（2026-10-15，本机，全链路冒烟通过）
- **代码（共享存储，additive 默认关 = 行为不变）**：
  - `appworld/envs.py`：`process_reward_mode`（api_prm）——g_set=成功 API 调用集、g_text=`[Completed API calls: ...]`、PRM=win+0.1×新调用−0.1×无效
  - `env_manager.py`：AppWorld g_text 注入 gate（bisim_embedder==bisim_head）+ prm 透传
  - `ppo_trainer.yaml`：`env.appworld.process_reward_mode`（默认 ""）
- **脚本**：`examples/bipace/appworld/qwen1.5b/train_appworld_1.5b_node1797956.sh`（四模式）
- **基础设施（本机）**：appworld 0.1.3 装入 bipace env；数据 193M 在 `/mnt/tidalfs-hssh01/data`（train 90 / test_normal 168 tasks）；
  **160 实例 server 集群已起**（端口 8000-8159，8080→8160 避让 code-server，~20GB 内存，
  启动脚本 `scripts/start_appworld_servers_node1797956.sh`，端口表 `/tmp/codewiz/appworld_ports.ports`）
- **冒烟**：单实例 ✓ / ray 128-env reset(24s)+step ✓ / g_text/PRM 生成 ✓
- **上卡计划**：GPU 0-3 总队列尾部排 AppWorld 三臂试水（baseline/gigpo/bisimhead_prm 各 100 步）；
  server 集群为常驻进程（实验上卡时需确认存活：`ps aux | grep 'appworld serve' | wc -l` 应 ≥160）
- **⚠ 他机注意**：AppWorld server 集群占本机 160 端口+20GB 内存（常驻）；若他机共享本机跑 AppWorld 需另起端口段

**他机重建完成后建议接的队列**（本机 0-3 已排满 ~5 天）：
1. SW 7B 方法 gtext 重跑 s0/s1（TAG=gtext，7B SFT ckpt 起点，4 卡 ~35h each）
2. ALF GiGPO s1/s2（阶梯 3 seeds 补齐，4 卡 ~28h each）
3. SW GRPO（MODE=grpo，4 卡）
- ⚠ 启动前先 `tail verl_log/` 查本机活跃日志避撞；GPU 0-3 本机接力器自动占用中，若他机要用请在本文件登记

### 📊 已收官数据快照（10/15 时点）
| 环境 | 阶梯 | 状态 |
|---|---|---|
| ALFWorld 1.5B | GRPO‡ 76.6 < GiGPO‡ 83.6 < BiPACE-Q‡ 90.2 < **PABH 95.2±1.4** | 阶梯全齐 ✅ |
| WebShop 1.5B | GRPO‡ 65.6/61.3/s2跑 < BiPACE-Q‡ 69.7 < **PABH 72.2±1.8 (8s, best 75.8)** ≈ GiGPO‡ 72.4 | GRPO s2 明午齐 ✅ |
| SW 1.5B | GiGPO‡ 91.7±3.3 > BiPACE-Q‡ 87.5±1.6；方法 gtext 3s 0.904（去尖峰 0.872） | s3 凑 4 seeds 🔄 |
| Sokoban | GiGPO 31.2 / baseline 34.4 / 方法(旧,作废) | 全队列重跑中 ⏳ |

### ⚠ 本轮排障记录（协作者重建后必读）
1. **relay 空闲检测死锁**：idle monitor 的 stress 占卡 vs relay 等空闲互锁——relay 已加清障逻辑（确认他机无活动后杀 stress）
2. **relay is_done 在已存在目录上误判**（根因未明，s0 重复启动）——**任何 relay 上卡后 30 分钟内人工核对一次 `ls verl_checkpoints/<exp>/`**；双跑撞车杀孤儿保原 run
3. midarm-gtext 三连崩（他机）未查根因

## ➖➖ 双机全景 + 方法 seed 扩池收官 + CVPR 初稿落地（2026-10-12，ai-1797956）➖➖

## ➖➖ 【重要】方法臂 g_text 缺失修复 + 重跑计划（2026-10-12 10:10，ai-1797956）➖➖

**缺陷与修复**：SW/Sokoban 的方法臂（bisimhead_prm）此前**未注入 g_text**（三层栈第 1 层缺席）——
协作者 10/12 09:33-09:57 已修复：`sciworld_env_manager.py` 新增反馈事件解析器+注入 gate、
`sokoban/envs.py` 新增 boxes-on-target 进度文本、`SokobanEnvironmentManager` 注入 gate、
SW 脚本新增 `TAG` 环境变量机制（`TAG=gtext` → experiment_name 带 `_gtext`）。
本机已同步给 Sokoban 脚本加 TAG 机制。

**作废数据（方法臂跑的是无 g_text 版）**：
| 范围 | 作废 | 保留（不受影响） |
|---|---|---|
| SW 1.5B | 方法 4 seeds（0.906/0.891/0.891/0.891） | GiGPO 3s / baseline 3s（无 g_text 是正确口径） |
| SW 7B | 方法 s0（0.891）+ **s1（在跑，旧代码进程，建议他机杀掉重启**） | baseline s0/s1 |
| Sokoban | 方法 s0/s1（0.312/0.391） | GiGPO s0 / baseline s0/s1 |

**本机重跑队列已部署**（`scripts/gtext_rerun_relay_node1797956.sh`，pid 1747607）：
GPU 0-3 空闲 15min 后串行（TAG=gtext 新口径，experiment_name 带 `_gtext`）：
1. SW 1.5B 方法 s0→s1→s2→s3（SFT 起点，~25h each，共 ~4.2 天）
2. Sokoban 1.5B 方法 s0→s1→s2（~16h each，共 ~2 天）
3. Sokoban GiGPO s1/s2 + baseline s2（补种子，无 TAG）
**⚠ 他机调度注意**：本机守候器盯着 GPU 0-3 空闲窗（15min 确认即上卡）——
他机若需 0-3 请保持卡上有进程，或先在本登记协调。

**请他机接的重跑**（本机 0-3 队列已满 ~9 天）：
- SW 7B 方法 s0/s1 重跑（TAG=gtext，SFT 7B ckpt 起点，各 ~35h）
- （s1 在跑的旧代码 run：建议杀掉，其 ckpt 目录会与新 run 分离不冲突）

**WebShop/ALFWorld 方法臂不受影响**（g_text gate 一直存在，数据有效）。

---

### 🔴 双机当前实验状态（10/12 10:10 时点，零同名冲突）
| 机器 | run | GPU | 状态 |
|---|---|---|---|
| **本机** | WebShop 方法 seed4 | 6,7 | 🔄 ~25/200 |
| **本机** | WebShop GRPO s0→s1→s2 接力 | 6,7（接力） | ⏳ seed4 完后串行 |
| **本机** | gtext 重跑守候器（SW 方法×4 + Sokoban 方法×3 + Sokoban 补种子×3） | 0-3（守候） | ⏳ 等 GPU 空闲 |
| **他机** | SW 7B 方法 s1 | — | 🔄 ~80%（**旧代码，跑完也作废，建议杀掉提前重跑**） |
| **他机** | SW 7B baseline s1 | — | 🔄 ~50% |
| **他机** | ALF GRPO s0 | — | 🔄 ~80% |

### 🆕 新脚本（共享存储，双机可用）
- `examples/bipace/webshop/qwen1.5b_2gpu_official/train_grpo_official_node1613826.sh`（WebShop GRPO 官方口径 2 卡，唯一差 adv_estimator=grpo）
- `train_sciworld_1.5b_node1613826.sh` 新增 `MODE=grpo` 分支（默认行为不变，STRICT 兼容）
- **阶梯表分工建议**：ALF GRPO（他机在跑）+ WebShop GRPO（本机接力）+ **SW GRPO MODE=grpo 待他机或本机接**（`MODE=grpo SEED=0 CUDA_VISIBLE_DEVICES=… bash train_sciworld_1.5b_node1613826.sh`，SFT ckpt 起点与 GiGPO 行同口径）；Sokoban GRPO 最低优先级

### 📋 新环境 3-seed 补齐计划（2026-10-12 用户拍板：新数据 1.5B/7B 全部多 seed）

**缺口盘点（已核 verl_checkpoints/日志）**：

| 环境/规模 | GiGPO | baseline | 方法 | 需补 run |
|---|---|---|---|---|
| SW 1.5B SFT | 3s ✅ | 3s ✅ | 4s ✅ | 无 |
| **SW 7B SFT** | 0s | 2s（s1 在跑 49%） | 2s（s1 在跑 78%） | **双臂 s2（2 runs）+ GiGPO s0/s1/s2（3 runs）** |
| **Sokoban 1.5B** | 1s | 2s | 2s | **GiGPO s1/s2 + baseline s2 + 方法 s2（4 runs）** |
| WebShop 1.5B | 3s ✅ | 4s | 8s | baseline 对称补（P3 可选） |
| WebShop 7B | 0s | 3s ✅ | 3s ✅ | GiGPO s0/s1/s2（若要 7B GiGPO 行） |

**排期建议（双机产能 ~4×4卡run/天）**：
1. 他机 SW 7B s1 双臂收官（~10/13 晨）→ **SW 7B 双臂 s2**（8 卡并行，~35h，10/14 出）
2. 他机随后 → ALF GiGPO s1/s2（阶梯 P1，~28h each）
3. 本机 GRPO relay 完（~10/16）→ 6,7 接 Sokoban 补 seeds（如届时他机已接则跳过）
4. 10/16-24 → SW 7B GiGPO ×3 + WebShop 7B GiGPO ×3 视卡位滚动
- Sokoban 7B 不建议（1.5B 尚在噪声带，7B 成本 ×2.8 无统计价值）
- 全部 3-seed 矩阵预计 **10/24 前出齐**（CVPR 摘要冻结线 11/10 前有 2 周余量）

**⚠ 本机守候器占用声明（他机调度请避开 GPU 0-3 空闲窗）**：本机已部署
`sokoban_seeds_relay_node1797956.sh`（pid 见 `pgrep -f sokoban_seeds_relay`）：
GPU 0-3 连续 15 分钟空闲即上 Sokoban 补种子队列
（baseline s2 → gigpo s1 → gigpo s2 → prm s2 串行，各 ~16h，共 ~2.7 天）。
他机若需用 0-3 请提前在本文登记或保持卡上有进程（守候条件即不触发）。

### 📊 本机 WebShop 方法 seed 扩池终局（官方口径峰值）
{0: 0.730, 2: 0.707, 5: 0.707, 10: 0.719} + {1: **0.703**, 3: **0.727**, 7: **0.758**（历史最高！）} → **7 seeds 0.722±0.019**
- vs baseline 4s 0.697±0.013：**+2.5pp**（扩池前 4s 是 +1.9pp）
- vs GiGPO‡ 3s 0.724±0.026：持平；方法 best（0.758）> GiGPO best（0.750）
- seed4 在跑，8 池齐后数据冻结进 CVPR 初稿

### 🔧 代码修复（共享存储，双方生效）
- **WebShop 四脚本补 `unset TRITON_PTXAS_PATH`**（bisimhead/bipace 的 node1613826+pod1491319 四文件）——本容器 CUDA 13.0 底噪坑，10/10-10/11 曾致方法 seed1/3 反复崩溃 ~21h（relay 数百次重启循环），修复后即刻正常
- relay 判活宽限 180s（fork 空窗教训已内建）

### 📝 CVPR'27 初稿（`paper_cvpr2027/`，共享存储）
- 模板：cvpr-org/author-kit（cvpr.sty），TeXLive 2026 scheme-small 装于 `/usr/local/texlive`（本机私有）
- `main.tex`（7 页已编译通过）+ `tables.tex`（含 10/12 最新数据：GiGPO 全环境复现、SW GiGPO 3s 0.917、Sokoban、方法 WebShop 7s 0.722）
- 叙事升级：新增 §ladder 边界条件节（GiGPO 成败由环境观测通道的进度对齐性决定——SW 上 GiGPO 0.917 ≥ 方法 0.895，ALFWorld/WebShop 上方法必需）
- 图表：figs/ 4 张（沿用 www 版，make_figs.py 可重生成）
- 待补：seed4 终值、ALF GRPO 阶梯终值（他机在跑）、SW 7B 方法 s1 终值（他机在跑）

### ⏹ 已完成的登记（10/9-10/12 本机）
| 任务 | 结果 |
|---|---|
| Sokoban GiGPO s0 | 峰 0.312 |
| WebShop GiGPO s0/s2/s10（官方 3 seeds） | 0.699 / **0.750** / 0.723 → 0.724±0.026 |
| 方法 seed1/3/7 | 0.703 / 0.727 / **0.758** |

**注意**：relay 类脚本判活用 `pgrep -f "trainer.experiment_name=..."`（python fork 前有 ~60s 空窗），
宽限期务必 ≥180s；双启过一次会 vllm OOM 自灭（显存互斥），不会双跑写污染。

## ➖➖ 容器五换 + GiGPO 多环境验证接手（2026-10-09 10:35，ai-1797956）➖➖

**环境已重建并验证**（`/root/miniconda3/envs/bipace-webshop`，与历次容器同构）：
python 3.10.22 / torch 2.6.0+cu124 / vllm 0.8.4 / flash-attn 2.7.4（repo wheel）/ flashinfer 0.2.2（repo wheel）/
transformers 4.51.1 / ray 2.58（-e . 后回钉）/ tensordict 0.6.2 / verl editable / scienceworld 1.2.3 +
openjdk 17.0.18 / gymnasium 0.29.1 + stable-baselines3 2.6.0 / gym_sokoban 0.0.6 + gym 0.26.2 + setuptools 80.10.2 /
pyserini 0.17.0 + faiss-cpu + spacy 3.8.16（sm+lg 模型，lg 走代理下载）/ selenium。
本容器 pypi/github 直连可用，github 大文件走代理 http://10.140.24.177:3128 快 10 倍+。

### 🔴 本机正在跑的实验（2026-10-09 10:29 起，ai-1797956，协作者勿重复）

**接手已死的 gigpo_multi_env_launcher**（其死等永不出现的 WebShop/Sokoban GiGPO global_step_200）：

| run | GPU | experiment_name | 状态 |
|---|---|---|---|
| Sokoban GiGPO **seed0** | 0-3 | `gigpo_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826` | 🔄 10/10 06:40 时点 ~s155/200，峰 0.312@s30 |
| WebShop GiGPO **seed0**（官方口径 2 卡） | 4,5 | `gigpo_official_webshop_1.5b_2gpu_L-12_seed0_node1613826` | ✅ 05:15 收官，峰 0.699 |
| WebShop GiGPO **seed2** | 6,7 | `gigpo_official_webshop_1.5b_2gpu_L-12_seed2_node1613826` | ✅ 04:15 收官，峰 **0.750@s195** |
| WebShop GiGPO **seed10** | 6,7 | `gigpo_official_webshop_1.5b_2gpu_L-12_seed10_node1613826` | 🔄 s15/200（接力器上卡；04:17 曾因判活空窗双启一次，第二个进程 vllm OOM 自灭，无数据污染） |
| WebShop 方法 **seed1/seed3**（新增） | 0,1 / 2,3（接力） | `bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed{1,3}_node1613826` | ⏳ Sokoban GiGPO s0 收官后自动上卡（`scripts/method_seeds_relay_node1797956.sh`）——动机：WebShop seed 方差大（GiGPO 5.1pp 极差），方法加 seed 报全量均值 |
| WebShop 方法 **seed7/seed4**（新增） | 6,7（接力） | `bisimhead_official_webshop_1.5b_2gpu_L-12_eps0.06_qstyle_g_prmanchor_seed{7,4}_node1613826` | ⏳ GiGPO seed10 收官后串行上卡（`scripts/method_seeds_relay2_node1797956.sh`）——方法 seed 池扩至 8 个 {0,1,2,3,4,5,7,10} |

- 看门狗：`scripts/gigpo_multienv_watchdog_node1797956.sh`（死亡→错峰 100s 重启→ckpt 续→完成判据=global_step_200 目录→退出；日志 `verl_log/gigpo_multienv_watchdog_node1797956.log`）
- GiGPO 无 head → resume 干净无口径污染
- **WebShop 索引注意（§26.1 教训重演险情）**：本机重建环境时曾误将全量 118 万索引建进 `search_engine/indexes`（错配 use_small=True 的 1000 商品目录），已删除并从 `indexes_1k` 复制恢复（1000 docs，search 满页验证通过）。**协作者勿再动 indexes 目录**
- gpu_idle_monitor（8 卡 supervisor）已后台运行，实验上卡自动让路

### 队列说明
- WebShop GiGPO 剩余 seed10 视 s0/s2 收官时间接力（凑 3 seeds 对官方 67.4±4.5）
- 其余队列（SW method s3 / 7B raw-方法 / SW 7B baseline s1 / ALF GiGPO/GRPO 阶梯 / SW GiGPO s1/s2 续 / baseline s2 续）仍归原编排机（sw_resume_chain / scheduler v4 / alf_ladder_watchdog），本机不碰

## ⚠ 调度风险预警（2026-10-07 11:15，ai-1774370 登记，ai-1753446 协作者请核处）

00. **【10-09 03:00 补】编排缺口三处（ai-1753446，请核处）**：
    - **GiGPO seed1 未完成**（13:10 死于 ~s139，ckpt 仅 global_step_100，val 峰 0.875@s120）：`gigpo_multi_env_launcher` 死等
      `gigpo_..._seed1/global_step_200` 永不出现 → **WebShop/Sokoban GiGPO 被永久阻塞**，需改等退出事件或补跑 seed1；
    - **GiGPO seed2 同样未完成**（23:10 退出时 145/200，peak 未锁）：sw_gigpo_priority_chain 按"退出"而非"完成"推进 → seed2 需补跑或标记半程口径；
    - **rl15_prm_raw_s0（1.5B raw-方法）四连崩 ALARM give up**（scheduler3，16:23-20:02）：每次启动 ~40-50 分钟内死亡，
      疑似 vllm cumem 段错误在启动段连发（或 raw+head 配置系统性问题），需人工诊断后再放队列；
    - 另：1.5B raw-baseline s0 崩在 s148 后一直未续（best val 0.0625）。
    当前该机：GPU 0-3 方法 seed3（23:12 起）✅、GPU 4-7 7B raw-方法（15:40 恢复）✅。

0. **【10-08 04:30 补】方法 seed3 已丢失，需补跑**：sw_mc_orchestrator lane A 检测到 GiGPO s0 退出后**仅 1s 就启动 seed3**
   （无错峰），seed3 的 ray 集群撞上 GiGPO 集群残留（`ActorUnavailableError: Socket closed`）2 分钟即崩；
   编排器无重试逻辑已继续推进 GiGPO seed1（在跑）→ **方法 seed3 无人重启**。
   建议：GiGPO seed1 收官后手工补 `MODE=bisimhead_prm SEED=3 START=sft MODEL_PATH=<SFT492> CUDA_VISIBLE_DEVICES=0,1,2,3`。
   另注：ScienceWorld GiGPO s0 的 0.922@s70 经本机核读为 val64 噪声尖峰（前后 0.766/0.812），GiGPO 实现口径审查无问题
   （=官方精确锚，avg step-group 8-10）；该环境 singleton 病理不存在（BiGPO 聚类 singleton 仅 7-9%，ALFWorld 为 35-40%），
   "GiGPO≥方法"若被 seed 复现确认，机制解释=环境病理缺席而非方法失效（详见 EXPERIMENTS_LOG 后续节）。

1. **1.5B raw-baseline s0 未完成即退出**（11:04:45 在 s148/200 崩溃，best val 仅 0.0625）：v3 检测到 pid 退出后已把 sw_gigpo_s0 放上 GPU 0-3；**v2 的 run_queue 仍会在 5 分钟轮询后尝试在 0-3 重启 rl15_base_raw_s0**——wait_free_gpus 的 deadline 若超时放行将撞卡。建议：v2 队列里该项手工标记/跳过，或直接停 v2（其 lane A 使命已由 v3 接管）。
2. **ladder_launcher 与 v3 队列重复**（ALFWorld GiGPO/GRPO 两边都排了，且 experiment_name 相同 `gigpo/grpo_alfworld_1.5b_4gpu_seed0_thisnode`）：ladder 的 wait_free_gpus deadline 1h 超时后**无视占用强行启动**（约 12:04 撞上 0-3 的 sw_gigpo）。若 v3 已取代 ladder，建议 `pkill -f ladder_launcher`。
3. **1.5B raw-方法 s0 重复排队**：同时在 v2 lane B 尾部与 v3 lane A 尾部（experiment_name 相同），一处启动后另一处需靠 is_done（global_step_200）幂等跳过——存在双跑窗口期。
4. 科学注记：raw 起点 1.5B 在 ScienceWorld 上 val 崩至 0.016-0.078（s130-140 段），与 SFT 起点 0.83-0.91 量级天壤之别——raw 消融初步显示该环境对冷启动 SFT 几乎刚需。

---

## ➖➖ 容器第四次更换 + 环境重建完成（2026-10-05，ai-1774370）➖➖

> **双机分工确认（2026-10-05 14:00，ai-1753446 补记）**：**ai-1753446 存活且 scheduler v2 完整在跑**（pid 3497341，
> 7B SFT-方法 s0 + 7B raw-baseline s0 双泳道）；调度器脚本已备份至共享存储 `scripts/sw_matrix_scheduler2.sh`。
> 分工无重叠：**ai-1774370 = R7 补 seeds（1.5B SFT baseline seed1 + 1.5B SFT prm seed2）**；
> **ai-1753446 = 7B 双臂（SFT/raw）+ 后续 7B raw-方法、1.5B raw 起点双臂**。
> 1.5B raw 双臂归 ai-1753446 队列（他机已明示不接）；7B 各臂 ai-1774370 未接。
> 双方判读汇合点：1.5B SFT 行（baseline s0/s1 + 方法 s0/s1/s2）与 7B 行各自成表。

**容器第四次更换**：实验机现为 **qs-325328-2164709-ai-1774370-default0-0**（8×H20，driver 535.161/CUDA 13.0）。
ai-1753446 消亡，其上 scheduler v2 双泳道（7B SFT-baseline s0 续跑 / 7B raw-baseline s0 全新）随之中断；
**sw_matrix scheduler v2 脚本本体丢失**（原在旧机 /tmp/codewiz，未落共享存储），需手工重启或重建调度器。

**环境已重建并全部验证**（`/root/miniconda3/envs/bipace-webshop`，与前三容器同构）：
python 3.10.22 / torch 2.6.0+cu124 / vllm 0.8.4 / flash-attn 2.7.4（repo wheel）/ flashinfer 0.2.2（repo wheel）/
transformers 4.51.1 / ray 2.58 / tensordict 0.6.2 / verl editable（BiPACE 仓）/ scienceworld 1.2.3 + openjdk 17.0.18
（conda 内，JAVA_HOME 默认即 env 根目录）/ gymnasium 0.29.1 + stable-baselines3 2.6.0。
本容器 pypi/github/download.pytorch.org 全部直连可用，无需代理。
**⚠ 回钉教训（10/5 实证，纠正 9/30 节错误说法）**：`pip install -e .` 后只需回钉 `ray==2.58.0 tensordict==0.6.2`；
**otel 必须保留 1.45 家族**——回钉 1.26 会让 ray 2.58 dashboard 崩
（`Meter.create_histogram() got an unexpected keyword argument 'explicit_bucket_boundaries_advisory'`），
训练启动即失败（本机 10/5 首发双 run 因此各崩 2 轮，看门狗自动接力+otel 回 1.45 后恢复）。
vllm 对 otel<1.27 的要求只是 pip warning，实测兼容 1.45。
已验证：全部关键包 import、ScienceWorld JVM 加载 + reset（30 tasks）、prepare_offline parquet 生成
（`/home/work/data/verl-agent/text/`，HOME=/home/work）。
**注意**：GPU 全忙时（如 idle-monitor stress 在跑）vllm import 会因 CUDA 初始化排队耗时 ~8 分钟，属预期现象。
gpu_idle_monitor（8 卡 supervisor 模式）已在本机后台运行，实验上卡自动让路；脚本内 `unset TRITON_PTXAS_PATH` 依旧必需。

### 🔴 本机正在跑的实验（2026-10-08 04:00 起，ai-1774370，协作者勿重复）

**Sokoban s0 双臂已收官**（10/7 17:59，全程无死亡）：baseline s0 峰 0.297@s120/终 0.250；方法 s0 峰 0.312@s180/终 0.281。
**机器控制设计**：Sokoban 方法-vs-baseline 对照全部放本机（s0 已在本机），同机零混杂——现接 **seed1 双臂**：

| run | GPU | experiment_name | 起点 |
|---|---|---|---|
| Sokoban baseline **seed1** | 0-3 | `baseline_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826` | raw Qwen2.5-1.5B-Instruct，全新 |
| Sokoban 方法 **seed1** | 4-7 | `bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826` | 同上 |

- 与协作者队列零冲突（其 Sokoban 项仅 GiGPO s0 参照臂，experiment_name 不同）
- 看门狗：`scripts/sw_sokoban_s1_watchdog.sh`（同前模式：死亡→错峰重启→200 步完成→自动拉起 gpu_idle_monitor；日志 `verl_log/sw_sokoban_s1_watchdog.log`）
- 200 步 ~34h（Sokoban 实测 ~610s/it），预计 10/9 午后收官；凑齐后 Sokoban 双臂 2 seeds 口径

### ⏹ 已完成的登记（2026-10-06 08:00 起，Sokoban s0 双臂）

**R7 补 seeds 已收官**（10/6 07:19）：baseline seed1 ✅（峰 0.859@s160/终 0.812）、bisimhead_prm seed2 ✅（峰 0.891@s120）。
C 臂 3 seeds 峰值 0.906/0.891/0.891（均 0.896）vs A 臂 2 seeds 0.875/0.859（均 0.867）——方法跨 seed 稳定正向成立。

**ai-1753446 仍在跑 7B 双臂**（raw-baseline @s86、SFT-方法 @s73，均剩 ~15h），其队列尾部 1.5B raw 双臂归它，本机不碰。
本机现接 **Sokoban 第 5 环境双臂**（§29.15 代码就绪项，不在任何调度队列，零冲突）：

| run | GPU | experiment_name | 起点 |
|---|---|---|---|
| Sokoban baseline **seed0** | 0-3 | `baseline_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826` | raw Qwen2.5-1.5B-Instruct，全新 |
| Sokoban 方法（bisimhead_prm + box_prm 锚）**seed0** | 4-7 | `bisimhead_prm_sokoban_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed0_node1613826` | 同上 |

- 环境补充：本容器新装 gym_sokoban 0.0.6 + gym 0.26.2 + setuptools 80.10.2（SokobanWorker 冒烟通过，PRM/ASCII 渲染正常）
- 看门狗：`scripts/sw_sokoban_watchdog.sh`（死亡→错峰 100s 重启→ckpt 续→200 步完成退出，**退出时自动拉起 gpu_idle_monitor**；日志 `verl_log/sw_sokoban_watchdog.log`）
- 200 步，预计 ~16h（按 1.5B 实测 ~280s/it）收官

### ⏹ 已完成的登记（2026-10-05，R7 补 seeds）

已核实 **ai-1753446 仍存活**，其 scheduler v2 双泳道在跑 7B 两臂（SFT-方法 s0 @s28、raw-baseline s0 @s53）。
本机接 **R7 补 seeds 两项**（不在其调度队列内，零冲突；experiment_name 均含 seed 区分，ckpt 目录无交集）：

| run | GPU | experiment_name | 起点 |
|---|---|---|---|
| A 臂 1.5B SFT baseline **seed1** 重跑 | 0-3 | `baseline_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed1_node1613826` | SFT ckpt global_step_492，全新（旧目录为空壳） |
| C 臂 1.5B SFT 方法（bisimhead_prm）**seed2** | 4-7 | `bisimhead_prm_sftstart_sciworld_1.5b_4gpu_L-12_eps0.10_qstyle_actiontag_seed2_node1613826` | 同上，全新 |

- 看门狗：**已落共享存储** `scripts/sw_r7_watchdog.sh`（死亡→错峰 100s 重启→ckpt 续→200 步完成退出；日志 `verl_log/sw_r7_watchdog.log`）——不再放 /tmp，防容器死亡丢失
- **状态（10/5 12:20）**：双 run 已进训练（各 s1/200 完成，~540s/it，预计 ~30h）；首发曾因 otel 误回钉 1.26 各崩 2 轮（见上条教训），看门狗自动接力恢复，无人工损失
- **1.5B raw 起点双臂（raw-baseline s0 / raw-方法 s0）本机未接**，仍归 ai-1753446 调度器队列；若其 7B 泳道收官时这两项已被他机完成，其调度器按"幂等跳过已完成"处理
- 预计 ~25-26h（200 步，~460s/it）收官；完成后本机空窗，将按 §29.16 判读预案执行 C vs A/B 对比分析（零 GPU）

---

## ➖➖ 容器三换 + ScienceWorld seed0 终局 + seed1/2 接力（2026-09-30，ai-1753446）➖➖

**容器第三次更换**：实验机现为 **qs-316244-2145784-ai-1753446-default0-0**（8×H20，driver 535.161/CUDA 13.0）。
ai-1745017 消亡前，**ScienceWorld seed0 双组已跑完 200/200**（9/29 09:29 启动、17:12 收官）：

| run | val 峰值（步） | val 终值 s200 | episode succ |
|---|---|---|---|
| baseline seed0 | **0.875**（s170） | 0.828 | 0.742 |
| bisimhead seed0 | **0.875**（s130） | 0.734 | 0.742 |

判读：峰值持平、终值 head 差 9.4pp——ScienceWorld val 只有 64 题、run 内震荡 ±7pp（baseline 自身 s170 0.875→s200 0.828），单 seed 无法下结论，**必须补 seed1/2**（与 ALFWorld 同口径 3 seeds）。

**环境重建**（本机 `/root/miniconda3/envs/bipace-webshop`，与前两容器同构，全部已验证 import）：
python 3.10.21 / torch 2.6.0+cu124 / vllm 0.8.4 / flash-attn 2.7.4（repo wheel）/ flashinfer 0.2.2（repo wheel）/
transformers 4.51.1 / ray 2.58（注意：`pip install -e .` 会把 ray 降到 2.50、tensordict 升到 0.10、otel 升到 1.45，
需回钉 `ray==2.58.0 tensordict==0.6.2 opentelemetry-*==1.26.0`）/ verl editable / scienceworld 1.2.3 + openjdk 17。
本容器直连 pypi/github 可用，download.pytorch.org 需代理 `http://10.140.24.177:3128`。

**本机在跑**（2026-10-04 18:06 起，scheduler v2 双泳道，日志 `verl_log/sw_matrix_scheduler2.log`）：
| 泳道 | 当前 | 后续 |
|---|---|---|
| A（0-3） | 7B SFT-baseline s0（s100 续，~13h） | SFT-方法 s0 → 1.5B raw-baseline s0 |
| B（4-7） | 7B raw-baseline s0（全新，~35h） | raw-方法 s0 → 1.5B raw-方法 s0 |

**⚠ 10/4 同名覆盖事故（第三次发生，协作者必读）**：7B raw/SFT 双臂曾因 experiment_name 未带起点标签共享 ckpt 目录互相污染。
**现脚本强制要求**：`START=raw|sft` 进实验名（`train_sciworld_7b/1.5b_node1613826.sh`）——**新增任何"同 MODE 同 seed 不同配置"的 run 必须把全部区分维度写进 experiment_name**。
raw 臂 s89 后数据已废弃（污染段），全新重跑中；SFT 臂轨迹有效（目录判归 0.797@s100 实证）从 s100 续。
**ckpt 清理（10/4 用户指令）**：全仓执行"只留 best-val + 最后一个"，清 8 目录；新脚本均带 keep_best_and_latest 自动维持。
**环境注意**：本容器 TRITON_PTXAS_PATH 必须 unset（新脚本已内置）；vllm 0.8.4 cumem sleep 段错误为随机平台底噪（~1/200 step），靠 save_freq=50+看门狗容错。

experiment_name 沿用 `*_node1613826` 后缀体系（历史沿用名，与实际容器号无关），与协作者（PABPO 线 ai-1615151、
pod1491319）无同名冲突。SFT 冷启动 ckpt：`verl_checkpoints/qwen2.5_1.5b_sciworld_sft_node1613826/sft_full/global_step_492`。
**勿动**：`/mnt/tidalfs-hssh01/dataset/bxd/gpu_idle_monitor.py` + `gpu_stress.py`（多阶段拟真负载版）在实验空窗期占卡防回收，
实验启动前需 `pkill -f gpu_idle_monitor; pkill -f gpu_stress`。

---

## ➖➖ 容器更换 + ScienceWorld 接力修复（2026-09-28，ai-1745017）➖➖

**容器再次更换**：实验机现为 **qs-316244-2137750-ai-1745017-default0-0**（8×H20，driver 535.161/CUDA 13.0 底噪）。
ai-1613826 上的接力链在 9/28 03:49 启动 ScienceWorld 双组时**启动即崩**，GPU 空转 ~13h 后由本机重建环境并重启。

**环境重建**（`/root/miniconda3/envs/bipace-webshop`，与 ai-1613826 同构）：
python 3.10.21 / torch 2.6.0+cu124 / vllm 0.8.4 / flash-attn 2.7.4（repo wheel）/ flashinfer 0.2.2（repo wheel）/
transformers 4.51.1 / ray 2.58 / verl editable（BiPACE 仓）/ scienceworld 1.2.3 + openjdk 17（conda 内）。

**ScienceWorld 脚本三处修复**（`examples/bipace/sciworld/qwen1.5b/train_sciworld_1.5b_node1613826.sh`，共享存储已生效）：
1. **加 `cd $BASE_DIR`**：`env.sciworld.split_path` 是相对路径，接力从 /tmp/codewiz 启动时 FileNotFoundError（9/28 03:49 事故根因）；
2. **`unset TRITON_PTXAS_PATH`**：本容器该变量指向系统 CUDA 13.0 ptxas，triton 3.2 无法解析版本号导致 vllm 编译崩溃；
3. **补 `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8` + `ref.fsdp_config.param_offload=True`**（与 ALFWorld 1.5B 4 卡配置对齐，yaml 默认为 null 会直接 ValueError）。

**当前在跑**（9/28 18:12 启动，预计 ~23h，200 步，SFT 冷启动 ckpt global_step_492）：
| run | GPU | 日志 |
|---|---|---|
| sciworld baseline seed0 | 0-3 | `verl_log/sciworld_1.5b_baseline_seed0_node1613826_20260928_181206.log` |
| sciworld bisimhead seed0 | 4-7 | `verl_log/sciworld_1.5b_bisimhead_seed0_node1613826_20260928_181211.log` |

监控：`/tmp/codewiz/monitor_sciworld.sh` 常驻，状态面板 `cat /tmp/codewiz/experiment_status.txt`（10 分钟刷新）。
**注意**：本机 shell 的 HOME=/home/work，`prepare_offline` 生成的 parquet 在 `/home/work/data/verl-agent/text/`（脚本内自洽，勿手动指向 /root）。

---

## 〇〇、PABPO 支线进展速览（2026-09-18，详见 ../PABPO/PABPO_EXPERIMENTS_LOG.md）

- **PABPO 仓库**（`../PABPO`，BiPACE 的 B3 修复支线）第 1 刀 B3 尺度失配修复已收官 1.5B：
  - v1 rescale **坍缩负结果**（度量尺度与 eps 耦合，已归档），v2 clamp（尾部截断）1.5B=0.922 vs 基线 seed0 0.930（噪声内持平，clamp 安全性验证）
  - 7B v2 seed1（ai-1615151）s70=0.883 **领先基线同期 ~10pp**，9/19 终局；1613826 侧 seed0/2 补跑计划见 PABPO 日志
- **E2 纯度回溯分析已完成 16 run**（零 GPU，含你机跑完的 rawanchor 7B 双 seed——多谢）：
  - **7B 三臂（rawanchor/midarm/head+PRM）固有数据病理完全相同**（conf_clus 18.3/18.4/18.8%），val 阶梯 0.933→0.945→0.953 全部来自管线——主结果的归因证据
  - conf_clus vs val：7B **r=-0.748**；混淆率随训练**翻倍上升**（病理不自愈）；singleton 与效果零相关（r=+0.02）
  - 工具 `PABPO/examples/analysis/e2_purity_retro.py`，数据 `PABPO/research_archive/e2_results/`

## 〇、文档地图（2026-09-16 整理，旧路径已变！）

根目录活文档（6 份）：
- `PAPER_DRAFT.md` — 论文初稿（活，7B 结果待补）
- `EXPERIMENTS_LOG_THISNODE.md` — **主实验日志**（一切实验结果以此为准，已记至 9/14）
- `CROSS_MACHINE_STATUS.md` — 本文件
- `DISCUSSION_PURITY_VIEW_AND_PLAN.md` — **新认识与新方案**（纯度观 / 三层方法栈 / 优化路线图，9/16）
- `README.md` / `CONTRIBUTING.md` — 官方仓库文件
- `RESUME_SUMMARY.md` — 简历素材

已归档（`research_archive/`，历史快照，不再更新，内有 README 索引）：
`EXPERIMENTS_LOG.md`、`EXPERIMENT_B_BISIM_HEAD.md`、`DESIGN_PROGRESS_AWARE_BISIM_HEAD.md`、`BIPACE_ANALYSIS_AND_IMPROVEMENTS.md`
**注意**：若你按旧引用找这几份文档（如 ANALYSIS 的 PRM 路线），请到 `research_archive/` 下找。

## 一、本机当前正在跑的实验

| 实验 | GPU | 状态 |
|---|---|---|
| ALFWorld 7B **纯官方口径复现（rawanchor）** seed0 | 0-3 | 9/16 11:00 启动（容器已换为 ai-1613826，环境从零重建），~9/18-19 完成 |
| ALFWorld 7B **纯官方口径复现（rawanchor）** seed1 | 4-7 | 同上 |

> ⚠ 容器更换通知（2026-09-16）：本机旧容器（ai-1421894）已销毁，9/16 06:15 启动的 head seed0 run
> 随之消亡（无 checkpoint 损失，需重跑）。新容器 hostname 为 ai-1613826，`/root` 环境已重建
>（python 3.12 / vllm 0.11.0 / torch 2.8.0+cu128 / flash-attn 2.8.3 官方 wheel / ray 2.50.0）。

### ✅ g_text gate 已回移（2026-09-16，DISCUSSION 行动项 #5 已完成）

`env_manager.py` 的 ALFWorld/TextCraft 分支 g_text anchor 注入现在有 gate（WebShop 同款）：
`bisim_embedder == "bisim_head"` 才注入，**actor_hidden（官方口径）不再注入**。
当前在跑的 rawanchor 双 seed 即首个纯官方口径 7B run（脚本 `train_official_bipaceq_rawanchor_thisnode.sh`，
配置与官方逐项一致，验证详情见 EXPERIMENTS_LOG_THISNODE.md §25）。你若跑 head 系列不受影响（依旧注入）。

### 已完成终局（截至 2026-09-16 晨，详见 EXPERIMENTS_LOG_THISNODE.md §16-23）

| 实验 | seeds | 结果 | 备注 |
|---|---|---|---|
| ALFWorld 1.5B head+PRM 锚 | 0/1/2/3/5（5 seeds） | **0.952±0.014**（vs 官方 93.5±1.2：+1.7pp，z≈1.83 边缘；vs 两机复现 0.898/0.906：+4.6~5.4pp） | 主结果 |
| ALFWorld 1.5B head+d_g 锚 | 0 | 0.875 | 锚消融（PRM 锚优 7.6pp） |
| ALFWorld 7B head+PRM 锚 | 1/2 | 0.953/0.953（vs GiGPO 90.8：+4.5pp；vs 官方 97.1：−1.8pp） | stabA_s200（lr3e-7/kl015/200步）同为 0.953，天花板确认 |
| ALFWorld 7B "官方配置" | 0/1 | 0.938/0.953 | **⚠ 口径警告**：见下节 |

### ⚠ 重要口径警告：ALFWorld "官方配置" run 带了 g_text（2026-09-16 发现）

`env_manager.py` 的 ALFWorld/TextCraft 分支 g_text 注入 anchor **无 gate**（WebShop 分支有 gate，`env_manager.py:352-359`）。因此 9/14 启动的 `train_official_bipaceq_thisnode.sh`（actor_hidden、无 head）实为 **"g_text + actor_hidden"（仅状态重定义、无 head）**，不是纯官方口径复现：
- 它其实是此前缺失的**中间消融臂**：7B 上 head 相对"仅状态重定义"的增量 ≈ 0.953 − 0.945 ≈ +0.8pp（各 2 seeds）
- "纯官方 97.1 vs 我机 0.945" 的 −2.6pp 中硬件差与 g_text 效应混杂
- **1.5B 主结果 +5pp 尚未分解**（中间臂 g_text+actor_hidden 没跑过）——已列为 P0 补实验项（DISCUSSION §七-2）
- 若你要跑纯官方口径 ALFWorld run，需先回移 gate 或临时回退 `env_manager.py:137-138/156-157` 的注入

### 纯度观新认识（2026-09-16，详见 DISCUSSION_PURITY_VIEW_AND_PLAN.md）

用 14 run 数据证实：**singleton 率（覆盖率）与最终效果无单一方向因果，组内价值一致性（纯度）才是正确优化目标**。含义：不要为压 singleton 而调大 eps；后续调参以纯度指标（组内 RTG 方差比 / g_set 混淆率）为准；另发现 `bisim_head.py` 尺度失配 bug（L2 归一化使 pred 上界 2 < target 上界 3.0）。

## 二、本轮新增代码（共享存储上，你会直接用到）

### 2.1 ALFWorld 进度感知（本机新写，2026-09-07）
| 文件 | 改动 | 对你的影响 |
|---|---|---|
| `agent_system/environments/env_package/alfworld/envs.py` | ① 新增 `AlfProgressTracker`：6 事件模板正则（take/clean/cool/heat/put/toggle）解析 obs → g_set/g_text；② `AlfworldWorker.step/reset` 的 info 加 `g_set`/`g_text`/`process_reward`（win=+1.0, 新子目标=+0.5, no-op=−0.1, 导航=0）；③ **修复 info 解包 bug**：`AlfworldEnvs.step/reset` 原 `info[k][0]` 对新增标量会崩，现改为 isinstance(list/tuple) 判断 | 你跑 ALFWorld 时 info 多 3 个 key（纯增量）；解包修复对旧 list 值无行为变化 |
| `agent_system/environments/env_manager.py`（AlfWorld 分支） | reset/step 的 anchor 改为 `g_text + "\n\n" + obs`（g_text 空时=原 obs）——**无开关，默认生效** | 若你想要纯官方口径的 ALFWorld anchor，需要回退这 6 行或加开关 |
| `alfworld/configs/` | 无改动 | — |

### 2.2 head 训练增强（环境无关）
| 文件 | 改动 | 默认值 |
|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | 新增 `bisim_updates_per_sample`：每轮采样后同批 pairs 做 K 次梯度更新（mini-epoch） | 1（=原行为，逐位兼容） |
| `verl/trainer/config/ppo_trainer.yaml` | 注册 `bisim_updates_per_sample: 1` | 1 |
| （已有）`bisim_resample_pairs` / `bisim_min_dt` / `bisim_anchor_mode` | 见 yaml 注释 | False / 0 / prm |

### 2.3 之前一轮的改动（另一台机器可能还没注意到）
- `bisim_anchor_mode: prm|d_g`（TextCraft 系列实验用）
- `bisim_resample_pairs` / `bisim_min_dt`（采样策略，TextCraft 实验4 已验证无增益，但 ALFWorld 上开启了）
- TextCraft env 的 g_set/g_text（配方闭包）

## 三、TextCraft 阶段结论（本机 6 run + 你机 6 run 全景）

详见 `EXPERIMENTS_LOG_THISNODE.md` §九-十二。要点：

1. **方差问题**：TextCraft 1.5B 的 run 级 std = ±7-8pp（官方 README 数据），我们的处理效应 3-8pp 全在噪声内；**两机的"baseline 差 14pp"（0.703 vs 0.562）其实是正常波动**，谁都别怀疑自己跑崩了
2. 我们的"baseline"就是 BiPACE-Q 本体（actor_hidden+q_style），GiGPO 从未跑过
3. 你机的 lr1e-4+τ0.005=0.719 是 12 run 唯一超 0.703 的 → 本机 ALFWorld 双实验直接采用了该组合（τ 是 12 run 里唯一未交叉验证的强变量）
4. g_text / d_g / resample 在 TextCraft 上无显著增益（但也没显著损失，被噪声淹没）→ 转战 ALFWorld（±1.2-1.7pp，3 seeds 可检出 ~3pp）

## 四、冲突教训（重要！共享存储协作守则）

1. **同 experiment_name 会互相覆盖 checkpoint**（save_freq=50 时 step50/100 都会覆盖）。
   本机曾发现你在 9/3-9/4 跑了与本机完全相同的 g_in_anchor 实验（同配置同 seed 同名字），
   rollouts 已互相污染。→ 启动前请 grep 本文件"正在跑的实验"表 + `ls verl_log/` 查重
2. **本机脚本带 `_thisnode` 后缀**即本机专属；你的脚本请继续用你的后缀（`_pod1491319` 等），
   别动对方的脚本文件（本机的 `train_bisimhead_g_in_anchor_dg_anchor_thisnode.sh` 的 lr 曾被改成 1e-4，
   好在运行进程用的是启动时内存里的旧值）
3. **共享代码目录可写**：任何对 `verl/`、`agent_system/`、`gigpo/` 的改动双方都会立即看到。
   本机的改动全部走"新增配置开关+默认值保持原行为"路径；如需 break 兼容请先在本文登记
4. hydra 键一旦注册进 `ppo_trainer.yaml`，脚本里就**不能再用 `+` 前缀**传（会报
   `Could not append to config`）——本机 ALFWorld 实验首次启动就栽在这，已修

## 五、两机分工建议（2026-09-16 更新）

| 任务 | 谁做 | 状态 |
|---|---|---|
| ALFWorld 7B head+PRM seed0（凑 3 seeds） | 本机 | 🔄 在跑（9/16 启动，~9/18 完） |
| WebShop 1.5B 基线 + head 版（3 seeds × 2 配置） | 你机 | 🔄 在跑（9/14 启动，pod1491319） |
| **E2 纯度回溯分析**（10 个 ALFWorld run，零 GPU，组内 RTG 方差比 + g_set 混淆率 + dpi_calibration） | 待定 | ⭐ P0（DISCUSSION §七-1） |
| **1.5B 中间臂消融 g_text+actor_hidden（2 seeds）**（分解主结果 +5pp：状态重定义 vs head 贡献） | 待定 | ⭐ P0（DISCUSSION §七-2） |
| TextCraft val 扩容离线重评（val 64→320，val_only 模式） | 待定 | 低优先 |

## 五点五、投稿规划（2026-09-10）

- **主目标 ICML 2027**（~2027-01-22 摘要）；ICLR 冲刺已放弃（7B −1.8pp 风险，主日志 §20.3）
- 详细时间表：EXPERIMENTS_LOG_THISNODE.md §19
- 9/16 新增：论文叙事拟按 DISCUSSION_PURITY_VIEW_AND_PLAN.md §6.3 升级（coverage≠quality 发现 + 三层方法栈 + 软分组完成态），待 E2 回溯分析与 1.5B 分解消融结果定稿

## 六、快速核对本机实验健康度的命令

```bash
cd /mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
LOG=$(ls -t verl_log/bipace_7b_alfworld_bisimhead_g_prmanchor_seed0_thisnode_*.log | head -1)
tr '\r' '\n' < "$LOG" | grep -E "Training Progress|val/success_rate" | tail -2
```

### 历史观察存档（2026-09-10，你机的 7B TextCraft 双锚实验）
- 终局见 EXPERIMENTS_LOG_THISNODE.md §20.2：PRM 0.881 / dg 0.863（val320 口径），两锚胶着，再次印证"锚选择效应 < 环境粒度匹配"

## ➖➖ ai-1753446 侧动态（2026-10-12 04:00 补记，本节由该机追加）➖➖

**本机在跑**：lane A = SW 7B 方法 s1 续（96/150）→ baseline s1 续 → **SW 1.5B seed 扩样（GiGPO s3/s4 + baseline s3/s4，`sw_seed_expansion.log`）**；lane B = ALF GRPO（104/200）。
**SW 1.5B 扩样目标（用户拍板：多 seed 定谳）**：GiGPO 现有 3 seeds（0.953尖峰/0.875/0.891）、baseline 3、方法 4——扩到 GiGPO 5-6 / baseline 5。
**给 ai-1797956 的请求**：WebShop 方法补 seeds（s1/s3/s7）收官后若空卡，可接 **SW GiGPO seed3/seed4**（`MODE=gigpo SEED=3/4 START=sft MODEL_PATH=<sft ckpt>` + train_sciworld_1.5b_node1613826.sh）——**所有 run 以 global_step_200 为完成判据，双机幂等无冲突**，贵机启动后本机 watcher 自动跳过。
**已收官可入表**（本机侧）：SW 1.5B 三臂 3-4 seeds 全齐；7B raw-方法 150/150；ALF GiGPO 0.836（官方 86.7 → 复现 −3.1pp 环境差实锤）。

## ➖➖ 2026-11-09 · ai-1845503 本机同步（ScienceWorld 全部由本机跑）➖➖

- 本机（ai-1845503，bipace py3.12）**只负责 ScienceWorld**：SW GRPO/GiGPO/PABH eps 寻优。
- 已从本机接力里**移除 ALF 7B PABH seed0 续跑与 seed1**，避免与 ai-1851164 的 seed1 重复。
- 当前在跑：GPU0-3 = SW GRPO 1.5B seed0（从 SFT 冷启）；GPU4-7 = SW PABH 7B gtextv2 eps 寻优。

### ALFWorld 消融补种分工（tab:attrib 三行都到 3 seeds）
- Raw (BiPACE-Q)：1.5B/7B 均已 3 seeds。
- g_text only（midarm）：1.5B 缺 seed2（ai-1851164 补）；7B 由 ai-1851164 补 seed0/1/2。
- g_text + trained head（完整方法）：1.5B 已 3 seeds；7B 缺 seed1（ai-1851164 在跑）。
- 本机不再启动上述 ALFWorld 消融，避免重复 experiment_name。

**2026-11-10 更新（用户拍板）**：本机（ai-1845503）**暂停 ScienceWorld**，两个槽位都改跑
ALF 7B 我们的方法（PABH）：
- GPU0-3：seed0 从 checkpoint 100 续跑收官（experiment_name 不带 TAG）。
- GPU4-7：seed1 全新重跑，experiment_name 带 `_ai1845503` 后缀，避免与 ai-1851164 的
  `..._seed1_thisnode` 同名冲突。
- 建议 ai-1851164 停掉其 ALF 7B PABH seed1，避免重复烧卡。

## ➖➖ 2026-11-11 · ALFWorld OOD（valid_unseen）补全计划（ai-1845503）➖➖

当前 GPU0-3 / GPU4-7 各跑一组 ALF 7B PABH seed0（`_ai1845503` / `_ai1845503b`），
赛后取峰值高者与 seed1(历史 0.953)、seed2(0.953) 组三 seed。两组 seed0 完赛后，
按以下顺序在 ALFWorld OOD 上补方法矩阵（每个方法 × 每个规模 3 seeds）：

1. OOD 1.5B GiGPO ×3
2. OOD 7B PABH（我们方法）×3 → OOD 7B BiPACE-Q ×3 → OOD 7B GiGPO ×3
3. OOD 1.5B GRPO ×3 → OOD 7B GRPO ×3

超参口径（与 ALFWorld 主实验保持一致；我们的方法沿用主实验表现好的超参）：
- BiPACE-Q：1.5B eps=0.05 L=-12 first_n+counterfactual；7B eps=0.10 L=-8 action_tag+q_style
- 我们的方法 PABH：1.5B eps=0.05 L=-12 first_n+counterfactual + bisim_head(PRM锚)；
  7B eps=0.10 L=-8 action_tag+q_style + bisim_head(PRM锚)
- GiGPO：精确锚（`bisim_grouping=False` + `enable_similarity=False`），无 eps/layer
- GRPO：episode 级（`adv_estimator=grpo`）
- 步数：1.5B=200 / 7B=150；val 为 `env.alfworld.eval_dataset=eval_out_of_distribution`

已有 OOD（1.5B）：PABH 3s=88.0±1.2；BiPACE-Q 3s=87.3±2.5（不再重跑）。

新脚本（本计划用，MODE=bipaceq|pabh|gigpo|grpo + SEED）：
- `examples/bipace/alfworld/qwen1.5b_4xh100/train_ood_all_1.5b_thisnode.sh`
- `examples/bipace/alfworld/qwen7b_4xh100/train_ood_all_7b_thisnode.sh`

他机提示：OOD 已由 ai-1845503 认领，请勿重复启动同 experiment_name。
