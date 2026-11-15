# 方法改进路线 + 智能体基准实验规划

> 更新：2026-10-19。本文记录"进度感知双模拟头（PABH）"的三个改进方向，
> 以及 WebArena / OSWorld / AndroidWorld 三个开放智能体基准的实验规划。
> 配套讨论见 EXPERIMENTS_LOG_THISNODE.md §29、DISCUSSION_PURITY_VIEW_AND_PLAN.md。

## 0. 必要性判据（insight）

我们的方法（g_text + 学度量的 head）只有当以下两条**同时成立**时才有意义：

1. 注入的状态信息 g_text 比裸观测更全（补上 obs 里没有的目标/进度/剩余）；
2. 这份更全的信息**无法被精确文本匹配分组消费掉**——否则 "g_text + 精确分组"
   （pivot 臂，`EST=gigpo_gtext`）就够了，head 是多余的。

推论：让 head "结构性必要"的唯一办法，是让它学一个**连续、任务间可比、
文本匹配天生吃不下**的信号——即价值（折现回报），而不是离散的进度列表。

## 1. 改进一：价值锚（MICo 目标从规则 PRM 换成步级折现回报）★最关键

**现状**：`verl/trainer/ppo/ray_trainer.py:1283-1333` 里 head 的 MICo 目标
`|r_i - r_j| + γ·‖z̄_i⁺ - z̄_j⁺‖` 的 reward 项 r 取自
`batch.non_tensor_batch["process_reward"]`（规则 PRM，4 级：−0.1/0/0.5/1.0）
或 `d_g`（g_set 的 Jaccard）。两者都是**离散、任务相对**的进度代理。

**改法**：reward 项改用 batch 里已算好的步级折现回报
`batch.batch["step_rewards"]`（`compute_step_discounted_returns`，ray_trainer.py:1454）。
新增 `bisim_anchor_mode=return`，`_compute_bisim_head_embedder` 从
`batch.batch["step_rewards"]` 取每步 return-to-go 作为 r。

**为什么这一步让 head 结构性必要**：

- 价值是连续量：文本精确匹配只能做离散等价，学度量是唯一能表达
  "距离 ≈ 价值差"的方式；
- 价值是任务间可比的（都归一化到 episode 回报量纲），跨任务配对不再冲突；
- 价值天然编码"剩余进度"（return-to-go 就是剩余价值），补齐 g_text 缺失的一维。

**细节/风险**：

- step_rewards 与 process_reward 要对齐到同一步粒度（多轮 rollout 每步一行）；
- 量纲：return 可能到 10（episode 成功）甚至更大，建议对 r 做
  per-batch 标准化或除以 max-episode-return 再进 loss；
- 与 `mode=mean_std_norm` 的步优势归一化口径要写清楚（head 的 r 用原始折现回报即可，
  不必与优势同口径）。

## 2. 改进二：状态补全（anchor 显式加入目标）

**现状**：step 步的 anchor = `g_text + 最近一条动作反馈`（env_manager.py
ALFWorld 分支）。**目标不在里面**（只有 reset 步的 obs 碰巧含任务描述）；
"剩余进度"也完全缺失。

**改法**：

- 目标：env_manager 已用 `extract_task` 把 `self.tasks[i]` 抽出来了，
  只需把 `[Task] {task}` 拼进 anchor：`anchor = f"[Task] {task}\n{g_text}\n{obs}"`；
- 剩余进度：由改进一的价值锚隐式提供（return-to-go 即剩余价值）；
  若要显式可读版本，可加 per-task-type 的"预期子目标模板 − 已完成 g_set"
  的规则化"Remaining subgoals"（与 TextCraft 的 Remaining crafting depth 同思路）。

**预期**：anchor 补全后，head 需要"猜"的维数减少；若此时 head 增益缩小，
则证明此前 head 的部分收益是在替我们学缺失的目标/剩余——这是机制证据。

## 3. 改进三：跨任务训练一致性

**现状问题**：`gigpo/bisim_head.py:sample_pairs` 从整个 batch（16 任务 × 8 轨迹）
抽对，而 PRM 是任务相对的局部进度，导致"捡蛋"(r=0.5) 与"捡锅"(r=0.5) 这种
跨任务同局部进度对被教导"应拉近"——训练信号与真实价值不一致。

**改法**（按优先级）：

1. 采用改进一的价值锚后，reward 项任务间可比，跨任务配对自然合法（首选）；
2. 若保留 PRM 锚做消融，给 `sample_pairs` 加 "same-task-only" 约束
   （用 episode-group uid，而非 traj_uid，限定配对在同一任务内）。

## 4. 三个改进的验证顺序

1. **pivot 臂**（`gigpo_gtext`，已设计未跑）：g_text 精确文本分组 vs head，
   先判定当前 head 是否必要；
2. **改进一 A/B**：`bisim_anchor_mode=prm` vs `return`（ALFWorld 1.5B，各 ≥3 seeds），
   看价值锚是否稳定提纯；
3. **改进二 A/B**：anchor 加/不加 `[Task]`，验证"目标缺失"是否曾是 head 的隐性负担；
4. **改进三**：在价值锚生效后，验证跨任务配对是否还有问题（可并入改进一消融）。

## 5. WebArena / OSWorld / AndroidWorld 实验规划

### 5.1 为什么是这三个

- 都是开放、真实、长视野的 agentic 基准，观测不是封闭模板反馈；
- 进度**无法写成离散的 g_text**（无 "pick up X" 这类闭集事件），
  恰好落在我们"文本匹配吃不下、必须学价值"的判断上；
- 与 ALFWorld/WebShop 形成"封闭模板 vs 开放观测"的对照，直接检验改进一的必要性。

### 5.2 各自特点（决定接入方式）

| 基准 | 任务规模 | 观测 | 进度/奖励信号 | 难点 |
|---|---|---|---|---|
| WebArena | 812 任务（电商/社交/内容/地图/论坛） | 网页 HTML/DOM + 截图 | 用 LLM-as-judge 评估部分完成 | 最轻、研究最成熟，首选 |
| OSWorld | 369 真实 OS 任务（Ubuntu/Win/macOS） | 截图 + accessibility tree | 成功二进制 + 状态检查 | 需 VM 集群，重 |
| AndroidWorld | 116 任务 × 20 App | 截图 + UI 树 | 应用状态 + 最终成功 | 需模拟器，中等 |

### 5.3 关键前提：没有模板反馈，怎么给 head 一个价值信号

三者都没有 ALFWorld 式封闭反馈，所以不能用规则 PRM。三个备选：

1. **learned value head（首选，与改进一一致）**：直接训一个 value/critic 估
   return-to-go，作为 MICo 的 reward 锚——不需要解析进度文本；
2. **LLM-as-judge 进度分（bootstrap）**：零训练但噪声大，只用于冷启动/离线标注
   （research_archive/BIPACE_ANALYSIS_AND_IMPROVEMENTS.md 已有此结论）；
3. **任务分解式 g_text**：对每个 task 指令做 LLM 子目标分解，再判完成
   （介于 1/2 之间，成本可控，可作 g_text 的开放环境版本）。

### 5.4 分阶段计划

- **Phase 0（可行性，零/少 GPU）**：只接 WebArena，跑通最小 rollout 管线 +
  E2 式"精确锚纯度/混淆率"分析，确认"obs 进度混淆 + 无法文本分组"两个前提成立；
- **Phase 1（方法验证，先 WebArena）**：GRPO / GiGPO / BiPACE-Q / PABH(改进一)
  阶梯，value head 作锚，看 head 是否在开放观测上拉开差距；
- **Phase 2（横向扩展）**：WebArena 结论成立后再上 OSWorld / AndroidWorld，
  验证"开放环境需要学价值"的跨基准一致性；
- **Phase 3（消融闭环）**：在 WebArena 上补 "learned value vs LLM-judge vs
  任务分解 g_text" 的锚消融，与 ALFWorld 的 PRM/Jaccard 消融呼应。

### 5.5 风险与依赖

- 环境基建重（VM/模拟器），建议 WebArena 先打样；
- learned value head 需要稳定训练信号，可能受稀疏成功奖励困扰，
  可先用 LLM-judge 的稠密分作 warm-up；
- 与当前 pydantic<2 vs vLLM>=2.11 的依赖冲突同源，接入新环境前先解决。

## 6. 结论

方法的价值主张应从"进度感知的离散分组"升级为"价值感知的连续度量"：
改进一把监督信号换成价值（结构性必要），改进二补全 anchor 的目标维，
改进三消除跨任务训练的不一致。验证顺序：pivot → 价值锚 → 目标补全，
然后把开放基准（WebArena 先行）作为"文本分组吃不下"的终极检验场。

## 7. 决策更新（2026-10-19，用户拍板）

**价值锚 / 状态补全 / 跨任务一致性三改进暂不实施**：

- 价值锚是此前已淘汰过的方案，不做；
- 状态补全会让所有已跑实验失去可比性、全部重跑，不做；
- 改进三依赖改进一，随之一并搁置。

**新方向**：不改方法，先在 WebArena / OSWorld / AndroidWorld 上验证
当前 PABH（g_text + head + PRM）是否相对 GiGPO / BiPACE 真正有效；
同时增加一个"补全状态"对照臂（goal + 完整进度 + obs，显式全状态）。

**可行性（2026-10-19 本容器核查）**：

- 无 Docker/nerdctl/podman → OSWorld（VM）、AndroidWorld（模拟器）、
  WebArena 官方（Docker web server）短期都不可用；
- WebArena 可走 Docker-free 的 `webarena-homepage`（自包含 HTML，本地 HTTP 服务）；
- 因此**先落地 WebArena-homepage**，OSWorld / AndroidWorld 等有 VM/模拟器环境后再补。

**接入工作项（WebArena-homepage）**：

1. 起本地 HTML server + 数据（任务清单）；
2. 写 `env_package/webarena/envs.py`：reset/step、admissible actions、
   进度定义（任务分解式 g_text：如"找到商品→加购→结账"的子目标完成判据）；
3. 写 `train_webarena_1.5b_*.sh`（MODE=grpo/gigpo/baseline/bisimhead_prm + 补全状态臂）；
4. 先 1-2 任务冒烟 rollout，再排阶梯实验。

## 8. AppWorld 实验规划（2026-10-19 起）

### 8.1 目标与科学问题

AppWorld 是第一个"开放观测 + 进度可形式化为 API 调用"的新数据集。核心问题：

> 在 obs 不干净暴露进度、步级信用是真实瓶颈的 API-agent 环境里，
> 当前 PABH（g_text + head + api_prm）是否相对 GiGPO / BiPACE-Q 真正有效？

同时用 AppWorld 检验方法的两条边界条件：

- obs 是否"进度混淆"（E2 式精确锚纯度分析）；
- g_text（已完成 API 调用集）是否"不可文本精确匹配消费"（API 调用是意图而非成功，有噪声）。

### 8.2 对照臂（同管线、单变量）

| 臂 | 配置 | 意义 |
|---|---|---|
| gigpo | bisim_grouping=False，精确锚 | 阶梯地板 |
| baseline | actor_hidden，BiGPO 聚类 | BiPACE-Q 官方口径 |
| bisimhead | bisim_head + g_text，无 api_prm | 冻结随机投影消融 |
| **bisimhead_prm** | bisim_head + g_text + api_prm | **我们的方法** |
| （待加）full-state | anchor 显式含 goal + 完整进度 + obs | 补全状态对照臂 |

### 8.3 配置

- 模型 Qwen2.5-1.5B-Instruct；4×H20（GPU 4-7，与 ALF OOD 的 GPU 0-3 并行）；
- eps=0.10、layer=-12、group_n=8、CACB action_tag+q_style；
- train 16 tasks、val 32 tasks（test_normal）、max_steps=40、100 步试水；
- 进度信号：g_set=(app, api) 调用对、g_text=`[Completed API calls: ...]`、
  PRM=win+0.1×新调用−0.1×无效（`process_reward_mode=api_prm`）。

### 8.4 分阶段执行

**Phase 1（进行中，GPU 4-7）**：三臂 × 1 seed × 100 步试水
（baseline → gigpo → bisimhead_prm，`appworld_relay_ai1845503.sh`）。
判定：步速是否可接受、效应方向、val 方差量级。

**Phase 2（Phase 1 效应可读后）**：

- 三臂补到 3 seeds × 200 步（若步速允许）；
- 加 bisimhead（冻结投影）消融，隔离"训练度量" vs "随机投影 + g_text"；
- 加 full-state 补全状态臂（锚含 goal + 完整进度）。

**Phase 3（可选，GPU 空出后）**：7B 扩展；per-app / per-difficulty 分解。

### 8.5 分析要点

1. 阶梯：GiGPO < BiPACE-Q < PABH（若成立，方法在第三类环境成立）；
2. 消融：bisimhead_prm vs bisimhead → head 训练的边际；
3. 机制：精确锚纯度/混淆率 + g_text 意图-成功噪声，回答"方法为何/为何不必要"；
4. 失败预案：若 PABH ≈ GiGPO（obs 进度对齐），则 AppWorld 归入 ScienceWorld 式边界例，
   方法必要性结论回退到 ALFWorld/WebShop。

### 8.6 风险

- AppWorld 步速慢（API 环境交互重），100 步/臂可能 1-2 天；
- val=32 方差可能偏大；
- server/client 的 pydantic v1/v2 分裂是长期维护约束（训练侧不能 import appworld）；
- g_text 从动作代码解析（意图）而非输出（成功），存在系统性噪声。

### 8.7 实测修正：raw 模型 0% → 必须 SFT 冷启动（2026-10-20）

raw Qwen2.5-1.5B 在 AppWorld 上 val=0.000、train episode reward 恒 0（模型不认识
AppWorld 专属 API），三臂比较无信号，已停。与 ScienceWorld raw 1.5B 坍缩同型。

**已建 SFT 冷启动（两个规模）**：

- 数据：`scripts/prep_appworld_sft.py`，用 `ground_truth/compiled_solution.py`
  （专家完整解法）构造 81 train / 9 val 一次性样本（prompt=APPWORLD 模板，
  response=`<think>+<code>解法</code>`），parquet 在 `/home/work/data/verl-agent/appworld_sft/`；
- 训练脚本：`examples/bipace/appworld/qwen1.5b/run_sft_coldstart_appworld.sh`、
  `examples/bipace/appworld/qwen7b/run_sft_coldstart_appworld.sh`（fsdp_sft_trainer，
  3 epochs，train_batch_size=8，max_length=8192）；
- 结果：
  - 1.5B：train/loss 1.11→0.12，val/loss 0.195，ckpt
    `qwen2.5_1.5b_appworld_sft_node1797956/sft_full/global_step_30`；
  - 7B：train/loss→0.006，val/loss 0.020，ckpt
    `qwen2.5_7b_appworld_sft_node1797956/sft_full/global_step_30`。

**下一步**：把 RL 训练脚本的 `MODEL_PATH` 指向 SFT checkpoint，从冷启动起点重跑
baseline / gigpo / bisimhead_prm 三臂（1.5B 先行；7B 后续再定）。

### 8.8 一次性 SFT 不够 → 多步 SFT（2026-10-21）

一次性 SFT 后模型会正确调用 API，但 val 仍 0%（即使 max_response_length 1024→4096），
因为一次性 SFT 教"整段解法"，与多轮 agent 接口不匹配、中途截断/不完整。

**已改多步 SFT**：

- `scripts/prep_appworld_multistep_sft.py`：回放专家 `solution()` 体，按顶层语句
  分块、在同一 session 里逐块执行并记录真实结果，构造多轮 state→action 样本
  （prompt=APPWORLD_TEMPLATE+历史，response=下一 `<code>` 块）；
- 数据：90 任务 → 234 样本（211 train / 23 val），
  `/home/work/data/verl-agent/appworld_sft_multistep/`；
- 训练：`examples/bipace/appworld/qwen1.5b/run_sft_coldstart_appworld_multistep.sh`
  → train/loss 0.009、val/loss 0.065，ckpt
  `qwen2.5_1.5b_appworld_sft_multistep_node1797956/sft_full/global_step_78`；
- RL 接力器已切到多步 ckpt（`TAG=sftms`），baseline → gigpo → bisimhead_prm 重跑中。
