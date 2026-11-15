# 简历陈述（Resume Bullet）— 进度感知双模拟头项目

> 源项目：BiPACE 改进（Progress-aware Bisimulation Head for Step-Level Credit Assignment）
> 论文初稿：`PAPER_DRAFT.md`
> 最后更新：2026-09-09

---

## 中文版（项目经历条目）

### 完整版（一段式，适合项目栏）

**LLM Agent 的步级信用分配强化学习方法研究（BiPACE 改进）**

针对分组式 RL（GiGPO/BiPACE）中状态等价类划分的"进度盲视"与"度量未训练"两大缺陷，提出**进度感知的双模拟头（Progress-aware Bisimulation Head）**：在冻结的 LLM backbone 上外挂 0.39M 参数线性投影，用基于任务进度的 MICo 度量损失在线训练（EMA target + 事件级过程奖励锚/进度集合 Jaccard 锚 + 改进的对比对采样策略），使状态聚类从句法相似升级为价值等价。在 ALFWorld（Qwen2.5-1.5B，3 seeds）上达到 **95.1%±1.8** 成功率，追平官方 BiPACE-Q 最优配置（93.5%±1.2），超出同管线复现基线 **+5.3pp**；通过锚消融实验揭示了"度量锚粒度需与环境进度结构匹配"的设计准则（PRM 锚 vs Jaccard 锚在两类环境上效应反转 7.6pp）；并完成 12 组 TextCraft 负结果实验与统计功效分析，论证短视野合成基准 ±7-8pp 种子方差对 <10pp 效应的结构性不可检测性，为该类 benchmark 建立评估协议建议。独立完成方法设计、跨环境进度形式化（TextCraft 配方闭包 / ALFWorld 模板事件解析）、8×H20 分布式训练、3 种子统计验证与 7B 规模扩展。

### 精简版（3-4 行，适合空间紧张）

- 提出进度感知双模拟头（0.39M 参数、冻结 backbone），以任务进度接地的 MICo 度量损失在线训练，将 LLM Agent 分组式 RL 的步级状态聚类从句法相似升级为价值等价
- ALFWorld（Qwen2.5-1.5B，3 seeds）达 **95.1%±1.8**，追平官方 BiPACE-Q（93.5%±1.2），超同管线复现基线 **+5.3pp**；锚消融揭示度量锚粒度与环境结构匹配准则（效应反转 7.6pp）
- 完成 12 组负结果实验与统计功效分析，论证高方差基准（±7-8pp）的结构性检测极限并建立评估协议建议；独立完成方法设计、跨环境形式化、8 卡分布式训练与 7B 扩展

---

## 英文版

### Resume bullet（一段式）

Proposed and validated a progress-aware bisimulation head (0.39M-param projection trained online with a MICo metric loss grounded in task-progress signals) that upgrades step-level state grouping in group-based RL for LLM agents from syntactic to value-equivalent clustering; reached **95.1±1.8%** on ALFWorld (Qwen2.5-1.5B, 3 seeds), matching the strongest published baseline (93.5±1.2%) with **+5.3pp** over same-pipeline reproduction, and derived anchor-granularity design guidelines via controlled ablations plus a 12-run statistical-power analysis of benchmark variance.

### Bullets 版（2-3 条）

- Designed a progress-aware bisimulation head for step-level credit assignment in group-based RL for LLM agents: a frozen-backbone linear projection trained online with a task-progress-grounded MICo loss (EMA target, dual metric anchors, improved pair sampling), replacing syntactic hidden-state clustering with learned value equivalence at <2% step-time cost
- Achieved **95.1±1.8%** on ALFWorld (Qwen2.5-1.5B, 3 seeds) — matching the strongest published BiPACE-Q baseline (93.5±1.2%) and exceeding same-pipeline reproduction by **+5.3pp**; extended to Qwen2.5-7B
- Conducted controlled anchor ablations (PRM vs. Jaccard, 7.6pp effect reversal across environments) and a 12-run statistical-power study establishing evaluation-protocol guidelines for high-variance agent benchmarks

---

## 数字备查（写简历/面试时引用）

| 关键数字 | 值 | 语境 |
|---|---|---|
| 主结果 | 95.1 ± 1.8%（3 seeds） | ALFWorld + Qwen2.5-1.5B + 200 步 |
| 官方基线 | 93.5 ± 1.2%（3 seeds） | BiPACE-Q 官方 |
| 复现基线 | 89.8% | 我们同管线 BiPACE-Q 复现（单种子） |
| 主效应 | +5.3pp | 相对同管线复现基线 |
| 锚消融 | 7.6pp | PRM 锚 vs Jaccard 锚（ALFWorld） |
| head 规模 | 0.39M 参数（1536→256 线性层） | 占 1.5B 模型 0.026% |
| 负结果研究 | 12 runs，±7-8pp 种子方差 | TextCraft，证明 <10pp 效应不可检测 |
| 训练规模 | 8×H20，3 seeds × 200 步 + 7B 2 seeds × 150 步 | 累计 >300 GPU 时 |
