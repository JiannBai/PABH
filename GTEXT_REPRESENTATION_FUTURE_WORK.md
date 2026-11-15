# g_text 表示局限、后续优化与 CVPR 2027 写作口径

> 日期：2026-11-03  
> 状态：仅记录表示改进方向；CVPR 2027 当前版本不因本文档新增实验  
> 适用范围：ALFWorld、WebShop、ScienceWorld

## 1. 核心结论

当前三个环境的 `g_text` 都是确定性的、环境相关的历史事件摘要，不是完整 MDP 状态，也不是任务进度或 bisimulation equivalence 的 ground truth。

它们的作用是为原本只含最新反馈的 anchor 补充任务相关的历史信息：

```text
anchor = [g_text; current_observation]
```

因此，当前方法不需要假设 `g_text` 完整或完全准确才能成立。更准确的论文主张是：

- 原始 anchor 缺少跨步骤的任务进度信息；
- 当前规则 tracker 能补充一部分确定性、任务相关的信息；
- 实验评估的是当前 progress proxy 与 trained head 的完整实现，而不是理想状态表示的上界；
- 更好的 progress serialization 可能进一步改善状态度量，留作后续工作。

CVPR 2027 当前稿件可以保留原研究动机、方法框架和现有实验，不需要因这些局限推翻原思路或立即重跑实验；但需要收紧“完整状态、完成子目标真值、保证 bisimilar”等潜在过强表述，并修正 ScienceWorld 的 g_text/PRM 通道描述。

## 2. 当前实现与主要局限

### 2.1 ALFWorld

代码：`agent_system/environments/env_package/alfworld/envs.py`

当前表示：

- 从 observation 第一行解析 `take/clean/cool/heat/put/toggle` 六类事件；
- 有序累积并去重为 `(predicate, object)`；
- 物体实例 ID 保留，例如 `kettle 1` 与 `kettle 3`；
- 导航、open/close、look、inventory 不进入事件集合；
- PRM 独立使用新事件、win 和无效动作信号。

主要局限：

1. `put(obj)` 丢弃目的地，正确放置与错误放置可能得到相同进度文本。
2. 事件历史不是当前环境状态；不会完整表示 inventory、物体位置和容器状态。
3. 有序文本会让相同事件集合的不同完成顺序产生不同序列化结果。
4. 排除导航等动作是有意的归纳偏置，不构成一般 MDP 状态等价保证。

后续优化：

- P0：将放置表示扩展为 `put(obj, receptacle)`，或记录目标 receptacle 的满足状态。
- P0：根据当前任务类型生成 goal-conditioned 子目标状态，区分“已发生事件”和“已满足任务条件”。
- P1：评估是否采用排序后的集合序列化，减少等价完成顺序带来的文本差异。
- P1：增加反例测试，例如相同 `put(obj)`、不同目的地；相同最新反馈、不同任务完成条件。
- P1：若后续任务需要，再补充当前 inventory/物体状态，而不是无差别加入所有导航历史。

### 2.2 WebShop

代码：`agent_system/environments/env_package/webshop/envs.py`

当前表示：

- `searches issued: n`；
- 有序去重的 viewed-item 列表，`g_text` 最多展示前八项；
- purchase 仅序列化为 `correct/wrong`；
- PRM 独立使用新 item view、正确/错误购买和无效动作信号。

主要局限：

1. 搜索 query 被丢弃，不同查询及不同候选结果集可能得到相同进度文本。
2. 代码实际累计所有 search action，并未实现注释所称的 distinct-search 去重。
3. viewed item 表示浏览行为，不表示商品对任务属性的满足程度。
4. 未表示当前 ASIN、buying options、目标属性匹配程度和当前页面状态。
5. view 更新只检查点击目标及结果页类型，未严格验证 target 一定是 ASIN；商品选项点击存在被误记为 view 的风险。
6. 购买后的 `correct/wrong` 丢失 ASIN、选项及原始部分得分信息。

后续优化：

- P0：严格区分商品链接与 buying-option 点击，只有确认商品身份时才记录 `view(asin)`。
- P0：使 search 语义与实现一致；明确选择“搜索动作次数”或“去重查询数”，不要混用。
- P0：加入当前商品和已选择 options 的结构化状态，避免将选项点击误当商品浏览。
- P1：在 anchor 可用信息范围内，引入 goal-conditioned 属性满足状态，而不仅是浏览历史。
- P1：评估是否保留规范化 query 或候选结果集摘要；需注意过度包含 query 可能使分组过碎。
- P1：消除前八项截断造成的信息碰撞，或评估截断是否属于有意的 token 预算权衡。

### 2.3 ScienceWorld

代码：

- `agent_system/environments/env_package/sciworld/sciworld_env_manager.py`
- `agent_system/environments/env_package/sciworld/envs.py`

当前表示：

- manager 中解析 `pickup/putdown/pour/dunk/mix/move/connect/activate/open/focus` 十类事件；
- 有序累积并去重为 `(predicate, object)` 后注入 anchor；
- 官方 score 增量在 env worker 中独立计算为 `process_reward`；
- 完整方法脚本通过 `env.sciworld.process_reward_mode=score_delta_prm` 启用 PRM；
- 官方 score 不进入 `g_text`，也不存在“离散 parser 不可用时自动改用 dense score 作为 g_text”的回退逻辑。

主要局限：

1. 十类模板不能覆盖 ScienceWorld 全部任务相关状态变化；类注释提到的 heat 并未在实际事件表中实现。
2. `pour/move/connect/dunk` 等事件丢弃目标容器、目的地或第二个对象。
3. `g_text` 不包含当前累计 official score 或官方子目标满足状态。
4. 与另外两环境一样，事件历史不是完整环境内部状态。

后续优化：

- P0：为 `pour/move/connect/dunk` 等事件保留任务关键的第二参数。
- P0：依据实际训练任务清单审计动作反馈覆盖率，补齐 heat 等缺失事件，避免只依据类注释判断覆盖范围。
- P1：若官方 API 能提供合法的任务进度/子目标状态，优先构造 goal-conditioned 表示；否则采用 task-specific schema。
- P1：评估是否将累计 official score 或分桶进度加入 anchor 文本；该改动属于新的表示臂，不能用现有实验结果直接背书。
- P1：继续保留 score-delta PRM 与 anchor serialization 的通道区分，避免论文把两种信号混写。

## 3. 跨环境统一优化原则

1. 区分“动作发生过”与“任务条件已满足”，后者通常更接近 credit assignment 所需状态。
2. 优先加入影响 continuation value 的任务变量，而不是追求完整重建所有环境细节。
3. 保持确定性序列化，并检查等价状态是否得到规范化表示。
4. 不因追求信息完整而盲目加入所有历史；过多无关细节可能使 BiGPO 分组重新变碎。
5. `g_text` 只进入 metric anchor，不进入 policy prompt；process reward 只监督 MICo head，不进入 PPO reward。
6. 新表示必须通过独立实验验证，不能 retroactively 用当前结果声称其有效。

建议的后续验证顺序：

1. 静态单元测试：模板覆盖、参数保留、reset、去重、规范化。
2. 反例状态对测试：相同最新 observation、不同真实任务状态时，检查新表示是否可区分。
3. 离线聚类诊断：在相同 rollout 上比较 raw observation、当前 g_text、改进 g_text 的进度混淆。
4. 小规模训练：先固定 head 配置比较表示版本，排除同时调参带来的混杂。
5. 正式多 seed：沿用当前环境的训练预算、评估协议和 PACE 配置。

以上实验均标记为 CVPR 2027 当前版本之后的工作，本轮不启动。

## 4. CVPR 2027 当前论文建议

### 4.1 可以保留的内容

- 原问题动机：最新 observation/raw hidden state 缺少显式任务进度，可能混淆不同 continuation value 的状态。
- 原方法框架：progress-event augmentation + PRM-anchored metric head + BiGPO/PACE。
- 当前实验结果及其结论：它们证明当前实现有收益或揭示适用边界，不依赖“tracker 完整准确”这一更强假设。
- ALFWorld 的具体失败案例和方法图，只要将其称为 illustrative progress events，而非环境 ground-truth state。

### 4.2 需要收紧的表述

建议统一使用：

- `progress-event summary`
- `environment-specific progress proxy`
- `deterministic task-relevant side information`
- `encourages progress-aware / value-aware grouping`

避免无限制地使用：

- `complete progress state`
- `ground-truth completed subgoals`（跨环境泛称）
- `ensures bisimilarity`
- `guarantees value-equivalent groups`

方法贡献仍然是“引入额外状态信息并学习度量”，不是“已经完美恢复任务状态”。

### 4.3 ScienceWorld 必须校准的代码口径

推荐论文表述：

> ScienceWorld serializes ten types of template-parsed action events into the anchor. Independently, non-negative official score increments provide PRM supervision for the MICo head.

不要写成 dense score 在 parser 不可用时替代或补全 `g_text`。

### 4.4 建议加入的统一限制说明

可在 progress extraction 附录或 Limitations 中加入：

> The progress text is an environment-specific event summary rather than a complete MDP-state reconstruction. It supplies deterministic task-relevant side information beyond the latest observation, but does not by itself certify bisimulation equivalence. Designing richer goal-conditioned serializers is a promising direction for future work.

该说明不会削弱当前实验，反而明确区分了“方法机制有效”和“状态表示仍可改进”两个命题。

## 5. 当前版本决策

- 保留当前三环境 g_text 实现及现有实验数据。
- 不把本文档中的优化点写成已完成组件或已验证结论。
- 当前论文只做概念与代码口径校准，不新增表示优化实验。
- 后续表示版本建议命名为独立实验臂，避免与当前 CVPR 2027 结果混用。
