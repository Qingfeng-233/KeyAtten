# 2026-03-25 Decoder-Only 落地主项目收口

> 目的：把 2026-03-24 到 2026-03-25 在 `测试沙箱/` 完成的 decoder-only / Qwen3 研究线收口到主项目，只保留已经证明稳定加分的部分。

## 一、真正纳进主线的增益

### 1. decoder-only causal attention 适配

- `cls_attn` 不再直接读取位置 `0`，而是改为 `last-token anchor + content masking`
- `fusion_attn` 使用“新 `cls_attn` × 原始 `received_attn`”的归一化融合
- `received_attn` / `samrank` 保持原始聚合，不引入额外修正项
- 模型构建时自动识别 `is_causal`

这是本轮唯一明确、稳定、可跨样本复现的 decoder-only 主线增益。

### 2. decoder-only 默认层位自动推荐

- 如果调用方没有显式传 `layer_index` / `layer_indices`
- 对 causal 模型，主库现在默认使用“中后层”推荐值，规则是约 `75%` 深度
- 对 `Qwen/Qwen3-Embedding-0.6B`（28 层）会落到 `layer 21`

这一步解决了“主库虽然已经支持 causal 适配，但默认仍落到最后层”的问题。

### 3. 中文 causal instruction prefix 默认保留

- 中文 causal 模型默认前缀保持为：`核心关键词、关键实体、主题：`
- 如果用户显式传入 `instruction_prefix`，仍以用户传入值为准

### 4. `@5` 可选去嵌套后处理

- 主库新增 `dedup_nested_for_topk5`
- 该开关只在 `top_k <= 5` 时生效
- 逻辑是对子串 / 超串短语做去重，不改动 `@10` 的默认排序行为

这一步不改变主线方法本身，只作为 `F1@5` 场景的可选工程后处理。

## 二、当前正式主结论

基于已收口的 `100 docs` 结果，当前不再说“decoder-only 全场绝对最强”，而是保留为“分数据集的最优方法不同”：

- `csl_test 100`：`received_attn_idf@layer_21 = 0.1630`
- `shencecup_labeled 100`：`fusion_attn_idf@layer_21 = 0.2718`

对应结论来自：

- `测试沙箱/docs/2026-03-24-当前研究收口总结.md`
- `测试沙箱/docs/2026-03-24-decoder-causal-attention-adaptation.md`
- `测试沙箱/docs/2026-03-25-无YAKE全方法实测汇总.md`

## 三、不升格进主线的实验项

以下方向已经做过，但没有形成稳定主增益，本次不纳入默认算法：

- `excess_attn`
- `received_attn_debiased`
- `sink_realloc_fusion_attn`
- head weighting / head utility 变体
- rise score
- 真双向 attention monkey-patch
- 纯 attention-gated candidates

这些结果保留在沙箱文档里，作为研究资产，不作为正式默认行为。

## 四、主库落地内容

### 代码

- `keyatten/attention.py`
  - 为 causal 模型补充推荐层位元数据
- `keyatten/extractor.py`
  - `extract_keywords()` / `extract_keywords_batch()` 默认方法改为 `received_attn`
  - 当 `layer_index` 留空时，causal 模型自动走推荐层位
  - 新增 `is_causal_override`，必要时可显式覆盖自动检测，防止模型配置误判
  - 新增 `dedup_nested_for_topk5`，作为 `top_k<=5` 的可选去嵌套后处理

### 文档

- `README.md`
- `README.zh-CN.md`
- `EVALUATION-PUBLIC.md`

主文档现在会明确区分：

- 轻量默认发布路线：`gte-small-zh + ONNX Runtime`
- 已正式支持的 decoder-only 路线：causal 自适配 + 中后层默认推荐

## 五、使用建议

- 如果你要轻量发布，继续优先用 `gte-small-zh`
- 如果你要追中文 decoder-only 的当前最好结果，优先看 `Qwen/Qwen3-Embedding-0.6B`
- 如果你有语料库，优先试 `_idf` 变体
- 如果你没有显式选层，主库现在会自动给 causal 模型选中后层，不需要再手工把默认值从最后层挪开
