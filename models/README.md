# models 目录说明

时间：2026-05-06

当前 `models/` 只保留主线相关模型与权重；历史实验产物已移到 `_archive/`。

## 当前保留

### `gte-small-zh`

- 用途：轻量 Attention 基座模型
- 主要服务：
  - `KeyAttenExtractor`
  - 轻量中文 Attention 方法

### `gte_small_zh_onnx`

- 用途：`gte-small-zh` 的 ONNX 导出版本
- 主要服务：
  - ONNX 推理
  - 轻量部署

### `Qwen3-Embedding-0.6B`

- 用途：Qwen 基座模型
- 主要服务：
  - `CandidateSegmentAttentionExtractor`
  - `QKLoRAExtractor`
  - Qwen 路线训练脚本

### `bio_ckipbert_extractive_ep13`

- 用途：BIO 候选模型 checkpoint
- 主要服务：
  - `BIOExtractor`
  - 主方法里的 BIO 候选生成

### `candidate_segment_attn`

- 用途：当前主方法 adapter
- 对应方法：
  - `BIO 候选 + Attention 微调排序`
- 主要服务：
  - `CandidateSegmentAttentionExtractor`

### `qk_qwen0.6B`

- 用途：QK 单方法 adapter
- 主要服务：
  - `QKLoRAExtractor`

## 已归档

历史实验产物统一放在：

- `_archive/`

当前归档内容包括：

- `attn_lora_gte`
- `attn_lora_gte_full`
- `ckiplab-bert-base-chinese-ner`
- `ppc_reranker_shence200_v1`
- `qk_qwen0.6B_hq_r16_nocsl8k_best_adapter`
- `qk_qwen4B`


1. `gte-small-zh`：轻量 Attention 基座
2. `bio_ckipbert_extractive_ep13`：BIO 候选模型
3. `candidate_segment_attn` + `Qwen3-Embedding-0.6B`：主方法微调排序模型
