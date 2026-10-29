# KeyAtten

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19451584.svg)](https://doi.org/10.5281/zenodo.19451584)

[English](README.md) | [中文](README.zh-CN.md)

基于 Transformer Attention 的关键词提取，支持中英双语。基础路线零训练、零标注、一次前向推理；需要更高精度时可用训练过的路线（BIO 候选、LoRA 重排）。

## 安装

```bash
pip install "keyatten[inference,zh]"              # 中文
pip install "keyatten[inference,en]"              # 英文
pip install "keyatten[inference,zh,lightweight]"  # 中文 + ONNX Runtime
pip install "keyatten[full]"                      # 全部依赖
```

| 可选组 | 包含 |
|---|---|
| `inference` | `torch`、`transformers` |
| `lightweight` | `onnx`、`onnxruntime`、`tokenizers` |
| `zh` | `jieba` |
| `en` | `scikit-learn`、`nltk` |
| `bio` | `torch`、`transformers`、`pytorch-crf` |

只装 `pip install keyatten` 时仅依赖 `numpy`；缺少可选组时，调用接口会给出安装提示。

## 快速开始

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(model="thenlper/gte-small-zh", language="zh")

ext.extract_keywords("自然语言处理是人工智能的重要方向", method="received_attn", top_k=10)

# 有语料时：先拟合 IDF，再用 _idf 方法
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])
ext.extract_keywords("自然语言处理是人工智能的重要方向", method="fusion_attn_idf", idf_lookup=idf)

ext.extract_keywords_batch(["文本一", "文本二"], method="fusion_attn")
ext.extract_word_weights("自然语言处理是人工智能的重要方向")  # list[WordWeight(word, index, weight, pos_tag)]
```

单次调用：`from keyatten import extract_keywords; extract_keywords(text, model="thenlper/gte-small-zh")`。

## 提取器

| 提取器 | 需要训练权重 | 适用场景 |
|---|:---:|---|
| `KeyAttenExtractor` | 否 | 默认，零样本 Attention 打分 |
| `KeyAttenExtractor(candidate_scoring="bio")` | BIO checkpoint | 用 BIO 候选替换结巴候选 |
| `CandidateSegmentAttentionExtractor` | BIO checkpoint + LoRA adapter | 主方法，中文精度最高 |
| `QKLoRAExtractor` | LoRA adapter | 独立的 Q·K 打分 |
| `BIOExtractor` | BIO checkpoint | 只做片段抽取（`extract_spans`） |

```python
from keyatten import CandidateSegmentAttentionExtractor

ext = CandidateSegmentAttentionExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    adapter_path="models/candidate_segment_attn/qwen06_v2_2k_len1024_c30/best_adapter",
    bio_model_path="models/bio_ckipbert_extractive_ep13/bio_model_full.pt",
)
ext.extract_keywords("水木年华被嘲讽已过气，卢庚戌回应称作品会留下来", random_seeds=[1, 2, 3])
```

默认 `candidate_order="random"`，传多个 `random_seeds` 可降低候选顺序带来的波动。

## Attention 方法

| 方法 | 打分方式 |
|---|---|
| `received_attn` | 每个 token 收到的注意力总和，推荐默认 |
| `cls_attn` | `[CLS]` 指向各 token 的注意力，偏向辨识度高的实体（适合标签云） |
| `samrank` | SAMRank 公式（[Kang & Shin, 2023](https://doi.org/10.18653/v1/2023.emnlp-main.630)） |
| `fusion_attn` | `cls_attn` 与 `received_attn` 归一化后相乘 |
| `voted_attn` | 屏蔽非内容 token 并按位置归一化后的接收注意力 |
| `excess_attn` | 高于均匀基线的注意力，排除 attention sink |

每种方法都有 `_idf` 变体（如 `received_attn_idf`），即乘以 TF-IDF，有语料时优先使用。`fit_idf()` 重建 IDF 状态，`update_idf()` 增量追加新文档。

## KeyAttenExtractor 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `model` | 必填 | HF 模型名或本地路径 |
| `language` | `"zh"` | `"zh"` / `"en"` |
| `device` | `"cpu"` | 计算设备 |
| `backend` | `"auto"` | `"torch"` / `"onnx"`；`auto` 找到 `.onnx` 文件时走 ONNX |
| `onnx_path` | `None` | 导出的 attention ONNX 文件 |
| `dtype` | `"auto"` | `"float32"` / `"float16"` / `"bfloat16"` |
| `layer_index` | `None` | 单层；`None` 时编码器取最后一层，decoder-only 模型取约 3/4 深度 |
| `layer_indices`、`layer_weights` | `None` | 多层加权融合 |
| `instruction_prefix` | `None` | causal 模型的前缀（中文默认 `核心关键词、关键实体、主题：`） |
| `is_causal_override` | `None` | 强制按 decoder（`True`）或 encoder（`False`）方式读取注意力 |
| `user_dict` | `None` | 领域词：词典文件路径、词列表或 `{词: 词性}` / `{词: (词频, 词性)}` |
| `candidate_scoring` | `"word"` | `"word"` / `"token_span"`（按候选字符区间打分）/ `"bio"`（需 `bio_model_path`） |
| `dedup_nested_for_topk5` | `False` | `top_k <= 5` 时去掉互相包含的重复词 |
| `cache_enabled`、`cache_dir` | `False`、`"cache"` | 磁盘缓存 attention 分数（IDF 变化时可复用）和最终关键词 |

`extract_keywords` 也接受已分好的词：`ext.extract_keywords(["空天信息", "系统", "优化"], pos_tags=["n", "n", "v"])`。`token_span` 和 `bio` 只对原始字符串输入生效。

## 模型与部署

| 用途 | 模型 |
|---|---|
| 中文轻量（约 33M） | `thenlper/gte-small-zh` |
| 英文轻量（约 22M） | `sentence-transformers/all-MiniLM-L6-v2` |
| 中文 decoder-only | `Qwen/Qwen3-Embedding-0.6B` + `fusion_attn_idf` |

不装 torch、只用 ONNX Runtime（`pip install "keyatten[zh,lightweight]"`）：

```python
ext = KeyAttenExtractor(
    model="/path/to/gte-small-zh",        # 含 tokenizer.json 的本地目录
    backend="onnx",
    onnx_path="/path/to/attention_last.onnx",  # 只支持导出的单层 attention
)
```

ONNX 文件用 [gte_onnx_probe.py](./benchmark/tools/gte_onnx_probe.py) 导出（还需要 `inference` 依赖）。

## 评测

在 7 个公开数据集上与 TF-IDF、TextRank、KeyBERT 等 14 种方法对比，指标为 F1@10：

| 场景 | KeyAtten 最优 | 方法 | vs 最强传统基线 | vs 最强外部方法 |
|------|:---:|--------|:---:|:---:|
| 中文新闻（news55） | **0.4994** | BIO Viterbi | +224% | — |
| 中文新闻（ShenCeCup 1000） | **0.3292** | QK LoRA | +113% | — |
| 中文学术（paper_test_800） | **0.2752** | CSA (high_recall) | — | — |
| 中文学术摘要（CSL，零样本） | **0.2106** | `samrank_idf` | +9% | — |
| 英文长文（SemEval2010-fulltext） | **0.1344** | `cls_attn_idf` | — | +78% |
| 英文长文（Krapivin2009-fulltext） | **0.1268** | `cls_attn_idf` | — | +79% |
| 英文短文（3 个数据集） | 0.1370 | `fusion_attn` | — | 持平 |

主方法（`CandidateSegmentAttentionExtractor`）在 news55 上 F1@10 = 0.4665，较 BIO clean 基线（0.3916）提升 13.7%。

完整报告见 [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md)。在源码目录下用 `python -m benchmark.main --help` 复现（见 [benchmark/README.md](./benchmark/README.md)）。

## 引用

`samrank` 引用自 Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.630)。其余方法及所有 `_idf` 混合策略为本项目原创。

感谢 [LinuxDo](https://linux.do) 社区的支持。许可证：[MIT](./LICENSE)。
