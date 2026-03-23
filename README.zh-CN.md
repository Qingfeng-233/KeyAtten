# KeyAtten

[English](README.md) | [中文](README.zh-CN.md)

基于 Transformer Attention 机制的关键词提取框架。零训练、零标注，仅需一次前向推理，支持中英双语。

在 7 个公开数据集、14 种方法的对比评测中，中文新闻场景 F1@10 较传统基线提升 67%，英文长文场景较外部最强方法提升约 78%。

## 特性

- 直接利用预训练模型的注意力权重提取关键词，无需额外训练或标注
- 提供 Attention-IDF 混合策略，在长文和有语料库的场景下效果显著
- 支持词级语义权重输出（含权重值、位置索引、词性标注）
- 支持单层或多层 Attention 加权融合
- 仅需 22M–33M 参数的小模型，单次前向推理完成

## 安装

```bash
pip install keyatten
```

默认安装现在只包含 `numpy`，不会在 `import keyatten` 时顺带拉起整套重量级推理依赖。

```bash
pip install "keyatten[inference,zh]"   # 中文关键词提取
pip install "keyatten[inference,en]"   # 英文关键词提取
pip install "keyatten[full]"           # 安装全部可选依赖
```

可选依赖分组：

- `inference`: `torch>=2.0`、`transformers>=4.30`
- `zh`: `jieba>=0.42`
- `en`: `scikit-learn>=1.0`、`nltk>=3.8`

如果缺少对应 extras 就直接调用提取接口，KeyAtten 现在会给出明确安装提示，而不是在 `import keyatten` 阶段就失败。

## 快速开始

### 关键词提取

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(model="thenlper/gte-small-zh", language="zh")

# 纯 Attention
keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="cls_attn",
)
```

### Attention-IDF 混合

```python
# 先从语料库拟合 IDF
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])

keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="samrank_idf",
    idf_lookup=idf,
)
```

### 词级权重

```python
weights = ext.extract_word_weights(
    "自然语言处理是人工智能的重要方向",
    method="received_attn",
)
for w in weights:
    print(w.word, w.weight, w.pos_tag)
```

### 批量提取

```python
results = ext.extract_keywords_batch(
    ["文本一", "文本二", "文本三"],
    method="fusion_attn",
)
```

### 便捷函数

```python
from keyatten import extract_keywords

keywords = extract_keywords(
    "自然语言处理是人工智能的重要方向",
    model="thenlper/gte-small-zh",
)
```

## 提取方法

| 方法 | 说明 |
|------|------|
| `cls_attn` | [CLS] token 对各 token 的注意力权重 |
| `received_attn` | 各 token 从所有 token 接收的注意力总和 |
| `samrank` | SAMRank 排序公式（全局注意力 + 比例分配） |
| `fusion_attn` | CLS 注意力与 received 注意力的归一化融合 |

以上每种方法均有对应的 `_idf` 混合变体（如 `cls_attn_idf`），将 Attention 分数与 TF-IDF 相乘，适合有语料库的场景。

> `samrank` 的排序公式引用自 [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630)，其余方法及所有 `_idf` 混合策略为本项目原创。

### 如何选择方法

`samrank` 系列在 Benchmark 上跑分最高（F1@10），因为它覆盖面广、recall 强。但在实际应用中，`cls_attn` 往往更实用——它提取的是最具辨识度的核心词，一眼就能看出文章在讲什么。

## 实战示例

以下为 `cls_attn` 与 `samrank` 在不同领域文本上的提取对比（模型：`gte-small-zh`，top_k=6）：

| 领域 | 输入文本（节选） | cls_attn | samrank |
|------|-----------------|----------|---------|
| 科技 | OpenAI发布了GPT-4o模型，支持多模态输入... | OpenAI, GPT, 模型 | OpenAI, 模型, GPT |
| 医学 | mRNA疫苗通过编码刺突蛋白...对新冠病毒Omicron变异株... | mRNA, mRNA疫苗, 新冠, **Omicron变异** | mRNA, mRNA疫苗, 新冠, 新冠病毒 |
| 金融 | 美联储宣布加息25个基点... | 加息, 基点, **全球股市**, 基金利率 | 加息, 基点, 利率, 全球股市 |
| 体育 | 梅西在世界杯决赛中上演帽子戏法...捧起大力神杯 | **梅西**, 大力神杯, 帽子戏法, **决赛** | 大力神杯, 梅西, 帽子戏法, **点球** |
| 历史 | 秦始皇统一六国...建立中央集权的封建王朝 | 中央集权, 封建王朝, 车同轨, 六国 | 中央集权, 车同轨, 书同文, 封建王朝 |
| 日常 | 今天下午在星巴克见面...去北京出差 | **星巴克**, 北京, 北京出差 | **见面**, 北京, **聊聊** |

`cls_attn` 倾向于抓最具辨识度的实体（梅西、星巴克、Omicron），适合标签云、摘要展示等需要一眼抓住主题的场景；`samrank` 覆盖面更广，适合需要全面召回的检索和评测场景。

## 推荐模型

| 语言 | 模型 | 参数量 |
|------|------|--------|
| 中文 | `thenlper/gte-small-zh` | ~33M |
| 英文 | `sentence-transformers/all-MiniLM-L6-v2` | ~22M |

## 评测摘要

在 7 个公开数据集上与 TF-IDF、TextRank、KeyBERT 等 14 种方法对比，指标为 F1@10：

| 场景 | KeyAtten 最优 | vs 最强传统基线 | vs 最强外部方法 |
|------|:---:|:---:|:---:|
| 中文新闻（ShenCeCup） | **0.2579** | +67% | — |
| 中文学术摘要（CSL） | **0.2106** | +9% | — |
| 英文长文（SemEval2010-fulltext） | **0.1344** | — | +78% |
| 英文长文（Krapivin2009-fulltext） | **0.1268** | — | +79% |
| 英文短文（3 个数据集） | 0.1370 | — | 持平 |

完整评测报告见 [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md)。

## API

### KeyAttenExtractor

```python
KeyAttenExtractor(
    model: str,                         # Hugging Face 模型名称
    language: str = "zh",               # "zh" 或 "en"
    device: str = "cpu",                # 计算设备
    layer_index: int = -1,              # 单层索引（-1 = 最后一层）
    layer_indices: list[int] = None,    # 多层索引列表
    layer_weights: list[float] = None,  # 多层权重列表
    attn_merge: bool = False,           # Attention 引导的中文单字合并
    merge_threshold: float = 0.3,       # 合并阈值（0.0–1.0）
)
```

| 方法 | 返回值 |
|------|--------|
| `extract_keywords(text, method, top_k, idf_lookup)` | `list[str]` |
| `extract_keywords_batch(texts, method, top_k, idf_lookup)` | `list[list[str]]` |
| `extract_word_weights(text, method)` | `list[WordWeight]` |
| `fit_idf(texts)` | `dict[str, float]` |

`WordWeight` 包含字段：`word`、`index`、`weight`、`pos_tag`。

## 引用

本项目的 `samrank` 方法引用了以下论文的排序公式：

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI: 10.18653/v1/2023.emnlp-main.630](https://doi.org/10.18653/v1/2023.emnlp-main.630)

`cls_attn`、`received_attn`、`fusion_attn` 及所有 `_idf` 混合策略为本项目原创。

## 许可证

[MIT](./LICENSE)
