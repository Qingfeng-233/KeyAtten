# KeyAtten：Attention-Based 关键词提取统一评测

> **版本**: V2
> **日期**: 2026-03-22

---

## 摘要

本项目构建了一个覆盖中英双语、短文与长文、Encoder 与 Decoder-only 架构的 Attention-based 关键词提取统一评测框架，在 7 个公开数据集上对 14 种方法进行了系统对比。

KeyAtten 提供 4 种纯 Attention 方法（`cls_attn`、`received_attn`、`samrank`、`fusion_attn`）及其 IDF 混合变体。其中 SAMRank 排序公式源自 [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630)，本项目为独立实现的变种（多头平均 + IDF 混合）。

### 核心发现

- **中文新闻场景**：Attention 类方法 F1@10 达到 0.2579，较最强传统基线提升 **67%**
- **中文学术摘要场景**：`samrank_idf` F1@10 达到 0.2106，超越最强传统基线 **9%**
- **英文长文场景**：两个独立 fulltext 数据集一致显示 `cls_attn_idf` 大幅领先外部基线（F1@10 分别为 0.1344 / 0.1268，提升 **+78% / +79%**）
- **架构泛化性**：Attention 关键词提取可在 Encoder 和 Decoder-only 两种架构上工作
- **极低成本**：仅需 22M–33M 参数的小模型，单次前向推理，零训练、零标注

---

## 评测覆盖

| 维度 | 覆盖内容 |
|------|----------|
| 语言 | 中文、英文 |
| 文本长度 | 短文（摘要）、长文（全文） |
| 模型架构 | Encoder、Decoder-only |
| 数据集数量 | 7 个公开数据集 |
| 方法数量 | 14 种（传统统计、图方法、向量相似度、Attention、Attention-IDF hybrid） |

### 评测数据集

| 数据集 | 语言 | 场景 |
|--------|------|------|
| CSL | 中文 | 学术摘要 |
| ShenCeCup | 中文 | 新闻 |
| SemEval2010 | 英文 | 学术短文 |
| PubMed | 英文 | 学术短文 |
| LIS2000 | 英文 | 学术短文 |
| SemEval2010-fulltext | 英文 | 学术长文（243 篇全文） |
| Krapivin2009-fulltext | 英文 | 学术长文 |

---

## 主榜

> 指标：F1@10 | 每个数据集取各方法类别下的最优结果

| 数据集 | 场景 | 最强传统基线 | 最强外部方法 | KeyAtten 纯 Attention 最优 | KeyAtten-IDF 最优 | 提升幅度 |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| CSL | 中文 / 学术摘要 | TF-IDF 0.1935 | KeyBERT 0.1296 | `samrank` 0.1773 | `samrank_idf` **0.2106** | +9% vs 传统 |
| ShenCeCup | 中文 / 新闻 | TermFreq 0.1543 | TextRank 0.0661 | `samrank` **0.2495** | `cls_attn_idf` 0.2357 | +67% vs 传统 |
| SemEval2010 | 英文 / 学术短文 | TF-IDF 0.1040 | KeyBERT 0.1448 | `fusion_attn` 0.1448 | — | 持平 |
| PubMed | 英文 / 学术短文 | TF-IDF 0.1211 | KeyBERT 0.1327 | `fusion_attn` 0.1327 | — | 持平 |
| LIS2000 | 英文 / 学术短文 | TermFreq 0.1040 | KeyBERT 0.1370 | `fusion_attn` 0.1370 | — | 持平 |
| SemEval2010-fulltext | 英文 / 学术长文 | TF-IDF 0.0604 | TextRank 0.0754 | `cls_attn` 0.0671 | `cls_attn_idf` **0.1344** | +78% vs 外部最强 |
| Krapivin2009-fulltext | 英文 / 学术长文 | TF-IDF 0.0565 | TextRank 0.0707 | `cls_attn` 0.0789 | `cls_attn_idf` **0.1268** | +79% vs 外部最强 |

### 主榜解读

1. **中文新闻**：纯 Attention 方法（`samrank`）已大幅领先传统基线，在 Encoder 和 Decoder-only 架构上均表现突出
2. **中文学术摘要**：`samrank_idf` 超过最强传统基线（TF-IDF），IDF 混合带来关键增益
3. **英文长文**：`cls_attn_idf` 在两个独立 fulltext 数据集上一致以约 80% 幅度领先外部基线，前四名均为 KeyAtten-IDF 方法
4. **英文短文**：`fusion_attn` 与外部最优方法（KeyBERT）持平，尚未拉开明显差距

---

## 中文场景完整排名

### CSL 中文学术摘要

| 排名 | 方法 | 模型 | F1@10 | R@10 |
|:---:|------|------|:---:|:---:|
| 1 | TF-IDF | — | 0.1935 | 0.2984 |
| 2 | TermFreq | — | 0.1834 | 0.2858 |
| 3 | `samrank` | gte-small-zh | 0.1773 | 0.2567 |
| 4 | `received_attn` | m3e-small | 0.1666 | 0.2298 |
| 5 | `received_attn` | Qwen3.5-2B | 0.1568 | 0.2347 |
| 6 | `cls_attn` | gte-small-zh | 0.1554 | 0.2215 |
| 7 | KeyBERT | m3e-small | 0.1531 | 0.2078 |
| 8 | `fusion_attn` | gte-small-zh | 0.1505 | 0.2150 |
| 9 | `received_attn` | Qwen3.5-0.8B | 0.1496 | 0.2222 |
| 10 | TextRank | — | 0.1237 | 0.1812 |

### ShenCeCup 中文新闻

| 排名 | 方法 | 模型 | F1@10 | R@10 |
|:---:|------|------|:---:|:---:|
| 1 | `received_attn` | Qwen3.5-0.8B | **0.2579** | 0.5633 |
| 2 | `samrank` | gte-small-zh | **0.2495** | 0.5367 |
| 3 | `received_attn` | gte-small-zh | **0.2424** | 0.5242 |
| 4 | `cls_attn` | gte-small-zh | **0.2269** | 0.4892 |
| 5 | `fusion_attn` | gte-small-zh | **0.2264** | 0.4867 |
| 6 | `received_attn` | Qwen3.5-2B | 0.2221 | 0.4975 |
| 7 | TF-IDF | — | 0.1543 | 0.3542 |
| 8 | TermFreq | — | 0.1543 | 0.3542 |
| 9 | KeyBERT | m3e-small | 0.0991 | 0.2167 |
| 10 | TextRank | — | 0.0661 | 0.1308 |

---

## 英文长文场景完整排名

以 SemEval2010-fulltext（243 篇全文）为例：

| 排名 | 方法 | F1@10 | R@10 |
|:---:|------|:---:|:---:|
| 1 | `cls_attn_idf` | **0.1344** | 0.1145 |
| 2 | `samrank_idf` | **0.1327** | 0.1129 |
| 3 | `received_attn_idf` | **0.1268** | 0.1080 |
| 4 | `fusion_attn_idf` | **0.1224** | 0.1044 |
| 5 | TextRank | 0.0754 | 0.0651 |
| 6 | `cls_attn` | 0.0671 | 0.0572 |
| 7 | `fusion_attn` | 0.0622 | 0.0527 |
| 8 | TF-IDF | 0.0604 | 0.0516 |
| 9 | TermFreq | 0.0570 | 0.0487 |
| 10 | `samrank` | 0.0481 | 0.0411 |
| 11 | `received_attn` | 0.0480 | 0.0410 |
| 12 | KeyBERT | 0.0445 | 0.0380 |

Krapivin2009-fulltext 排序模式一致，`cls_attn_idf` 最优 F1@10 = 0.1268。

---

## IDF 混合增益分析

为确保评测公平，对外部方法也做了同类 IDF 增强对照（SemEval2010-fulltext）：

| 方法 | 原始 F1@10 | + IDF 后 | 变化幅度 |
|------|:---:|:---:|:---:|
| `cls_attn`（KeyAtten） | 0.0671 | **0.1344** | **+100%** |
| `samrank`（KeyAtten） | 0.0481 | **0.1327** | **+176%** |
| `received_attn`（KeyAtten） | 0.0480 | **0.1268** | **+164%** |
| `fusion_attn`（KeyAtten） | 0.0622 | **0.1224** | **+97%** |
| KeyBERT | 0.0445 | 0.0690 | +55% |
| TextRank | 0.0754 | 0.0617 | -18% |

结论：

- Attention 方法与 IDF 的协同效应最强（提升 +97% 至 +176%），远超 KeyBERT 的 +55%
- TextRank + IDF 反而退化 18%，并非所有方法都受益于 IDF 混合
- 最强外部 Hybrid（KeyBERT+IDF = 0.0690）仍远低于 KeyAtten 最弱的 Hybrid（`fusion_attn_idf` = 0.1224），差距达 77%

---

## 成本分析

| 方法类型 | 需要模型 | 参数量 | 推理次数 | 额外计算 |
|----------|:---:|:---:|:---:|----------|
| TF-IDF / TextRank | 否 | 0 | 0 | 词频统计 / 图迭代 |
| KeyBERT | 是 | ~33M | 1 次 forward | 余弦相似度 |
| **KeyAtten Attention 系列** | 是 | **22M–33M** | **1 次 forward** | **无** |
| **KeyAtten-IDF hybrid** | 是 | **22M–33M** | **1 次 forward** | **+ IDF（近零开销）** |

- 与 KeyBERT 推理成本完全相同，但优势场景效果高 67%–176%
- 使用小模型（22M–33M 参数），非大模型
- 零训练、零标注、零微调

---

## 架构泛化性

Attention 关键词提取已验证可在 Decoder-only 长上下文 Transformer 上工作：

| 架构 | 模型 | 模型规模 | CSL F1@10 | ShenCeCup F1@10 |
|------|------|:---:|:---:|:---:|
| Decoder-only | Qwen3.5-0.8B | 0.8B | 0.1496 | **0.2579** |
| Decoder-only | Qwen3.5-2B | 2B | **0.1568** | 0.2221 |
| Encoder | gte-small-zh | ~33M | 0.1773 | 0.2495 |

模型规模不是唯一因素：Qwen3.5-0.8B 在新闻场景反超 2B 版本，说明 Attention 机制和领域匹配同样关键。

---

## 场景适用指南

| 场景 | 推荐方法 | 预期 F1@10 | 相对传统基线提升 |
|------|----------|:---:|:---:|
| 中文新闻 | `samrank` | ~0.25 | +67% |
| 中文学术摘要 | `samrank_idf` | ~0.21 | +9% |
| 英文长文 | `cls_attn_idf` | ~0.13 | +78%–79% |
| 标签云 / 摘要展示 | `cls_attn` | — | 辨识度最高 |
| Decoder-only | `received_attn` | ~0.16–0.26 | 视场景而定 |

---

## 局限性

- 中文数据集覆盖偏窄（学术摘要 + 新闻），缺更多领域验证
- Decoder-only 结果基于小样本，尚未扩展到正式 benchmark 规模
- 英文短文场景下优势不明显

---

## 引用

SAMRank 排序公式源自以下论文，本项目的 `samrank` 方法为独立实现的变种：

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). DOI: 10.18653/v1/2023.emnlp-main.630

---

## 可验证性

完整评测代码已打包存档，SHA-256：

```
keyatten-benchmark-code.tar.gz  1760e974241209a85f74fd94ff2aecd1f4f9c7704bcc5045eed8e162cf1aef6e
```

计划通过 GitHub Release 或 OpenTimestamps 建立公开时间锚。代码将在后续开源，届时可通过上述 hash 验证一致性。

---

*本评测基于公开数据集，所有数据集均可通过原始来源获取。*
