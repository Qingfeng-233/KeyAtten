# KeyAtten：基于 Transformer Attention 的关键词提取方法评测报告

> **版本**: V2
> **更新日期**: 2026-03-22
> **评测状态**: 已完成 V1 级别大横屏评测，覆盖中英双语、短文与长文、Encoder 与 Decoder-only 架构
> **作者**: 蒋林浩

---

## 核心发现

我们提出的基于 Transformer Attention 机制的关键词提取方法，在多个公开数据集上展现出稳定的竞争力，并在以下场景中表现出明确优势：

1. **中文新闻场景**：纯 Attention 方法显著优于传统基线（F1@10 提升 60%+）
2. **中文学术摘要场景**：Attention + IDF 混合方法超越最强传统基线
3. **英文长文场景**：Attention + IDF 混合方法以近 2 倍优势领先外部基线
4. **Decoder-only 架构**：方法已验证可扩展至无 [CLS] 的长上下文 Transformer

**关键结论**：不同场景适合不同的 Attention 落地形态——新闻类任务中纯 Attention 更强，学术摘要与长文任务中 Attention + IDF 混合更强。这不是"一个方法全场景碾压"，而是"Attention 机制在不同场景下可通过不同落地形态持续发挥作用"。

---

## 评测覆盖范围

| 维度 | 覆盖内容 |
|------|----------|
| **语言** | 中文、英文 |
| **文本长度** | 短文（摘要）、长文（全文） |
| **模型架构** | Encoder（BERT 系列）、Decoder-only（Qwen 系列） |
| **方法类型** | 传统统计、图方法、向量相似度、Attention、Attention + IDF 混合 |
| **数据集数量** | 7 个公开数据集 |
| **模型数量** | 9 个预训练模型 |
| **对比方法** | 14 种方法（含变体） |

### 数据集列表

| 数据集 | 语言 | 场景 | 规模 |
|--------|------|------|------|
| CSL | 中文 | 学术摘要 | 完整测试集 |
| ShenCeCup | 中文 | 新闻 | 完整标注集 |
| SemEval2010 | 英文 | 学术短文 | 完整测试集 |
| PubMed | 英文 | 学术短文 | 完整测试集 |
| LIS2000 | 英文 | 学术短文 | 完整测试集 |
| SemEval2010-fulltext | 英文 | 学术长文 | 243 篇全文 |
| Krapivin2009-fulltext | 英文 | 学术长文 | 完整集 |

---

## 主榜（V1 大横屏）

> 主指标：F1@10 | 辅助指标：R@10 | 每个数据集取各方法组下的最优代表

| 数据集 | 场景 | 最强传统基线 | 最强图/相似度基线 | 最强KeyAtten纯 Attention | 最强KeyAtten Hybrid | 最强 Decoder-only |
|--------|------|:---:|:---:|:---:|:---:|:---:|
| **CSL** | 中文 / 学术摘要 | TF-IDF `0.1935` | KeyBERT `0.1296` | SAMRank `0.1773` | **samrank_idf `0.2106`** | Qwen-2B received `0.1568` |
| **ShenCeCup** | 中文 / 新闻 | TermFreq `0.1543` | TextRank `0.0661` | SAMRank `0.2495` | cls_attn_idf `0.2357` | **Qwen-0.8B received `0.2579`** |
| **SemEval2010** | 英文 / 学术短文 | TF-IDF `0.1040` | KeyBERT `0.1448` | Fusion-Attn `0.1448` | — | — |
| **PubMed** | 英文 / 学术短文 | TF-IDF `0.1211` | KeyBERT `0.1327` | Fusion-Attn `0.1327` | — | — |
| **LIS2000** | 英文 / 学术短文 | TermFreq `0.1040` | KeyBERT `0.1370` | Fusion-Attn `0.1370` | — | — |
| **SemEval2010-fulltext** | 英文 / 学术长文 | TF-IDF `0.0604` | TextRank `0.0754` | CLS-Attn `0.0671` | **cls_attn_idf `0.1344`** | — |
| **Krapivin2009-fulltext** | 英文 / 学术长文 | TF-IDF `0.0565` | TextRank `0.0707` | CLS-Attn `0.0789` | **cls_attn_idf `0.1268`** | — |

### 主榜解读

1. **中文新闻**：纯 Attention 和 Decoder-only 均显著强于传统基线，其中 Qwen-0.8B Received-Attn 以 F1@10 = 0.2579 夺得最高分
2. **中文学术摘要**：Attention + IDF 混合方法（samrank_idf F1@10 = 0.2106）已超过最强传统基线（TF-IDF F1@10 = 0.1935）
3. **英文长文**：两个 fulltext 数据集均显示 Attention + IDF 混合方法大幅领先外部基线（约 +78% ~ +80%）
4. **英文短文**：Fusion-Attn 与 KeyBERT 持平，但未拉开明显差距

---

## 亮点结果展示

### 英文长文场景：前四名全部是KeyAtten方法

以 SemEval2010-fulltext（243 篇全文）为例：

| 排名 | 方法 | F1@10 | R@10 |
|:---:|------|:---:|:---:|
| **1** | **cls_attn_idf** | **0.1344** | 0.1145 |
| **2** | **samrank_idf** | **0.1327** | 0.1129 |
| **3** | **received_attn_idf** | **0.1268** | 0.1080 |
| **4** | **fusion_attn_idf** | **0.1224** | 0.1044 |
| 5 | textrank | 0.0754 | 0.0651 |
| 6 | cls_attn | 0.0671 | 0.0572 |
| 7 | tfidf | 0.0604 | 0.0516 |
| 8 | keybert | 0.0445 | 0.0380 |

纯 Attention 在长文上不够强，但与 IDF 混合后提升极为显著，且外部方法即使同样加上 IDF 也无法追上。

### 混合增益并非KeyAtten专属——但KeyAtten受益最大

为验证公平性，我们给外部方法也加上了 IDF 混合：

| 方法 | 原始 F1@10 | + IDF 后 F1@10 | 变化 |
|------|:---:|:---:|:---:|
| TextRank | 0.0754 | 0.0617 | **退化** |
| KeyBERT | 0.0445 | 0.0690 | +55% |
| CLS-Attn（KeyAtten） | 0.0671 | **0.1344** | **+100%** |

结论：IDF 不是KeyAtten方法的专属增益来源，但不同方法对 IDF 的受益程度差异极大。KeyAtten Attention + IDF 在混合后仍以绝对优势领先。

---

## 方法族概览

| 方法 | 类型 | 是否需要 [CLS] | 支持架构 | 上下文窗口 | 可解释性 |
|------|------|:---:|----------|:---:|:---:|
| TermFreq / TF-IDF | 传统统计 | 否 | 无模型依赖 | 不限 | 强 |
| TextRank | 图方法 | 否 | 无模型依赖 | 不限 | 强 |
| KeyBERT | 向量相似度 | 否 | Encoder | 受模型限制 | 中 |
| CLS-Attn | Attention | 是 | Encoder | 512 | 强 |
| Received-Attn | Attention | 否 | Encoder / Decoder-only | 512 / 262K | 强 |
| SAMRank | Attention | 否 | Encoder | 512 | 中 |
| Fusion-Attn | Attention | 部分 | Encoder | 512 | 中 |
| *_idf 变体 | Attention + IDF | 同上 | 同上 | 同上 | 中 |

### Decoder-only 扩展

KeyAtten方法已验证可在无 [CLS] 的 Decoder-only 长上下文 Transformer 上工作：

| 方法 | 模型 | CSL F1@10 | ShenCeCup F1@10 |
|------|------|:---:|:---:|
| Received-Attn | Qwen3.5-0.8B | 0.1496 | **0.2579** |
| Received-Attn | Qwen3.5-2B | **0.1568** | 0.2221 |
| Last-Token Attn | Qwen3.5-2B | 0.1100 | 0.0694 |
| EOS-Token Attn | Qwen3.5-2B | 0.1100 | 0.0694 |

关键发现：
- `Received-Attn` 是当前最值得推进的无 [CLS] 方案
- Attention 关键词提取不依赖 [CLS]，只要能获取 token-to-token Attention 即可工作
- 模型规模增大不保证效果单调提升——0.8B 在新闻场景反超 2B

---

## 场景推荐

| 场景 | 推荐方法 | 预期表现 |
|------|----------|----------|
| 中文新闻关键词提取 | SAMRank / Received-Attn（纯 Attention） | F1@10 ≈ 0.25 |
| 中文学术摘要关键词提取 | samrank_idf / cls_attn_idf（混合） | F1@10 ≈ 0.21 |
| 英文长文关键词提取 | cls_attn_idf（混合） | F1@10 ≈ 0.13 |
| 长上下文 / Decoder-only | Received-Attn（无 [CLS]） | F1@10 ≈ 0.16–0.26 |
| 低成本快速上线 | TextRank（传统基线） | F1@10 ≈ 0.07–0.10 |

---

## 局限性与后续计划

### 当前局限

- 中文数据集覆盖偏窄（仅学术摘要 + 新闻两类），缺少中文专业领域或长文集
- Decoder-only 结果基于小样本（20 docs），尚未扩展到正式 benchmark 规模
- 英文短文场景下优势不明显，尚需进一步优化

### 后续优先级

1. **P0**：补充更多中文关键词数据集（优先中文专业领域或中文长文）
2. **P1**：扩大 Decoder-only 正式评测样本规模
3. **P2**：探索更多 Encoder 模型和层级组合

---

## 支撑材料

详细的方法对比矩阵、混合基线分析、实验复现指南等支撑文档请见本目录：

- [`benchmark-leaderboard.md`](benchmark-leaderboard.md) — 完整主榜数据（含 Recall 指标）
- [`method-comparison.md`](method-comparison.md) — 方法能力对比矩阵与同口径效果表
- [`hybrid-baseline-analysis.md`](hybrid-baseline-analysis.md) — 混合基线公平对照实验
- [`reproduction-guide.md`](reproduction-guide.md) — 实验复现指南（模型下载、运行命令）

## 引用

本项目的 `samrank` 方法引用了以下论文的排序公式：

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. DOI: 10.18653/v1/2023.emnlp-main.630

`cls_attn`、`received_attn`、`fusion_attn` 及所有 `_idf` 混合策略为本项目原创。
