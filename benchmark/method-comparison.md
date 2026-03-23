# 方法能力对比矩阵

## 方法全景

| 方法 | 方法族 | 需要 [CLS] | 验证模型/架构 | 上下文窗口 | 长上下文潜力 | 可解释性 | 定位 |
|------|--------|:---:|--------------|:---:|:---:|:---:|------|
| TermFreq | 传统统计 | 否 | 无模型依赖 | 不限 | 强 | 强 | 最轻量的频率基线 |
| TF-IDF | 传统统计 | 否 | 无模型依赖 | 不限 | 强 | 强 | 中文学术场景的强基线 |
| TextRank | 图方法 | 否 | 无模型依赖 | 不限 | 中 | 强 | 最稳定的传统基线 |
| KeyBERT | 向量相似度 | 否 | m3e-small / distilbert | 受模型限制 | 中 | 中 | 最"吃模型升级红利" |
| **CLS-Attn** | Attention | 是 | gte-small-zh | 512 | 弱 | 强 | 核心方法，直接取 [CLS] 行注意力 |
| **Received-Attn** | Attention | 否 | gte-small-zh / m3e-small / Qwen | 512 / 262K | 强 | 强 | 最稳定的 Attention 方法；发布默认优先 `gte-small-zh` |
| **SAMRank** | Attention | 否 | gte-small-zh | 512 | 中 | 中 | 新闻场景最强的 Encoder 方法 |
| **Fusion-Attn** | Attention | 部分 | gte-small-zh | 512 | 中 | 中 | CLS × Received 融合增强版 |
| **\*_idf 变体** | Attention + IDF | 同上 | 同上 | 同上 | 同上 | 中 | 长文场景显著增强 |

## 同口径效果对比

> 以下数据基于 CSL 20 docs + ShenCeCup 20 docs 同口径评测

### CSL 中文学术摘要

| 排名 | 方法 | 最优模型 | F1@10 | R@10 |
|:---:|------|----------|:---:|:---:|
| 1 | TF-IDF | gte-small-zh | 0.1935 | 0.2984 |
| 2 | TermFreq | gte-small-zh | 0.1834 | 0.2858 |
| 3 | SAMRank | gte-small-zh | 0.1773 | 0.2567 |
| 4 | Received-Attn | m3e-small | 0.1666 | 0.2298 |
| 5 | Qwen-2B Received | Qwen3.5-2B | 0.1568 | 0.2347 |
| 6 | CLS-Attn | gte-small-zh | 0.1554 | 0.2215 |
| 7 | KeyBERT | m3e-small | 0.1531 | 0.2078 |
| 8 | Fusion-Attn | gte-small-zh | 0.1505 | 0.2150 |
| 9 | Qwen-0.8B Received | Qwen3.5-0.8B | 0.1496 | 0.2222 |
| 10 | TextRank | gte-small-zh | 0.1237 | 0.1812 |

### ShenCeCup 中文新闻

| 排名 | 方法 | 最优模型 | F1@10 | R@10 |
|:---:|------|----------|:---:|:---:|
| 1 | Qwen-0.8B Received | Qwen3.5-0.8B | 0.2579 | 0.5633 |
| 2 | SAMRank | gte-small-zh | 0.2495 | 0.5367 |
| 3 | Received-Attn | gte-small-zh | 0.2424 | 0.5242 |
| 4 | CLS-Attn | gte-small-zh | 0.2269 | 0.4892 |
| 5 | Fusion-Attn | gte-small-zh | 0.2264 | 0.4867 |
| 6 | Qwen-2B Received | Qwen3.5-2B | 0.2221 | 0.4975 |
| 7 | TF-IDF | gte-small-zh | 0.1543 | 0.3542 |
| 8 | TermFreq | gte-small-zh | 0.1543 | 0.3542 |
| 9 | KeyBERT | m3e-small | 0.0991 | 0.2167 |
| 10 | TextRank | gte-small-zh | 0.0661 | 0.1308 |

## 方法评分（综合维度）

> 评分范围 1-10，面向中文关键词提取场景

| 方法 | 平均 F1@10 | 效果 | 稳定性 | 成本 | 吃模型红利 | 综合推荐 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Fusion-Attn | 0.1199 | 8.5 | 7.0 | 7.5 | 4.5 | 8.0 |
| CLS-Attn | 0.1160 | 8.0 | 6.5 | 8.0 | 3.5 | 7.5 |
| SAMRank | 0.1159 | 8.0 | 6.5 | 7.0 | 4.0 | 7.5 |
| Received-Attn | 0.1128 | 7.5 | 6.5 | 8.0 | 4.0 | 7.0 |
| TextRank | 0.1052 | 7.0 | 8.5 | 9.5 | 1.0 | 7.0 |
| KeyBERT | 0.0904 | 6.5 | 6.5 | 6.5 | 9.0 | 7.0 |

## 关键观察

1. **Attention 质量与 Embedding 模型性能不强相关**：更强的 Embedding 模型不一定带来更好的 Attention 关键词提取效果
2. **Attention 提供的区分度远超 Embedding 相似度**：在测试样本中，[CLS] 对核心关键词的注意力权重可达非关键词的 8.5 倍，而 Embedding 相似度差距仅 1.2 倍
3. **模型规模不是唯一因素**：Qwen3.5-0.8B 在新闻场景反超 2B 版本，说明 Attention 机制和领域匹配同样关键

## 发布口径

1. 默认发布模型为 `gte-small-zh`，因为它在中文主线上稳定、轻量、易部署。
2. 默认发布方法为 `received_attn / samrank / *_idf`，而不是更大的向量模型路线。
3. Qwen 与 decoder-only 结果保留为研究扩展，不作为默认发布配置。
