# 完整主榜数据

> 本表为 V1 大横屏主榜的完整版本，包含 F1@10 和 R@10 双指标。
> 每个数据集仅保留各方法组下的最优代表。

## 指标说明

- **F1@10**：取 Top-10 关键词后计算的 F1 值（主指标）
- **R@10**：取 Top-10 关键词后的召回率（辅助指标）

## 主榜

| 数据集 | 语言/场景 | 最强传统基线 | 最强图/相似度基线 | 最强KeyAtten纯 Attention | 最强KeyAtten Hybrid | 最强 Decoder-only | 备注 |
|--------|-----------|-------------|-------------------|---------------------|----------------|-------------------|------|
| CSL | 中文 / 学术摘要 | TF-IDF @ gte-small-zh | KeyBERT @ gte-small-zh | SAMRank @ gte-small-zh | samrank_idf @ gte-small-zh | Qwen3.5-2B received_attn | Hybrid 已超传统基线 |
| | | F1=0.1935 R=0.2984 | F1=0.1296 R=0.1739 | F1=0.1773 R=0.2567 | **F1=0.2106 R=0.3167** | F1=0.1568 R=0.2347 | |
| ShenCeCup | 中文 / 新闻 | TermFreq @ gte-small-zh | TextRank @ gte-small-zh | SAMRank @ gte-small-zh | cls_attn_idf @ gte-small-zh | Qwen3.5-0.8B received_attn | 纯 Attention 更强 |
| | | F1=0.1543 R=0.3542 | F1=0.0661 R=0.1308 | F1=0.2495 R=0.5367 | F1=0.2357 R=0.5142 | **F1=0.2579 R=0.5633** | |
| SemEval2010 | 英文 / 学术短文 | TF-IDF @ MiniLM | KeyBERT @ distilbert | Fusion-Attn @ distilbert | — | — | 无 Hybrid/Decoder 口径 |
| | | F1=0.1040 R=0.1968 | F1=0.1448 R=0.2814 | F1=0.1448 R=0.2814 | | | |
| PubMed | 英文 / 学术短文 | TF-IDF @ MiniLM | KeyBERT @ distilbert | Fusion-Attn @ distilbert | — | — | 无 Hybrid/Decoder 口径 |
| | | F1=0.1211 R=0.1838 | F1=0.1327 R=0.2063 | F1=0.1327 R=0.2063 | | | |
| LIS2000 | 英文 / 学术短文 | TermFreq @ MiniLM | KeyBERT @ distilbert | Fusion-Attn @ distilbert | — | — | 无 Hybrid/Decoder 口径 |
| | | F1=0.1040 R=0.1652 | F1=0.1370 R=0.2179 | F1=0.1370 R=0.2179 | | | |
| SemEval2010-fulltext | 英文 / 学术长文 | TF-IDF @ MiniLM | TextRank @ MiniLM | CLS-Attn @ MiniLM | cls_attn_idf @ MiniLM | — | Hybrid 明显最强 |
| | | F1=0.0604 R=0.0516 | F1=0.0754 R=0.0651 | F1=0.0671 R=0.0572 | **F1=0.1344 R=0.1145** | | |
| Krapivin2009-fulltext | 英文 / 学术长文 | TF-IDF @ MiniLM | TextRank @ MiniLM | CLS-Attn @ MiniLM | cls_attn_idf @ MiniLM | — | 第二个长文集继续支持 Hybrid 最强 |
| | | F1=0.0565 R=0.0902 | F1=0.0707 R=0.1143 | F1=0.0789 R=0.1252 | **F1=0.1268 R=0.2030** | | |

## 结果来源

- CSL / ShenCeCup Hybrid：`测试沙箱/outputs_round10_hybrid_mean/`
- SemEval2010-fulltext：`测试沙箱/outputs_round12_semeval_fulltext_minilm/`
- Krapivin2009-fulltext：`测试沙箱/outputs_round12_krapivin_fulltext_minilm/`
- 英文短文：`测试沙箱/archive/experiments/outputs_round6_en_crosslingual/`
- Decoder-only：`测试沙箱/transformer_generalization/results/`

## SemEval2010-fulltext 完整排名

| 排名 | 方法 | F1@10 | R@10 |
|:---:|------|:---:|:---:|
| 1 | **cls_attn_idf** | 0.1344 | 0.1145 |
| 2 | **samrank_idf** | 0.1327 | 0.1129 |
| 3 | **received_attn_idf** | 0.1268 | 0.1080 |
| 4 | **fusion_attn_idf** | 0.1224 | 0.1044 |
| 5 | textrank | 0.0754 | 0.0651 |
| 6 | **cls_attn** | 0.0671 | 0.0572 |
| 7 | **fusion_attn** | 0.0622 | 0.0527 |
| 8 | tfidf | 0.0604 | 0.0516 |
| 9 | termfreq | 0.0570 | 0.0487 |
| 10 | **samrank** | 0.0481 | 0.0411 |
| 11 | **received_attn** | 0.0480 | 0.0410 |
| 12 | keybert | 0.0445 | 0.0380 |
