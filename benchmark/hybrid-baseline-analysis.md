# 混合基线公平对照实验

## 实验目的

验证"IDF 混合"是否只对KeyAtten Attention 系方法有效。如果外部方法加上 IDF 也能达到同等效果，那KeyAtten的优势就不成立。

为此，我们给两个代表性外部方法也补上了 IDF 混合变体：
- TextRank + IDF
- KeyBERT + IDF

## 实验配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | SemEval2010-fulltext（243 篇全文） |
| 模型 | sentence-transformers/all-MiniLM-L6-v2 |
| 主指标 | F1@10 |
| 辅指标 | R@10 |

## 结果

| 方法 | F1@10 | R@10 | 类型 |
|------|:---:|:---:|------|
| **cls_attn_idf** | **0.1344** | 0.1145 | KeyAtten Hybrid |
| **samrank_idf** | **0.1327** | 0.1129 | KeyAtten Hybrid |
| **received_attn_idf** | **0.1268** | 0.1080 | KeyAtten Hybrid |
| **fusion_attn_idf** | **0.1224** | 0.1044 | KeyAtten Hybrid |
| textrank | 0.0754 | 0.0651 | 外部基线 |
| keybert_idf | 0.0690 | 0.0595 | 外部 Hybrid |
| cls_attn | 0.0671 | 0.0572 | KeyAtten纯 Attention |
| textrank_idf | 0.0617 | 0.0529 | 外部 Hybrid |
| fusion_attn | 0.0622 | 0.0527 | KeyAtten纯 Attention |
| tfidf | 0.0604 | 0.0516 | 外部基线 |
| termfreq | 0.0570 | 0.0487 | 外部基线 |
| samrank | 0.0481 | 0.0411 | KeyAtten纯 Attention |
| received_attn | 0.0480 | 0.0410 | KeyAtten纯 Attention |
| keybert | 0.0445 | 0.0380 | 外部基线 |

## IDF 混合增益对比

| 方法 | 原始 F1@10 | + IDF 后 | 变化幅度 | 结论 |
|------|:---:|:---:|:---:|------|
| CLS-Attn（KeyAtten） | 0.0671 | 0.1344 | **+100%** | 显著增强 |
| SAMRank（KeyAtten） | 0.0481 | 0.1327 | **+176%** | 显著增强 |
| Received-Attn（KeyAtten） | 0.0480 | 0.1268 | **+164%** | 显著增强 |
| Fusion-Attn（KeyAtten） | 0.0622 | 0.1224 | **+97%** | 显著增强 |
| KeyBERT（外部） | 0.0445 | 0.0690 | +55% | 有一定提升 |
| TextRank（外部） | 0.0754 | 0.0617 | **-18%** | 反而退化 |

## 结论

1. **IDF 不是KeyAtten方法的专属增益**：KeyBERT + IDF 也获得了 55% 的提升
2. **但增益程度差异极大**：KeyAtten Attention 方法与 IDF 混合后提升 97%–176%，远超 KeyBERT 的 55%
3. **并非所有方法都受益于 IDF**：TextRank + IDF 反而退化了 18%
4. **外部方法混合后仍无法追上**：最强外部 Hybrid（keybert_idf = 0.0690）与KeyAtten最弱 Hybrid（fusion_attn_idf = 0.1224）之间仍有 77% 的差距
5. **最稳表述**：Attention 信号与 IDF 在长文场景中具有强互补性，这种互补性在KeyAtten方法上体现得最为充分
