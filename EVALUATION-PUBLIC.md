# KeyAtten: Unified Evaluation of Attention-Based Keyword Extraction

> **Version**: V2
> **Date**: 2026-03-25
> **Author**: Linhao Jiang

---

## Abstract

This project presents a unified evaluation framework for attention-based keyword extraction, covering Chinese and English, short and long documents, and both Encoder and Decoder-only architectures. 14 methods were systematically compared across 7 public datasets.

KeyAtten provides 4 pure attention methods (`cls_attn`, `received_attn`, `samrank`, `fusion_attn`) and their IDF hybrid variants. The `samrank` ranking formula is referenced from [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630). The other methods (`cls_attn`, `received_attn`, `fusion_attn`) and all `_idf` hybrid strategies are original to this project.

### 2026-03-25 Update

- The only stable decoder-side gain from the latest Qwen3 study is the causal adaptation itself: last-token anchor, content masking, automatic causal detection, and middle-upper layer selection.
- This decoder-only adaptation has now been landed in the main library.
- A shortlist-only nested-phrase de-dup post-processing option is now available in the main library for `top_k<=5`; it is not part of the default `@10` evaluation path.
- Latest stable 100-document decoder-only results with `Qwen/Qwen3-Embedding-0.6B`:
  - `csl_test`: `received_attn_idf@layer_21 = 0.1630`
  - `shencecup_labeled`: `fusion_attn_idf@layer_21 = 0.2718`
- Experimental branches such as `excess_attn`, head-weighting, rise score, attention-gated candidates, and true bidirectional monkey-patches are not promoted to the default algorithm.
- Rollout summary: [benchmark/decoder-only-rollout-summary.md](./benchmark/decoder-only-rollout-summary.md)

### Key Findings

- **Chinese News**: Attention methods achieve F1@10 of 0.2579, **+67%** over the strongest traditional baseline
- **Chinese Academic Abstracts**: `samrank_idf` achieves F1@10 of 0.2106, surpassing the strongest traditional baseline by **+9%**
- **English Long Documents**: `cls_attn_idf` significantly outperforms external baselines on two independent fulltext datasets (F1@10: 0.1344 / 0.1268, **+78% / +79%**)
- **Architecture Generalization**: Attention-based keyword extraction works on both Encoder and Decoder-only architectures
- **Ultra-Low Cost**: Only 22M–33M parameter models, single forward pass, zero training, zero labeling

### Release Position

- The default release model is `gte-small-zh`
- The default release method is `received_attn`, with `_idf` variants as the main corpus-aware route
- `gte-small-zh + ONNX Runtime` remains the lightweight default production path
- Decoder-only causal adaptation is now part of the main library, but the historical tables below remain a V2 benchmark snapshot
- The lightweight deployment route is `gte-small-zh + ONNX Runtime`

---

## Evaluation Coverage

| Dimension | Coverage |
|-----------|----------|
| Languages | Chinese, English |
| Document Length | Short (abstracts), Long (full text) |
| Model Architecture | Encoder, Decoder-only |
| Datasets | 7 public datasets |
| Methods | 14 (traditional statistics, graph methods, vector similarity, Attention, Attention-IDF hybrid) |

### Datasets

| Dataset | Language | Scenario |
|---------|----------|----------|
| CSL | Chinese | Academic abstracts |
| ShenCeCup | Chinese | News |
| SemEval2010 | English | Academic short documents |
| PubMed | English | Academic short documents |
| LIS2000 | English | Academic short documents |
| SemEval2010-fulltext | English | Academic long documents (243 full texts) |
| Krapivin2009-fulltext | English | Academic long documents |

---

## Main Leaderboard

> Metric: F1@10 | Best result per method category for each dataset

| Dataset | Scenario | Best Traditional | Best External | KeyAtten Pure Attention | KeyAtten-IDF Best | Improvement |
|---------|----------|:---:|:---:|:---:|:---:|:---:|
| CSL | Chinese / Academic | TF-IDF 0.1935 | KeyBERT 0.1296 | `samrank` 0.1773 | `samrank_idf` **0.2106** | +9% vs traditional |
| ShenCeCup | Chinese / News | TermFreq 0.1543 | TextRank 0.0661 | `samrank` **0.2495** | `cls_attn_idf` 0.2357 | +67% vs traditional |
| SemEval2010 | English / Short | TF-IDF 0.1040 | KeyBERT 0.1448 | `fusion_attn` 0.1448 | — | On par |
| PubMed | English / Short | TF-IDF 0.1211 | KeyBERT 0.1327 | `fusion_attn` 0.1327 | — | On par |
| LIS2000 | English / Short | TermFreq 0.1040 | KeyBERT 0.1370 | `fusion_attn` 0.1370 | — | On par |
| SemEval2010-fulltext | English / Long | TF-IDF 0.0604 | TextRank 0.0754 | `cls_attn` 0.0671 | `cls_attn_idf` **0.1344** | +78% vs external |
| Krapivin2009-fulltext | English / Long | TF-IDF 0.0565 | TextRank 0.0707 | `cls_attn` 0.0789 | `cls_attn_idf` **0.1268** | +79% vs external |

### Interpretation

1. **Chinese News**: Pure attention (`samrank`) significantly outperforms traditional baselines on both Encoder and Decoder-only architectures
2. **Chinese Academic**: `samrank_idf` surpasses the strongest traditional baseline (TF-IDF), with IDF hybrid providing the key gain
3. **English Long Documents**: `cls_attn_idf` consistently leads by ~80% over external baselines on two independent fulltext datasets; top 4 are all KeyAtten-IDF methods
4. **English Short Documents**: `fusion_attn` matches the best external method (KeyBERT) but does not pull ahead

For release and deployment, the project standardizes on the encoder route around `gte-small-zh`. Decoder-only results remain useful as research evidence, but not as the default shipped configuration.

---

## Chinese Full Rankings

### CSL Chinese Academic Abstracts

| Rank | Method | Model | F1@10 | R@10 |
|:---:|--------|-------|:---:|:---:|
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

### ShenCeCup Chinese News

| Rank | Method | Model | F1@10 | R@10 |
|:---:|--------|-------|:---:|:---:|
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

## English Long Document Full Rankings

SemEval2010-fulltext (243 full texts):

| Rank | Method | F1@10 | R@10 |
|:---:|--------|:---:|:---:|
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

Krapivin2009-fulltext shows the same ranking pattern, with `cls_attn_idf` best at F1@10 = 0.1268.

---

## IDF Hybrid Gain Analysis

To ensure fairness, external methods were also given IDF augmentation (SemEval2010-fulltext):

| Method | Original F1@10 | + IDF | Change |
|--------|:---:|:---:|:---:|
| `cls_attn` (KeyAtten) | 0.0671 | **0.1344** | **+100%** |
| `samrank` (KeyAtten) | 0.0481 | **0.1327** | **+176%** |
| `received_attn` (KeyAtten) | 0.0480 | **0.1268** | **+164%** |
| `fusion_attn` (KeyAtten) | 0.0622 | **0.1224** | **+97%** |
| KeyBERT | 0.0445 | 0.0690 | +55% |
| TextRank | 0.0754 | 0.0617 | -18% |

Conclusions:

- Attention methods show the strongest synergy with IDF (+97% to +176%), far exceeding KeyBERT's +55%
- TextRank + IDF actually degrades by 18% — not all methods benefit from IDF augmentation
- The strongest external hybrid (KeyBERT+IDF = 0.0690) still falls far below KeyAtten's weakest hybrid (`fusion_attn_idf` = 0.1224), a 77% gap

---

## Cost Analysis

| Method Type | Requires Model | Parameters | Inference | Extra Computation |
|-------------|:---:|:---:|:---:|-------------------|
| TF-IDF / TextRank | No | 0 | 0 | Word frequency / graph iteration |
| KeyBERT | Yes | ~33M | 1 forward | Cosine similarity |
| **KeyAtten Attention** | Yes | **22M–33M** | **1 forward** | **None** |
| **KeyAtten-IDF hybrid** | Yes | **22M–33M** | **1 forward** | **+ IDF (near-zero overhead)** |

- Same inference cost as KeyBERT, but 67%–176% better in advantageous scenarios
- Uses small models (22M–33M parameters), not large language models
- Zero training, zero labeling, zero fine-tuning

---

## Architecture Generalization

Attention-based keyword extraction has been validated on Decoder-only long-context Transformers:

| Architecture | Model | Scale | CSL F1@10 | ShenCeCup F1@10 |
|-------------|-------|:---:|:---:|:---:|
| Decoder-only | Qwen3.5-0.8B | 0.8B | 0.1496 | **0.2579** |
| Decoder-only | Qwen3.5-2B | 2B | **0.1568** | 0.2221 |
| Encoder | gte-small-zh | ~33M | 0.1773 | 0.2495 |

Model scale is not the only factor: Qwen3.5-0.8B outperforms the 2B version on news, suggesting attention mechanism and domain fit matter as much as size. These decoder-only results are kept as exploratory validation; the default release recommendation remains `gte-small-zh`.

---

## Scenario Guide

| Scenario | Recommended Method | Expected F1@10 | vs Traditional Baseline |
|----------|-------------------|:---:|:---:|
| Chinese News | `samrank` | ~0.25 | +67% |
| Chinese Academic | `samrank_idf` | ~0.21 | +9% |
| English Long Documents | `cls_attn_idf` | ~0.13 | +78%–79% |
| Tag Clouds / Summaries | `cls_attn` | — | Highest distinctiveness |
| Decoder-only | `received_attn` | ~0.16–0.26 | Varies by scenario |

---

## Limitations

- Chinese dataset coverage is narrow (academic abstracts + news only), lacking more domain-specific validation
- Decoder-only results are based on small samples, not yet scaled to formal benchmark size
- No clear advantage on English short document scenarios

---

## Citation

The `samrank` method in this project references the ranking formula from:

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). DOI: 10.18653/v1/2023.emnlp-main.630

`cls_attn`, `received_attn`, `fusion_attn` and all `_idf` hybrid strategies are original to this project.

---

## Reproducibility

Full evaluation code is archived, SHA-256:

```
keyatten-benchmark-code.tar.gz  1760e974241209a85f74fd94ff2aecd1f4f9c7704bcc5045eed8e162cf1aef6e
```

The benchmark code is available in the [`benchmark/`](./benchmark/) directory. The archive hash can be used to verify consistency.

---

*This evaluation is based on public datasets, all obtainable from their original sources.*
