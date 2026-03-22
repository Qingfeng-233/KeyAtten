# KeyAtten

[English](README.md) | [中文](README.zh-CN.md)

Attention-based keyword extraction framework. Zero training, zero labeling, single forward pass. Supports Chinese and English.

Evaluated on 7 public datasets against 14 methods: +67% F1@10 over traditional baselines on Chinese news, +78% over the strongest external method on English long documents.

## Features

- Extracts keywords directly from pretrained model attention weights — no fine-tuning or labeling required
- Attention-IDF hybrid strategy for significant gains on long documents and corpus-aware scenarios
- Word-level semantic weight output (weight value, position index, POS tag)
- Single-layer or multi-layer attention weighted fusion
- Lightweight: 22M–33M parameter models, single forward pass

## Installation

```bash
pip install .
```

Dependencies: `torch>=2.0` `transformers>=4.30` `jieba` `scikit-learn` `nltk` `numpy`

## Quick Start

### Keyword Extraction

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(model="thenlper/gte-small-zh", language="zh")

# Pure attention
keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="cls_attn",
)
```

### Attention-IDF Hybrid

```python
# Fit IDF from a corpus first
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])

keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="samrank_idf",
    idf_lookup=idf,
)
```

### Word-Level Weights

```python
weights = ext.extract_word_weights(
    "自然语言处理是人工智能的重要方向",
    method="received_attn",
)
for w in weights:
    print(w.word, w.weight, w.pos_tag)
```

### Batch Extraction

```python
results = ext.extract_keywords_batch(
    ["文本一", "文本二", "文本三"],
    method="fusion_attn",
)
```

### Convenience Function

```python
from keyatten import extract_keywords

keywords = extract_keywords(
    "自然语言处理是人工智能的重要方向",
    model="thenlper/gte-small-zh",
)
```

## Methods

| Method | Description |
|--------|-------------|
| `cls_attn` | Attention weights from [CLS] token to each token |
| `received_attn` | Total attention each token receives from all tokens |
| `samrank` | SAMRank formula (global attention + proportional redistribution) |
| `fusion_attn` | Normalized product of CLS and received attention |

Each method has a corresponding `_idf` hybrid variant (e.g., `cls_attn_idf`) that multiplies attention scores with TF-IDF, suitable for corpus-aware scenarios.

> The `samrank` formula is referenced from [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630). The other methods (`cls_attn`, `received_attn`, `fusion_attn`) and all `_idf` hybrid strategies are original to this project.

### Choosing a Method

`samrank` achieves the highest benchmark scores (F1@10) due to broader coverage and stronger recall. In practice, `cls_attn` is often more useful — it extracts the most distinctive core terms, making it ideal for tag clouds and summaries.

## Practical Examples

Side-by-side comparison of `cls_attn` vs `samrank` across domains (model: `gte-small-zh`, top_k=6):

| Domain | Input (excerpt) | cls_attn | samrank |
|--------|----------------|----------|---------|
| Tech | OpenAI released GPT-4o with multimodal input... | OpenAI, GPT, model | OpenAI, model, GPT |
| Medical | mRNA vaccine encodes spike protein... Omicron variant... | mRNA, mRNA vaccine, COVID, **Omicron variant** | mRNA, mRNA vaccine, COVID, COVID virus |
| Finance | Fed announces 25bp rate hike... | rate hike, basis points, **global stocks**, rate | rate hike, basis points, rate, global stocks |
| Sports | Messi scores hat-trick in World Cup final... lifts trophy | **Messi**, trophy, hat-trick, **final** | trophy, Messi, hat-trick, **penalty** |
| History | Qin Shi Huang unified six states... centralized dynasty | centralization, feudal dynasty, standardization | centralization, standardization, feudal dynasty |
| Daily | Meet at Starbucks at 3pm... business trip to Beijing | **Starbucks**, Beijing, business trip | **meet**, Beijing, **chat** |

`cls_attn` favors the most distinctive entities (Messi, Starbucks, Omicron), ideal for tag clouds and summary displays. `samrank` provides broader coverage, better suited for retrieval and evaluation scenarios.

## Recommended Models

| Language | Model | Parameters |
|----------|-------|------------|
| Chinese | `thenlper/gte-small-zh` | ~33M |
| English | `sentence-transformers/all-MiniLM-L6-v2` | ~22M |

## Evaluation Summary

Compared against TF-IDF, TextRank, KeyBERT and 14 methods total on 7 public datasets (F1@10):

| Scenario | KeyAtten Best | vs Strongest Traditional | vs Strongest External |
|----------|:---:|:---:|:---:|
| Chinese News (ShenCeCup) | **0.2579** | +67% | — |
| Chinese Academic (CSL) | **0.2106** | +9% | — |
| English Long-doc (SemEval2010-fulltext) | **0.1344** | — | +78% |
| English Long-doc (Krapivin2009-fulltext) | **0.1268** | — | +79% |
| English Short-doc (3 datasets) | 0.1370 | — | On par |

Full evaluation report: [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md)

## API

### KeyAttenExtractor

```python
KeyAttenExtractor(
    model: str,                         # Hugging Face model name
    language: str = "zh",               # "zh" or "en"
    device: str = "cpu",                # compute device
    layer_index: int = -1,              # single layer index (-1 = last layer)
    layer_indices: list[int] = None,    # multi-layer indices
    layer_weights: list[float] = None,  # multi-layer weights
    attn_merge: bool = False,           # attention-guided char merging for Chinese
    merge_threshold: float = 0.3,       # merge threshold (0.0–1.0)
)
```

| Method | Returns |
|--------|---------|
| `extract_keywords(text, method, top_k, idf_lookup)` | `list[str]` |
| `extract_keywords_batch(texts, method, top_k, idf_lookup)` | `list[list[str]]` |
| `extract_word_weights(text, method)` | `list[WordWeight]` |
| `fit_idf(texts)` | `dict[str, float]` |

`WordWeight` fields: `word`, `index`, `weight`, `pos_tag`.

## Citation

The `samrank` method in this project references the ranking formula from:

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI: 10.18653/v1/2023.emnlp-main.630](https://doi.org/10.18653/v1/2023.emnlp-main.630)

`cls_attn`, `received_attn`, `fusion_attn` and all `_idf` hybrid strategies are original to this project.

## License

[MIT](./LICENSE)
