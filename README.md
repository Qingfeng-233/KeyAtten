# KeyAtten

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19451584.svg)](https://doi.org/10.5281/zenodo.19451584)

[English](README.md) | [中文](README.zh-CN.md)

Keyword / keyphrase extraction from Transformer attention, for Chinese and English. The base route needs no training or labels and runs in a single forward pass; trained routes (BIO candidates, LoRA reranking) are available when higher accuracy is needed.

## Installation

```bash
pip install "keyatten[inference,zh]"              # Chinese
pip install "keyatten[inference,en]"              # English
pip install "keyatten[inference,zh,lightweight]"  # Chinese + ONNX Runtime
pip install "keyatten[full]"                      # everything
```

| Extra | Packages |
|---|---|
| `inference` | `torch`, `transformers` |
| `lightweight` | `onnx`, `onnxruntime`, `tokenizers` |
| `zh` | `jieba` |
| `en` | `scikit-learn`, `nltk` |
| `bio` | `torch`, `transformers`, `pytorch-crf` |

`pip install keyatten` alone only pulls `numpy`; a missing extra raises an install hint when the API is called.

## Quick Start

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(model="thenlper/gte-small-zh", language="zh")

ext.extract_keywords("自然语言处理是人工智能的重要方向", method="received_attn", top_k=10)

# With a corpus: fit IDF, then use an _idf method
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])
ext.extract_keywords("自然语言处理是人工智能的重要方向", method="fusion_attn_idf", idf_lookup=idf)

ext.extract_keywords_batch(["文本一", "文本二"], method="fusion_attn")
ext.extract_word_weights("自然语言处理是人工智能的重要方向")  # list[WordWeight(word, index, weight, pos_tag)]
```

One-off call: `from keyatten import extract_keywords; extract_keywords(text, model="thenlper/gte-small-zh")`.

## Extractors

| Extractor | Needs trained weights | Use when |
|---|:---:|---|
| `KeyAttenExtractor` | No | Default; zero-shot attention scoring |
| `KeyAttenExtractor(candidate_scoring="bio")` | BIO checkpoint | Better Chinese candidates than jieba |
| `CandidateSegmentAttentionExtractor` | BIO checkpoint + LoRA adapter | Main method, best Chinese accuracy |
| `QKLoRAExtractor` | LoRA adapter | Standalone Q·K scoring |
| `BIOExtractor` | BIO checkpoint | Span extraction only (`extract_spans`) |

```python
from keyatten import CandidateSegmentAttentionExtractor

ext = CandidateSegmentAttentionExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    adapter_path="models/candidate_segment_attn/qwen06_v2_2k_len1024_c30/best_adapter",
    bio_model_path="models/bio_ckipbert_extractive_ep13/bio_model_full.pt",
)
ext.extract_keywords("水木年华被嘲讽已过气，卢庚戌回应称作品会留下来", random_seeds=[1, 2, 3])
```

With `candidate_order="random"` (default), multiple `random_seeds` reduce sensitivity to candidate order.

## Attention Methods

| Method | Score |
|---|---|
| `received_attn` | Total attention a token receives — recommended default |
| `cls_attn` | Attention from `[CLS]`; favors distinctive entities (tag clouds) |
| `samrank` | SAMRank formula ([Kang & Shin, 2023](https://doi.org/10.18653/v1/2023.emnlp-main.630)) |
| `fusion_attn` | Normalized product of `cls_attn` and `received_attn` |
| `voted_attn` | Received attention, content-masked and position-normalized |
| `excess_attn` | Attention above a uniform baseline, excluding attention sinks |

Every method has an `_idf` variant (e.g. `received_attn_idf`) that multiplies by TF-IDF; prefer it whenever a corpus is available. `fit_idf()` rebuilds IDF state, `update_idf()` adds new documents incrementally.

## KeyAttenExtractor Options

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | HF model name or local path |
| `language` | `"zh"` | `"zh"` / `"en"` |
| `device` | `"cpu"` | Compute device |
| `backend` | `"auto"` | `"torch"` / `"onnx"`; `auto` picks ONNX when an `.onnx` file is found |
| `onnx_path` | `None` | Exported attention ONNX file |
| `dtype` | `"auto"` | `"float32"` / `"float16"` / `"bfloat16"` |
| `layer_index` | `None` | Single layer; `None` = last layer for encoders, ~3/4 depth for decoder-only models |
| `layer_indices`, `layer_weights` | `None` | Weighted multi-layer fusion |
| `instruction_prefix` | `None` | Prefix for causal models (Chinese default `核心关键词、关键实体、主题：`) |
| `is_causal_override` | `None` | Force decoder (`True`) or encoder (`False`) readout |
| `user_dict` | `None` | Domain terms: file path, list, or `{term: tag}` / `{term: (freq, tag)}` |
| `candidate_scoring` | `"word"` | `"word"` / `"token_span"` (score candidate char spans) / `"bio"` (needs `bio_model_path`) |
| `dedup_nested_for_topk5` | `False` | Drop nested duplicates when `top_k <= 5` |
| `cache_enabled`, `cache_dir` | `False`, `"cache"` | Disk cache of attention scores (reused across IDF changes) and final keywords |

`extract_keywords` also accepts pre-tokenized input: `ext.extract_keywords(["空天信息", "系统", "优化"], pos_tags=["n", "n", "v"])`. `token_span` and `bio` apply only to raw string input.

## Models and Deployment

| Use | Model |
|---|---|
| Chinese, lightweight (~33M) | `thenlper/gte-small-zh` |
| English, lightweight (~22M) | `sentence-transformers/all-MiniLM-L6-v2` |
| Chinese, decoder-only | `Qwen/Qwen3-Embedding-0.6B` + `fusion_attn_idf` |

ONNX Runtime without torch (`pip install "keyatten[zh,lightweight]"`):

```python
ext = KeyAttenExtractor(
    model="/path/to/gte-small-zh",        # local dir containing tokenizer.json
    backend="onnx",
    onnx_path="/path/to/attention_last.onnx",  # single exported attention layer
)
```

Export the ONNX file with [gte_onnx_probe.py](./benchmark/tools/gte_onnx_probe.py) (needs `inference` too).

## Evaluation

F1@10 on 7 public datasets, against 14 methods including TF-IDF, TextRank and KeyBERT:

| Scenario | KeyAtten Best | Method | vs Strongest Traditional | vs Strongest External |
|----------|:---:|--------|:---:|:---:|
| Chinese News (news55) | **0.4994** | BIO Viterbi | +224% | — |
| Chinese News (ShenCeCup 1000) | **0.3292** | QK LoRA | +113% | — |
| Chinese Academic (paper_test_800) | **0.2752** | CSA (high_recall) | — | — |
| Chinese Academic (CSL, zero-shot) | **0.2106** | `samrank_idf` | +9% | — |
| English Long-doc (SemEval2010-fulltext) | **0.1344** | `cls_attn_idf` | — | +78% |
| English Long-doc (Krapivin2009-fulltext) | **0.1268** | `cls_attn_idf` | — | +79% |
| English Short-doc (3 datasets) | 0.1370 | `fusion_attn` | — | On par |

The main method (`CandidateSegmentAttentionExtractor`) reaches F1@10 = 0.4665 on news55, +13.7% over the BIO-only clean baseline (0.3916).

Full report: [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md). Reproduce from a source checkout with `python -m benchmark.main --help` ([benchmark/README.md](./benchmark/README.md)).

## Citation

`samrank` follows Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI](https://doi.org/10.18653/v1/2023.emnlp-main.630). All other methods and the `_idf` hybrids are original to this project.

Thanks to the [LinuxDo](https://linux.do) community. License: [MIT](./LICENSE).
