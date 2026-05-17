# KeyAtten

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19451584.svg)](https://doi.org/10.5281/zenodo.19451584)

KeyAtten: Attention-based Keyword/Keyphrase Extraction
[English](README.md) | [中文](README.zh-CN.md)

Attention-based keyword extraction framework. Zero training, zero labeling, single forward pass. Supports Chinese and English.

Evaluated on 7 public datasets against 14 methods: +67% F1@10 over traditional baselines on Chinese news, +78% over the strongest external method on English long documents.

## Default Release Path

- Default Chinese model: `thenlper/gte-small-zh`
- Default release method: `received_attn`, plus `_idf` variants when a corpus is available
- Default deployment path: small encoder + interpretable attention + lightweight operators

The repository still treats `gte-small-zh` as the lightweight default production model, but the main library now ships decoder-only causal attention adaptation. When no layer is specified for a causal model, KeyAtten automatically recommends a middle-upper layer at roughly 3/4 depth instead of falling back to the last layer. For example, `Qwen/Qwen3-Embedding-0.6B` defaults to the middle-upper band around layer 21 rather than layer 27.

## Method Categories

The README now groups the library into three public method categories:

### 1. Main Method: BIO Candidates + Fine-Tuned Attention Reranking

This is the current primary route.

In plain terms:

- replace the default candidate set with BIO candidates
- then use fine-tuned attention to rank them

Main entrypoints:

- `KeyAttenExtractor(candidate_scoring="bio")`
- `CandidateSegmentAttentionExtractor`

### 2. Standalone Methods

- Attention-series methods
- `BIOExtractor`
- `QKLoRAExtractor`

Attention-series methods include:

- `cls_attn`
- `received_attn`
- `samrank`
- `fusion_attn`
- their `_idf` variants

### 3. Baselines / Other Methods

- used for comparison, legacy experiments, or external baselines

## Features

- Extracts keywords directly from pretrained model attention weights — no fine-tuning or labeling required
- Attention-IDF hybrid strategy for significant gains on long documents and corpus-aware scenarios
- Word-level semantic weight output (weight value, position index, POS tag)
- Single-layer or multi-layer attention weighted fusion
- Candidate-segment attention reranking with BIO-generated phrase candidates
- Lightweight: 22M–33M parameter models, single forward pass

## Installation

```bash
pip install keyatten
```

Minimal install only includes `numpy` so importing the package does not pull the full ML stack by default.

```bash
pip install "keyatten[inference,zh]"   # Chinese keyword extraction
pip install "keyatten[inference,en]"   # English keyword extraction
pip install "keyatten[inference,zh,lightweight]"  # Chinese lightweight deployment
pip install "keyatten[full]"           # All optional dependencies
```

Optional dependency groups:

- `inference`: `torch>=2.0`, `transformers>=4.30`
- `lightweight`: `onnx>=1.16`, `onnxruntime>=1.18`, `tokenizers>=0.15`
- `zh`: `jieba>=0.42`
- `en`: `scikit-learn>=1.0`, `nltk>=3.8`

If you call extraction APIs without the required extras installed, KeyAtten now raises a direct install hint instead of failing during `import keyatten`.

## Quick Start

### Keyword Extraction

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(model="thenlper/gte-small-zh", language="zh")

# Pure attention
keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="received_attn",
)
```

### Attention-IDF Hybrid

```python
# Fit IDF from a corpus first
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])

keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="fusion_attn_idf",
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

### External Token Input

```python
keywords = ext.extract_keywords(
    ["空天信息", "系统", "优化"],
    pos_tags=["n", "n", "v"],
    method="received_attn",
)
```

### Domain Dictionary

```python
ext = KeyAttenExtractor(
    model="thenlper/gte-small-zh",
    language="zh",
    user_dict=["空天信息", "星闪技术"],
)

keywords = ext.extract_keywords(
    "空天信息系统优化方法",
    method="received_attn",
)
```

### Token-Span Candidate Scoring

```python
ext = KeyAttenExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    language="zh",
    candidate_scoring="token_span",
)

keywords = ext.extract_keywords(
    "水木年华被嘲讽已过气，卢庚戌回应称作品会留下来",
    method="fusion_attn_idf",
    idf_lookup=idf,
)
```

### BIO Candidates Instead of Jieba Candidates

```python
ext = KeyAttenExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    language="zh",
    candidate_scoring="bio",
    bio_model_path="models/bio_ckipbert_extractive_ep13/bio_model_full.pt",
)

keywords = ext.extract_keywords(
    "水木年华被嘲讽已过气，卢庚戌回应称作品会留下来",
    method="received_attn",
)
```

### Candidate-Segment Attention Reranking

```python
from keyatten import CandidateSegmentAttentionExtractor

ext = CandidateSegmentAttentionExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    adapter_path="models/candidate_segment_attn/qwen06_v2_2k_len1024_c30/best_adapter",
    bio_model_path="models/bio_ckipbert_extractive_ep13/bio_model_full.pt",
    max_candidates=30,
)

keywords = ext.extract_keywords(
    "水木年华被嘲讽已过气，卢庚戌回应称作品会留下来",
    random_seeds=[1, 2, 3],
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

## Attention-Series Methods

| Method | Description |
|--------|-------------|
| `cls_attn` | Attention weights from [CLS] token to each token |
| `received_attn` | Total attention each token receives from all tokens |
| `samrank` | SAMRank formula (global attention + proportional redistribution) |
| `fusion_attn` | Normalized product of CLS and received attention |

Each method has a corresponding `_idf` hybrid variant (e.g., `cls_attn_idf`) that multiplies attention scores with TF-IDF, suitable for corpus-aware scenarios.

> The `samrank` formula is referenced from [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630). The other methods (`cls_attn`, `received_attn`, `fusion_attn`) and all `_idf` hybrid strategies are original to this project.

### Using Attention as a Secondary Method

`received_attn` is now the safest default starting point. When a corpus is available, `_idf` variants should be tried first; in the latest Chinese decoder-only rollup, `received_attn_idf` is the main CSL path and `fusion_attn_idf` is the main ShenCeCup path. `cls_attn` is still useful for high-distinctiveness tag-cloud style outputs, but it is no longer the default keyword-extraction method.

If your main metric is `F1@5`, the library now also exposes an optional nested-phrase de-dup post-ranking step. It only activates when `top_k <= 5`, filters substring/superstring duplicates such as `natural language processing / natural language / language processing`, and stays off by default so the `@10` path is unchanged.

For raw string input, the library now also exposes an optional `candidate_scoring="token_span"` route. Candidate generation still follows the segmenter and POS filter, but ranking aggregates token attention directly over each candidate's character span, bypassing the previous word-level mean-of-means path.

For Chinese raw string input, the library also exposes `candidate_scoring="bio"`. This replaces the default jieba/POS candidate generator with `BIOExtractor` candidates first, then scores those BIO candidates with attention.

For trained Chinese reranking, the library also exposes `CandidateSegmentAttentionExtractor`. This route uses `BIOExtractor` only for candidate generation, then reranks the explicit candidate list with attention over the full `document + candidate segment` input. If you use random candidate order, prefer multi-seed inference such as `random_seeds=[1, 2, 3]` to reduce order sensitivity.

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

## Decoder-Only Support

The main library now includes the stable decoder-only gains:

- automatic causal model detection
- default Chinese causal prefix `核心关键词、关键实体、主题：`
- automatic middle-upper layer recommendation when `layer_index` is omitted, using a band at roughly 3/4 depth for causal models
- current recommended Chinese decoder-only combination: `Qwen/Qwen3-Embedding-0.6B + fusion_attn_idf`

Latest rollout details are documented in the project's internal experiment notes under `docs/`.

## Lightweight Deployment

The recommended lightweight deployment path is `gte-small-zh + ONNX Runtime`. Internal validation shows that `gte-small-zh` can export token attention and reproduce `received_attn` word scores with stable numerical agreement, making it the default route for lightweight operators and deployment work.

Recommended install:

```bash
pip install "keyatten[zh,lightweight]"
```

Lightweight backend example:

```python
from keyatten import KeyAttenExtractor

ext = KeyAttenExtractor(
    model="/path/to/thenlper__gte-small-zh",
    language="zh",
    backend="onnx",
    onnx_path="/path/to/attention_last.onnx",
)

keywords = ext.extract_keywords(
    "自然语言处理用于关键词提取与文本分析",
    method="received_attn",
)
```

Notes:

- `model` should point to a local `gte-small-zh` directory so KeyAtten can read `tokenizer.json`
- `onnx_path` should point to the exported attention ONNX file
- the lightweight backend currently supports a single exported attention layer, which matches the default `gte-small-zh` release path
- if you want to export the ONNX file yourself, install `keyatten[inference,zh,lightweight]` instead

See:

- [gte_onnx_probe.py](./benchmark/tools/gte_onnx_probe.py)

## Benchmark Entry

Use one professional entrypoint instead of browsing scripts under `benchmark/`:

```bash
python -m keyatten.benchmark_cli --help
python -m keyatten.benchmark_cli keyword --root-dir "." --output-dir "outputs_smoke" --datasets csl_test --models thenlper/gte-small-zh --skip-yake --device cpu
```

After editable install, you can use:

```bash
keyatten-benchmark --help
keyatten-benchmark gte-onnx-probe
```

Main command mapping:

- `keyword` -> `benchmark/eval/run_keyword_benchmark.py`
- `hidden-head` -> `benchmark/eval/run_hidden_head_benchmark.py`
- `gte-onnx-probe` -> `benchmark/tools/gte_onnx_probe.py`
- `llm-keyword` -> `benchmark/eval/llm_keyword_benchmark.py`

Full benchmark usage notes: [benchmark/README.md](./benchmark/README.md)

## Evaluation Summary

Compared against TF-IDF, TextRank, KeyBERT and 14 methods total on 7 public datasets (F1@10):

| Scenario | KeyAtten Best | Method | vs Strongest Traditional | vs Strongest External |
|----------|:---:|--------|:---:|:---:|
| Chinese News (news55) | **0.4994** | BIO Viterbi | +224% | — |
| Chinese News (ShenCeCup 1000) | **0.3292** | QK LoRA | +113% | — |
| Chinese Academic (paper_test_800) | **0.2752** | CSA (high_recall) | — | — |
| Chinese Academic (CSL, zero-shot) | **0.2106** | `samrank_idf` | +9% | — |
| English Long-doc (SemEval2010-fulltext) | **0.1344** | `cls_attn_idf` | — | +78% |
| English Long-doc (Krapivin2009-fulltext) | **0.1268** | `cls_attn_idf` | — | +79% |
| English Short-doc (3 datasets) | 0.1370 | `fusion_attn` | — | On par |

The **main method** (BIO candidates + fine-tuned Candidate-Segment Attention reranking) achieves F1@10 = 0.4665 on news55, a +13.7% improvement over BIO-only clean baseline (0.3916).

Full evaluation report: [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md)

## API

### KeyAttenExtractor

```python
KeyAttenExtractor(
    model: str,                         # Hugging Face model name
    language: str = "zh",               # "zh" or "en"
    device: str = "cpu",                # compute device
    backend: str = "auto",              # "auto" / "torch" / "onnx"
    onnx_path: str | None = None,       # ONNX attention file path
    user_dict: str | list[str] | dict = None,  # domain dictionary path / term list / term config
    layer_index: int | None = None,     # None = auto; causal models default to the middle-upper band at roughly 3/4 depth, -1 = explicit last layer
    layer_indices: list[int] = None,    # multi-layer indices
    layer_weights: list[float] = None,  # multi-layer weights
    instruction_prefix: str | None = None,  # optional prefix for causal models
    is_causal_override: bool | None = None,  # None=auto detect; False=force encoder-style readout; True=force decoder-style readout
    dedup_nested_for_topk5: bool = False,    # enable substring de-dup post-processing only when top_k<=5
    candidate_scoring: str = "word",   # "word" / "token_span" / "bio"
)
```

| Method | Returns |
|--------|---------|
| `extract_keywords(text, method, top_k, idf_lookup)` | `list[str]` |
| `extract_keywords_batch(texts, method, top_k, idf_lookup)` | `list[list[str]]` |
| `extract_word_weights(text, method)` | `list[WordWeight]` |
| `fit_idf(texts)` | `dict[str, float]` |

`WordWeight` fields: `word`, `index`, `weight`, `pos_tag`.

Notes:

- `extract_keywords` and `extract_word_weights` also accept pre-tokenized `list[str]`
- when external tokens are provided, `pos_tags` is optional; Chinese defaults to `n`, English defaults to `eng`
- `user_dict` accepts a dictionary file path, a term list, or mappings like `{term: tag}` / `{term: (freq, tag)}`
- `extract_keywords()` and `extract_keywords_batch()` now default to `received_attn`
- if `layer_index` is omitted for a causal model, KeyAtten automatically uses the recommended middle-upper layer at roughly 3/4 depth
- `is_causal_override` only overrides the attention readout mode; it does not change the underlying model architecture
- when `dedup_nested_for_topk5=True`, substring/superstring de-dup is applied only for `top_k<=5`, not for `@10`
- `candidate_scoring="token_span"` only applies to raw string input; external token input stays on the word-based ranking path
- `candidate_scoring="bio"` requires `bio_model_path` and only applies to raw string input

## Citation

The `samrank` method in this project references the ranking formula from:

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI: 10.18653/v1/2023.emnlp-main.630](https://doi.org/10.18653/v1/2023.emnlp-main.630)

`cls_attn`, `received_attn`, `fusion_attn` and all `_idf` hybrid strategies are original to this project.

## Acknowledgments

Thanks to the [LinuxDo](https://linux.do) community for their support.

## License

[MIT](./LICENSE)
