# KeyAtten

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19451584.svg)](https://doi.org/10.5281/zenodo.19451584)

[English](README.md) | [中文](README.zh-CN.md)

基于 Transformer Attention 机制的关键词提取框架。零训练、零标注，仅需一次前向推理，支持中英双语。

在 7 个公开数据集、14 种方法的对比评测中，中文新闻场景 F1@10 较传统基线提升 67%，英文长文场景较外部最强方法提升约 78%。

## 默认发布路线

- 默认中文模型：`thenlper/gte-small-zh`
- 默认发布方法：`received_attn`，以及有语料库时的 `_idf` 变体
- 默认部署方向：小模型 + 可解释 Attention + 轻量算子

当前仓库仍把 `gte-small-zh` 作为轻量默认发布模型，但主库已经正式支持 decoder-only 的 causal attention 自适配。对于未显式指定层位的 causal 模型，默认会自动推荐约 3/4 深度附近的中上层，而不是继续落在最后层。以 `Qwen/Qwen3-Embedding-0.6B` 为例，默认会优先落在约第 21 层附近，而不是第 27 层。

## 方法分类

当前 README 只按 3 类方法理解主库：

### 1. 主方法：BIO 候选 + Attention 微调排序

这是当前主推路线。

含义很简单：

- 先把默认候选换成 BIO 候选
- 再对 attention 做微调，用它来排序

主入口：

- `KeyAttenExtractor(candidate_scoring="bio")`
- `CandidateSegmentAttentionExtractor`

### 2. 单方法

- Attention 系列方法
- `BIOExtractor`
- `QKLoRAExtractor`

其中 Attention 系列方法包括：

- `cls_attn`
- `received_attn`
- `samrank`
- `fusion_attn`
- 对应 `_idf` 变体

### 3. Baseline / 其他方法

- 用于对照、历史实验或外部比较

## 特性

- 直接利用预训练模型的注意力权重提取关键词，无需额外训练或标注
- 提供 Attention-IDF 混合策略，在长文和有语料库的场景下效果显著
- 支持词级语义权重输出（含权重值、位置索引、词性标注）
- 支持单层或多层 Attention 加权融合
- 支持基于 BIO 候选的 Candidate-Segment Attention 重排
- 仅需 22M–33M 参数的小模型，单次前向推理完成

## 安装

```bash
pip install keyatten
```

默认安装现在只包含 `numpy`，不会在 `import keyatten` 时顺带拉起整套重量级推理依赖。

```bash
pip install "keyatten[inference,zh]"   # 中文关键词提取
pip install "keyatten[inference,en]"   # 英文关键词提取
pip install "keyatten[inference,zh,lightweight]"  # 中文轻量部署
pip install "keyatten[full]"           # 安装全部可选依赖
```

可选依赖分组：

- `inference`: `torch>=2.0`、`transformers>=4.30`
- `lightweight`: `onnx>=1.16`、`onnxruntime>=1.18`、`tokenizers>=0.15`
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
    method="received_attn",
)
```

### Attention-IDF 混合

```python
# 先从语料库拟合 IDF
idf = ext.fit_idf(["自然语言处理是人工智能的重要方向", "关键词提取是文本挖掘任务"])

keywords = ext.extract_keywords(
    "自然语言处理是人工智能的重要方向",
    method="fusion_attn_idf",
    idf_lookup=idf,
)
```

### 缓存与增量 IDF

```python
ext = KeyAttenExtractor(
    model="Qwen/Qwen3-Embedding-0.6B",
    language="zh",
    device="cuda",
    dtype="float16",
    cache_enabled=True,
    cache_dir="cache",
)

# fit_idf 会从零重建 IDF 状态
idf = ext.fit_idf(["旧文本一", "旧文本二"])

# update_idf 只追加新文本的 document frequency
idf = ext.update_idf(["新增文本三"])

keywords = ext.extract_keywords(
    "新增文本三",
    method="fusion_attn_idf",
    top_k=8,
    idf_lookup=idf,
)
```

开启缓存后，KeyAtten 会写入两层缓存：

- `cache/keyatten_documents/`：IDF 前缓存，保存分词、候选、`token_counts` 和 attention 词分数；IDF 变化后可复用，不重跑模型。
- `cache/keyatten_keywords/`：IDF 后缓存，保存某个 IDF 指纹下的最终关键词；同配置同 IDF 再次调用可直接返回。

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

### 外部分词输入

```python
keywords = ext.extract_keywords(
    ["空天信息", "系统", "优化"],
    pos_tags=["n", "n", "v"],
    method="received_attn",
)
```

### 领域词典

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

### Token-Span 候选打分

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

### 用 BIO 候选替换结巴候选

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

### Candidate-Segment Attention 重排

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

### 便捷函数

```python
from keyatten import extract_keywords

keywords = extract_keywords(
    "自然语言处理是人工智能的重要方向",
    model="thenlper/gte-small-zh",
)
```

## Attention 系列方法

| 方法 | 说明 |
|------|------|
| `cls_attn` | [CLS] token 对各 token 的注意力权重 |
| `received_attn` | 各 token 从所有 token 接收的注意力总和 |
| `samrank` | SAMRank 排序公式（全局注意力 + 比例分配） |
| `fusion_attn` | CLS 注意力与 received 注意力的归一化融合 |

以上每种方法均有对应的 `_idf` 混合变体（如 `cls_attn_idf`），将 Attention 分数与 TF-IDF 相乘，适合有语料库的场景。

> `samrank` 的排序公式引用自 [Kang & Shin (2023, EMNLP)](https://doi.org/10.18653/v1/2023.emnlp-main.630)，其余方法及所有 `_idf` 混合策略为本项目原创。

### 如何使用 Attention 副方法

当前更稳的默认起点是 `received_attn`。如果你有语料库，优先试 `_idf` 变体；在本轮中文 decoder-only 收口里，`csl_test` 主看 `received_attn_idf`，`shencecup_labeled` 主看 `fusion_attn_idf`。`cls_attn` 仍然适合做“一眼看主题”的高辨识度展示，但不再是主库关键词接口的默认方法。

如果你的主指标是 `F1@5`，现在还可以把“嵌套短语去重”作为可选后处理打开。这个开关只会在 `top_k <= 5` 时生效，用来过滤“自然语言处理 / 自然语言 / 语言处理”这类文本包含关系，默认关闭，不影响现有 `@10` 路线。

对于原始字符串输入，主库现在还支持可选的 `candidate_scoring="token_span"` 路线。候选生成仍然走分词与词性过滤，但候选排序会直接聚合候选字符跨度内的 token attention，绕过原来的词级 mean-of-means 打分。

对于中文原始字符串输入，主库也支持 `candidate_scoring="bio"` 路线。这条方法会先用 `BIOExtractor` 替换默认的结巴/POS 候选生成，再用 attention 对 BIO 候选打分。

对于需要训练后重排的中文路线，主库现在还提供 `CandidateSegmentAttentionExtractor`。这条路线里 `BIOExtractor` 只负责产出候选，排序完全由 `文章全文 + 候选段` 上的 attention 完成。如果使用随机候选顺序，建议推理时使用 `random_seeds=[1, 2, 3]` 这类多 seed 集成，以降低顺序敏感性。

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

## Decoder-Only 支持

主库现在已经把 decoder-only 的稳定增益并进来了：

- 自动识别 causal 模型
- 中文 causal 模型默认使用前缀 `核心关键词、关键实体、主题：`
- 未显式传 `layer_index` 时，默认推荐约 3/4 深度附近的中上层，而不是最后层
- 对中文 decoder-only 的当前推荐组合，优先看 `Qwen/Qwen3-Embedding-0.6B + fusion_attn_idf`

最新收口详见项目内部实验文档 `docs/`。

## 轻量部署

当前推荐的轻量部署路线是 `gte-small-zh + ONNX Runtime`。在测试验证中，`gte-small-zh` 已经可以稳定导出 attention 并复现 `received_attn` 词分数，适合作为后续轻量算子和服务化部署的默认路线。

推荐安装命令：

```bash
pip install "keyatten[zh,lightweight]"
```

轻量后端示例：

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

说明：

- `model` 指向本地 `gte-small-zh` 模型目录，用于读取 `tokenizer.json`
- `onnx_path` 指向导出的 attention ONNX 文件
- 当前轻量后端支持单层 attention，适合默认发布路线的 `gte-small-zh`
- 如果要自行导出 ONNX 文件，再额外安装 `keyatten[inference,zh,lightweight]`

主仓库说明见：

- [gte_onnx_probe.py](./benchmark/tools/gte_onnx_probe.py)

## Benchmark 统一入口

无需再翻 `benchmark/` 里的零散脚本，统一从专业入口执行：

```bash
python -m keyatten.benchmark_cli --help
python -m keyatten.benchmark_cli keyword --root-dir "." --output-dir "outputs_smoke" --datasets csl_test --models thenlper/gte-small-zh --skip-yake --device cpu
```

可编辑安装后也可直接用：

```bash
keyatten-benchmark --help
keyatten-benchmark gte-onnx-probe
```

主要命令映射：

- `keyword` -> `benchmark/eval/run_keyword_benchmark.py`
- `hidden-head` -> `benchmark/eval/run_hidden_head_benchmark.py`
- `gte-onnx-probe` -> `benchmark/tools/gte_onnx_probe.py`
- `llm-keyword` -> `benchmark/eval/llm_keyword_benchmark.py`

完整说明见：[benchmark/README.md](./benchmark/README.md)

## 评测摘要

在 7 个公开数据集上与 TF-IDF、TextRank、KeyBERT 等 14 种方法对比，指标为 F1@10：

| 场景 | KeyAtten 最优 | 方法 | vs 最强传统基线 | vs 最强外部方法 |
|------|:---:|--------|:---:|:---:|
| 中文新闻（news55） | **0.4994** | BIO Viterbi | +224% | — |
| 中文新闻（ShenCeCup 1000） | **0.3292** | QK LoRA | +113% | — |
| 中文学术摘要（CSL） | **0.2106** | `samrank_idf` | +9% | — |
| 英文长文（SemEval2010-fulltext） | **0.1344** | `cls_attn_idf` | — | +78% |
| 英文长文（Krapivin2009-fulltext） | **0.1268** | `cls_attn_idf` | — | +79% |
| 英文短文（3 个数据集） | 0.1370 | `fusion_attn` | — | 持平 |

**主方法**（BIO 候选 + Candidate-Segment Attention 微调排序）在 news55 上 F1@10 = 0.4665，较 BIO clean 基线（0.3916）提升 +13.7%。

完整评测报告见 [EVALUATION-PUBLIC.md](./EVALUATION-PUBLIC.md)。

## API

### KeyAttenExtractor

```python
KeyAttenExtractor(
    model: str,                         # Hugging Face 模型名称
    language: str = "zh",               # "zh" 或 "en"
    device: str = "cpu",                # 计算设备
    backend: str = "auto",              # "auto" / "torch" / "onnx"
    onnx_path: str | None = None,       # ONNX attention 文件路径
    user_dict: str | list[str] | dict = None,  # 领域词典路径 / 术语列表 / 术语配置
    layer_index: int | None = None,     # None = 自动；causal 模型默认约 3/4 深度附近的中上层，-1 = 显式最后层
    layer_indices: list[int] = None,    # 多层索引列表
    layer_weights: list[float] = None,  # 多层权重列表
    instruction_prefix: str | None = None,  # causal 模型可选前缀
    is_causal_override: bool | None = None,  # None=自动检测；False=强制按 encoder 读；True=强制按 decoder 读
    dedup_nested_for_topk5: bool = False,    # 仅在 top_k<=5 时启用子串去重后处理
    candidate_scoring: str = "word",         # "word" / "token_span" / "bio"
    cache_enabled: bool = False,             # 是否启用磁盘缓存
    cache_dir: str | Path = "cache",         # 缓存目录
)
```

| 方法 | 返回值 |
|------|--------|
| `extract_keywords(text, method, top_k, idf_lookup)` | `list[str]` |
| `extract_keywords_batch(texts, method, top_k, idf_lookup)` | `list[list[str]]` |
| `extract_word_weights(text, method)` | `list[WordWeight]` |
| `fit_idf(texts)` | `dict[str, float]` |
| `update_idf(texts)` | `dict[str, float]` |

`WordWeight` 包含字段：`word`、`index`、`weight`、`pos_tag`。

说明：

- `extract_keywords` / `extract_word_weights` 支持直接传入外部分词后的 `list[str]`
- 这时可选传 `pos_tags`；若不传，中文默认按名词 `n`、英文默认按 `eng` 处理
- `user_dict` 支持三种形式：词典文件路径、术语列表、`{term: tag}` / `{term: (freq, tag)}` 配置
- `extract_keywords()` / `extract_keywords_batch()` 默认方法现在是 `received_attn`
- 对 causal 模型，如果 `layer_index` 留空，主库会自动使用约 3/4 深度附近的推荐中上层
- `is_causal_override` 只覆盖 attention 读取模式，不会改变模型本身结构
- `dedup_nested_for_topk5=True` 时，仅在 `top_k<=5` 应用子串/超串去重，不影响 `@10`
- `candidate_scoring="token_span"` 仅适用于原始字符串输入；外部分词输入仍然走原有词级排序路径
- `candidate_scoring="bio"` 需要配套传入 `bio_model_path`，且仅适用于原始字符串输入
- `fit_idf()` 会重建 IDF 状态；`update_idf()` 会在当前状态上增量追加新文档
- `cache_enabled=True` 时，word 候选路径会同时缓存 IDF 前文档分数和 IDF 后最终关键词

## 引用

本项目的 `samrank` 方法引用了以下论文的排序公式：

> Kang, B., & Shin, H. (2023). *SAMRank: Unsupervised Keyphrase Extraction using Self-Attention Map in BERT and GPT-2.* EMNLP 2023. [DOI: 10.18653/v1/2023.emnlp-main.630](https://doi.org/10.18653/v1/2023.emnlp-main.630)

`cls_attn`、`received_attn`、`fusion_attn` 及所有 `_idf` 混合策略为本项目原创。

## 致谢

感谢 [LinuxDo](https://linux.do) 社区的支持。

## 许可证

[MIT](./LICENSE)
