# 实验复现指南

## 环境要求

- Python 3.8+
- PyTorch（支持 CUDA）
- transformers
- 其他依赖见代码 import

## 项目结构

```
benchmark/
├── run_keyword_benchmark.py       # 主评测脚本
├── generate_attention_case_study.py  # 注意力可视化
├── download_hf_assets.py          # 模型下载
├── keyword_bench/
│   ├── data.py                    # 数据加载
│   ├── methods.py                 # 方法实现
│   └── metrics.py                 # 评测指标
├── data/                          # 数据集目录
│   ├── CSL/                       # 中文学术关键词
│   ├── SemEval2010/               # 英文学术
│   ├── Krapivin2009/              # 英文长文
│   ├── cake/                      # 中文医学摘要
│   └── shencecup/                 # 中文新闻
├── models/                        # 预训练模型
└── transformer_generalization/    # Decoder-only 扩展实验
    └── scripts/
        └── run_decoder_attention_benchmark.py
```

## 第 1 步：下载模型

使用 Hugging Face 镜像站加速下载：

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'

# 中文模型
python download_hf_assets.py --model "thenlper/gte-small-zh" --root-dir "." --timeout 600 --workers 6
python download_hf_assets.py --model "BAAI/bge-small-zh-v1.5" --root-dir "." --timeout 600 --workers 6
python download_hf_assets.py --model "thenlper/gte-base-zh" --root-dir "." --timeout 600 --workers 8
python download_hf_assets.py --model "moka-ai/m3e-base" --root-dir "." --timeout 600 --workers 4
python download_hf_assets.py --model "moka-ai/m3e-small" --root-dir "." --timeout 600 --workers 1

# 英文模型
python download_hf_assets.py --model "sentence-transformers/all-MiniLM-L6-v2" --root-dir "." --timeout 900 --workers 4
python download_hf_assets.py --model "distilbert-base-uncased" --root-dir "." --timeout 900 --workers 4
```

## 第 2 步：运行评测

### 中文主线评测

```powershell
# CSL 学术摘要 + ShenCeCup 新闻（含 IDF 混合方法）
python run_keyword_benchmark.py `
  --root-dir "." `
  --output-dir "outputs_round10_hybrid_mean" `
  --datasets csl_test shencecup_labeled `
  --models thenlper/gte-small-zh `
  --shencecup-limit 100 `
  --skip-yake `
  --device cuda
```

### 英文长文评测

```powershell
# SemEval2010 全文
python run_keyword_benchmark.py `
  --root-dir "." `
  --output-dir "outputs_round12_semeval_fulltext_minilm" `
  --datasets semeval2010_fulltext `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --skip-yake `
  --device cuda

# Krapivin2009 全文
python run_keyword_benchmark.py `
  --root-dir "." `
  --output-dir "outputs_round12_krapivin_fulltext_minilm" `
  --datasets krapivin2009_fulltext `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --skip-yake `
  --device cuda
```

### 英文短文跨语言评测

```powershell
python run_keyword_benchmark.py `
  --root-dir "." `
  --output-dir "outputs_round6_en_crosslingual" `
  --datasets semeval2010_test pubmed_test lis2000_test `
  --models sentence-transformers/all-MiniLM-L6-v2 distilbert-base-uncased `
  --english-limit 60 `
  --skip-yake `
  --device cuda
```

### 混合基线对照实验

```powershell
python run_keyword_benchmark.py `
  --root-dir "." `
  --output-dir "outputs_round13_baseline_hybrid_compare" `
  --datasets semeval2010_fulltext `
  --models sentence-transformers/all-MiniLM-L6-v2 `
  --skip-yake `
  --device cuda
```

### 注意力可视化（热力图）

```powershell
python generate_attention_case_study.py `
  --root-dir "." `
  --dataset "shencecup_labeled" `
  --model "thenlper/gte-small-zh" `
  --attention-layer-spec "mean_last3" `
  --doc-limit 3 `
  --output-dir "outputs_case_study" `
  --device cuda
```

## 第 3 步：查看结果

评测结果以 JSON 格式存储在各 `output-dir` 下的 `keyword_benchmark_results.json` 中。

### 主要结果文件

| 实验 | 结果路径 |
|------|----------|
| 中文主线 + Hybrid | `outputs_round10_hybrid_mean/keyword_benchmark_results.json` |
| SemEval2010-fulltext | `outputs_round12_semeval_fulltext_minilm/keyword_benchmark_results.json` |
| Krapivin2009-fulltext | `outputs_round12_krapivin_fulltext_minilm/keyword_benchmark_results.json` |
| 英文短文 | `outputs_round6_en_crosslingual/keyword_benchmark_results.json` |(archived)
| 混合基线对照 | `outputs_round13_baseline_hybrid_compare/keyword_benchmark_results.json` |
| Decoder-only (0.8B) | `transformer_generalization/results/qwen_decoder_benchmark_20.json` |
| Decoder-only (2B) | `transformer_generalization/results/qwen2b_decoder_benchmark_20.json` |

## 注意事项

- `m3e-small` 下载时建议使用 `--workers 1` 以提高稳定性
- 所有评测命令需在 `benchmark/` 目录下执行
- 需要 CUDA GPU 以获得合理的运行速度
