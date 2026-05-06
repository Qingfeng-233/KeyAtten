# Benchmark Workspace

`benchmark/` contains reusable benchmark modules plus local evaluation and utility commands.

## Entry Point

Use one command entry point:

```bash
python -m benchmark.main <command> [command args...]
```

Available commands:

| Command | Module | Purpose |
|---|---|---|
| `keyword` | `benchmark.eval.run_keyword_benchmark` | Run the main keyword benchmark. |
| `hidden-head` | `benchmark.eval.run_hidden_head_benchmark` | Evaluate a hidden-state keyword head checkpoint. |
| `llm-keyword` | `benchmark.eval.llm_keyword_benchmark` | Benchmark an OpenAI-compatible LLM keyword extractor. |
| `bio-local-matrix` | `benchmark.scripts.eval_bio_local_matrix` | Run the local BIO/QK matrix evaluation. |
| `bio-qk-combo` | `benchmark.scripts.eval_bio_qk_combo` | Run BIO-only, BIO+QK, and fused BIO/QK evaluation. |
| `shence-heldout` | `benchmark.scripts.run_shence_heldout_eval` | Run ShenCe strict held-out evaluation. |
| `gemini-heldout` | `benchmark.scripts.gemini_heldout_100` | Run Gemini held-out helper. |
| `test-llm-keywords` | `benchmark.scripts.test_llm_keywords` | Run the LLM keyword smoke script. |
| `download-hf-assets` | `benchmark.tools.download_hf_assets` | Download Hugging Face model assets. |
| `gte-onnx-probe` | `benchmark.tools.gte_onnx_probe` | Validate `gte-small-zh` ONNX attention export. |
| `remote-hidden-head` | `benchmark.tools.remote_hidden_head_runner` | Run the remote hidden-head helper. |
| `render-embedding-comparison` | `benchmark.tools.render_embedding_comparison` | Render embedding comparison reports. |

Examples:

```bash
python -m benchmark.main keyword --root-dir "." --output-dir "outputs_smoke" --datasets csl_test --models thenlper/gte-small-zh --skip-yake --device cpu
python -m benchmark.main hidden-head --checkpoint "/path/to/best_hidden_head.pt" --datasets csl_dev csl_test --device cuda
python -m benchmark.main gte-onnx-probe --model-path "/path/to/gte-small-zh" --output-path "outputs/attention.onnx" --words 自然语言处理 关键词 提取
```

## Layout

```text
benchmark/
  main.py                 # command dispatcher
  keyword_bench/          # reusable data, method, metric, and output helpers
  eval/                   # primary benchmark evaluation jobs
  scripts/                # experiment-specific benchmark jobs
  tools/                  # download, rendering, ONNX, and remote helper tools
```

## Rules

- Keep reusable benchmark logic under `keyword_bench/`.
- Put stable evaluation jobs under `eval/`.
- Put experiment-specific jobs under `scripts/`.
- Put operational helpers under `tools/`.
- New runnable files should be exposed through `benchmark/main.py`.
- Do not put generated outputs or model adapters under `benchmark/`.

## Outputs

- Most benchmark jobs write to `测试沙箱/Outputs/` or the requested `--output-dir`.
- Main result files are JSON; logs are written beside the output when supported.
- Primary metrics are Precision@K, Recall@K, and F1@K.
