# Train Workspace

`train/` is the local command package for training, labeling, and experiment evaluation.

## Entry Point

Use one command entry point:

```bash
python -m train.main <command> [command args...]
```

Available commands:

| Command | Module | Purpose |
|---|---|---|
| `eval-news55` | `train.eval.news55` | Evaluate the clean `data/news_annotated.jsonl` set. |
| `eval-bio` | `train.eval.bio` | Run the legacy BIO-only benchmark evaluation. |
| `eval-fusion` | `train.eval.fusion` | Evaluate attention LoRA + BIO fusion. |
| `train-bio` | `train.jobs.bio_boundary` | Train the BIO boundary model. |
| `train-attn-lora` | `train.jobs.attn_lora` | Train attention LoRA from supervised labels. |
| `train-attn-lora-llm` | `train.jobs.attn_lora_llm` | Train attention LoRA from LLM labels. |
| `train-qk-lora` | `train.jobs.qk_lora` | Run the legacy QK LoRA experiment. |
| `label-llm` | `train.jobs.run_llm_labeling` | Generate LLM keyword labels. |

Examples:

```bash
python -m train.main eval-news55 --methods bio --device cpu
python -m train.main eval-news55 --methods bio attn --device cuda
python -m train.main train-attn-lora --smoke --device cuda
```

## Layout

```text
train/
  main.py                 # command dispatcher
  eval/
    bio.py                # BIO-only evaluation
    fusion.py             # attention LoRA + BIO fusion evaluation
    news55.py             # clean news55 evaluation
  jobs/
    bio_boundary.py       # BIO model training
    attn_lora.py          # attention LoRA training
    attn_lora_llm.py      # LLM-label attention LoRA training
    qk_lora.py            # legacy QK LoRA training
    llm_teacher.py        # LLM teacher helpers
    run_llm_labeling.py   # LLM labeling runner
  bio_ckpt_ep13_production/
```

## Rules

- Keep new runnable entry points behind `train/main.py`.
- Put evaluation code under `train/eval/`.
- Put training or labeling jobs under `train/jobs/`.
- Do not add one-off diagnostics back into the root of `train/`.
- Model files and adapters belong under `models/` unless they are intentionally curated production checkpoints.
