# Embedding 同口径分数表

> 对比 `thenlper/gte-small-zh` 与 `Qwen/Qwen3-Embedding-0.6B` 在同一数据集、同一方法、同一评测口径下的表现。

## 当前结论

- 本页结果基于主仓库 benchmark 代码、`20 docs` 同口径抽样。
- `Qwen3-Embedding-0.6B` 在 `KeyBERT` 这类 embedding 相似度路线里有竞争力。
- `gte-small-zh` 在项目主线的 attention 方法上明显更稳、更强，适合作为默认发布模型。

## 评测口径

- 数据根目录：`D:/工作区/项目/Keyatten/测试沙箱`
- 数据集：`csl_test`、`shencecup_labeled`
- baseline 模型：`thenlper/gte-small-zh`
- candidate 模型：`Qwen/Qwen3-Embedding-0.6B`
- 方法：
  - `keybert`
  - `keybert_idf`
  - `cls_attn`
  - `received_attn`
  - `samrank`
  - `fusion_attn`
  - `cls_attn_idf`
  - `received_attn_idf`
  - `samrank_idf`
  - `fusion_attn_idf`
- 主指标：`F1@10`
- 辅助指标：`R@10`

## 运行命令

```powershell
python benchmark/run_keyword_benchmark.py `
  --root-dir "D:/工作区/项目/Keyatten/测试沙箱" `
  --output-dir "D:/工作区/项目/Keyatten/benchmark/outputs_embedding_compare_gte_vs_qwen3_0_6b_20" `
  --datasets csl_test shencecup_labeled `
  --models "D:/工作区/项目/Keyatten/测试沙箱/models/thenlper__gte-small-zh" "D:/工作区/项目/Keyatten/测试沙箱/models/Qwen__Qwen3-Embedding-0.6B" `
  --test-limit 20 `
  --shencecup-limit 20 `
  --skip-yake `
  --device cuda `
  --attention-batch-size 4 `
  --embedding-batch-size 16
```

## csl_test

| 方法 | gte-small-zh F1@10 | gte-small-zh R@10 | Qwen3-Embedding-0.6B F1@10 | Qwen3-Embedding-0.6B R@10 | ΔF1@10 | ΔR@10 |
|------|-------------------:|------------------:|---------------------------:|--------------------------:|-------:|------:|
| `keybert` | 0.1296 | 0.1739 | 0.0960 | 0.1287 | -0.0336 | -0.0452 |
| `keybert_idf` | 0.1819 | 0.2700 | 0.1795 | 0.2717 | -0.0024 | +0.0017 |
| `cls_attn` | 0.1554 | 0.2215 | 0.0967 | 0.1284 | -0.0587 | -0.0930 |
| `received_attn` | 0.1597 | 0.2292 | 0.1271 | 0.1778 | -0.0326 | -0.0514 |
| `samrank` | 0.1773 | 0.2567 | 0.0750 | 0.1009 | -0.1023 | -0.1558 |
| `fusion_attn` | 0.1505 | 0.2150 | 0.0967 | 0.1284 | -0.0538 | -0.0866 |
| `cls_attn_idf` | 0.2063 | 0.3100 | 0.0967 | 0.1284 | -0.1096 | -0.1816 |
| `received_attn_idf` | 0.1929 | 0.2892 | 0.1770 | 0.2694 | -0.0159 | -0.0198 |
| `samrank_idf` | 0.2106 | 0.3167 | 0.1430 | 0.2115 | -0.0676 | -0.1052 |
| `fusion_attn_idf` | 0.1787 | 0.2642 | 0.0967 | 0.1284 | -0.0819 | -0.1358 |

## shencecup_labeled

| 方法 | gte-small-zh F1@10 | gte-small-zh R@10 | Qwen3-Embedding-0.6B F1@10 | Qwen3-Embedding-0.6B R@10 | ΔF1@10 | ΔR@10 |
|------|-------------------:|------------------:|---------------------------:|--------------------------:|-------:|------:|
| `keybert` | 0.0600 | 0.1250 | 0.1522 | 0.3367 | +0.0922 | +0.2117 |
| `keybert_idf` | 0.1461 | 0.3333 | 0.1543 | 0.3542 | +0.0082 | +0.0208 |
| `cls_attn` | 0.2269 | 0.4892 | 0.0386 | 0.0875 | -0.1884 | -0.4017 |
| `received_attn` | 0.2424 | 0.5242 | 0.0930 | 0.2125 | -0.1493 | -0.3117 |
| `samrank` | 0.2495 | 0.5367 | 0.0392 | 0.0958 | -0.2103 | -0.4408 |
| `fusion_attn` | 0.2264 | 0.4867 | 0.0386 | 0.0875 | -0.1879 | -0.3992 |
| `cls_attn_idf` | 0.2357 | 0.5142 | 0.0386 | 0.0875 | -0.1971 | -0.4267 |
| `received_attn_idf` | 0.2121 | 0.4617 | 0.1757 | 0.3892 | -0.0364 | -0.0725 |
| `samrank_idf` | 0.2193 | 0.4742 | 0.1395 | 0.3250 | -0.0798 | -0.1492 |
| `fusion_attn_idf` | 0.2351 | 0.5100 | 0.0386 | 0.0875 | -0.1966 | -0.4225 |

## 解读

- `CSL` 学术摘要里，`gte-small-zh` 几乎全面领先，尤其是 attention 主线。
- `ShenCeCup` 中文新闻里，`Qwen3-Embedding-0.6B` 只在 `keybert` 和 `keybert_idf` 上占优。
- 一旦切回项目主线方法，特别是 `received_attn`、`samrank`、`*_idf`，`gte-small-zh` 依然整体更强。
- 这说明 `Qwen3-Embedding-0.6B` 更像强 embedding 相似度模型，不适合作为当前默认 attention 发布模型。

## 自动渲染

跑完新的 benchmark 结果后，可用下面命令重新生成对照表：

```powershell
python benchmark/render_embedding_comparison.py `
  --results "benchmark/outputs_embedding_compare_gte_vs_qwen3_0_6b_20/keyword_benchmark_results.json" `
  --baseline-model "D:/工作区/项目/Keyatten/测试沙箱/models/thenlper__gte-small-zh" `
  --candidate-model "D:/工作区/项目/Keyatten/测试沙箱/models/Qwen__Qwen3-Embedding-0.6B" `
  --datasets csl_test shencecup_labeled `
  --output "benchmark/outputs_embedding_compare_gte_vs_qwen3_0_6b_20/embedding_comparison.md"
```
