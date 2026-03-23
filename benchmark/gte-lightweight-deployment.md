# gte-small-zh 轻量部署路线

## 目标

把 KeyAtten 的默认发布路线收敛到：

- 模型：`thenlper/gte-small-zh`
- 方法：`received_attn`、`samrank`、以及 `_idf` 变体
- 部署方向：`ONNX Runtime` 轻量推理

## 为什么选 gte-small-zh

1. 中文主线效果稳定，是当前仓库默认推荐模型。
2. 参数量约 `33M`，远小于大模型 embedding 路线。
3. 已验证可导出 token attention，并在 `ONNX Runtime` 中复现 `received_attn` 词分数。
4. 更适合做轻量算子、边缘部署、以及低资源服务。

## 已验证结果

在 `gte-small-zh` 的 ONNX attention probe 中：

- ONNX 文件大小：约 `109.6 MB`
- 导出时间：约 `0.80s`
- ORT 单次推理：约 `0.008s`
- `attention_max_abs_diff = 7.15e-07`
- `received_max_abs_diff = 1.13e-06`

这说明 `attention -> ONNX Runtime -> received_attn` 这条链路在数值上已经足够稳定，可以作为轻量部署基础。

## 仓库内工具

- 轻量验证脚本：`benchmark/gte_onnx_probe.py`
- Benchmark 主线：`benchmark/run_keyword_benchmark.py`
- 方法实现：`benchmark/keyword_bench/methods.py`

推荐安装命令：

```bash
pip install "keyatten[inference,zh,lightweight]"
```

## 建议发布口径

- 默认中文模型：`gte-small-zh`
- 默认方法：`received_attn / samrank / *_idf`
- 默认工程叙事：
  - 小模型
  - 可解释 Attention
  - 单次前向
  - 可落到 ONNX Runtime

## 暂不纳入默认发布的路线

- `Qwen3-Embedding-0.6B`
- decoder-only 长上下文模型
- 其他更大 embedding 模型

这些路线保留在 benchmark 和实验中即可，不作为默认发布模型。
