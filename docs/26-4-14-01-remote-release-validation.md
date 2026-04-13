# 远端发布验证启动记录

## 思路

- 目标：在临时云端 `RTX 3090` 环境上跑一轮可作为发布证据的正式验证。
- 范围：
  - `gte-small-zh` 的 attention 系列与常规基线全量 benchmark
  - `QK LoRA` 两个模型的正式评估
- 原则：
  - 使用 `screen` 常驻，避免 SSH 断开导致任务中断
  - 远端优先修复环境，再下载基座模型，再顺序执行 benchmark 与 QK 评估
  - 模型下载优先 `ModelScope`，失败后回退 `hf-mirror`

## 结果

- 已连接远端服务器：
  - 主机：`root@link.lanyun.net:41831`
  - GPU：`NVIDIA GeForce RTX 3090`
- 已同步内容到远端：
  - 主项目代码
  - 评测所需精简数据集
  - `QK LoRA` 适配器目录
- 已修复远端运行环境：
  - 通过 `.venv --system-site-packages` 复用现有 CUDA / torch 环境
  - 修复了损坏的 `scikit-learn`
  - 修复了 `peft` 导入链
  - 清除了与当前 `torch` 冲突的 `torchvision` / `torchaudio`
- 已启动远端常驻任务：
  - `screen` 会话：`keyatten_release_full`
  - 运行脚本：`run_release_full.sh`
  - 当前输出目录前缀：`outputs/release_evidence_20260414_005947`

## 结论

- 远端发布验证任务已经成功启动，不是停留在准备阶段。
- 当前首个阶段已进入大模型下载，说明：
  - SSH 会话正常
  - `screen` 正常
  - Python 环境可运行
  - 下载链路可用
- 待任务完成后，可直接从远端 `outputs/release_evidence_*` 汇总 benchmark 与 QK 评估结果，作为发布前验证证据。
