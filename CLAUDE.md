# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

KeyAtten 是基于 Transformer Attention 机制的关键词提取框架，支持中英双语，零训练零标注，单次前向推理完成。提供 4 种纯 Attention 方法和 4 种 Attention-IDF 混合方法。

## 构建与安装

```bash
pip install .
```

项目无测试套件、无 lint 配置、无 Makefile。当前没有 `test`、`lint`、`format` 命令。

## 架构

### 数据流

```
文本 → segment_text() → build_candidates() → attention_word_scores() → [IDF混合] → rank → 关键词
```

### 模块职责与依赖

```
extractor.py  ← 唯一公共入口，编排整个流程
  ├── attention.py   ← Transformer 前向推理 + 4种Attention分数计算 + 子词→词聚合
  ├── candidates.py  ← 分词(jieba/regex) + n-gram候选词生成 + 候选词评分排序
  └── hybrid.py      ← IDF计算 + Attention×TF-IDF 分数融合

candidates.py → utils.py (短语规范化、词干化)
hybrid.py     → candidates.py (复用词有效性检查函数)
```

`extractor.py` 是唯一对外模块，其余模块不应被外部直接导入。

### 关键类型

- `KeyAttenExtractor`: 主类，持有模型和配置，提供 `extract_keywords`、`extract_keywords_batch`、`extract_word_weights`、`fit_idf`
- `WordWeight(word, index, weight, pos_tag)`: 词级权重数据类，`@dataclass(slots=True)`
- `Candidate(text, word_start, word_end)`: 候选词数据类，记录在词序列中的起止位置

### 8 种提取方法

纯 Attention: `cls_attn`、`received_attn`、`samrank`、`fusion_attn`
混合变体: 上述各加 `_idf` 后缀，将 Attention 分数与 TF-IDF 相乘

### 多层融合

`KeyAttenExtractor` 支持 `layer_indices` + `layer_weights` 参数，对多层 Attention 分数做加权平均（`np.average`）。单层时用 `layer_index`（默认 -1，最后一层）。

## 代码约定

- 类型注解使用 PEP 604 语法（`str | None`，非 `Optional[str]`）
- 数据类使用 `@dataclass(slots=True)`
- 每个模块定义 `__all__` 控制导出
- 向量化优先：用 NumPy 数组操作代替 Python 循环
- Python >= 3.10

## 语言处理差异

- 中文：jieba.posseg 分词，有效词性前缀 `n, eng, v`，最大 4-gram
- 英文：正则表达式分词，过滤停用词和单字符，Porter 词干化用于规范化

## 声明

本项目核心代码由 AI 辅助生成，由 蒋林浩 完成架构设计、逻辑调试与优化