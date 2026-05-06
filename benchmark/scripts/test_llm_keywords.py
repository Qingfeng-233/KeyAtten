#!/usr/bin/env python3
"""
测试多个 LLM API 的关键词抽取效果
使用方法:
  python test_llm_keywords.py
  # 然后在终端交互输入 API Key

模型列表 (yunwu.ai):
  - gemini-3-flash-preview
  - gpt-5.4-nano-2026-03-17
  - qwen3.6-plus
  - MiniMax-M2.7
  - claude-sonnet-4-6
"""

import os
import json
import time
import requests
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ========== 配置区 ==========

# ShenCeCup 原始数据路径 (测试沙箱)
SHENCE_DATA_ROOT = Path("D:/工作区/项目/Keyatten/测试沙箱/data/shencecup/raw")
ALL_DOCS_FILE = SHENCE_DATA_ROOT / "all_docs.txt"
LABELS_FILE = SHENCE_DATA_ROOT / "train_docs_keywords.txt"

# 输出路径 (相对当前工作目录)
OUTPUT_DIR = Path("outputs/llm_keyword_comparison")


# 要测试的模型 (官方名称)
MODELS = {
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "gpt-5.4-nano-2026-03-17": "gpt-5.4-nano-2026-03-17",
    "qwen3.6-plus": "qwen3.6-plus",
    "MiniMax-M2.7": "MiniMax-M2.7",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}

# API 配置
API_BASE = "https://yunwu.ai/v1"
REQUEST_TIMEOUT = 60
CONCURRENT_REQUESTS = 5  # 并发数

# Prompt 模板 (抽取式约束)
PROMPT_TEMPLATE = """请从以下新闻正文中提取5-10个关键词。

要求：
1. 关键词必须在原文中出现
2. 优先选择名词短语
3. 按重要性排序

正文：
{content}

请只输出关键词列表，每行一个，不要有编号、解释或其他内容。"""

SYSTEM_PROMPT = "你是一个关键词提取助手。只输出关键词列表，每行一个，不要有编号。"

# ========== 核心代码 ==========


def get_api_key() -> str:
    """交互式获取 API Key (不在代码中硬编码)"""
    key = os.getenv("YUNWU_API_KEY")
    if key:
        print(f"从环境变量读取 API Key: {key[:10]}...")
        return key
    
    print("\n请输入 yunwu.ai API Key (输入不会显示):")
    try:
        import getpass
        key = getpass.getpass("API Key: ").strip()
    except:
        key = input("API Key: ").strip()
    
    if not key.startswith("sk-"):
        print("警告: Key 格式可能不正确，通常以 sk- 开头")
    
    return key


def call_llm_api(content: str, model: str, api_key: str) -> Tuple[Optional[List[str]], float, Optional[str]]:
    """
    调用 LLM API 提取关键词
    
    Returns:
        (keywords_list, latency_seconds, error_message)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT_TEMPLATE.format(content=content[:2000])}  # 截断防超限
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }
    
    try:
        start = time.time()
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        latency = time.time() - start
        
        if resp.status_code != 200:
            error_text = resp.text[:200]
            return None, latency, f"HTTP {resp.status_code}: {error_text}"
        
        result = resp.json()
        
        if "choices" not in result or not result["choices"]:
            return None, latency, "Invalid response format"
        
        keywords_text = result["choices"][0]["message"]["content"]
        
        # 解析关键词列表
        keywords = []
        for line in keywords_text.strip().split("\n"):
            line = line.strip()
            # 去掉编号前缀 (1. 、- 、* 等)
            line = line.lstrip("0123456789.-*• ")
            if line and len(line) > 1:
                keywords.append(line)
        
        return keywords[:10], latency, None
        
    except requests.exceptions.Timeout:
        return None, REQUEST_TIMEOUT, "Timeout"
    except Exception as e:
        return None, 0, str(e)


# 引入标准评测函数
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from keyword_bench.metrics import normalize_phrase, _prf_at_k


def evaluate_single_doc(predictions: List[str], ground_truth: List[str]) -> Dict:
    """计算单个文档的 F1@5 和 F1@10 (使用标准 normalize_phrase)"""
    # 使用标准评测逻辑
    p5, r5, f1_5 = _prf_at_k(predictions, ground_truth, k=5)
    p10, r10, f1_10 = _prf_at_k(predictions, ground_truth, k=10)
    
    # 计算匹配数用于显示 (k=10)
    pred_normalized = {normalize_phrase(p) for p in predictions[:10] if normalize_phrase(p)}
    gt_normalized = {normalize_phrase(g) for g in ground_truth if normalize_phrase(g)}
    matched = len(pred_normalized & gt_normalized)
    
    return {
        "f1@5": round(f1_5, 4),
        "f1@10": round(f1_10, 4),
        "precision@5": round(p5, 4),
        "precision@10": round(p10, 4),
        "recall@5": round(r5, 4),
        "recall@10": round(r10, 4),
        "matched": matched,
        "predicted_count": len(predictions),
        "ground_truth_count": len(ground_truth),
    }


def load_shencecup_data(max_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    从 ShenCeCup 原始数据加载
    格式:
      - all_docs.txt: doc_id\x01title\x01body
      - train_docs_keywords.txt: doc_id\tkeyword1,keyword2,...
    """
    docs = []
    
    # 检查文件存在
    if not ALL_DOCS_FILE.exists():
        print(f"错误: 正文文件不存在 {ALL_DOCS_FILE}")
        return []
    if not LABELS_FILE.exists():
        print(f"错误: 标签文件不存在 {LABELS_FILE}")
        return []
    
    # 加载关键词标注
    labels = {}
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            doc_id, raw_keywords = parts
            labels[doc_id] = [k.strip() for k in raw_keywords.split(',') if k.strip()]
    
    print(f"加载标注: {len(labels)} 篇文档有关键词")
    
    # 加载正文
    all_docs = []
    with open(ALL_DOCS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\x01')
            if len(parts) >= 3:
                doc_id = parts[0]
                title = parts[1]
                body = parts[2]
                # 只保留有关键词标注的文档
                if doc_id in labels:
                    all_docs.append({
                        "id": doc_id,
                        "content": f"{title}\n{body}",
                        "keywords": labels[doc_id]
                    })
    
    print(f"加载正文: {len(all_docs)} 篇文档匹配标注")
    
    # 随机采样
    if len(all_docs) > max_samples:
        random.seed(seed)
        docs = random.sample(all_docs, max_samples)
    else:
        docs = all_docs
    
    return docs


def test_single_model(model_name: str, model_id: str, docs: List[Dict], api_key: str) -> Dict:
    """测试单个模型 (并发版本)"""
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"{'='*50}")
    print(f"并发数: {CONCURRENT_REQUESTS}")
    
    results = []
    errors = 0
    print_lock = Lock()
    completed = [0]  # 使用列表包装以便在闭包中修改
    
    def process_single_doc(args):
        """处理单个文档"""
        i, doc = args
        doc_id = doc.get("id", f"doc_{i}")
        content = doc.get("content", doc.get("text", ""))
        ground_truth = doc.get("keywords", doc.get("keyphrases", []))
        
        if not content or not ground_truth:
            return None
        
        keywords, latency, error = call_llm_api(content, model_id, api_key)
        
        with print_lock:
            completed[0] += 1
            progress = f"[{completed[0]}/{len(docs)}]"
            
            if error:
                print(f"  {progress} {doc_id} ... 失败: {error}")
                return {"error": True, "doc_id": doc_id}
            
            eval_result = evaluate_single_doc(keywords, ground_truth)
            eval_result["doc_id"] = doc_id
            eval_result["latency_ms"] = round(latency * 1000, 1)
            eval_result["predicted_keywords"] = keywords
            eval_result["ground_truth"] = ground_truth
            
            print(f"  {progress} {doc_id} ... F1={eval_result['f1@10']:.3f} ({eval_result['matched']}/{eval_result['ground_truth_count']} match)")
            return eval_result
    
    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = {executor.submit(process_single_doc, (i, doc)): i for i, doc in enumerate(docs)}
        
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            if result.get("error"):
                errors += 1
            else:
                results.append(result)
    
    # 汇总统计
    if results:
        avg_f1 = sum(r["f1@10"] for r in results) / len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        avg_precision = sum(r["precision@10"] for r in results) / len(results)
        avg_recall = sum(r["recall@10"] for r in results) / len(results)
    else:
        avg_f1 = avg_latency = avg_precision = avg_recall = 0.0
    
    summary = {
        "model_name": model_name,
        "model_id": model_id,
        "total_samples": len(docs),
        "successful": len(results),
        "errors": errors,
        "avg_f1@10": round(avg_f1, 4),
        "avg_precision@10": round(avg_precision, 4),
        "avg_recall@10": round(avg_recall, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "details": results
    }
    
    print(f"\n{model_name} 汇总:")
    print(f"  成功率: {len(results)}/{len(docs)} ({len(results)/len(docs)*100:.1f}%)")
    print(f"  F1@10: {avg_f1:.4f}")
    print(f"  Precision@10: {avg_precision:.4f}")
    print(f"  Recall@10: {avg_recall:.4f}")
    print(f"  平均延迟: {avg_latency:.0f}ms")
    
    return summary


def process_doc_for_model(args) -> Dict:
    """处理单个文档的通用函数（多模型并行使用）"""
    doc, model_name, model_id, api_key = args
    doc_id = doc.get("id", "unknown")
    content = doc.get("content", "")
    ground_truth = doc.get("keywords", [])
    
    if not content or not ground_truth:
        return {"error": True, "doc_id": doc_id, "model": model_name}
    
    keywords, latency, error = call_llm_api(content, model_id, api_key)
    
    if error:
        return {"error": True, "doc_id": doc_id, "model": model_name, "error_msg": error}
    
    eval_result = evaluate_single_doc(keywords, ground_truth)
    eval_result["doc_id"] = doc_id
    eval_result["model"] = model_name
    eval_result["latency_ms"] = round(latency * 1000, 1)
    eval_result["predicted_keywords"] = keywords
    eval_result["ground_truth"] = ground_truth
    
    return eval_result


def main():
    print("="*70)
    print("LLM 关键词抽取对比测试 (多模型并行)")
    print("="*70)
    print(f"配置: {len(MODELS)} 个模型 × {CONCURRENT_REQUESTS} 并发 = {len(MODELS) * CONCURRENT_REQUESTS} 总并发")
    
    # 获取 API Key
    api_key = get_api_key()
    if not api_key:
        print("错误: 未提供 API Key")
    
    # 加载测试数据 (ShenCeCup)
    print(f"\n从 {ALL_DOCS_FILE} 加载数据")
    docs = load_shencecup_data(max_samples=30)
    
    if not docs:
        print("没有加载到测试数据，退出")
        return
    
    print(f"加载成功: {len(docs)} 篇文档")
    
    # 确认测试
    sample = docs[0]
    print(f"\n样例数据:")
    print(f"  ID: {sample.get('id', 'N/A')}")
    print(f"  内容长度: {len(sample.get('content', ''))} 字符")
    print(f"  关键词: {sample.get('keywords', [])}")
    
    total_calls = len(docs) * len(MODELS)
    print(f"\n预计 API 调用次数: {len(docs)} 篇 × {len(MODELS)} 模型 = {total_calls} 次")
    
    confirm = input(f"确认开始测试? (y/N): ")
    if confirm.lower() != 'y':
        print("取消测试")
        return
    
    # 准备所有任务
    print(f"\n{'='*70}")
    print("开始多模型并行测试")
    print(f"{'='*70}")
    
    # 为每个模型准备任务列表
    model_tasks = {}
    for model_name, model_id in MODELS.items():
        model_tasks[model_name] = {
            "model_id": model_id,
            "tasks": [(doc, model_name, model_id, api_key) for doc in docs],
            "completed": 0,
            "errors": 0,
            "results": []
        }
    
    # 使用大线程池同时处理所有模型的所有请求
    total_workers = len(MODELS) * CONCURRENT_REQUESTS  # 25 并发
    print(f"线程池大小: {total_workers}")
    
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        # 提交所有任务
        future_to_info = {}
        for model_name, info in model_tasks.items():
            print(f"  提交 {model_name}: {len(info['tasks'])} 个任务")
            for task in info['tasks']:
                future = executor.submit(process_doc_for_model, task)
                future_to_info[future] = model_name
        
        print(f"\n总任务数: {len(future_to_info)}，开始执行...\n")
        
        # 收集结果（按完成顺序）
        model_counters = {name: 0 for name in MODELS}
        
        for future in as_completed(future_to_info):
            model_name = future_to_info[future]
            try:
                result = future.result()
                model_counters[model_name] += 1
                counter = model_counters[model_name]
                total_counter = sum(model_counters.values())
                
                if result.get("error"):
                    model_tasks[model_name]["errors"] += 1
                    print(f"[{total_counter}/{total_calls}] [{model_name}] {result.get('doc_id')} ... 失败: {result.get('error_msg', 'unknown')}")
                else:
                    model_tasks[model_name]["results"].append(result)
                    print(f"[{total_counter}/{total_calls}] [{model_name}] {result['doc_id']} ... F1={result['f1@10']:.3f} ({result['matched']}/{result['ground_truth_count']} match)")
                    
            except Exception as e:
                model_tasks[model_name]["errors"] += 1
                print(f"[{total_counter}/{total_calls}] [{model_name}] 异常: {e}")
    
    print(f"\n{'='*70}")
    print("测试完成，汇总结果")
    print(f"{'='*70}")
    
    # 汇总每个模型的结果
    all_results = {}
    
    for model_name, info in model_tasks.items():
        results = info["results"]
        errors = info["errors"]
        total = len(docs)
        
        if results:
            avg_f1_5 = sum(r["f1@5"] for r in results) / len(results)
            avg_f1_10 = sum(r["f1@10"] for r in results) / len(results)
            avg_p_10 = sum(r["precision@10"] for r in results) / len(results)
            avg_r_10 = sum(r["recall@10"] for r in results) / len(results)
            avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        else:
            avg_f1_5 = avg_f1_10 = avg_p_10 = avg_r_10 = avg_latency = 0.0
        
        all_results[model_name] = {
            "model_name": model_name,
            "model_id": info["model_id"],
            "total_samples": total,
            "successful": len(results),
            "errors": errors,
            "avg_f1@5": round(avg_f1_5, 4),
            "avg_f1@10": round(avg_f1_10, 4),
            "avg_precision@10": round(avg_p_10, 4),
            "avg_recall@10": round(avg_r_10, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "details": results
        }
        
        print(f"\n{model_name}:")
        print(f"  成功率: {len(results)}/{total} ({len(results)/total*100:.1f}%)")
        print(f"  F1@5:  {avg_f1_5:.4f}")
        print(f"  F1@10: {avg_f1_10:.4f}")
        print(f"  P@10:  {avg_p_10:.4f}")
        print(f"  R@10:  {avg_r_10:.4f}")
    
    # 保存结果
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/final_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印对比表
    print(f"\n{'='*70}")
    print("模型对比汇总表")
    print(f"{'='*70}")
    print(f"{'模型':<30} {'F1@10':>8} {'P@10':>8} {'R@10':>8} {'延迟(ms)':>10} {'成功率':>8}")
    print("-"*70)
    
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]['avg_f1@10'], reverse=True)
    
    for name, data in sorted_models:
        success_rate = data['successful'] / data['total_samples'] * 100
        print(f"{name:<30} {data['avg_f1@10']:>8.4f} {data['avg_precision@10']:>8.4f} {data['avg_recall@10']:>8.4f} {data['avg_latency_ms']:>10.0f} {success_rate:>7.1f}%")
    
    print(f"\n结果已保存: {output_file}")
    print(f"\n最佳模型: {sorted_models[0][0]} (F1@10 = {sorted_models[0][1]['avg_f1@10']:.4f})")


if __name__ == "__main__":
    main()
