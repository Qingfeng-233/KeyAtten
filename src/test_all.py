#!/usr/bin/env python3
"""Full test suite for KeyAtten - tests all methods and interfaces.

Models are checked automatically before running.
Download missing models with:
    python -m keyatten.download_models
    python -m keyatten.download_models --model gte-small-zh
    python -m keyatten.download_models --source huggingface

Usage:
    python src/test_all.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from keyatten import CandidateSegmentAttentionExtractor, KeyAttenExtractor, QKLoRAExtractor
from keyatten.download_models import check_all_models, download_model

TEST_TEXT = "注意力机制是Transformer模型的核心，它通过计算查询和键的相似度来分配权重。"
MODEL_PATH = "models/gte-small-zh"
ONNX_PATH = "models/gte_small_zh_onnx"
QWEN_MODEL = "models/Qwen3-Embedding-0.6B"
QK_ADAPTER = "models/qk_qwen0.6B/best_adapter"
SEGMENT_ADAPTER = "models/candidate_segment_attn/qwen06_v2_2k_len1024_c30/best_adapter"
BIO_CKPT = "models/bio_ckipbert_extractive_ep13/bio_model_full.pt"


def test_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_result(method: str, keywords: list[str], elapsed: float) -> bool:
    ok = len(keywords) > 0
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {method}: {keywords} ({elapsed:.2f}s)")
    return ok


def ensure_models() -> None:
    """Check and download required models before testing."""
    print("Checking model assets...")
    results = check_all_models()
    
    missing = [name for name, r in results.items() if not r["ok"]]
    essential = ["gte-small-zh", "gte-small-zh-onnx"]
    missing_essential = [m for m in missing if m in essential]
    
    if missing_essential:
        print(f"  Missing essential models: {missing_essential}")
        print("  Downloading from ModelScope (魔搭)...")
        for name in missing_essential:
            result = download_model(name)
            status = "OK" if result["ok"] else "FAILED"
            print(f"    [{status}] {name}: {result['message']}")
        # Re-check
        results = check_all_models()
        if any(not r["ok"] for r in results.values() if r.get("dir", "").endswith("gte-small-zh") or r.get("dir", "").endswith("gte_small_zh_onnx")):
            print("\n  ERROR: Failed to download required models.")
            print("  Try: python -m keyatten.download_models --model gte-small-zh --source huggingface")
            sys.exit(1)
    
    # Report QK status (non-fatal if missing)
    for qk in ["qk_qwen0.6B/best_adapter"]:
        if qk in results and not results[qk]["ok"]:
            print(f"  Note: {qk} not found (QK LoRA test will be skipped)")
    
    print("  Model check complete.\n")


def main() -> None:
    ensure_models()
    passed = 0
    failed = 0

    # ── 1. Attention methods ──
    test_section("1. Attention Methods (PyTorch)")
    extractor = KeyAttenExtractor(MODEL_PATH, device="cpu")

    for method in ["cls_attn", "received_attn", "samrank", "fusion_attn", "voted_attn", "excess_attn"]:
        t0 = time.time()
        try:
            kw = extractor.extract_keywords(TEST_TEXT, method=method, top_k=5)
            elapsed = time.time() - t0
            if test_result(method, kw, elapsed):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {method}: {e} ({elapsed:.2f}s)")
            failed += 1

    # ── 2. Hybrid IDF methods ──
    test_section("2. Hybrid IDF Methods")
    # Fit IDF on a small corpus
    corpus = [
        "注意力机制是Transformer的核心",
        "自然语言处理是人工智能的重要分支",
        "关键词提取技术广泛应用于文本分析",
    ]
    extractor.fit_idf(corpus)

    for method in ["cls_attn_idf", "received_attn_idf", "samrank_idf"]:
        t0 = time.time()
        try:
            kw = extractor.extract_keywords(TEST_TEXT, method=method, top_k=5)
            elapsed = time.time() - t0
            if test_result(method, kw, elapsed):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] {method}: {e} ({elapsed:.2f}s)")
            failed += 1

    # ── 3. ONNX backend ──
    test_section("3. ONNX Backend")
    try:
        onnx_extractor = KeyAttenExtractor(ONNX_PATH, device="cpu", backend="onnx")
        t0 = time.time()
        kw = onnx_extractor.extract_keywords(TEST_TEXT, method="samrank", top_k=5)
        elapsed = time.time() - t0
        if test_result("onnx_samrank", kw, elapsed):
            passed += 1
        else:
            failed += 1
    except Exception as e:
        print(f"  [FAIL] onnx_samrank: {e}")
        failed += 1

    # ── 4. Batch interface ──
    test_section("4. Batch Interface")
    texts = [
        "深度学习在计算机视觉领域取得巨大成功",
        "Transformer架构改变了自然语言处理方式",
    ]
    t0 = time.time()
    try:
        results = extractor.extract_keywords_batch(texts, method="samrank", top_k=5)
        elapsed = time.time() - t0
        ok = len(results) == 2 and all(len(r) > 0 for r in results)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] batch_samrank: {results} ({elapsed:.2f}s)")
        if ok:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] batch_samrank: {e} ({elapsed:.2f}s)")
        failed += 1

    # ── 5. Word weights ──
    test_section("5. Word Weights")
    t0 = time.time()
    try:
        weights = extractor.extract_word_weights(TEST_TEXT, method="samrank")
        elapsed = time.time() - t0
        ok = len(weights) > 0
        status = "OK" if ok else "FAIL"
        top_words = [f"{w.word}={w.weight:.3f}" for w in weights[:5]]
        print(f"  [{status}] word_weights: {top_words} ({elapsed:.2f}s)")
        if ok:
            passed += 1
        else:
            failed += 1
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] word_weights: {e} ({elapsed:.2f}s)")
        failed += 1

    # ── 6. QK LoRA (if available) ──
    test_section("6. QK LoRA Method")
    qwen_exists = Path(QWEN_MODEL).exists()
    adapter_exists = Path(QK_ADAPTER).exists()
    if not qwen_exists or not adapter_exists:
        print(f"  [SKIP] QK LoRA: Qwen model exists={qwen_exists}, adapter exists={adapter_exists}")
        print(f"        Run: python -m keyatten.download_models --model qwen3-embed-0.6b")
    else:
        try:
            qk_extractor = QKLoRAExtractor(
                model=QWEN_MODEL,
                adapter_path=QK_ADAPTER,
                device="cpu",
            )
            t0 = time.time()
            kw = qk_extractor.extract_keywords(TEST_TEXT, top_k=5)
            elapsed = time.time() - t0
            if test_result("qk_lora", kw, elapsed):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] qk_lora: {e}")
            failed += 1

    # ── 7. Candidate Segment Attention (if available) ──
    test_section("7. Candidate Segment Attention")
    segment_adapter_exists = Path(SEGMENT_ADAPTER).exists()
    bio_ckpt_exists = Path(BIO_CKPT).exists()
    if not qwen_exists or not segment_adapter_exists or not bio_ckpt_exists:
        print(
            "  [SKIP] candidate_segment_attn: "
            f"Qwen model exists={qwen_exists}, adapter exists={segment_adapter_exists}, bio exists={bio_ckpt_exists}"
        )
    else:
        try:
            segment_extractor = CandidateSegmentAttentionExtractor(
                model=QWEN_MODEL,
                adapter_path=SEGMENT_ADAPTER,
                bio_model_path=BIO_CKPT,
                device="cpu",
                max_candidates=20,
            )
            t0 = time.time()
            kw = segment_extractor.extract_keywords(TEST_TEXT, top_k=5, random_seeds=[1, 2])
            elapsed = time.time() - t0
            if test_result("candidate_segment_attn", kw, elapsed):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [FAIL] candidate_segment_attn: {e}")
            failed += 1

    # ── Summary ──
    test_section("Summary")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if failed == 0:
        print("\n  All tests passed! ✓")
    else:
        print(f"\n  {failed} test(s) failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
