from __future__ import annotations

from .scoring import ATTENTION_METHODS


ATTENTION_IDF_METHODS = tuple(f"{method}_idf" for method in ATTENTION_METHODS)

METHOD_CATEGORIES: dict[str, dict] = {
    "attention_series": {
        "label": "Attention 系列方法",
        "description": "无监督 attention 方法，可选叠加 IDF。",
        "entrypoints": ["KeyAttenExtractor"],
        "methods": list(ATTENTION_METHODS),
        "idf_methods": list(ATTENTION_IDF_METHODS),
        "supports_idf": True,
    },
    "baselines": {
        "label": "Baseline / 其他方法",
        "description": "主要用于 benchmark 对照，不是当前主库主推推理线。",
        "entrypoints": [],
        "methods": [],
        "supports_idf": False,
    },
    "standalone_methods": {
        "label": "单方法",
        "description": "独立成线的方法，不与其他方法混合表达。",
        "entrypoints": ["BIOExtractor", "QKLoRAExtractor"],
        "methods": ["BIO", "QK"],
        "supports_idf": False,
    },
    "main_method": {
        "label": "主方法：BIO 候选 + Attention 排序",
        "description": "当前主推路线，统一按 BIO 候选 + Attention 排序 对外表达。",
        "entrypoints": [
            "KeyAttenExtractor(candidate_scoring='bio')",
            "CandidateSegmentAttentionExtractor",
        ],
        "methods": ["bio_attention_rerank"],
        "supports_idf": True,
    },
}


def get_method_catalog() -> dict[str, dict]:
    """Return the public method-category catalog used by the main library."""
    return {
        key: {
            **value,
            "entrypoints": list(value.get("entrypoints", [])),
            "methods": list(value.get("methods", [])),
            "idf_methods": list(value.get("idf_methods", [])),
        }
        for key, value in METHOD_CATEGORIES.items()
    }


__all__ = [
    "ATTENTION_IDF_METHODS",
    "METHOD_CATEGORIES",
    "get_method_catalog",
]
