from __future__ import annotations

__all__ = [
    "BIOExtractor",
    "CandidateSegmentAttentionExtractor",
    "KeyAttenExtractor",
    "QKLoRAExtractor",
    "extract_keywords",
]


def __getattr__(name: str):
    if name in {"KeyAttenExtractor", "extract_keywords"}:
        from .attention import KeyAttenExtractor, extract_keywords

        return {
            "KeyAttenExtractor": KeyAttenExtractor,
            "extract_keywords": extract_keywords,
        }[name]
    if name == "BIOExtractor":
        from .bio import BIOExtractor

        return BIOExtractor
    if name == "CandidateSegmentAttentionExtractor":
        from .candidate_segment import CandidateSegmentAttentionExtractor

        return CandidateSegmentAttentionExtractor
    if name == "QKLoRAExtractor":
        from .qk_lora import QKLoRAExtractor

        return QKLoRAExtractor
    raise AttributeError(f"module 'keyatten.extractors' has no attribute '{name}'")
