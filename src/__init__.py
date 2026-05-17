# Suppress known deprecation warnings from third-party dependencies
import warnings as _warnings
_warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

__version__ = "0.3.0"
__all__ = [
    "KeyAttenExtractor",
    "QKLoRAExtractor",
    "CandidateSegmentAttentionExtractor",
    "BIOExtractor",
    "METHOD_CATEGORIES",
    "get_method_catalog",
    "WordWeight",
    "extract_keywords",
]


def __getattr__(name: str):
    if name == "WordWeight":
        from .candidates import WordWeight

        return WordWeight
    if name in {"KeyAttenExtractor", "extract_keywords"}:
        from .extractors.attention import KeyAttenExtractor, extract_keywords

        return {
            "KeyAttenExtractor": KeyAttenExtractor,
            "extract_keywords": extract_keywords,
        }[name]
    if name == "QKLoRAExtractor":
        from .extractors.qk_lora import QKLoRAExtractor

        return QKLoRAExtractor
    if name == "CandidateSegmentAttentionExtractor":
        from .extractors.candidate_segment import CandidateSegmentAttentionExtractor

        return CandidateSegmentAttentionExtractor
    if name == "BIOExtractor":
        from .extractors.bio import BIOExtractor

        return BIOExtractor
    if name in {"METHOD_CATEGORIES", "get_method_catalog"}:
        from .method_catalog import METHOD_CATEGORIES, get_method_catalog

        return {
            "METHOD_CATEGORIES": METHOD_CATEGORIES,
            "get_method_catalog": get_method_catalog,
        }[name]
    raise AttributeError(f"module 'keyatten' has no attribute '{name}'")
