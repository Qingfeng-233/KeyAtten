"""HuggingFace 模型加载的共用步骤（QKLoRAExtractor / CandidateSegmentAttentionExtractor 共享）。

只负责：可选依赖检查、dtype 解析、tokenizer 加载回退、LoRA adapter 挂载。
"""
from __future__ import annotations

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


def require_torch(component: str) -> None:
    """缺少 torch / transformers 时抛出带安装提示的 ImportError。"""
    if torch is None:
        raise ImportError(f"torch is required for {component}. Install with: pip install torch>=2.0")
    if AutoModel is None or AutoTokenizer is None:
        raise ImportError(
            f"transformers is required for {component}. Install with: pip install transformers>=4.30"
        )


def resolve_amp_dtype(dtype: str, device: str):
    """把 dtype 字符串转换成 torch dtype；'auto' 在支持 bf16 的 CUDA 上取 bfloat16，其余返回 None。"""
    if dtype == "auto":
        if device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    return None


def load_tokenizer(model_name: str):
    """优先加载 fast tokenizer，失败时回退到 slow tokenizer。"""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)


def load_adapter(base_model, adapter_path: str | None):
    """adapter_path 非空时用 peft 挂载 LoRA adapter，否则原样返回模型。"""
    if adapter_path is None:
        return base_model
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError(
            "peft is required to load LoRA adapters. "
            "Install with: pip install peft>=0.10.0"
        )
    return PeftModel.from_pretrained(base_model, adapter_path)
