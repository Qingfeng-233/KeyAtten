#!/usr/bin/env python3
"""Qwen teacher for extractive keyword ranking."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """给定以下文本，请按重要性从高到低输出最多 {top_k} 个关键词。
- 必须从原文抽取（不要改写、不要总结）
- 每行一个关键词，格式："排名. 关键词"
- 越靠前越重要

文本：{text}"""


def parse_ranked_keywords(output: str, source_text: str, top_k: int = 20) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\s]*", "", line)
        line = re.sub(r"^\d+\s*[\.\、\):：-]\s*", "", line).strip()
        line = line.strip("\"'“”‘’` ，,。;；")
        if not line or line in seen:
            continue
        if len(line) > 32:
            continue
        if line in source_text:
            keywords.append(line)
            seen.add(line)
        if len(keywords) >= top_k:
            break
    return keywords


class QwenKeywordTeacher:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        max_gpu_memory: str | None = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.max_new_tokens = max_new_tokens
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["device_map"] = "cuda" if device == "cuda" else None
        elif device == "cuda" and max_gpu_memory:
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "48GiB"}
        elif device == "cuda":
            load_kwargs["device_map"] = "cuda"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_kwargs)
        if device != "cuda" and not load_in_4bit:
            self.model.to(device)
        self.model.eval()

    def generate_keywords(self, text: str, top_k: int = 20) -> tuple[list[str], str]:
        prompt = PROMPT_TEMPLATE.format(text=text, top_k=top_k)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                model_input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            model_input = prompt
        inputs = self.tokenizer(model_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return parse_ranked_keywords(output, text, top_k=top_k), output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/Keyatten/models/Qwen3.5-4B")
    parser.add_argument("--text", default="我国新能源汽车产业快速发展，比亚迪、宁德时代等企业在全球市场竞争力持续提升。")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32", "auto"), default="bfloat16")
    parser.add_argument("--max-gpu-memory", default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    teacher = QwenKeywordTeacher(
        args.model,
        device=args.device,
        dtype=args.dtype,
        max_gpu_memory=args.max_gpu_memory,
        load_in_4bit=args.load_in_4bit,
    )
    keywords, raw = teacher.generate_keywords(args.text, top_k=args.top_k)
    print("[raw]")
    print(raw)
    print("[keywords]")
    for idx, kw in enumerate(keywords, 1):
        print(f"{idx}. {kw}")


if __name__ == "__main__":
    main()
