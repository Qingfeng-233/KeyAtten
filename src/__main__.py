"""
KeyAtten: Attention-based Keyword/Keyphrase Extraction

Usage:
    python -m keyatten                          # show version and available extractors
    python -m keyatten extract "你的文本"        # quick keyword extraction
    python -m keyatten benchmark --help         # benchmark CLI
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in {"--help", "-h"}:
        from keyatten import __version__
        print(f"KeyAtten v{__version__}")
        print()
        print("Usage:")
        print("  python -m keyatten                           show this help")
        print("  python -m keyatten extract \"文本\"            quick extraction (default: received_attn)")
        print("  python -m keyatten extract \"文本\" -m samrank use specific method")
        print("  python -m keyatten benchmark ...             run benchmark CLI")
        print()
        print("Available extractors:")
        print("  KeyAttenExtractor                   attention-based (zero-shot)")
        print("  QKLoRAExtractor                     QK contrastive LoRA")
        print("  BIOExtractor                        BIO sequence labeling")
        print("  CandidateSegmentAttentionExtractor  BIO + attention reranking (main method)")
        return 0

    command = args[0]

    if command == "extract":
        return _cmd_extract(args[1:])
    elif command == "benchmark":
        from keyatten.benchmark_cli import main as bench_main
        return bench_main(args[1:])
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `python -m keyatten --help` for usage.", file=sys.stderr)
        return 1


def _cmd_extract(args: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="keyatten extract")
    parser.add_argument("text", help="Text to extract keywords from")
    parser.add_argument("-m", "--method", default="received_attn", help="Attention method (default: received_attn)")
    parser.add_argument("--model", default="thenlper/gte-small-zh", help="Model name or path")
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Number of keywords (default: 10)")
    parser.add_argument("--language", default="zh", choices=["zh", "en"], help="Language (default: zh)")
    parsed = parser.parse_args(args)

    from keyatten import KeyAttenExtractor
    ext = KeyAttenExtractor(model=parsed.model, language=parsed.language)
    keywords = ext.extract_keywords(parsed.text, method=parsed.method, top_k=parsed.top_k)
    print(", ".join(keywords))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
