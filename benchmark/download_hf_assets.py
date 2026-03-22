from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests


DEFAULT_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "model.safetensors",
    "pytorch_model.bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF repo id, such as BAAI/bge-small-zh-v1.5")
    parser.add_argument("--root-dir", default=".", help="Project root")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def parallel_download(
    session: requests.Session,
    url: str,
    target_path: Path,
    total_bytes: int,
    timeout: int,
    workers: int,
) -> int:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        handle.truncate(total_bytes)

    chunk_size = max(total_bytes // workers, 1)
    ranges = []
    for index in range(workers):
        start = index * chunk_size
        end = total_bytes - 1 if index == workers - 1 else min(((index + 1) * chunk_size) - 1, total_bytes - 1)
        if start > end:
            continue
        ranges.append((start, end))

    def _download_one(byte_range: tuple[int, int]) -> int:
        start, end = byte_range
        headers = {"Range": f"bytes={start}-{end}"}
        with session.get(url, stream=True, headers=headers, timeout=(30, timeout)) as response:
            response.raise_for_status()
            written = 0
            with target_path.open("r+b") as handle:
                handle.seek(start)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written += len(chunk)
            return written

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return sum(executor.map(_download_one, ranges))


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).resolve()
    target_dir = root_dir / "models" / args.model.replace("/", "__")
    target_dir.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

    session = requests.Session()
    for filename in DEFAULT_FILES:
        url = f"{endpoint}/{args.model}/resolve/main/{filename}?download=1"
        target_path = target_dir / filename
        try:
            head = session.head(url, allow_redirects=True, timeout=(30, args.timeout))
            if head.status_code == 404:
                continue
            head.raise_for_status()
            content_length = int(head.headers.get("content-length", "0") or "0")
            accept_ranges = head.headers.get("accept-ranges", "")
            if content_length > 20 * 1024 * 1024 and accept_ranges.lower() == "bytes":
                total = parallel_download(
                    session,
                    head.url,
                    target_path,
                    content_length,
                    args.timeout,
                    args.workers,
                )
            else:
                with session.get(url, stream=True, timeout=(30, args.timeout)) as response:
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    total = 0
                    with target_path.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            total += len(chunk)
            print(f"{filename}: {total} bytes")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                continue
            raise
    (target_dir / "download.ok").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
