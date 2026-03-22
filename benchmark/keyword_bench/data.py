from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class Document:
    doc_id: str
    text: str
    keywords: List[str]
    language: str = "zh"
    meta: dict = field(default_factory=dict)


def _parse_csl_keywords(raw: str) -> List[str]:
    return [item.strip() for item in raw.split("_") if item.strip()]


def load_csl_split(path: Path, split_name: str, limit: int | None = None) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            _, text, labels = parts
            docs.append(
                Document(
                    doc_id=f"{split_name}-{index}",
                    text=text.strip(),
                    keywords=_parse_csl_keywords(labels),
                    meta={
                        "split": split_name,
                        "char_len": len(text.strip()),
                        "keyword_count": len(_parse_csl_keywords(labels)),
                    },
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs


def _percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    position = min(max(int(round((len(sorted_values) - 1) * q)), 0), len(sorted_values) - 1)
    return sorted_values[position]


def _slice_by_length(docs: List[Document]) -> Dict[str, List[Document]]:
    lengths = sorted(doc.meta["char_len"] for doc in docs)
    low = _percentile(lengths, 1 / 3)
    high = _percentile(lengths, 2 / 3)
    return {
        "csl_test_short": [doc for doc in docs if doc.meta["char_len"] <= low],
        "csl_test_medium": [doc for doc in docs if low < doc.meta["char_len"] <= high],
        "csl_test_long": [doc for doc in docs if doc.meta["char_len"] > high],
    }


def _slice_by_keyword_count(docs: List[Document]) -> Dict[str, List[Document]]:
    return {
        "csl_test_kw_le_4": [doc for doc in docs if doc.meta["keyword_count"] <= 4],
        "csl_test_kw_ge_5": [doc for doc in docs if doc.meta["keyword_count"] >= 5],
    }


def _maybe_trim(docs: Iterable[Document], limit: int | None) -> List[Document]:
    docs_list = list(docs)
    if limit is None:
        return docs_list
    return docs_list[:limit]


def build_csl_eval_sets(
    root_dir: str | Path,
    train_limit: int = 300,
    dev_limit: int = 300,
    test_limit: int = 500,
    derived_limit: int | None = 300,
) -> Dict[str, List[Document]]:
    root = Path(root_dir)
    kg_dir = root / "external" / "CSL" / "benchmark" / "kg"
    train_docs = load_csl_split(kg_dir / "train.tsv", "train", limit=train_limit)
    dev_docs = load_csl_split(kg_dir / "dev.tsv", "dev", limit=dev_limit)
    test_docs = load_csl_split(kg_dir / "test.tsv", "test", limit=test_limit)

    datasets: Dict[str, List[Document]] = {
        "csl_train_sample": train_docs,
        "csl_dev": dev_docs,
        "csl_test": test_docs,
    }
    for name, docs in {**_slice_by_length(test_docs), **_slice_by_keyword_count(test_docs)}.items():
        datasets[name] = _maybe_trim(docs, derived_limit)
    return datasets


def _compose_english_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    if title:
        return title
    if abstract:
        return abstract
    return (record.get("full_text") or "").strip()


def load_english_jsonl_split(path: Path, dataset_name: str, limit: int | None = None) -> List[Document]:
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            text = _compose_english_text(record)
            keywords = [item.strip() for item in record.get("keywords", []) if item.strip()]
            docs.append(
                Document(
                    doc_id=str(record.get("id") or f"{dataset_name}-{index}"),
                    text=text,
                    keywords=keywords,
                    language="en",
                    meta={
                        "split": "test",
                        "dataset": dataset_name,
                        "char_len": len(text),
                        "keyword_count": len(keywords),
                        "text_source": "title+abstract",
                    },
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs


def load_docsutf8_key_dataset(
    root_dir: str | Path,
    dataset_dirname: str,
    dataset_name: str,
    limit: int | None = None,
) -> List[Document]:
    root = Path(root_dir)
    dataset_root = root / "data" / dataset_dirname
    docs_dir = dataset_root / "docsutf8"
    keys_dir = dataset_root / "keys"
    if not docs_dir.exists() or not keys_dir.exists():
        return []

    docs: List[Document] = []
    for text_path in sorted(docs_dir.glob("*.txt")):
        key_path = keys_dir / f"{text_path.stem}.key"
        if not key_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8", errors="ignore").strip()
        keywords = [
            line.strip()
            for line in key_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip()
        ]
        if not text or not keywords:
            continue

        docs.append(
            Document(
                doc_id=text_path.stem,
                text=text,
                keywords=keywords,
                language="en",
                meta={
                    "split": "test",
                    "dataset": dataset_name,
                    "char_len": len(text),
                    "keyword_count": len(keywords),
                    "text_source": "full_text",
                    "source_format": "docsutf8+keys",
                },
            )
        )
        if limit and len(docs) >= limit:
            break
    return docs


def build_english_eval_sets(root_dir: str | Path, english_limit: int | None = None) -> Dict[str, List[Document]]:
    root = Path(root_dir)
    english_root = root / "external" / "Keyphrase_Extraction" / "Dataset"
    datasets = {
        "semeval2010_test": load_english_jsonl_split(
            english_root / "SemEval-2010" / "test.json",
            "SemEval-2010",
            limit=english_limit,
        ),
        "pubmed_test": load_english_jsonl_split(
            english_root / "PubMed" / "test.json",
            "PubMed",
            limit=english_limit,
        ),
        "lis2000_test": load_english_jsonl_split(
            english_root / "LIS-2000" / "test.json",
            "LIS-2000",
            limit=english_limit,
        ),
    }
    extra_datasets = {
        "semeval2010_fulltext": load_docsutf8_key_dataset(
            root_dir,
            "SemEval2010",
            "SemEval2010-fulltext",
            limit=english_limit,
        ),
        "krapivin2009_fulltext": load_docsutf8_key_dataset(
            root_dir,
            "Krapivin2009",
            "Krapivin2009-fulltext",
            limit=english_limit,
        ),
    }
    datasets.update({name: docs for name, docs in extra_datasets.items() if docs})
    return datasets


def _parse_shencecup_keywords(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_shencecup_labeled(root_dir: str | Path, limit: int | None = None) -> List[Document]:
    root = Path(root_dir)
    raw_dir = root / "data" / "shencecup" / "raw"
    labels_path = raw_dir / "train_docs_keywords.txt"
    docs_path = raw_dir / "all_docs.txt"
    if not labels_path.exists() or not docs_path.exists():
        return []

    labels: Dict[str, List[str]] = {}
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) != 2:
                continue
            doc_id, raw_keywords = parts
            labels[doc_id] = _parse_shencecup_keywords(raw_keywords)

    docs: List[Document] = []
    with docs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\x01")
            if len(parts) != 3:
                continue
            doc_id, title, body = parts
            if doc_id not in labels:
                continue
            text = f"{title.strip()}。{body.strip()}".strip("。")
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=text,
                    keywords=labels[doc_id],
                    language="zh",
                    meta={
                        "split": "labeled",
                        "dataset": "ShenCeCup",
                        "char_len": len(text),
                        "keyword_count": len(labels[doc_id]),
                        "title": title.strip(),
                    },
                )
            )
            if limit and len(docs) >= limit:
                break
    return docs


def build_shencecup_eval_sets(root_dir: str | Path, shencecup_limit: int | None = None) -> Dict[str, List[Document]]:
    docs = load_shencecup_labeled(root_dir, limit=shencecup_limit)
    if not docs:
        return {}
    return {
        "shencecup_labeled": docs,
        "shencecup_short": _slice_by_length(docs)["csl_test_short"],
        "shencecup_medium": _slice_by_length(docs)["csl_test_medium"],
        "shencecup_long": _slice_by_length(docs)["csl_test_long"],
        "shencecup_kw_le_2": [doc for doc in docs if doc.meta["keyword_count"] <= 2],
        "shencecup_kw_ge_4": [doc for doc in docs if doc.meta["keyword_count"] >= 4],
    }


def build_all_eval_sets(
    root_dir: str | Path,
    train_limit: int = 300,
    dev_limit: int = 300,
    test_limit: int = 500,
    derived_limit: int | None = 300,
    english_limit: int | None = None,
    shencecup_limit: int | None = None,
) -> Dict[str, List[Document]]:
    datasets = build_csl_eval_sets(
        root_dir,
        train_limit=train_limit,
        dev_limit=dev_limit,
        test_limit=test_limit,
        derived_limit=derived_limit,
    )
    datasets.update(build_english_eval_sets(root_dir, english_limit=english_limit))
    datasets.update(build_shencecup_eval_sets(root_dir, shencecup_limit=shencecup_limit))
    return datasets
