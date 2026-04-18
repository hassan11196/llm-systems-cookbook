"""Small dataset loaders used across notebooks.

All loaders are cache-friendly and return in-memory Python objects (lists of
dicts). Nothing is downloaded at module import time — loaders pull data only
when called.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from llm_systems_cookbook._utils import data_dir


def _cache_file(name: str) -> Path:
    return data_dir() / f"{name}.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_wikitext2_mini(n_docs: int = 200, split: str = "test") -> list[str]:
    """Return the first ``n_docs`` non-empty documents from wikitext-2."""

    from datasets import load_dataset  # noqa: PLC0415

    cache = data_dir() / ".hf_cache"
    cache.mkdir(exist_ok=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=str(cache))
    docs: list[str] = []
    buf: list[str] = []
    for row in ds:
        text = row["text"]
        if text.startswith(" = ") and not text.startswith(" = ="):
            if buf:
                docs.append("".join(buf))
                buf = []
            if len(docs) >= n_docs:
                break
        buf.append(text)
    if buf and len(docs) < n_docs:
        docs.append("".join(buf))
    return docs[:n_docs]


def load_beir_scifact_dev(n: int = 300) -> dict[str, list[dict[str, Any]]]:
    """Return a small BEIR/scifact dev slice.

    Returns ``{"queries": [...], "corpus": [...], "qrels": [...]}``. Each query
    and corpus row has an ``id`` and a ``text`` field; qrels rows record
    ``query_id``, ``doc_id``, and ``score``.

    Uses the HF ``BeIR/scifact`` dataset under the hood and caches a compact
    JSONL per split.
    """

    from datasets import load_dataset  # noqa: PLC0415

    cache = data_dir() / ".hf_cache"
    cache.mkdir(exist_ok=True)
    q_path = _cache_file("scifact_dev_queries")
    c_path = _cache_file("scifact_dev_corpus")
    r_path = _cache_file("scifact_dev_qrels")

    if not (q_path.exists() and c_path.exists() and r_path.exists()):
        queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", cache_dir=str(cache))
        corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", cache_dir=str(cache))
        qrels_ds = load_dataset("BeIR/scifact-qrels", split="test", cache_dir=str(cache))

        q_rows = [{"id": str(r["_id"]), "text": r["text"]} for r in queries_ds]
        c_rows = [
            {"id": str(r["_id"]), "text": r["text"], "title": r.get("title", "")} for r in corpus_ds
        ]
        r_rows = [
            {
                "query_id": str(r["query-id"]),
                "doc_id": str(r["corpus-id"]),
                "score": int(r["score"]),
            }
            for r in qrels_ds
        ]

        _write_jsonl(q_path, q_rows)
        _write_jsonl(c_path, c_rows)
        _write_jsonl(r_path, r_rows)

    queries = _read_jsonl(q_path)[:n]
    keep_qids = {q["id"] for q in queries}
    qrels = [r for r in _read_jsonl(r_path) if r["query_id"] in keep_qids]
    keep_docs = {r["doc_id"] for r in qrels}
    corpus_all = _read_jsonl(c_path)
    corpus = [d for d in corpus_all if d["id"] in keep_docs]
    # Pad corpus with random distractors so retrieval is non-trivial.
    rng = random.Random(0)
    distractors = [d for d in corpus_all if d["id"] not in keep_docs]
    rng.shuffle(distractors)
    target_size = max(len(corpus) * 5, 1000)
    corpus.extend(distractors[: max(0, target_size - len(corpus))])

    return {"queries": queries, "corpus": corpus, "qrels": qrels}


def load_squad_mini(n: int = 50) -> list[dict[str, Any]]:
    """Return a small SQuAD-v2 dev slice as question/context/answers tuples."""

    from datasets import load_dataset  # noqa: PLC0415

    cache = data_dir() / ".hf_cache"
    ds = load_dataset("squad_v2", split="validation", cache_dir=str(cache))
    out: list[dict[str, Any]] = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context": row["context"],
                "answers": row["answers"]["text"],
            }
        )
    return out
