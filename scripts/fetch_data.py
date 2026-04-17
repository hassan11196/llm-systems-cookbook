"""Idempotent dataset fetcher.

Downloads the small corpora used by Phase 2 notebooks into ``data/``:

* wikitext-2-raw-v1 (perplexity, mixed-precision training token source)
* A BEIR/scifact dev slice (chunking, reranking)
* A SQuAD-v2 validation slice (RAGAS)

Safe to re-run: cached HuggingFace artefacts live under ``data/.hf_cache``.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="skip BEIR/scifact download (useful when RAG deps are not installed)",
    )
    parser.add_argument(
        "--skip-squad",
        action="store_true",
        help="skip SQuAD download",
    )
    args = parser.parse_args()

    try:
        from llm_infra_lab._utils import download_wikitext2
        from llm_infra_lab.datasets import load_beir_scifact_dev, load_squad_mini
    except ImportError as e:
        print(f"Install the package first: pip install -e . ({e})", file=sys.stderr)
        return 1

    print("[fetch] wikitext-2 (test split)")
    path = download_wikitext2(split="test")
    print(f"  -> {path}")

    if not args.skip_rag:
        print("[fetch] BEIR/scifact dev slice (n=300)")
        d = load_beir_scifact_dev(n=300)
        print(f"  -> queries={len(d['queries'])} corpus={len(d['corpus'])} qrels={len(d['qrels'])}")

    if not args.skip_squad:
        print("[fetch] SQuAD-v2 validation mini (n=50)")
        rows = load_squad_mini(n=50)
        print(f"  -> {len(rows)} rows")

    print("[fetch] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
