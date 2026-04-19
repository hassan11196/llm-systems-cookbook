"""Pre-pull tokenizers and configs for the model shortlist.

Downloads only tokenizer + config artefacts (not weights) so the command
stays fast and can be run on machines without a GPU. Weights are fetched
lazily by individual notebooks.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="also download model weights (multi-GB; use with care)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from llm_systems_cookbook.models import REGISTRY
    except ImportError as e:
        print(f"Install the package first: pip install -e . ({e})", file=sys.stderr)
        return 1

    for key, spec in REGISTRY.items():
        print(f"[warm] {key} ({spec.hf_id}, ~{spec.params_m}M params)")
        try:
            AutoTokenizer.from_pretrained(spec.hf_id)
            AutoConfig.from_pretrained(spec.hf_id)
            if args.weights:
                # transformers 4.46 renamed torch_dtype -> dtype; keep
                # a fallback so older pinned versions still work.
                try:
                    AutoModelForCausalLM.from_pretrained(spec.hf_id, dtype="auto")
                except TypeError:
                    AutoModelForCausalLM.from_pretrained(spec.hf_id, torch_dtype="auto")
        except Exception as e:  # noqa: BLE001 - best-effort prefetch
            print(f"  WARN: {type(e).__name__}: {e}")

    print("[warm] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
