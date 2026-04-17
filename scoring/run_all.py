"""Aggregate all ``scores/*.json`` files into a single markdown table.

Usage::

    python -m scoring.run_all

Exits non-zero if any notebook has ``passed < total``.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES_DIR = REPO_ROOT / "scores"

TRACK_NAMES: dict[str, str] = {
    "01_inference": "Inference engines",
    "02_rag": "Retrieval-augmented generation",
    "03_training": "Training and fine-tuning",
    "04_agents": "Agent frameworks",
    "05_serving": "Serving and scaling",
    "06_eval": "Evaluation",
    "07_gpu": "GPU programming",
}


def _track_for(notebook_id: str) -> str:
    parts = notebook_id.split("_", 2)
    if len(parts) < 2:
        return "other"
    return f"{parts[0]}_{parts[1]}"


def main() -> int:
    if not SCORES_DIR.exists():
        print("No scores/ directory. Run some notebooks first.")
        return 0

    files = sorted(SCORES_DIR.glob("*.json"))
    if not files:
        print("No scoring artifacts found in scores/.")
        return 0

    by_track: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"WARN: skipping malformed {path.name}: {e}", file=sys.stderr)
            continue
        by_track[_track_for(data["notebook_id"])].append(data)

    lines: list[str] = ["# Scoring summary", ""]
    total_passed = total_total = total_skipped = 0
    any_failures = False

    for track_key in sorted(by_track):
        track_label = TRACK_NAMES.get(track_key, track_key)
        lines.append(f"## {track_label}")
        lines.append("")
        lines.append("| Notebook | Passed | Total | Skipped | Score | Elapsed (s) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for data in sorted(by_track[track_key], key=lambda d: d["notebook_id"]):
            # Coerce defensively: some notebooks produce numpy ints that the
            # Scorer serialises via ``default=str``, leaving "passed" / "total"
            # as strings in the JSON file.
            passed = int(data["passed"])
            total = int(data["total"])
            skipped = int(data.get("skipped", 0))
            score = float(data.get("score", 0.0))
            elapsed = float(data.get("elapsed_s", 0.0))
            total_passed += passed
            total_total += total
            total_skipped += skipped
            if total and passed < total:
                any_failures = True
            lines.append(
                f"| {data['notebook_id']} | {passed} | {total} | {skipped} | "
                f"{score:.0%} | {elapsed:.1f} |"
            )
        lines.append("")

    overall = (total_passed / total_total) if total_total else 0.0
    lines.append(
        f"**Overall:** {total_passed}/{total_total} passed ({overall:.0%}); "
        f"skipped={total_skipped} across {len(files)} notebooks."
    )
    print("\n".join(lines))
    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())
