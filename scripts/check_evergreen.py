"""Check prose documentation for month-stamped content.

The cookbook's documentation is written *evergreen*: prose must not pin
claims to specific months ("released May 2026", "as of June 2026",
"(published May 2026)") because date-stamped statements go stale within
weeks and add no pedagogical value. Year-level and event-level references
("2026", "H2 2026", "GTC 2025") are fine.

Scope: Markdown files, notebook markdown cells, and comment lines in
notebook code cells. Fixture data (``_fixtures/``) and Python sources are
exempt — dates there are data, not documentation.

Usage::

    python scripts/check_evergreen.py [paths...]

With no arguments, scans every ``*.md`` and ``*.ipynb`` under the repo
root. Exits 1 if violations are found.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

MONTHS_FULL = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
MONTHS_ABBR = ("Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec")

# Full month names are flagged anywhere (title-case avoids the modal verb
# "may"); abbreviations and numeric YYYY-MM forms only when attached to a
# year, to keep false positives low.
PATTERN = re.compile(
    r"\b(?:" + "|".join(MONTHS_FULL) + r")\b"
    r"|\b(?:" + "|".join(MONTHS_ABBR) + r")\.?\s+20\d{2}\b"
    r"|\b20\d{2}-(?:0[1-9]|1[0-2])\b"
)

EXCLUDED_PARTS = {".git", "_build", "_fixtures", ".ipynb_checkpoints", "node_modules"}

# Inline code spans may legitimately contain dated identifiers (API beta
# headers, spec version strings); only the surrounding prose is linted.
CODE_SPAN = re.compile(r"`[^`]*`")


def _notebook_lines(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (location, line) pairs for the prose-bearing lines of a notebook."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        src = cell.get("source", "")
        text = "".join(src) if isinstance(src, list) else src
        for lineno, line in enumerate(text.splitlines(), 1):
            if cell.get("cell_type") == "markdown":
                yield f"cell {idx} line {lineno}", line
            elif cell.get("cell_type") == "code" and line.lstrip().startswith("#"):
                yield f"cell {idx} line {lineno} (comment)", line


def _markdown_lines(path: Path) -> Iterator[tuple[str, str]]:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        yield str(lineno), line


def check_file(path: Path) -> list[str]:
    """Return formatted violation messages for one file."""
    lines = _notebook_lines(path) if path.suffix == ".ipynb" else _markdown_lines(path)
    violations: list[str] = []
    for location, line in lines:
        match = PATTERN.search(CODE_SPAN.sub("`…`", line))
        if match:
            violations.append(f"{path}:{location}: {match.group(0)!r} in: {line.strip()[:100]}")
    return violations


def collect_targets(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    root = Path(__file__).resolve().parent.parent
    targets = sorted(root.rglob("*.md")) + sorted(root.rglob("*.ipynb"))
    return [p for p in targets if not EXCLUDED_PARTS.intersection(p.parts)]


def main(args: list[str]) -> int:
    violations: list[str] = []
    for path in collect_targets(args):
        violations.extend(check_file(path))
    if violations:
        print(f"{len(violations)} month-stamped line(s) found — docs must stay evergreen:")
        print("\n".join(violations))
        return 1
    print("evergreen check passed: no month-stamped prose found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
