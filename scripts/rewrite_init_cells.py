"""One-shot migration: collapse each notebook's init boilerplate into a
single ``bootstrap()`` call.

Usage::

    python scripts/rewrite_init_cells.py              # rewrite in place
    python scripts/rewrite_init_cells.py --check      # dry-run, exit 1 on diff
    python scripts/rewrite_init_cells.py <path.ipynb> # single file

The rewriter is idempotent: running it on an already-migrated notebook is a
no-op. Notebook-specific imports (``math``, ``torch``, ``numpy as np``,
``from dataclasses import ...``, etc.) are preserved verbatim — only the
canonical boilerplate slab is replaced.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BOILERPLATE_EXACT = {
    "from __future__ import annotations",
    "import sys",
    "from pathlib import Path",
    "REPO = Path.cwd()",
    'while not (REPO / "scoring" / "harness.py").exists() and REPO != REPO.parent:',
    "    REPO = REPO.parent",
    "sys.path.insert(0, str(REPO))",
    'sys.path.insert(0, str(REPO / "src"))',
    "from scoring.harness import Scorer",
    "print(hardware_check())",
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
    "DEVICE = get_device()",
    'IS_CUDA = DEVICE.type == "cuda"',
}

BOILERPLATE_RE = [
    re.compile(r"^from llm_systems_cookbook\._utils import [\w,\s]+$"),
    re.compile(r"^set_seed\(\d+\)$"),
    re.compile(r'^s = Scorer\("[^"]+"\)$'),
]

SCORER_RE = re.compile(r'Scorer\("([^"]+)"\)')
BOOTSTRAP_RE = re.compile(r'^s = bootstrap\("([^"]+)"\)$')


def is_boilerplate(line: str) -> bool:
    stripped = line.rstrip("\n")
    if stripped in BOILERPLATE_EXACT:
        return True
    return any(pat.match(stripped) for pat in BOILERPLATE_RE)


def rewrite_cell_source(src: str) -> tuple[str, bool]:
    """Return (new_source, changed).

    ``changed`` is False if the cell is already in the post-migration shape
    (contains ``s = bootstrap("...")``) or has no Scorer to migrate.
    """

    if BOOTSTRAP_RE.search(src):
        return src, False

    m = SCORER_RE.search(src)
    if not m:
        return src, False
    notebook_id = m.group(1)

    kept: list[str] = []
    for line in src.splitlines():
        if is_boilerplate(line):
            continue
        kept.append(line)

    # Collapse runs of blank lines and strip leading/trailing blanks.
    collapsed: list[str] = []
    prev_blank = True
    for line in kept:
        if line.strip() == "":
            if prev_blank:
                continue
            collapsed.append("")
            prev_blank = True
        else:
            collapsed.append(line)
            prev_blank = False
    while collapsed and collapsed[-1] == "":
        collapsed.pop()

    new_lines: list[str] = ["from llm_systems_cookbook.nb import bootstrap", ""]
    if collapsed:
        new_lines.extend(collapsed)
        new_lines.append("")
    new_lines.append(f's = bootstrap("{notebook_id}")')

    return "\n".join(new_lines) + "\n", True


def source_to_jupyter_lines(src: str) -> list[str]:
    """Jupyter stores ``source`` as a list of lines; every line but the last
    ends with ``\\n``. Reproduce that convention."""

    parts = src.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return [p + "\n" for p in parts[:-1]] + [parts[-1]] if parts else []


def rewrite_notebook(path: Path) -> bool:
    raw = path.read_text()
    nb = json.loads(raw)
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new_src, did = rewrite_cell_source(src)
        if did:
            cell["source"] = source_to_jupyter_lines(new_src)
            changed = True
        # Only the first code cell carries the slab; stop after the first one.
        break
    if changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", type=Path, help="notebooks to rewrite")
    ap.add_argument("--check", action="store_true", help="dry-run; exit 1 on diff")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    targets = [p.resolve() for p in args.paths] or sorted((repo / "notebooks").rglob("*.ipynb"))

    changed: list[Path] = []
    for p in targets:
        if args.check:
            raw = p.read_text()
            nb = json.loads(raw)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                src = "".join(cell.get("source", []))
                _, did = rewrite_cell_source(src)
                if did:
                    changed.append(p)
                break
        else:
            if rewrite_notebook(p):
                changed.append(p)

    def _rel(pth: Path) -> str:
        try:
            return str(pth.relative_to(repo))
        except ValueError:
            return str(pth)

    if args.check:
        for p in changed:
            print(f"would rewrite {_rel(p)}")
        return 1 if changed else 0

    for p in changed:
        print(f"rewrote {_rel(p)}")
    print(f"{len(changed)}/{len(targets)} notebooks modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
