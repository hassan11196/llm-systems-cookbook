# Contributing

Short and practical.

## Branches

Work on topic branches off `main`. Name them `<user>/<topic>`
(e.g. `hassan11196/paged-attention`). Do not push directly to `main`.

## Commits

Conventional commits style, present tense, lower case.

- `feat(01_inference): add paged attention allocator notebook`
- `fix(scoring): correct rel_err reporting in assert_close`
- `chore(ci): pin ruff to v0.8.4`
- `docs: refresh 02_rag README`

One notebook per commit where possible. Do not add yourself or any AI
assistant as co-author.

## Notebook hygiene

Before committing, clear notebook outputs:

    make nbclear

`nbstripout` is installed as a pre-commit hook to enforce this automatically.
CI re-executes notebooks, so the committed `.ipynb` files contain only cell
source.

## Style

- PEP-604 type hints on every user-defined function.
- `from __future__ import annotations` at the top of every `.py` file.
  (Notebook cells no longer need it — annotations aren't used inside the
  bootstrap cell.)
- `make lint` must pass.
- Tone: neutral and technical. Avoid marketing copy, quiz-style framing, or
  apologetic prose. The repo reads as a reference, not a diary.

## Notebook init cell

Every notebook's first code cell uses a single `bootstrap(...)` call:

```python
from llm_systems_cookbook.nb import bootstrap
# ...any notebook-specific imports (torch, numpy, math, dataclasses, ...)...
s = bootstrap("<track>_<NN>_<slug>")
```

`bootstrap` seeds the RNGs, prints a hardware summary, returns a `Scorer`,
and injects `DEVICE` / `IS_CUDA` into the notebook's globals for cells that
need them. New notebooks should follow the same shape; `scripts/rewrite_init_cells.py`
is available to migrate any that drift.

## Scoring

Every notebook ends with a `Scorer.save()` call that writes
`scores/{track}_{NN}_{slug}.json`. Scoring checks must be deterministic given
the seed set by `bootstrap` in cell 2.
