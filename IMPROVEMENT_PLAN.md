# Cookbook improvement plan: readability and learning

A working plan for raising the cookbook's quality as a *learning resource*.
Each phase is independently mergeable; items reference the files they touch.
This file is a maintainers' document and is not part of the book build
(`only_build_toc_files: true` in `_config.yml`).

## Principles

1. **Evergreen prose.** Documentation must read correctly a year from now.
   No month-stamped claims (`released May 2026`, `as of June 2026`) — facts
   are stated with year-level granularity at most, and anything that changes
   monthly (top benchmark scores, patch-release notes, launch schedules)
   does not belong in the book at all.
2. **Single source of truth.** Version pins live in `pyproject.toml` /
   `environment.yml`; model defaults live in `src/llm_systems_cookbook/models.py`.
   Prose references these rather than restating exact versions.
3. **Concepts over news.** The glossary and track indexes teach durable
   concepts; ecosystem context is welcome but boxed, brief, and clearly
   marked as a snapshot.
4. **Learning is active.** Every chapter gives the reader something to
   predict, run, and check — not just read. The scoring harness is the
   self-assessment mechanism.

## Phase 1 — Evergreen content (in progress)

- [x] **De-date the ecosystem docs.** PR #33 rewrites `intro.md`,
  `glossary.md`, `CURRICULUM_SPEC.md`, and the inference/agents/serving/eval
  track indexes without month stamps, keeping all substantive content.
  This branch de-dates the remaining files (`notebooks/08_production/index.md`,
  notebooks 01, 06, and 09 in the production track).
- [x] **Evergreen lint.** `scripts/check_evergreen.py` flags month-stamped
  prose in Markdown files and notebook markdown/comment cells; inline code
  spans (API headers, spec version strings) and `_fixtures/` data are exempt.
  Run with `make check-evergreen`.
- [ ] **Wire the lint into CI** (`.github/workflows/ci.yml`) once PR #33 and
  this branch are both merged — the check is verified green against the
  combined state.
- [ ] **Move the changelog out of `intro.md`.** Replace the "What's new"
  admonition with a `CHANGELOG.md` (or GitHub Releases) and keep a stable
  two-line pointer in the intro. Release notes are the right home for
  time-stamped news; the book's landing page is not.
- [ ] **Reconcile version claims with actual pins.** `CURRICULUM_SPEC.md`
  describes `vllm==0.22.*` while `pyproject.toml` pins `vllm==0.8.*`. Decide
  which is right, fix the other, and add a small consistency check
  (`scripts/`) that asserts every `pkg==X.Y.*` mentioned in the spec matches
  the authoritative pin.

## Phase 2 — Readability

- [ ] **Standard chapter header cell.** One markdown template for all 64
  notebooks: *what you'll build → prerequisites (with `{doc}` links) →
  hardware + estimated runtime → reference papers*. Most chapters have parts
  of this; an audit pass makes it uniform. Template lives in
  `CONTRIBUTING.md`; `scripts/rewrite_init_cells.py` is the precedent for
  mechanical notebook edits.
- [ ] **Slim the track indexes.** Standard shape: *what you'll learn → key
  terms → reading order → (optional) ecosystem context box*. The ecosystem
  sections should be one paragraph each, not release-note lists — details
  belong in the glossary entries they link to.
- [ ] **Turn the `CURRICULUM_SPEC.md` pins paragraph into a table.** The
  framework-pins paragraph is a ~450-word single sentence; a table
  (package · pin · why it matters · notebooks affected) is scannable and
  diff-friendly.
- [ ] **Render the prerequisite DAG.** The spec describes the
  notebook-dependency DAG in prose; add a Mermaid diagram to `intro.md` and
  one per track index so readers can see the path (MyST supports Mermaid via
  `sphinxcontrib-mermaid`).
- [ ] **Glossary cross-link audit.** Ensure each chapter's first use of a
  glossary concept uses the `{term}` role; the glossary is the book's
  connective tissue and currently linkage is inconsistent across tracks.

## Phase 3 — Learning aids

- [ ] **"Check your understanding" blocks.** 3–5 conceptual questions per
  chapter with `{admonition}`-dropdown answers (e.g. *"Why does decode stay
  memory-bound even at batch 32?"*). Questions target the concept, not the
  code.
- [ ] **Solution sketches for exercises.** Chapters end with exercises; add
  collapsed-by-default solution outlines so self-learners can verify their
  approach without a classroom.
- [ ] **"Common pitfalls" section per chapter.** One short list per notebook
  (dtype mismatches in KV cache code, Triton autotune on T4, tokenizer
  whitespace in eval harnesses…). High value-per-line for learners debugging
  alone.
- [ ] **Document `make score` as self-assessment.** The deterministic scoring
  harness is the cookbook's best learning feature and is barely surfaced in
  the reader-facing docs; add a section to `intro.md` ("How to know you got
  it right") explaining `Scorer`, thresholds, and `scores/*.json`.
- [ ] **Difficulty and time-to-complete in reading orders.** Each track index
  reading-order list gains ★ difficulty and the existing runtime estimates
  from the spec, so readers can plan sessions.
- [ ] **Capstone per track.** One integrative exercise per track (e.g.
  inference: combine continuous batching + paged KV + speculative decoding
  on SmolLM2 and measure goodput). Specified in `CURRICULUM_SPEC.md` as v0.3
  candidates.

## Phase 4 — Maintenance guards

- [ ] `make check-evergreen` in CI (Phase 1 follow-through).
- [ ] **Treat Jupyter Book build warnings as errors** in `book.yml` for
  broken `{term}` / `{doc}` references, so cross-links can't silently rot.
- [ ] **Ecosystem-update policy** in `CONTRIBUTING.md`: updates summarize
  *what changed conceptually* with year-level dates; per-release patch notes,
  launch dates, and leaderboard deltas are out of scope for the book. PRs
  titled "<month> update" should be rewritten evergreen before merge.

## Sequencing

1. Merge PR #33 (evergreen rewrite of the ecosystem docs).
2. Merge this branch (plan, lint, remaining de-dating).
3. Wire `check-evergreen` into CI — from here the policy is enforced.
4. Phase 2 and 3 items are per-track and parallelizable; each is a small PR.
