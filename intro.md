# Welcome to llm-infra-lab

Hands-on notebooks that teach modern LLM systems engineering — **inference,
retrieval, training, agents, serving, evaluation, and GPU programming** —
from first principles. Every notebook explains the idea, builds it in code,
and scores itself with numerical checks.

The book is written for a CS undergrad who wants to go from *"I know what
softmax is"* to *"I can reason about LLM serving economics."* No prior
deep-learning experience assumed.

## How to read this book

Each chapter follows the same shape so you always know where you are:

1. **Motivation** — the problem in plain English.
2. **CS analogy** — the "aha" hook (KV cache ≈ memoised Fibonacci,
   embedding ≈ soft hash, agent ≈ REPL).
3. **First-principles warm-up** — tiny code that proves the limit cases by
   hand before any library appears.
4. **Real implementation** — the actual technique, narrated cell by cell.
5. **Self-scoring cell** — deterministic numeric checks that pass or fail.
6. **Exercises + further reading** — what to try next and which papers
   to read.

## Run any notebook in one click

Every notebook's title cell opens with an **Open in Colab** badge. Click
it to run the notebook end-to-end on a free Colab T4 — no install needed.
The repo's landing page on GitHub has the same badges in a table for
direct access.

```{note}
The Colab badges, and the rocket-ship launch button in the top right of
each chapter, point at the `main` branch of
[hassan11196/llm-infra-lab](https://github.com/hassan11196/llm-infra-lab).
```

## Suggested reading order

| Step | Chapter | Why this order |
|---|---|---|
| 1 | *GPU architecture tour* | Establishes the bandwidth-vs-compute mental model everything else relies on. |
| 2 | *Roofline for LLM serving* | Applies that mental model to LLM-specific workloads. |
| 3 | *Autoregressive decoding and KV cache* | First contact with a real LLM; explains why decode is memory-bound. |
| 4 | *Perplexity from scratch* | The canonical LLM metric, derived and implemented three ways. |
| 5 | *Mixed precision, grad accum, and activation checkpointing* | The three memory tricks every training run uses. |
| 6 | *Chunking strategies for retrieval* | Your introduction to retrieval-augmented generation. |
| 7 | *ReAct from scratch* | Your introduction to agents — and a reminder that "agent" is a three-line loop. |

## What's in the book

This book currently covers **seven hello-world notebooks**, one per track
of the full curriculum. The complete 61-notebook spec lives in the
[curriculum specification](CURRICULUM_SPEC.md); the remaining 54
notebooks are authored in follow-up work.

## Install locally (optional)

Running in Colab is recommended. If you want a local setup:

```bash
pip install -e ".[book,dev]"
make book          # builds the book into _build/html/
open _build/html/index.html
```

The source is on [GitHub](https://github.com/hassan11196/llm-infra-lab)
under the MIT license.
