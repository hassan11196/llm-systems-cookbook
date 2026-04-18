# llm-infra-lab

A collection of Jupyter notebooks covering modern LLM systems engineering:
inference engines, serving, scaling, training, retrieval, agents, evaluation,
and GPU programming. Each notebook reimplements a core technique from scratch,
compares it against a production tool, and self-scores via numerical checks.

The full 61-notebook specification lives in
[`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md). This repository builds that
curriculum incrementally — see the **Status** section below for what is
currently available.

## Tracks

| Track | Notebooks | Focus |
|---|---|---|
| 01 inference | 10 | KV cache, PagedAttention, FlashAttention, continuous batching, speculative decoding, disaggregated serving |
| 02 rag       |  9 | Chunking, dense/sparse/hybrid retrieval, ColBERT, reranking, HyDE, RAPTOR, GraphRAG |
| 03 training  |  8 | Mixed precision, FSDP2, tensor/pipeline parallel, LoRA, QLoRA, DPO, GRPO |
| 04 agents    |  7 | ReAct, structured outputs, LangGraph, DSPy, MCP, multi-agent, τ-bench/SWE-bench |
| 05 serving   | 11 | Roofline, KV variants, compression, quantization, MoE, disaggregation, SLO/autoscaling |
| 06 eval      |  8 | Perplexity, MMLU, HumanEval, LLM-as-judge, Arena, NIAH/RULER, contamination |
| 07 gpu       |  8 | GPU arch, Triton, FlashAttention, fused kernels, torch.compile, Nsight, JAX sharding |

## Install

Pick the tracks you need. Optional dependency groups are defined so that
version-incompatible stacks (e.g., `transformers==4.46` for RAG vs. later
releases for training) can coexist via separate virtualenvs.

    pip install -e ".[inference,serving,gpu]"

Or everything at once (may require a recent CUDA toolkit for `vllm` and
`bitsandbytes`):

    pip install -e ".[inference,rag,training,agents,serving,eval,gpu,jax,dev]"

For CPU-only exploration (RAG, eval, agents, scoring harness):

    pip install -e ".[rag,eval,agents,dev]"

## Run

    make install-dev       # harness + linting only
    make warm-cache        # pre-pull HuggingFace models used in notebooks
    jupyter lab notebooks/

Each notebook ends with a scoring cell that writes
`scores/{track}_{NN}_{slug}.json`. Aggregate with:

    python -m scoring.run_all

## Hardware

Most notebooks run on a free Colab T4. Exceptions are declared in each
notebook's header cell:

- `01_inference/05_flashattention2_triton.ipynb` — Ampere or newer
- `07_gpu/04_triton_flashattention.ipynb` — Ampere or newer
- `07_gpu/07_nsight_profiling.ipynb` — local GPU or Colab Pro

## Open in Colab

Every notebook has an "Open in Colab" link so you can run it end-to-end on
a free T4 without installing anything locally. Click the badge next to the
notebook below.

The Colab URL template is:

```
https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/<path-to-notebook>
```

Replace `main` with another branch name if you want to try a work-in-progress
revision. The first cell of every notebook auto-discovers the repo root and
adds `scoring/` and `src/` to `sys.path`, so Colab execution works without a
`pip install` of the package.

## Status

Six of seven tracks fully authored; the last one lands with this PR.

| Track | Done | Notebooks in the book |
|---|---:|---|
| 01 inference | 10 / 10 | ✅ |
| 02 rag       | 9 / 9   | ✅ |
| 03 training  | 2 / 8   | six more remain |
| 04 agents    | 7 / 7   | ✅ |
| 05 serving   | 11 / 11 | ✅ (this PR) |
| 06 eval      | 8 / 8   | ✅ |
| 07 gpu       | 8 / 8   | ✅ |

Every completed notebook has an **Open in Colab** badge at the top
of its first markdown cell and appears in the published Jupyter
Book at the TOC entry for its track.

Jump into any track by name:

- [Foundations](notebooks/07_gpu/) — GPU architecture, Triton 101,
  tiled matmul, FlashAttention-2, fused RoPE + RMSNorm, torch.compile,
  Nsight profiling, JAX sharding, roofline analysis.
- [Inference engines](notebooks/01_inference/) — autoregressive
  decoding, attention roofline, PagedAttention, continuous batching,
  FA2-in-layer, radix prefix cache, speculative decoding, tree
  speculation, SARATHI chunked prefill, disaggregated serving.
- [Serving and scaling](notebooks/05_serving/) — KV variants and
  compression, KIVI, GPTQ/AWQ, SmoothQuant / FP8 / NF4,
  QuaRot/SpinQuant, batching strategies, MoE, DistServe,
  observability + autoscaler.
- [Retrieval-augmented generation](notebooks/02_rag/) — chunking,
  FAISS, BM25/SPLADE/RRF, ColBERTv2, reranking, HyDE, RAPTOR,
  GraphRAG, RAGAS.
- [Agent frameworks](notebooks/04_agents/) — ReAct, structured
  outputs, state machines, DSPy/MIPROv2, MCP, AutoGen vs CrewAI,
  evaluation suite.
- [Evaluation](notebooks/06_eval/) — perplexity, MMLU, HumanEval,
  judge bias, Arena Elo, NIAH/RULER, contamination, lm-eval vs
  Inspect AI.
- [Training](notebooks/03_training/) — mixed precision and
  checkpointing, DDP vs FSDP2 (six more notebooks authored in a
  future PR).

## Layout

```
llm-infra-lab/
├── CURRICULUM_SPEC.md         # per-notebook specification
├── src/llm_infra_lab/         # shared helpers (hardware_check, set_seed, ModelSpec registry)
├── scoring/                   # Scorer harness + aggregator + unit tests
├── notebooks/                 # per-track directories
├── scripts/                   # fetch_data.py, warm_cache.py
└── .github/workflows/         # CI (lint + CPU-safe notebook execution)
```

## License

MIT
