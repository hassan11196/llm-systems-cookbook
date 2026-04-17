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

## Status

This session delivers Phase 1 (scaffolding) and Phase 2 (one hello-world
notebook per track). Remaining notebooks are authored in follow-on sessions
in the order declared by `CURRICULUM_SPEC.md`.

| Track | Hello-world notebook |
|---|---|
| 01 inference | `01_autoregressive_decoding_kv_cache.ipynb` |
| 02 rag       | `01_chunking_strategies.ipynb` |
| 03 training  | `01_mixed_precision_accum_checkpointing.ipynb` |
| 04 agents    | `01_react_from_scratch.ipynb` |
| 05 serving   | `01_roofline_analysis.ipynb` |
| 06 eval      | `01_perplexity_from_scratch.ipynb` |
| 07 gpu       | `01_gpu_architecture_tour.ipynb` |

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
