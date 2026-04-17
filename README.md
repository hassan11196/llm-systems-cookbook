# llm-infra-lab

**Hands-on notebooks that teach modern LLM systems engineering — inference,
RAG, training, agents, serving, evaluation, and GPU programming — from first
principles.** Every notebook explains the idea, builds it in code, and scores
itself with numerical checks.

Written for a CS undergrad who wants to go from "I know what softmax is" to
"I can reason about LLM serving economics." No prior deep-learning experience
assumed.

---

## ▶️ Run it in Colab — one click, no install

Click any badge to open that notebook in a free Colab session. CPU-only
notebooks run without switching runtime; GPU ones auto-select a free T4.

| | Notebook | Topic | Runtime | Hardware |
|---|---|---|---|---|
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/07_gpu/01_gpu_architecture_tour.ipynb) | **Start here →** GPU architecture tour | bandwidth vs compute, roofline | 10 min | T4 or CPU |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/06_eval/01_perplexity_from_scratch.ipynb) | Perplexity from scratch | bits-per-char, sliding window | 1 min | CPU |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/05_serving/01_roofline_analysis.ipynb) | Roofline for LLM serving | prefill vs decode intensity | 30 s | CPU |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/01_inference/01_autoregressive_decoding_kv_cache.ipynb) | KV cache = memoised Fibonacci | autoregressive decoding | 20 min | T4 |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/03_training/01_mixed_precision_accum_checkpointing.ipynb) | Mixed precision + checkpointing | bf16, grad accum, memory tricks | 10 min | T4 or CPU |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/02_rag/01_chunking_strategies.ipynb) | Chunking for retrieval | embeddings as soft hashing | 3 min | CPU |
| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-infra-lab/blob/main/notebooks/04_agents/01_react_from_scratch.ipynb) | ReAct = three-line REPL | parser, tools, agent loop | 30 s | CPU |

> **Suggested order:** GPU arch → Roofline → KV cache → Perplexity → the rest
> in any order. The first three share the same mental model (bandwidth vs
> compute); the rest apply it to specific problems.

---

## Tracks

Each of the seven tracks below has one hello-world notebook ready today
(above). The full spec in [`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md) covers
**61 notebooks total**; the remaining 54 are authored in follow-up work.

| Track | # | Focus |
|---|---|---|
| 01 inference | 10 | KV cache, PagedAttention, FlashAttention, continuous batching, speculative decoding, disaggregated serving |
| 02 rag       |  9 | Chunking, dense/sparse/hybrid retrieval, ColBERT, reranking, HyDE, RAPTOR, GraphRAG |
| 03 training  |  8 | Mixed precision, FSDP2, tensor/pipeline parallel, LoRA, QLoRA, DPO, GRPO |
| 04 agents    |  7 | ReAct, structured outputs, LangGraph, DSPy, MCP, multi-agent, τ-bench/SWE-bench |
| 05 serving   | 11 | Roofline, KV variants, compression, quantization, MoE, disaggregation, SLO/autoscaling |
| 06 eval      |  8 | Perplexity, MMLU, HumanEval, LLM-as-judge, Arena, NIAH/RULER, contamination |
| 07 gpu       |  8 | GPU arch, Triton, FlashAttention, fused kernels, torch.compile, Nsight, JAX sharding |

## How each notebook is structured

Every notebook follows the same shape so you always know where you are:

1. **One-paragraph motivation** — the problem in plain English.
2. **CS analogy / intuition** — the "aha" hook (KV cache ≈ memoised
   Fibonacci, embedding ≈ soft hash, agent ≈ REPL).
3. **First-principles warm-up** — tiny code that proves the limit cases
   by hand before touching any library.
4. **Real implementation** — the actual technique, narrated cell by cell.
5. **Self-scoring cell** — deterministic numeric checks that pass or fail,
   written to `scores/{track}_{NN}_{slug}.json`.
6. **Exercises + further reading** — what to try next and which papers
   to read.

---

## Install locally (optional)

You only need this if you want to run outside Colab. Pick the tracks you
care about — the optional-dependency groups let incompatible pins coexist
in separate virtualenvs:

```bash
pip install -e ".[inference,serving,gpu]"        # inference + serving + GPU kernels
pip install -e ".[rag,eval,agents,dev]"          # CPU-only path
pip install -e ".[inference,rag,training,agents,serving,eval,gpu,jax,dev]"  # everything
```

Then:

```bash
make install-dev           # harness + linting
make warm-cache            # pre-pull HuggingFace models (optional)
jupyter lab notebooks/
python -m scoring.run_all  # aggregate all scores/*.json into a markdown table
```

## Hardware

Most notebooks run on a free Colab T4. Three exceptions are declared in
their own notebook headers:

- `01_inference/05_flashattention2_triton.ipynb` — Ampere or newer
- `07_gpu/04_triton_flashattention.ipynb` — Ampere or newer
- `07_gpu/07_nsight_profiling.ipynb` — local GPU or Colab Pro

## Layout

```
llm-infra-lab/
├── CURRICULUM_SPEC.md         # per-notebook specification
├── src/llm_infra_lab/         # shared helpers (hardware_check, set_seed, ModelSpec registry)
├── scoring/                   # Scorer harness + aggregator + unit tests
├── notebooks/                 # per-track directories, one hello-world each
├── scripts/                   # fetch_data.py, warm_cache.py
└── .github/workflows/         # CI (lint + CPU-safe notebook execution)
```

## License

MIT
