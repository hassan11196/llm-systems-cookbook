# The LLM Systems Cookbook

[![Read the Book](https://img.shields.io/badge/Read-The%20Book-2ea44f?style=for-the-badge)](https://hassan11196.github.io/llm-systems-cookbook/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-systems-cookbook/blob/main/notebooks/07_gpu/01_gpu_architecture_tour.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://github.com/hassan11196/llm-systems-cookbook/actions/workflows/book.yml/badge.svg)](https://github.com/hassan11196/llm-systems-cookbook/actions/workflows/book.yml)

A hands-on curriculum for **LLM systems engineering**. It has 64 Jupyter notebooks covering inference optimization, retrieval-augmented generation, agent frameworks, serving and scaling, evaluation, GPU programming, and production LLM patterns. Each notebook reimplements a core technique from first principles or exercises a production library, with self-scoring numerical checks throughout.

**[Read the book online →](https://hassan11196.github.io/llm-systems-cookbook/)**

> **Who this is for:** engineers and researchers who want to understand how production LLM systems work under the hood, not just how to call an API.

## Table of contents

- [What is this?](#what-is-this)
- [What makes this different](#what-makes-this-different)
- [Who is it for?](#who-is-it-for)
- [Topics covered](#topics-covered)
- [Learning paths](#learning-paths)
- [Quick start](#quick-start)
- [Run locally](#run-locally)
- [Hardware requirements](#hardware-requirements)
- [Production patterns track](#production-patterns-track)
- [Repository layout](#repository-layout)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## What is this?

The LLM Systems Cookbook is a structured curriculum covering the engineering side of large language models: GPU kernel programming, KV-cache mechanics, production RAG pipelines, AI agent frameworks, LLM evaluation harnesses, and deployment patterns.

Each notebook follows a consistent six-step structure:

1. **Motivation**: why the technique matters in production LLM systems
2. **Reference**: the paper or spec that introduced it
3. **First-principles warm-up**: reimplement the core idea in ≤50 lines
4. **Real implementation**: exercise the production library or GPU kernel
5. **Deterministic scoring**: numerical checks via a built-in harness
6. **Exercises**: stretch goals for deeper exploration

The curriculum focuses on building reliable LLM applications: finding where systems break, measuring them, and scaling them cost-effectively.

## What makes this different

Most LLM tutorials show you how to call an API. This cookbook teaches you what is happening inside the API and how to reason about the systems that serve it.

Every notebook answers a specific engineering question with a numerical result, not just a prose explanation. You reimplement FlashAttention-2 as a Triton GPU kernel, benchmark the roofline model with real memory bandwidth numbers, build a RAG pipeline you can evaluate end-to-end with RAGAS, and run PagedAttention's block allocator in pure Python. The built-in scoring harness verifies that each notebook produces correct numerical outputs before you move on.

The result is a curriculum that covers the production LLM stack (GPU kernel programming, KV-cache mechanics, multi-provider serving, LLM observability, and agent evaluation) in a single structured sequence, with code you can run on a free Colab T4.

## Who is it for?

The primary audience is a computer-science undergraduate who knows what softmax is but hasn't studied LLM serving economics or GPU kernel programming. No prior deep-learning background assumed.

In practice the notebooks are useful for:

- Engineers moving from ML research into **production LLM systems**
- Backend engineers adding **RAG pipelines** or **AI agent frameworks** to their stack
- Practitioners who want to understand **LLM inference optimization** rather than just calling an API
- Teams evaluating **LLM serving infrastructure** and deployment options
- Researchers who need hands-on practice with **LLM evaluation methodology**

## Topics covered

| Part | Chapters | Theme |
|---|---|---|
| **I · Foundations** | [9](notebooks/07_gpu/index.md) | GPU architecture and roofline model, Triton 101, tiled matmul, FlashAttention-2 kernel, fused RoPE/RMSNorm, torch.compile, Nsight profiling, JAX sharding |
| **II · Inference engines** | [10](notebooks/01_inference/index.md) | Autoregressive decoding, KV cache mechanics, attention roofline, PagedAttention, continuous batching, FlashAttention-2, radix prefix cache, speculative decoding, SARATHI chunked prefill, disaggregated prefill/decode |
| **III · Serving & scaling** | [10](notebooks/05_serving/index.md) | KV variants (MHA/GQA/MLA), KV compression (StreamingLLM/H2O/SnapKV), KIVI, GPTQ/AWQ quantization, SmoothQuant/FP8, QuaRot, batching strategies, MoE serving, DistServe/Dynamo, LLM observability |
| **IV · Training** | [2 of 8](notebooks/03_training/index.md) | Mixed precision training, gradient accumulation, checkpointing, DDP vs FSDP2; LoRA/DoRA, QLoRA, DPO/ORPO, GRPO in v0.2 |
| **V · Retrieval-augmented generation** | [9](notebooks/02_rag/index.md) | Chunking strategies, FAISS dense retrieval, BM25/SPLADE/RRF hybrid search, ColBERTv2 late interaction, two-stage reranking, HyDE query rewriting, RAPTOR hierarchical retrieval, GraphRAG, RAGAS evaluation |
| **VI · Agent frameworks** | [7](notebooks/04_agents/index.md) | ReAct from scratch, structured outputs (3 approaches), LangGraph state machines, DSPy/MIPROv2 prompt optimization, Model Context Protocol (MCP), AutoGen vs CrewAI, agent evaluation with τ-bench/SWE-bench |
| **VII · Evaluation** | [8](notebooks/06_eval/index.md) | Perplexity, MMLU + calibration, HumanEval pass@k, LLM-as-judge bias, Arena Elo + Bradley-Terry, NIAH/RULER needle-in-haystack, contamination detection, lm-eval vs Inspect AI |
| **VIII · Production patterns** | [9](notebooks/08_production/index.md) | Anthropic SDK prompt caching, LiteLLM multi-provider routing, native tool use, structured outputs comparison, hybrid RAG with citations, MCP server + client, DSPy MIPROv2, Inspect AI eval harness, GPU cost modeling |
| **v0.3 planned** | ~10 | Test-time compute, BitNet/sub-2-bit serving, multimodal/VLM track (SigLIP 2, LLaVA, VLM eval), safety and red-teaming |

Full curriculum specification with per-chapter scoring thresholds and paper citations: [`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md).

v0.1 ships 64 notebooks; the remaining 6 training-track notebooks are specified for v0.2; the v0.3 roadmap adds ~10 more.

## Learning paths

**New to GPU programming?**
Start with [Part I (Foundations)](notebooks/07_gpu/index.md), which covers the GPU architecture tour, roofline model, and Triton kernels, then move to [Part II (Inference engines)](notebooks/01_inference/index.md).

**Know PyTorch but want to understand LLM serving?**
Start at [05_serving/01_roofline_kv_budget](notebooks/05_serving/index.md) to understand serving economics, then work through Part II.

**Building a RAG system?**
Go straight to [Part V (RAG)](notebooks/02_rag/index.md). Production RAG code with real API fixtures is in [Part VIII](notebooks/08_production/index.md).

**Adding AI agents to your stack?**
[Part VI (Agent frameworks)](notebooks/04_agents/index.md) covers ReAct, LangGraph, DSPy, MCP, and multi-agent patterns from first principles.

**Evaluating LLM quality?**
[Part VII (Evaluation)](notebooks/06_eval/index.md) covers every major benchmark and evaluation methodology in use today, from perplexity to LLM-as-judge to NIAH/RULER.

## Quick start

Every notebook has an **Open in Colab** badge. Click it to run on a free T4 GPU with no local install needed.

**[→ Browse the full book](https://hassan11196.github.io/llm-systems-cookbook/)**

Start with [`07_gpu/01_gpu_architecture_tour`](notebooks/07_gpu/01_gpu_architecture_tour.ipynb) if you're new to GPU programming; otherwise jump to the track you want from the landing page.

## Run locally

```bash
# Install book dependencies
pip install -e ".[book,dev]"

# Build the HTML book
make book
open _build/html/index.html

# Run only the tracks you need (mix and match)
pip install -e ".[inference,serving,gpu]"    # GPU-bound parts
pip install -e ".[rag,eval,agents,dev]"      # CPU-safe
pip install -e ".[production,dev]"           # Part VIII (uses real API clients)

jupyter lab notebooks/
python -m scoring.run_all                    # Aggregate scores/*.json
```

## Hardware requirements

Most notebooks run on a free Colab T4. Three require more:

| Notebook | Requirement |
|---|---|
| `01_inference/05_flashattention2_triton.ipynb` | Ampere+ GPU (A100 / H100) |
| `07_gpu/04_triton_flashattention.ipynb` | Ampere+ GPU |
| `07_gpu/07_nsight_profiling.ipynb` | Local GPU or Colab Pro |

Each notebook declares its hardware requirement in its header cell.

**Gated models:** A few notebooks use `meta-llama/Llama-3.2-1B` which requires accepting Meta's license on Hugging Face and running `huggingface-cli login` (or setting `HF_TOKEN`). Notebooks using open models (SmolLM2, Qwen2.5, Qwen3, Phi-3.5) need no token.

## Production patterns track

Part VIII is different from the rest of the cookbook: instead of reimplementing from scratch, each notebook exercises a real production LLM library on a practical task.

Each Part VIII notebook works in two modes:

- **LIVE**: when the relevant API key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.), it hits the real provider and reports fresh numbers.
- **Replay**: without keys, it loads recorded responses from `notebooks/08_production/_fixtures/` so the notebook still runs end-to-end on a fresh Colab. The fixtures are real responses from a previous run, not mocks.

All nine notebooks pass their `s.check()` assertions in replay mode (46 checks total).

## Repository layout

```
llm-systems-cookbook/
├── intro.md                      # Book landing page
├── _toc.yml                      # Eight-part table of contents
├── _config.yml                   # Jupyter Book configuration
├── CURRICULUM_SPEC.md            # Per-chapter specification with paper citations
├── glossary.md                   # Cross-cutting terminology reference (100+ entries)
├── notebooks/                    # 64 chapters grouped by track
│   ├── 01_inference/             # Inference engine internals (10 notebooks)
│   ├── 02_rag/                   # Retrieval-augmented generation (9 notebooks)
│   ├── 03_training/              # Training and fine-tuning (2 shipped, 6 in v0.2)
│   ├── 04_agents/                # AI agent frameworks (7 notebooks)
│   ├── 05_serving/               # Serving and scaling (10 notebooks)
│   ├── 06_eval/                  # Evaluation methodology (8 notebooks)
│   ├── 07_gpu/                   # GPU programming and Triton (8 notebooks)
│   └── 08_production/            # Production LLM patterns + replay fixtures (9 notebooks)
├── src/llm_systems_cookbook/     # Shared utilities (model registry, dataset loaders)
├── scoring/                      # Self-scoring harness + tests
├── scripts/                      # Data fetching, fixture refresh utilities
└── .github/workflows/            # CI: lint, notebook execution, GitHub Pages deploy
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch naming conventions, notebook hygiene, commit style, and the scoring harness API.

The v0.2 training track (6 notebooks: tensor parallel, pipeline parallel, LoRA/DoRA, QLoRA, DPO/ORPO, GRPO) and the v0.3 roadmap (test-time compute, VLM track, safety/red-teaming) are the highest-priority areas for contributions.

## Citation

```bibtex
@misc{llm_systems_cookbook,
  author  = {Ahmed, Muhammad Hassan},
  title   = {The LLM Systems Cookbook},
  year    = {2026},
  url     = {https://github.com/hassan11196/llm-systems-cookbook},
}
```

## Acknowledgements

Format and structure inspired by [Project Pythia cookbooks](https://projectpythia.org/), [EECS 245 notes](https://notes.eecs245.org/), and [IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).

## License

MIT. See [LICENSE](LICENSE).
