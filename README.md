# LLM Systems Cookbook

[![Book](https://img.shields.io/badge/Read-The%20Book-2ea44f?style=for-the-badge)](https://hassan11196.github.io/llm-systems-cookbook/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-systems-cookbook/blob/main/intro.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A hands-on curriculum for modern LLM systems engineering.** 64 Jupyter
notebooks covering inference, retrieval, training, agents, serving,
evaluation, GPU programming, and production LLM patterns. Each notebook
either reimplements a core technique from first principles or exercises
a real production library, and self-scores with numerical checks.

Target audience: a computer-science undergrad who wants to go from
"I know what softmax is" to "I can reason about LLM serving
economics." No prior deep-learning background assumed.

> **May 2026 update:** Glossary extended with test-time compute,
> reasoning models, BitNet/FP4, VLM/SigLIP, NVIDIA Dynamo/NIXL, DoRA,
> and ORPO. Framework pins refreshed (torch 2.7, vLLM 0.20, TRL 0.26,
> SGLang 0.5 + XGrammar-2). v0.3 roadmap published: inference-time
> scaling, BitNet serving, VLM track, and safety/red-teaming track.
> **Google I/O 2026 (May 19):** Gemini 3.5 Flash GA — frontier Flash
> performance, 1 M-token context, 4× faster output than peer frontier
> models; Gemini Spark persistent agent; ADK v1.0 stable across Python,
> Go, Java, TypeScript; A2A v1.0 in production at 150+ organisations.
> **Gemini 3.5 Pro** expected June 2026 (announced at I/O 2026).
> **OpenAI (May 5):** GPT-5.5 Instant became the default ChatGPT model
> for all tiers — 52.5% fewer hallucinated claims vs GPT-5.3 Instant,
> 30% more concise output, and personalization via past conversations,
> files, and Gmail. GPT-5.5 Thinking (unified successor to the o-series)
> integrates extended chain-of-thought into the GPT-5 family; the
> standalone o-series (o4-mini etc.) enters retirement.
> **Inference (vLLM MRV2):** Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`,
> vLLM ≥ 0.20) achieves 56% higher throughput on GB200 via GPU-native
> Triton kernels and zero-synchronization speculative decoding.
> **Hardware:** NVIDIA Vera Rubin platform announced (GTC 2026) — Rubin
> GPU with 288 GB HBM4, 50 PFLOPS FP4, NVLink 6; 5× Blackwell
> inference throughput at 10× lower token cost; H2 2026 availability.
> **Infrastructure:** PegaFlow (Novita AI, May 2026) — Rust-core
> external KV cache for vLLM/SGLang with GPU offload, SSD tiering, and
> RDMA cross-node sharing. See [`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md).

## ▶️ Run in Colab - no install

Every notebook has an Open-in-Colab badge at the top. Landing page:

👉 **[hassan11196.github.io/llm-systems-cookbook](https://hassan11196.github.io/llm-systems-cookbook/)**

Start with [`07_gpu/01_gpu_architecture_tour`](notebooks/07_gpu/01_gpu_architecture_tour.ipynb)
if you're new to GPU programming; otherwise jump to the track you
want from the book's landing page.

## The eight parts

| Part | Chapters | Theme |
|---|---|---|
| **I · Foundations** | [9](notebooks/07_gpu/index.md) | GPU architecture, Triton, roofline |
| **II · Inference engines** | [10](notebooks/01_inference/index.md) | KV cache, PagedAttention, speculative decoding, SARATHI, disaggregated prefill/decode |
| **III · Serving & scaling** | [10](notebooks/05_serving/index.md) | KV variants/compression, KIVI, GPTQ/AWQ, FP8, QuaRot, MoE, DistServe/Dynamo |
| **IV · Training** | [2 of 8](notebooks/03_training/index.md) | Mixed precision, FSDP2; LoRA/DoRA, QLoRA, DPO/ORPO, GRPO in v0.2 |
| **V · Retrieval-augmented generation** | [9](notebooks/02_rag/index.md) | Chunking, FAISS, BM25/SPLADE, ColBERTv2, reranking, HyDE, RAPTOR, GraphRAG, RAGAS |
| **VI · Agent frameworks** | [7](notebooks/04_agents/index.md) | ReAct, structured outputs, LangGraph, DSPy/MIPROv2, MCP, AutoGen, τ-bench/SWE-bench |
| **VII · Evaluation** | [8](notebooks/06_eval/index.md) | Perplexity, MMLU, HumanEval, LLM-as-judge, Arena Elo, NIAH/RULER, contamination |
| **VIII · Production patterns** | [9](notebooks/08_production/index.md) | SDK prompt caching, LiteLLM routing, native tool use, structured outputs, hybrid RAG, MCP server, DSPy MIPROv2, Inspect AI, GPU pricing |
| **v0.3 planned** | ~10 | Test-time compute, BitNet/FP4 serving, VLM track (SigLIP 2, LLaVA, VLM eval), safety & red-teaming |

Full curriculum spec: [`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md). v0.1 ships 64 notebooks; the remaining 6 training-track notebooks are specified for v0.2; the v0.3 roadmap adds ~10 more.

### Production patterns track

The Part VIII notebooks are different from the rest of the cookbook: instead of
re-implementing a technique from scratch, each one exercises a real production
LLM library on a real task. Every notebook works in two modes:

- **LIVE** — when the relevant API key is set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  etc.), hits the real provider and reports fresh numbers.
- **Replay** — without keys, loads recorded responses from `notebooks/08_production/_fixtures/`
  so the notebook still runs end-to-end on a fresh Colab. The fixtures are real
  responses from a previous run, not mocks.

All nine notebooks pass their `s.check()` assertions in replay mode (46 checks total).

## Run it locally

```bash
# Book build (jupyter-book + sphinx-book-theme + myst-nb)
pip install -e ".[book,dev]"
make book          # builds into _build/html/
open _build/html/index.html

# Or run notebooks directly (pick only the tracks you need)
pip install -e ".[inference,serving,gpu]"          # GPU-bound parts
pip install -e ".[rag,eval,agents,dev]"            # fully CPU-safe
pip install -e ".[production,dev]"                  # Part VIII (anthropic, litellm, instructor, outlines)
jupyter lab notebooks/
python -m scoring.run_all                          # aggregate scores/*.json
```

## Hardware

Most notebooks run on a free Colab T4. Three require more:

- `01_inference/05_flashattention2_triton.ipynb` - Ampere+ GPU
- `07_gpu/04_triton_flashattention.ipynb` - Ampere+ GPU
- `07_gpu/07_nsight_profiling.ipynb` - local GPU or Colab Pro

Each chapter declares its requirements in its header cell.

## Gated models

A few notebooks (training track, some inference notebooks) download
`meta-llama/Llama-3.2-1B[-Instruct]` from HuggingFace. That repo is
**gated** - you must accept Meta's Llama community license on the
model page and then authenticate once with `huggingface-cli login`
(or set `HF_TOKEN`) before the notebook will run. Notebooks that
only use open models (SmolLM2, Qwen2.5, Qwen3, Phi-3.5) need no
token.

`Qwen/Qwen3-1.7B` is added to the model registry as of v0.2. It
supports hybrid thinking/non-thinking via `/think` and `/no_think`
prefixes; use `/no_think` for latency-sensitive benchmarks. Larger
Llama 4 models (Scout 17B-16E, Maverick 17B-128E) are documented in
the GPU pricing notebook and in the model registry but are too large
for free Colab T4 — they require A100 80 GB+ or H100.

## Layout

```
llm-systems-cookbook/
├── intro.md                      # book landing page
├── _toc.yml                      # eight-part table of contents
├── _config.yml                   # Jupyter Book config (launch buttons, theme)
├── environment.yml               # Binder / conda-forge reproducible env
├── CITATION.cff                  # academic citation metadata
├── CURRICULUM_SPEC.md            # per-chapter specification
├── notebooks/                    # the 63 chapters, grouped by track
│   ├── 01_inference/index.md
│   ├── 02_rag/index.md
│   ├── …
│   ├── 08_production/            # production-pattern notebooks + recorded fixtures
│   │   ├── _fixtures/            # JSON of real API responses for replay mode
│   │   └── index.md
├── src/llm_systems_cookbook/     # shared helpers (hardware_check, seed, etc.)
├── scoring/                      # Scorer harness + aggregator + unit tests
├── scripts/                      # fetch_data.py, warm_cache.py
└── .github/workflows/            # CI (lint + CPU-safe notebook execution + Pages deploy)
```

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

Format and structure inspired by
[Project Pythia cookbooks](https://projectpythia.org/),
[EECS 245 notes](https://notes.eecs245.org/), and
[IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).

## License

MIT.
