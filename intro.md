# LLM Systems Cookbook

```{image} https://img.shields.io/badge/book-llm--systems--cookbook-2ea44f?style=for-the-badge
:alt: LLM Systems Cookbook
```

**A hands-on curriculum for modern LLM systems engineering.** 64 Jupyter
notebooks that teach inference, RAG, training, agents, serving,
evaluation, GPU programming, and production LLM patterns — each by
reimplementing the core technique from first principles, exercising a
real production library, or both, with deterministic numerical checks.

Written for a computer-science undergraduate who wants to go from
"I know what softmax is" to "I can reason about LLM serving economics,"
with no prior deep-learning background assumed.

```{admonition} What's new — June 2026
:class: note

- **GPT-5** achieves 100% on AIME 2026 and leads Chatbot Arena (Elo 1,561); **GPT-5.5
  Thinking** retires the o-series into a single router that auto-selects fast vs extended
  chain-of-thought inference.
- **Gemini 3.5 Pro** in limited Vertex preview — 2 M-token context, "Deep Think"
  reasoning mode, frontier multimodal understanding; GA expected before end of June 2026.
  **Gemini 3.1 Pro** leads GPQA-Diamond at 94.3% ($2 / $12 per M tokens).
- **Llama 4** (April 5, 2026): Scout (10 M-token context, 2,600 tok/s, MLA + FP8 native)
  and Maverick (Chatbot Arena Elo ≈ 1,417) openly released by Meta; Behemoth in training.
- **Qwen 3.5** MoE (397 B / 17 B active): Qwen3.5-plus scores 91.3% on AIME 2026 —
  strongest open-weight result on the math-reasoning leaderboard.
- **DeepSeek V4**: 1 T parameters, 1 M-token context; supported by Megatron Core, vLLM
  0.20, and SGLang 0.5.
- **Benchmark landscape**: MMLU, HumanEval, and GSM8K are saturated (≥ 92 %, ≥ 99 %,
  ≥ 99 %); active frontier is GPQA-Diamond (~94% SOTA), Humanity's Last Exam (~65%
  SOTA), SWE-bench Verified (93.9% SOTA), and AIME 2026.
- **TGI maintenance**: Hugging Face Text Generation Inference moves to maintenance mode —
  vLLM and SGLang are the recommended production engines. SGLang serves xAI Grok, Azure,
  LinkedIn, and Cursor across 400,000+ GPUs; **RadixArk** (SGLang company) raised a
  $100 M seed round at $400 M valuation.
- **MCP** crosses 200 server implementations; **ACP** merges into **A2A** under the
  Linux Foundation.
- **Microsoft Agent Framework v1.0** (April 2026): AutoGen and Semantic Kernel unified
  into one library. AutoGen 0.4 and Semantic Kernel enter maintenance mode; LangGraph
  surpasses CrewAI in GitHub stars.
- **Google Cloud Next 2026** (April 2026): **TPU 8t (Sunfish)** — 12.6 FP4 PFLOPS/chip,
  1 M-chip training clusters — and **TPU 8i (Zebrafish)** — 288 GB HBM, 10.1 FP4
  PFLOPS/chip, purpose-built for long-context reasoning — announced. **AMD MI400**
  (MI455X): 432 GB HBM4, 19.6 TB/s bandwidth, CES 2026 announcement.
- **Vera Rubin NVL72** entering volume production (Q3 2026): 288 GB HBM4, NVLink 6
  (3.6 TB/s per GPU), 336 B transistors; HBM4 supplied by SK Hynix, Samsung, and Micron.
  **NVLink Fusion** opens NVLink to third-party silicon (Intel, Qualcomm, Marvell et al.).
- **Blackwell B300** shipping since Q1 2026: 288 GB HBM3e, 8 TB/s bandwidth, 15 PFLOPS
  FP4 per card.
```

```{admonition} What's new — May 2026
:class: note

- **Glossary** extended with 2025–2026 terms: test-time compute, reasoning
  models, BitNet / ternary quantization, FP4 (Blackwell), VLM / SigLIP,
  NVIDIA Dynamo / NIXL, DoRA, ORPO, Vera Rubin GPU, PegaFlow, Gemini 3.5
  Flash, and more.
- **Curriculum spec** updated with the v0.3 roadmap: inference-time scaling
  notebook, BitNet/sub-2-bit serving notebook, a 5-notebook multimodal track,
  and a safety/red-teaming track.
- **Framework pins** refreshed to the May 2026 ecosystem (torch 2.7, vLLM
  0.20, SGLang 0.5 + XGrammar-2, TRL 0.26, PEFT 0.14, JAX 0.6).
- **Training track (v0.2):** six remaining notebooks (tensor parallel,
  pipeline parallel, LoRA/DoRA, QLoRA, DPO/ORPO, GRPO) are fully specified
  and in active development.
- **Google I/O 2026 (May 19):** Gemini 3.5 Flash GA — frontier Flash-tier
  speed with Pro-tier coding/agentic accuracy; 1 M-token context window;
  dynamic thinking on by default; Gemini Spark persistent 24/7 agent; ADK v1.0
  stable across Python, Go, Java, and TypeScript; A2A v1.0 in production at
  150+ organisations. **Gemini 3.5 Pro** expected June 2026.
- **OpenAI (May 5):** GPT-5.5 Instant is now the default ChatGPT model for
  all tiers — 52.5% fewer hallucinated claims, 30% more concise output,
  personalisation via past conversations, files, and Gmail. GPT-5.5 Thinking
  is OpenAI's unified reasoning model: a single router auto-selects between
  fast and extended chain-of-thought inference, retiring the standalone
  o-series (o4-mini et al.).
- **vLLM Model Runner V2 (MRV2):** opt-in via `VLLM_USE_V2_MODEL_RUNNER=1`
  in vLLM ≥ 0.20; replaces CPU PyTorch ops with GPU-native Triton kernels,
  delivering 56% higher throughput on GB200 and zero-synchronization
  speculative decoding (6.3% lower TPOT on 4×GB200).
- **Hardware roadmap:** NVIDIA Vera Rubin platform (announced GTC 2026) —
  Rubin GPU (288 GB HBM4, 50 PFLOPS FP4), Vera CPU (72-core ARM), NVLink 6;
  targeting 5× Blackwell inference throughput at 10× lower cost; H2 2026.
  Rubin CPX variant optimised for massive-context inference now documented in
  the glossary.
- **Serving infrastructure:** PegaFlow (Novita AI, May 2026) — GIL-free Rust
  external KV cache for vLLM/SGLang with GPU offload, SSD tiering, and RDMA
  cross-node KV sharing.
```

```{admonition} How to read this book
:class: tip

1. Click the **rocket 🚀 button** at the top-right of any chapter and
   choose *Colab* to run it end-to-end on a free T4 - no install.
2. Start with the **Foundations** part if you're new to GPU
   programming; otherwise jump directly into the track that interests
   you.
3. Each chapter follows the same six-step shape: motivation →
   reference paper → first-principles warm-up → real implementation
   → deterministic scoring → exercises + further reading.
4. The repo source is at
   [hassan11196/llm-systems-cookbook](https://github.com/hassan11196/llm-systems-cookbook).
```

## What's inside

```{grid} 1 1 2 2
:gutter: 3
:class-container: full-width

:::{grid-item-card} 🏗️ Foundations
:link: notebooks/07_gpu/index
:link-type: doc

GPU architecture, roofline analysis, Triton 101 and tiled matmul,
FlashAttention-2, fused RoPE + RMSNorm, torch.compile, Nsight
profiling, JAX sharding.

*9 chapters · mostly GPU, two CPU-friendly*
:::

:::{grid-item-card} ⚡ Inference engines
:link: notebooks/01_inference/index
:link-type: doc

KV cache, attention roofline, PagedAttention, continuous batching,
FA2-in-layer, radix prefix cache, speculative + tree decoding,
SARATHI chunked prefill, disaggregated serving.
FlashAttention-3 (H100/Hopper, FP8) and test-time compute scaling
are covered in the glossary; v0.2 will add dedicated chapters.

*10 chapters · one Ampere-only*
:::

:::{grid-item-card} 📈 Serving and scaling
:link: notebooks/05_serving/index
:link-type: doc

KV variants (MHA/GQA/MLA), compression (StreamingLLM/H2O/SnapKV),
KIVI, GPTQ/AWQ, SmoothQuant/FP8/NF4, QuaRot/SpinQuant, batching,
MoE, DistServe, observability + autoscaler. The 2025 **NVIDIA
Dynamo** framework (KV-aware routing, NIXL, SLO Planner) extends
these patterns to datacenter scale; the **Blackwell B200** GPU
(180 GB HBM3e, NV-FP4 tensor cores) sets the new hardware baseline.

*10 chapters · CPU-safe*
:::

:::{grid-item-card} 🔍 Retrieval-augmented generation
:link: notebooks/02_rag/index
:link-type: doc

Chunking strategies, FAISS indices, BM25/SPLADE/RRF, ColBERTv2
late interaction, two-stage reranking, HyDE, RAPTOR, GraphRAG,
RAGAS. Agentic RAG and corrective RAG patterns are documented in
the glossary; production examples appear in Part VIII.

*9 chapters · CPU-safe*
:::

:::{grid-item-card} 🤖 Agent frameworks
:link: notebooks/04_agents/index
:link-type: doc

ReAct from scratch, structured outputs, LangGraph state machines,
DSPy/MIPROv2, MCP server+client, AutoGen vs CrewAI,
τ-bench/SWE-bench evaluation. Patterns apply directly to OpenAI
Agents SDK, Google ADK, Pydantic AI, smolagents, and the
**Microsoft Agent Framework 1.0** (AutoGen + Semantic Kernel,
April 2026).

*7 chapters · CPU-safe*
:::

:::{grid-item-card} 🎯 Evaluation
:link: notebooks/06_eval/index
:link-type: doc

Perplexity, MMLU + calibration, HumanEval pass@k, LLM-as-judge bias,
Arena Elo + Bradley-Terry, NIAH/RULER, contamination detection,
lm-eval vs Inspect AI.

*8 chapters · CPU-safe*
:::

:::{grid-item-card} 🔧 Training and fine-tuning
:link: notebooks/03_training/index
:link-type: doc

Mixed precision + gradient accumulation + checkpointing,
DDP vs FSDP2. Tensor parallel, pipeline parallel, LoRA/DoRA,
QLoRA, DPO/ORPO, GRPO (DeepSeek-R1 style), and RLVR (reinforcement
learning from verifiable rewards) are fully specified in
CURRICULUM_SPEC.md and in active v0.2 development.

*2 of 8 chapters shipped · 6 specified, in progress*
:::

:::{grid-item-card} 🚢 Production patterns
:link: notebooks/08_production/index
:link-type: doc

Anthropic SDK prompt caching, LiteLLM multi-provider routing,
native tool-use, structured outputs head-to-head (tool-use /
Instructor / Outlines), hybrid RAG with citations, MCP server +
client, DSPy 3 + MIPROv2, Inspect AI, GPU pricing.

*9 chapters · CPU-safe, replay-mode fixtures included*
:::

:::{grid-item-card} 📚 Reference
:link: CURRICULUM_SPEC
:link-type: doc

The full curriculum specification with per-chapter scoring
thresholds, paper citations, and prerequisite DAG.

*If a chapter and the spec disagree, the chapter is authoritative.*
:::

:::{grid-item-card} 🔭 Coming in v0.3
:link: CURRICULUM_SPEC
:link-type: doc

**Test-time compute & reasoning models** — best-of-N, process reward
models, budget forcing (S1 "wait" trick), MCTS over reasoning steps.

**BitNet & sub-2-bit serving** — ternary `BitLinear`, ternary-aware
training, `bitnet.cpp` CPU inference.

**Multimodal / VLM track** — SigLIP 2 fine-tune, LLaVA projection,
VLM evaluation (MMBench / POPE / HallusionBench), Phi-4-Multimodal
document QA, Vision-Language-Action overview.

**Safety & red-teaming** — HarmBench, constitutional self-critique,
hard-list watermarking, toxicity scoring.

*Fully specified in CURRICULUM_SPEC.md · contributions welcome*
:::
```

## Prerequisites

- **Programming.** Comfortable reading Python; a little PyTorch helps
  but isn't required.
- **Math.** High-school algebra; the notebooks explain every ML-
  specific equation the first time it appears.
- **Computer architecture.** Helpful to have seen cache hierarchy and
  memory bandwidth concepts once; the GPU architecture tour
  re-introduces them from first principles.
- **Hardware.** Free Colab T4 is enough for 61 of 64 chapters. Three
  chapters (the two FA2 Triton kernel notebooks in 01_inference/05 and
  07_gpu/04, plus 07_gpu/07 Nsight profiling) note their requirements
  in their chapter header.

## Citation

If you use this cookbook in teaching or research, please cite:

```bibtex
@misc{llm_systems_cookbook,
  author  = {Ahmed, Muhammad Hassan},
  title   = {The LLM Systems Cookbook},
  year    = {2026},
  url     = {https://github.com/hassan11196/llm-systems-cookbook},
}
```

## Acknowledgements

Paper authors cited in each chapter. The notebook style draws from
[Project Pythia cookbooks](https://projectpythia.org/), the
[EECS 245 Jupyter Book](https://notes.eecs245.org/), and the
[IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).
Scoring-harness pattern inspired by the `pytest` + `nbmake`
community. MIT-licensed; contributions welcome.
