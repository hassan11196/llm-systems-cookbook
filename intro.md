# The LLM Systems Cookbook

```{image} https://img.shields.io/badge/book-llm--systems--cookbook-2ea44f?style=for-the-badge
:alt: LLM Systems Cookbook
```

**A hands-on curriculum for modern LLM systems engineering.** 64 Jupyter
notebooks that teach inference, RAG, training, agents, serving,
evaluation, GPU programming, and production LLM patterns. Each chapter
reimplements the core technique from first principles, exercises a
real production library, or both, with deterministic numerical checks.

Written for a computer-science undergraduate who wants to go from
"I know what softmax is" to "I can reason about LLM serving economics,"
with no prior deep-learning background assumed.

```{admonition} How to read this book
:class: tip

1. Click the **rocket 🚀 button** at the top-right of any chapter and
   choose *Colab* to run it end-to-end on a free T4, with no install.
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

**Test-time compute & reasoning models**: best-of-N, process reward
models, budget forcing (S1 "wait" trick), MCTS over reasoning steps.

**BitNet & sub-2-bit serving**: ternary `BitLinear`, ternary-aware
training, `bitnet.cpp` CPU inference.

**Multimodal / VLM track**: SigLIP 2 fine-tune, LLaVA projection,
VLM evaluation (MMBench / POPE / HallusionBench), Phi-4-Multimodal
document QA, Vision-Language-Action overview.

**Safety & red-teaming**: HarmBench, constitutional self-critique,
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

```{admonition} What's new in July 2026
:class: note

- **MCP 2026-07-28**: the Model Context Protocol's largest revision since
  launch ships this month — a stateless protocol core (no more
  `Mcp-Session-Id`), the `initialize`/`initialized` handshake removed
  entirely, authorization rebuilt on standard OAuth/OIDC RFCs, and a new
  extensions framework for independently-versioned protocol additions.
- **Anthropic model refresh**: Claude Sonnet 5 (June 30) and Claude Fable 5
  (GA July 1) lead SWE-bench Pro at 63.2% and 80.3% respectively, at
  roughly half the per-token price of their predecessors.
- **Benchmark saturation**: as of July 2, 2026, 37% of the 154
  percentage-scaled benchmarks tracked by BenchLM.ai are saturated
  (top model ≥ 90%) — GSM8K is effectively solved (99%), MMLU sits at
  93%. GPQA Diamond, SWE-bench Verified/Pro, and Humanity's Last Exam
  remain the discriminating frontier evaluations; see the updated
  benchmark table in [Part VII](notebooks/06_eval/index.md).
- **NVIDIA Vera Rubin DSX AI Factory** reference design and the
  **Omniverse DSX Blueprint** reach general availability, packaging the
  Vera Rubin platform into a rack-to-datacenter build/simulate/operate
  workflow for continuously-running inference deployments.
- **Glossary** extended with 2025 to 2026 terms: test-time compute, reasoning
  models, BitNet / ternary quantization, FP4 (Blackwell), VLM / SigLIP,
  NVIDIA Dynamo / NIXL, DoRA, ORPO, Vera Rubin GPU, PegaFlow, Gemini 3.5
  Flash, Claude Sonnet 5 / Fable 5, and more.
- **Curriculum spec** updated with the v0.3 roadmap: inference-time scaling
  notebook, BitNet/sub-2-bit serving notebook, a 5-notebook multimodal track,
  and a safety/red-teaming track.
- **Framework pins** refreshed to the May 2026 ecosystem (torch 2.7, vLLM
  0.20, SGLang 0.5 + XGrammar-2, TRL 0.26, PEFT 0.14, JAX 0.6).
- **Training track (v0.2):** six remaining notebooks (tensor parallel,
  pipeline parallel, LoRA/DoRA, QLoRA, DPO/ORPO, GRPO) are fully specified
  and in active development.
- **Google I/O 2026 (May 19):** Gemini 3.5 Flash GA, with Flash-tier
  speed and Pro-tier coding/agentic accuracy; 1 M-token context window;
  dynamic thinking on by default; Gemini Spark persistent 24/7 agent; ADK v1.0
  stable across Python, Go, Java, and TypeScript; A2A v1.0 in production at
  150+ organisations. **Gemini 3.5 Pro** expected June 2026.
- **OpenAI (May 5):** GPT-5.5 Instant is now the default ChatGPT model for
  all tiers, with 52.5% fewer hallucinated claims, 30% more concise output,
  and personalisation via past conversations, files, and Gmail. GPT-5.5 Thinking
  is OpenAI's unified reasoning model: a single router auto-selects between
  fast and extended chain-of-thought inference, retiring the standalone
  o-series (o4-mini et al.).
- **vLLM Model Runner V2 (MRV2):** opt-in via `VLLM_USE_V2_MODEL_RUNNER=1`
  in vLLM ≥ 0.20; replaces CPU PyTorch ops with GPU-native Triton kernels,
  delivering 56% higher throughput on GB200 and zero-synchronization
  speculative decoding (6.3% lower TPOT on 4×GB200).
- **Hardware roadmap:** NVIDIA Vera Rubin platform (announced GTC 2026):
  Rubin GPU (288 GB HBM4, 50 PFLOPS FP4), Vera CPU (72-core ARM), NVLink 6;
  targeting 5× Blackwell inference throughput at 10× lower cost; H2 2026.
  Rubin CPX variant optimised for massive-context inference now documented in
  the glossary.
- **Serving infrastructure:** PegaFlow (Novita AI, May 2026): GIL-free Rust
  external KV cache for vLLM/SGLang with GPU offload, SSD tiering, and RDMA
  cross-node KV sharing.
```

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
