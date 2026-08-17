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

```{admonition} What's new in August 2026
:class: note

- **Week of August 17, 2026 refresh**: The first **Humanity's Last Exam**
  leaderboard snapshot naming Anthropic's newest models shows **Claude
  Fable 5** leading at 55.5% (August 11), just ahead of **Claude Opus 5**
  (54.9%) and **GPT-5.6 Sol** (49.5%) — the benchmark table in
  [Part VII](notebooks/06_eval/index.md) and the HLE glossary entry are
  updated accordingly. **Gemini 3.5 Pro**'s delay continues past the
  rumored August 12 date, with reporting pointing to coding-performance
  shortfalls and a disappointing training-data refresh that have left it
  "months behind schedule"; Google shipped **Gemini 3.7 Flash** instead
  on August 13, extending the Flash tier while Pro stays stuck in limited
  preview. xAI released **Grok 4.6** (long-running agents, deeper coding,
  500K-token context, $2/$0.50/$6 per M tokens input/cached/output) and
  added it to GitHub Copilot and Cursor; Grok 4.7 (2.1T parameters) is
  expected within weeks. OpenAI gave **GPT-5.6 Sol** a user-facing effort
  slider (August 6) and previewed an **Ultrafast mode** (up to 14× faster,
  August 13); o3 retires from ChatGPT August 26 after a 90-day sunset, and
  OpenAI expanded its Daybreak cybersecurity initiative with a new
  **GPT-5.6-Cyber** model for vetted defenders. **NVIDIA Vera Rubin**'s
  initial shipments went out in July to Microsoft and Google (Meta also
  holds allocation), with volume shipments ramping through the rest of
  2026 — noted in [Part IV](notebooks/05_serving/index.md). ByteDance
  shipped **Seed 2.1 Turbo** (August 10). The EU AI Act's **Article 50**
  transparency obligations took effect August 2, 2026, requiring
  disclosure of AI interaction and machine-readable marking of
  AI-generated content — relevant context for teams deploying the
  production notebooks in [Part VIII](notebooks/08_production/index.md)
  into the EU.
- **Week of August 10, 2026 refresh**: OpenAI teased its next model,
  **Astra**, on August 1 — not with a launch, but by publishing
  machine-checked Lean 4 proofs for ten mathematics and theoretical-CS
  problems that had stood open for a decade or more (including an
  explicit non-sofic group construction), for roughly $2,000 in
  inference cost; Astra remains unreleased and is framed as an
  extension of long-horizon, multi-agent test-time reasoning. Sakana
  AI's **Fugu-Ultra v1.1** (a multi-model orchestration system, not a
  single trained network) now leads the GPQA-Diamond leaderboard at
  95.5% (August 7), with the top three models clustered within 0.9
  points — confirming GPQA-Diamond has crossed into saturation at the
  top. Alibaba shipped **Qwen3.8-Max** (August 3, 2.4T-parameter MoE)
  as a closed-API release beating GPT-5.6 Sol Max and Claude Fable 5 on
  OSWorld-Verified computer use, with open weights following August 12
  — breaking the closed-only pattern of prior Qwen3.x releases.
  **Gemini 3.5 Pro** has now slipped a fourth time past its original
  June target and remains unreleased as of August 10, with the latest
  rumor pointing to August 12. Anthropic began assembling an internal
  AI chip design team (August 5) to co-design hardware with future
  models, and confirmed that Claude Sonnet 5's introductory $2/$10
  pricing ends August 31, 2026 — standard $3/$15 pricing (plus a newer,
  more token-hungry tokenizer) takes effect September 1, relevant to
  the GPU/API cost-modeling notebook in
  [Part VIII](notebooks/08_production/index.md). Meta shipped **Muse
  Spark 1.2** (August 6).
- **Week of August 3, 2026 refresh**: Meta's **Muse Spark 1.1** (a
  multimodal reasoning model for agentic workflows, closed US-only
  preview) ties OpenAI's GPT-5.6 Luna at 51 on the Artificial Analysis
  Intelligence Index v4.1. Alibaba previewed **Qwen3.8-Max** at the World
  AI Conference in Shanghai (July 19); the full Qwen3.8 release is
  expected this month, with Qwen 4.0 targeted for September — Qwen3.7
  remains closed-weight. DeepSeek shipped a **DeepSeek-V4-Flash-0731**
  refresh on July 31. **Benchmark saturation has widened**: MMLU,
  HumanEval, and MBPP no longer meaningfully separate frontier models;
  GPQA Diamond is approaching saturation at the very top but still
  differentiates the 60-90% band where most procurement decisions live;
  **Humanity's Last Exam (HLE)** is emerging as the primary frontier
  differentiator, currently led by Grok 4 at 50.7%. **NVIDIA Vera Rubin**
  is ramping into full production — NVIDIA claims 35× inference
  performance-per-watt and 10× more revenue per trillion-parameter model
  versus Blackwell — with first cloud availability expanding to AWS,
  Google Cloud, Microsoft, and OCI alongside cloud partners CoreWeave,
  Lambda, Nebius, and Nscale. **TurboQuant** (ICLR 2026) is now a major
  reference point in KV-cache quantization research, as KV-cache memory
  has become the binding constraint for long-context serving.
- **MCP 2026-07-28 (finalized)**: the Model Context Protocol's largest
  revision since launch shipped as the final spec on July 28, 2026 — a
  stateless protocol core (no more `Mcp-Session-Id`, no
  `initialize`/`initialized` handshake), Multi Round-Trip Requests,
  header-based routing, cacheable list results, authorization rebuilt on
  standard OAuth/OIDC RFCs, and a reverse-DNS-namespaced extensions
  framework. The first two extensions riding that framework are the
  **Tasks extension** (long-running async tool calls via `tasks/get` /
  `tasks/update` / `tasks/cancel`) and **MCP Apps** (server-rendered
  interactive UIs in sandboxed iframes). Tier 1 SDKs (Python, TypeScript,
  Go, C#) now ship stable 2026-07-28 support and see close to half a
  billion downloads a month, with the TypeScript and Python SDKs each
  past 1 billion total downloads. The **Enterprise-Managed Authorization**
  extension is stable and adopted by Anthropic, Microsoft, and Okta; X
  (formerly Twitter) shipped a hosted MCP server for its platform API in
  July.
- **New frontier entrants**: xAI released **Grok 4.5** (July 8) as its
  first model built specifically for coding and agentic work, priced
  over 60% below Claude Opus 4.8 or GPT-5.5 while landing fourth on the
  Artificial Analysis Intelligence Index. OpenAI released the **GPT-5.6**
  family — Sol, Terra, and Luna — on July 9; Sol sets a new state of the
  art on the Artificial Analysis Coding Agent Index (80, edging out
  Claude Fable 5) at roughly a third of the cost and under half the
  output tokens. Anthropic followed on July 24 with **Claude Opus 5**,
  which holds Opus 4.8's $5/$25-per-M-token price while more than
  doubling its score on the Frontier-Bench v0.1 agentic terminal-coding
  eval (43.3% vs 21.1%) — ahead of both GPT-5.6 Sol (34.4%) and Claude
  Fable 5 (33.7%) on that eval, at a third to half Fable 5's per-token
  cost; it carries a May 2026 knowledge cutoff, the freshest of any model
  in Anthropic's lineup.
- **Late-July open-weight wave**: Moonshot AI's **Kimi K3** (2.8T-parameter
  MoE) edged past Claude Opus 4.8 on Artificial Analysis's independent
  ranking at its July 16 launch, and shipped its full 594 GB MXFP4
  safetensors release on Hugging Face on July 27 under a Modified MIT
  license — the largest open-weight release to date. Z.ai's **GLM-5.2**
  (744B MoE) is the new top open-weight model overall, scoring 91.2% on
  GPQA Diamond and 62.1% on SWE-bench Pro at a fraction of frontier API
  pricing. **Inkling**, the first flagship model from Mira Murati's
  Thinking Machines Lab, shipped open-weight around July 15 as a
  general-purpose reasoning and coding model.
- **Anthropic model refresh**: Claude Sonnet 5 (June 30) and Claude Fable 5
  (GA July 1) lead SWE-bench Pro at 63.2% and 80.3% respectively, and
  Claude Opus 5 (July 24) now tops Frontier-Bench v0.1 agentic coding —
  all three at roughly half the per-token price of their predecessors.
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
- **Agent frameworks (Q2–Q3 2026):** the busiest quarter since agent
  frameworks began shipping. The Claude Agent SDK added hierarchical
  subagent spawning (up to 3 levels deep), fallback model chains, and a
  community MCP tool marketplace. CrewAI 1.14.6 (June 11) added pluggable
  memory/knowledge/RAG/flow backends, a Chat API, and native Snowflake
  Cortex support. Pydantic AI V2 (June 23) shipped a harness-first
  redesign with capabilities as a core primitive, and LlamaIndex
  Workflows 1.0 landed June 22. LangGraph followed with per-node timeouts,
  a `DeltaChannel` for incremental state updates, and a typed v2 streaming
  API.
- **Glossary** extended with 2025 to 2026 terms: test-time compute, reasoning
  models, BitNet / ternary quantization, FP4 (Blackwell), VLM / SigLIP,
  NVIDIA Dynamo / NIXL, DoRA, ORPO, Vera Rubin GPU, PegaFlow, Gemini 3.5
  Flash, Claude Sonnet 5 / Fable 5, and more.
- **Curriculum spec** updated with the v0.3 roadmap: inference-time scaling
  notebook, BitNet/sub-2-bit serving notebook, a 5-notebook multimodal track,
  and a safety/red-teaming track.
- **Framework pins** refreshed to the July 2026 ecosystem (torch 2.7, vLLM
  0.25, SGLang 0.5.15 + XGrammar-2, TRL 0.26, PEFT 0.14, JAX 0.6).
- **Training track (v0.2):** six remaining notebooks (tensor parallel,
  pipeline parallel, LoRA/DoRA, QLoRA, DPO/ORPO, GRPO) are fully specified
  and in active development.
- **Google I/O 2026 (May 19):** Gemini 3.5 Flash GA, with Flash-tier
  speed and Pro-tier coding/agentic accuracy; 1 M-token context window;
  dynamic thinking on by default; Gemini Spark persistent 24/7 agent; ADK v1.0
  stable across Python, Go, Java, and TypeScript; A2A v1.0 in production at
  150+ organisations. **Gemini 3.5 Pro** has since slipped past its original
  June 2026 target for a third time — Google DeepMind reportedly rebuilt the
  base model after it fell short of internal hallucination and reliability
  goals — and remains unreleased as of July 27. In its place, Google shipped
  three Flash-tier models on July 21: **Gemini 3.6 Flash** (the new default
  model; 17% fewer output tokens than 3.5 Flash, a March 2026 knowledge
  cutoff, 58.7% on SWE-bench Pro, and output pricing cut to $7.50/M tokens),
  **Gemini 3.5 Flash-Lite** (high-throughput, low-latency tier at
  $0.30/$2.50 per M tokens), and **Gemini 3.5 Flash Cyber** (a
  vulnerability-finding model limited to a government/partner pilot).
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
  targeting 5× Blackwell inference throughput at 10× lower cost. Rubin is
  now in full production and, per NVIDIA (July 21), "going gigascale":
  NVL72 racks are live at CoreWeave, Google Cloud, Microsoft Azure, OCI,
  and Mistral, alongside NVIDIA Cloud Partners Lambda, Nebius, and Nscale.
  A July 16 partnership with Japan's Noetra Corp — backed by Japan's METI
  — will build a national Vera Rubin AI factory (13,750 Vera CPUs, 27,500
  Rubin GPUs) for the country's FRONTia Project. Rubin CPX variant
  optimised for massive-context inference now documented in the glossary.
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
