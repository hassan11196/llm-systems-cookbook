# Glossary

A reference for the cross-cutting terms that show up in more than one
chapter. Entries are deliberately short - enough to orient you, not a
substitute for reading the chapter that introduces the concept. Each
term includes a pointer to the notebook where it first appears.

Notebook cells reference this page with the MyST `{term}` role, so
clicking a linked term (e.g. **{term}`prefill`**) jumps here.

## Hardware and GPUs

```{glossary}
HBM
  High Bandwidth Memory. The DRAM stack on a GPU. Modern chips reach
  1-3 TB/s of HBM bandwidth, which is one or two orders of magnitude
  faster than CPU DRAM but still the bottleneck for LLM decode. First
  introduced in {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

SM
  Streaming Multiprocessor. An NVIDIA GPU is a grid of SMs (40 on T4,
  108 on A100, 132 on H100 SXM5 / 114 on H100 PCIe). Each SM runs
  many warps concurrently and has its own L1 cache and shared memory.
  First introduced in {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

SIMT
  Single Instruction, Multiple Threads. The execution model where 32
  threads (a warp) execute the same instruction in lockstep on
  different data. Like SIMD but with lane-level divergence allowed.
  First introduced in {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

warp
  A group of 32 threads that an SM schedules as one unit. The smallest
  piece of work the hardware actually runs. First introduced in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

tensor core
  A specialized matrix-multiply unit on modern NVIDIA GPUs. Supported
  precisions expand per generation: Volta (cc 7.0) added FP16; Turing
  (cc 7.5) added INT8/INT4; Ampere (cc 8.0) added BF16 and TF32; Ada
  Lovelace (cc 8.9) and Hopper (cc 9.0) added FP8. Delivers most of
  the advertised tensor-core TFLOPs; non-tensor-core FP32 is far
  slower. First introduced in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

compute capability
  A `major.minor` version tag on NVIDIA GPUs (e.g. 7.5 on T4, 9.0 on
  H100) that gates feature availability. FlashAttention-2 needs ≥ 8.0;
  FP8 tensor cores need ≥ 8.9.

kernel
  A function that runs on the GPU. You launch kernels from the host;
  each launch has fixed overhead, so batching work per launch matters.

ThunderKittens
  A tile-based GPU kernel DSL from Hazy Research / Stanford (2024).
  The atomic unit of work is the 16×16 tile that maps to one H100
  Tensor Core instruction; full kernels fit in ~100 lines of C++.
  Their FlashAttention implementation reaches 855 TFLOPs/s — matching
  FlashAttention-3 — and outperforms best Triton implementations of
  Mamba-2, RoPE, and LayerNorm by 6–14×.

Blackwell / GB200
  NVIDIA's 2025 GPU architecture (compute capability 10.0). Adds NV-FP4
  (4-bit float) tensor cores and a new NVLink Switch interconnect.
  Two GB200 dies share 192 GB of HBM3e on a single NVL2 board,
  reaching 14.4 TB/s of aggregate memory bandwidth.
```

## Roofline, throughput, latency

```{glossary}
arithmetic intensity
  FLOPs executed per byte moved from HBM. Abbreviated AI or FLOPs/byte.
  Determines whether a kernel is memory- or compute-bound. Defined in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

ridge intensity
  The arithmetic intensity at the knee of the roofline, equal to
  `peak_compute / peak_bandwidth`. Kernels below the ridge are
  memory-bound; above it, compute-bound.

roofline
  A log-log plot with arithmetic intensity on the x-axis and throughput
  on the y-axis. The "roof" is `min(bw × AI, peak_compute)` - the
  achievable ceiling for any kernel on a given chip. Introduced in
  Williams, Waterman & Patterson (2009).

memory-bound
  A kernel whose runtime is dominated by HBM reads/writes, not math.
  LLM decode is the textbook example.

compute-bound
  A kernel whose runtime is dominated by math throughput, not data
  movement. Large-matmul prefill is the textbook example.

FLOPs
  Floating-point operations. A matmul of `(m, k) @ (k, n)` costs
  `2 × m × k × n` FLOPs.

TFLOPs
  Tera-FLOPs per second. Common unit for GPU compute throughput.

TTFT
  Time To First Token. The latency from request arrival to the first
  generated token - dominated by prefill.

TPOT
  Time Per Output Token. The steady-state decode latency - dominated
  by weight reads from HBM.

SLO
  Service Level Objective. A latency target (e.g. p95 TTFT < 1 s)
  that serving systems are designed and autoscaled against.
```

## Inference phases

```{glossary}
prefill
  The parallel forward pass over the entire input prompt. Computes
  hidden states and fills the KV cache for every prompt token. High
  arithmetic intensity - compute-bound. First introduced in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`; covered in depth
  in {doc}`notebooks/01_inference/01_autoregressive_decoding_kv_cache`.

decode
  The autoregressive loop that emits output tokens one at a time,
  reading the cached KV of prior tokens. Low arithmetic intensity -
  memory-bound. First introduced in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`; covered in depth
  in {doc}`notebooks/01_inference/01_autoregressive_decoding_kv_cache`.

autoregressive
  A generative model that conditions each new token on all previous
  ones. The sequential nature of decode is what makes LLMs memory-bound.

KV cache
  The per-layer Key and Value tensors cached during decode so attention
  doesn't recompute them for prior tokens. Size is linear in context
  length; its memory footprint is the dominant serving constraint.
  Introduced in {doc}`notebooks/01_inference/01_autoregressive_decoding_kv_cache`.

PagedAttention
  A KV-cache allocator that stores K/V in fixed-size blocks (like OS
  virtual memory pages) to eliminate fragmentation and enable sharing.
  Introduced by vLLM (Kwon et al. 2023). Covered in
  {doc}`notebooks/01_inference/03_pagedattention_block_allocator`.

continuous batching
  A scheduler (ORCA, vLLM) that admits and evicts requests at
  per-iteration granularity so short and long requests share GPU time
  efficiently, instead of waiting for a whole batch to finish.

chunked prefill
  Interleaving small prefill chunks with decode steps (SARATHI) so long
  prompts don't starve decoders. Improves tail latency under load.

disaggregated prefill/decode
  Running prefill and decode on separate GPU pools (DistServe) so they
  don't interfere. Prefill wants big batches and compute; decode wants
  low-latency weight reads.

speculative decoding
  Propose `k` draft tokens with a small cheap model, then verify them
  in one pass with the target model. Rejected tokens are redrafted.
  When accept rate is high, net speedup is 2-3×.

draft model
  The small model used to propose tokens in speculative decoding.

target model
  The model whose distribution speculative decoding must preserve
  bit-for-bit (via rejection sampling).

reasoning model
  An LLM trained with reinforcement learning to emit an extended
  chain-of-thought (CoT) trace before producing its final answer.
  Examples: OpenAI o1/o3, DeepSeek-R1, Qwen-QwQ. Accuracy scales
  with the compute budget allocated at inference, not just model size.

test-time compute
  Extra inference-time FLOPs spent on extended reasoning traces, best-
  of-N sampling, beam search, or process-reward-guided tree search.
  The dominant post-2024 scaling axis for hard reasoning tasks —
  often more FLOPs-efficient than proportionally scaling parameters.

inference-time scaling
  The empirical observation that harder tasks benefit from longer
  generation budgets (more "thinking" tokens). Motivates test-time
  compute strategies and the training recipe for reasoning models.
```

## Attention variants

```{glossary}
attention
  The operation `softmax(QKᵀ / √d) · V` that lets each position read
  from all others. O(N²) in sequence length by default; FlashAttention
  makes it O(N) in memory.

MHA
  Multi-Head Attention. Each of `H` heads has its own Q, K, V
  projections. Classic transformer. High KV memory at inference time.

GQA
  Grouped-Query Attention. Several query heads share one K/V head.
  Used by Llama-2-70B and most modern open models to shrink KV cache
  ~4-8×. Covered in
  {doc}`notebooks/05_serving/02_kv_cache_variants_mha_gqa_mla`.

MLA
  Multi-head Latent Attention (DeepSeek-V2). Compresses K and V into
  a low-rank latent and reconstructs heads on the fly. Smallest KV
  footprint of the three. Covered in
  {doc}`notebooks/05_serving/02_kv_cache_variants_mha_gqa_mla`.

FlashAttention
  A tiled, recomputation-based attention kernel that never materialises
  the `N × N` attention matrix. O(N²) FLOPs but O(N) HBM traffic.
  Covered in {doc}`notebooks/07_gpu/04_triton_flashattention`.

FlashAttention-3
  H100/Hopper-specific attention kernel (Shah et al., NeurIPS 2024;
  arXiv 2407.08608). Exploits Hopper's async TMA units and
  warp-specialization to pipeline compute and data movement, and adds
  block-wise FP8 quantization. Reaches 740 TFLOPs/s in FP16 (1.5–2×
  faster than FA2 on the same chip) and ~1.2 PFLOPs/s in FP8.

RoPE
  Rotary Position Embedding. Rotates Q and K by position-dependent
  angles so attention is translation-equivariant. Used by Llama,
  Qwen, Mistral.

RMSNorm
  Root-Mean-Square LayerNorm. A cheaper LayerNorm variant (no mean
  subtraction, no bias). Standard in modern LLMs.
```

## Precision and quantization

```{glossary}
FP32
  32-bit float. 4 bytes per value. Default for training in legacy
  frameworks. Mostly obsolete on modern hardware.

FP16
  IEEE half precision. 2 bytes. Fast on tensor cores but narrow
  dynamic range - used with loss scaling.

BF16
  BFloat16. Same 8-bit exponent as FP32, 7-bit mantissa. Wider range
  than FP16 at the same size. Standard for LLM training since 2022.

FP8
  8-bit float (E4M3 or E5M2 encoding). Half the bandwidth of FP16.
  Supported on Ada Lovelace (cc 8.9; RTX 4090, L4, L40) and Hopper
  (cc 9.0; H100) tensor cores. Covered in
  {doc}`notebooks/05_serving/06_smoothquant_fp8_nf4`.

FP4
  4-bit float. Native tensor-core support on Blackwell (NVIDIA GB200 /
  B200, cc 10.0). Delivers 2× the throughput of FP8 for inference
  with acceptable quality loss on most frontier models (ΔPPL < 0.5
  vs FP8 with block-scaling).

INT8 / INT4
  8- or 4-bit integer weights. Used for post-training weight
  quantization (GPTQ, AWQ).

INT2 / ternary
  2-bit or ternary {-1, 0, +1} weight encoding. Extreme compression
  reduces weight memory 8× vs FP16. KIVI covers 2-bit KV caches;
  BitNet covers ternary training-from-scratch models. Post-training
  2-bit (QuIP#, AQLM) with codebook lookup is an active research area.

NF4
  4-bit NormalFloat. A non-uniform 4-bit encoding tuned for normal-
  distributed weights; used by QLoRA.

GPTQ
  Post-Training Quantization via approximate second-order
  information. Reduces weights to 3-4 bits with small accuracy loss.
  Covered in {doc}`notebooks/05_serving/05_gptq_awq_weight_quant`.

AWQ
  Activation-aware Weight Quantization. Protects the small fraction of
  salient weight channels from quantization error.

SmoothQuant
  Scales activations and weights jointly so INT8 quantization works
  without fine-tuning.

QuaRot / SpinQuant
  Rotation-based quantization schemes that apply Hadamard rotations to
  flatten outlier distributions before quantizing. Covered in
  {doc}`notebooks/05_serving/07_quarot_spinquant_rotations`.

KV quantization
  Storing K and V in fewer bits per value (2, 4, or 8) to shrink the
  KV cache. Representative method: KIVI (asymmetric per-channel K,
  per-token V at 2 bits). Covered in
  {doc}`notebooks/05_serving/04_2bit_kv_quantization_kivi`.

KV eviction
  Shrinking the KV cache by *dropping* tokens rather than
  re-encoding them. StreamingLLM keeps attention sinks + a recent
  window; H2O keeps "heavy hitter" tokens by historical attention
  mass; SnapKV uses an observation window to score prompt tokens.
  Covered in
  {doc}`notebooks/05_serving/03_kv_compression_streamingllm_h2o_snapkv`.
```

## Batching, parallelism, serving

```{glossary}
static batching
  Pad all requests in a batch to the longest one and run them together.
  Simple but wasteful when lengths vary.

dynamic batching
  Combine requests that arrive within a small time window into one
  batch. Better utilisation than static, but still head-of-line blocks.

DDP
  Distributed Data Parallel. Each GPU holds a full copy of the model;
  gradients are all-reduced per step. Covered in
  {doc}`notebooks/03_training/02_ddp_vs_fsdp2`.

FSDP
  Fully Sharded Data Parallel. Shards weights, gradients, and optimizer
  state across GPUs (PyTorch's ZeRO-3). Halves memory vs DDP at the
  cost of an all-gather per forward pass. FSDP2 is the rewritten v2.

tensor parallel
  Split each matmul row-wise or column-wise across GPUs (Megatron
  style). Low latency but chatty; used within a node with NVLink.

pipeline parallel
  Split the model into sequential stages, one per GPU; pipeline micro-
  batches through them. Scales across nodes but adds bubble overhead.

expert parallel
  In Mixture-of-Experts models, shard experts across GPUs. Each token
  is routed to a handful of experts, so per-GPU compute is low even
  at huge parameter counts. Covered in
  {doc}`notebooks/05_serving/09_moe_expert_parallelism`.

MoE
  Mixture of Experts. A sparse architecture where each token activates
  only a few of many "expert" FFN blocks. Parameters scale decoupled
  from per-token FLOPs.
```

## Training

```{glossary}
mixed precision
  Compute forward/backward in FP16 or BF16 for speed; keep a master
  FP32 copy of weights and optimizer state for numerical stability.
  Covered in {doc}`notebooks/03_training/01_mixed_precision_accum_checkpointing`.

gradient accumulation
  Run `k` micro-batches and sum their gradients before stepping the
  optimizer. Emulates a larger batch without the memory cost.

activation checkpointing
  Recompute activations during the backward pass instead of storing
  them. Trades 1.3× compute for large memory savings.

SFT
  Supervised Fine-Tuning. Fine-tune on labelled (prompt, completion)
  pairs with the standard cross-entropy objective. The first stage of
  most alignment pipelines.

LoRA
  Low-Rank Adaptation. Freeze the base model; learn rank-r update
  matrices `B·A` added to selected linears. Reduces trainable params
  100-1000×.

DoRA
  Weight-Decomposed Low-Rank Adaptation. Splits each pre-trained weight
  into magnitude and direction components, then applies LoRA only to the
  direction, improving fine-tuning quality especially at very low ranks
  (r ≤ 8) without increasing inference cost.

QLoRA
  LoRA on top of a 4-bit-quantized base model. Enables fine-tuning
  65B-class models on one consumer GPU.

DPO
  Direct Preference Optimization. Learns from preference pairs without
  an explicit reward model, by reweighting the base policy's log-probs.

ORPO
  Odds-Ratio Preference Optimization. A single-stage SFT+alignment
  recipe that adds a log-odds-ratio penalty term to the cross-entropy
  loss, eliminating the need for a separate reference model. More
  memory-efficient than DPO.

GRPO
  Group Relative Policy Optimization. A reinforcement learning
  algorithm (DeepSeek-R1, 2501.12948) that estimates advantage from
  within-group reward statistics instead of a value network, enabling
  RL fine-tuning of reasoning models at lower compute than PPO. Covered
  in {doc}`notebooks/03_training/02_ddp_vs_fsdp2` prerequisites; fully
  specified in `CURRICULUM_SPEC.md` notebook 03_training/08.

RLHF
  Reinforcement Learning from Human Feedback. Classic 3-stage recipe:
  SFT → reward model → PPO. DPO and GRPO are simpler alternatives.

GRPO
  Group Relative Policy Optimization. A memory-efficient RL algorithm
  (DeepSeek, 2024) that replaces the PPO value network with group-level
  reward baselines: sample a group of G outputs, compute rewards, and
  normalize. Eliminates the critic model entirely. TRL ships
  `GRPOTrainer`; covered in
  {doc}`notebooks/03_training/08_grpo_deepseek_r1_style`.

RLVR
  Reinforcement Learning from Verifiable Rewards. Uses reward functions
  whose ground truth can be checked programmatically — math correctness,
  code test passing, structured-output validity — instead of a trained
  preference model. The post-training recipe behind DeepSeek-R1 and
  most 2025-2026 reasoning models.

DoRA
  Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024). Decomposes
  each weight matrix into magnitude and direction, updating both with
  LoRA-style efficiency. Typically outperforms LoRA at equal parameter
  budget; supported by PEFT ≥ 0.10.

BitNet
  A training-from-scratch quantization scheme (Microsoft, 2402.17764)
  where each weight is constrained to {-1, 0, +1} (ternary / 1.58-bit).
  BitNet b1.58-2B-4T (April 2025) is the first openly released 1.58-bit
  model at production scale (2B params, 4T tokens). Enables 15× better
  energy efficiency than FP16 equivalents on CPU inference.
```

## Retrieval

```{glossary}
RAG
  Retrieval-Augmented Generation. Fetch relevant documents and prepend
  them to the prompt, instead of relying solely on parametric memory.

chunking
  Splitting long documents into retrieval-sized pieces (by tokens,
  sentences, or semantic breakpoints). Covered in
  {doc}`notebooks/02_rag/01_chunking_strategies`.

dense retrieval
  Encode queries and documents into vectors; retrieve by cosine
  similarity. Covered in
  {doc}`notebooks/02_rag/02_faiss_dense_retrieval`.

sparse retrieval
  Lexical matching (BM25, TF-IDF) on tokenised terms. Still strong,
  especially hybridised with dense.

BM25
  Okapi BM25. The canonical sparse lexical retrieval score. Used as
  a baseline in every RAG paper. Covered in
  {doc}`notebooks/02_rag/03_bm25_splade_rrf_hybrid`.

SPLADE
  Sparse Lexical and Expansion model. A learned sparse retriever
  that outputs BERT-weighted term scores in the vocabulary.

RRF
  Reciprocal Rank Fusion. Combine multiple ranked lists by summing
  `1 / (k + rank)`. Simple and effective for hybrid retrieval.

ColBERT
  Late-interaction retriever. Scores a query by sum-of-max over per-
  token embeddings. Retrieval quality close to cross-encoders at a
  fraction of the cost. Covered in
  {doc}`notebooks/02_rag/04_colbertv2_late_interaction`.

reranking
  Re-score a retriever's top-k with a stronger (usually cross-encoder)
  model. Two-stage retrieval is standard in production.

agentic RAG
  A RAG pattern where a reasoning agent controls the retrieval loop:
  it decides whether to retrieve, what query to issue, whether the
  results are sufficient, and whether to iterate. Enables multi-hop
  reasoning, self-correction, and dynamic query reformulation (arXiv
  2501.09136 provides a 2025 survey).

corrective RAG
  A self-improvement variant (Shi et al. 2024) where a lightweight
  evaluator grades retrieved documents as relevant, ambiguous, or
  irrelevant and triggers a web-search fallback for low-quality
  retrievals before generation.

HyDE
  Hypothetical Document Embeddings. Generate a pseudo-answer with an
  LLM, embed that, retrieve against documents. Helps when the query
  is too short to embed well.

recall@k
  Fraction of relevant documents that appear in the top-k results.
  The primary retriever metric.

nDCG
  Normalised Discounted Cumulative Gain. A rank-weighted relevance
  metric. Rewards placing highly-relevant docs near the top.
```

## Evaluation

```{glossary}
perplexity
  `exp(mean cross-entropy)` over a held-out corpus. Lower is better.
  The standard language-modelling metric. Covered in
  {doc}`notebooks/06_eval/01_perplexity_from_scratch`.

calibration
  How well a model's predicted probabilities match actual correctness
  rates. A model is well-calibrated if "I'm 70% confident" is right
  70% of the time.

Expected Calibration Error (ECE)
  Bin predicted confidence scores and measure the weighted gap between
  confidence and empirical accuracy in each bin. Lower is better.

pass@k
  Probability that at least one of `k` code samples passes all unit
  tests. Standard HumanEval metric; unbiased estimator in Chen et al.
  2021. Covered in {doc}`notebooks/06_eval/03_humaneval_unbiased_pass_k`.

LLM-as-judge
  Using a strong LLM to score outputs of other models against a
  rubric. Cheap but biased (position, length, verbosity).

Elo
  Pairwise rating system from chess, adapted to LLM arenas. A model's
  Elo changes after each pairwise win/loss based on expected score.

Bradley-Terry
  The statistical model under Elo. Fitting BT to a pool of pairwise
  comparisons gives maximum-likelihood ratings.

NIAH
  Needle In A Haystack. A long-context probe: hide a fact in a long
  document and ask the model to retrieve it.

RULER
  A composite long-context benchmark that goes beyond NIAH with
  multi-key, multi-hop, and tracing tasks.

contamination
  When benchmark examples leak into training data, inflating scores.
  Detectable via membership-inference or length-canary tests.

GPQA
  Graduate-Level Google-Proof Q&A. 448 expert-authored questions in
  biology, chemistry, and physics that PhD-level domain experts answer
  correctly only ~65% of the time. Became a standard frontier benchmark
  as MMLU saturated in 2025.

HLE
  Humanity's Last Exam. A 2,500-question expert benchmark released Jan
  2025 by the Center for AI Safety and Scale AI. Covers math, science,
  and humanities at the level of PhD qualifying exams; frontier models
  scored below 10% on release, making it a long-term frontier target.

LiveCodeBench
  A contamination-resistant coding benchmark that continuously adds new
  problems from competitive programming contests (LeetCode, Codeforces,
  AtCoder) after the training cutoffs of all evaluated models.

ARC-AGI
  Abstraction and Reasoning Corpus for Artificial General Intelligence
  (François Chollet, 2019). Grid transformation tasks designed to
  require novel analogy-making; no model broke 5% until o3 reached
  96.7% in late 2024, after which ARC-AGI-2 was released as the
  successor frontier challenge.
```

## Agents

```{glossary}
ReAct
  Reasoning + Acting. A prompting pattern that interleaves thought,
  action, and observation steps. Still the baseline agent loop.
  Covered in {doc}`notebooks/04_agents/01_react_from_scratch`.

tool use
  Letting an LLM invoke external functions (search, code exec, APIs)
  via structured outputs. The foundation of all agent frameworks.

structured outputs
  Constraining LLM output to valid JSON or a typed schema. Three
  common methods: JSON mode, tool-call schemas, grammar-constrained
  decoding. Covered in
  {doc}`notebooks/04_agents/02_structured_outputs_three_ways`.

finite-state machine (FSM)
  A graph of allowed states and transitions used to constrain decoding
  to valid output formats. If a token would violate the graph, it is
  disallowed.

MCP
  Model Context Protocol. An open standard for exposing tools and
  data sources to LLM clients over JSON-RPC. Covered in
  {doc}`notebooks/04_agents/05_mcp_server_client`.

DSPy
  A framework that compiles high-level program-like agents into
  optimised prompt+weights pairs. Covered in
  {doc}`notebooks/04_agents/04_dspy_3_miprov2`.

A2A
  Agent-to-Agent Protocol. An open specification (Google, April 2025)
  for agents to discover each other via JSON "Agent Cards" and delegate
  subtasks over a REST interface. Horizontal complement to MCP: where
  MCP connects a single agent to tools/data, A2A connects agents to
  other agents. Merged under the Linux Foundation in late 2025.

handoff
  Transferring control and conversation state from one agent to another.
  The core primitive in the OpenAI Agents SDK (released March 2025);
  implemented as a specialized tool call `transfer_to_<agent>`.

guardrail
  A validation function that runs before or after an LLM call to
  enforce safety or format constraints without model retraining.
  First-class primitive in the OpenAI Agents SDK; analogous to
  middleware in web frameworks.

Pydantic AI
  An agent framework from the Pydantic team (2024) with FastAPI-style
  dependency injection and first-class Pydantic validation. Supports
  tool calls, structured outputs, streaming, and multi-agent graphs.

smolagents
  A minimalist agent library by Hugging Face (~1,000 lines of Python).
  Its `CodeAgent` generates executable Python snippets that invoke
  tools directly rather than emitting JSON tool-call objects, closing
  the execution loop in one step.

computer use
  A tool-use modality where an LLM controls a GUI: clicks, types, and
  takes screenshots to drive desktop applications. Standardised as a
  special tool type in the same protocol as JSON tool calls.
```

## Reasoning and inference-time scaling

```{glossary}
test-time compute
  Spending additional GPU flops at inference (rather than training)
  to improve output quality. Strategies include longer chain-of-thought,
  best-of-N sampling, beam search over reasoning steps, and parallel
  coordinated reasoning (PaCoRe). The dominant 2025 scaling axis.

reasoning model
  An LLM that emits an extended internal chain-of-thought
  ("thinking tokens") before its final answer. Trained with RL reward
  signals (GRPO, REINFORCE) on verifiable tasks. Representative
  models: DeepSeek-R1 (2501.12948), o1/o3 (OpenAI), QwQ-32B.

thinking tokens
  Tokens generated by a reasoning model during its latent scratchpad
  phase that are shown to the user but not part of the final answer.
  Budget scales linearly with problem difficulty; pruning techniques
  (MatryoshkaThinking) reduce unnecessary token spend.

best-of-N
  Generate N independent completions and return the one scored highest
  by a verifier or reward model. Simple but effective test-time scaling
  strategy; acceptance rate is `1 − (1 − p)^N` for iid Bernoulli pass.

parallel coordinated reasoning
  A test-time scaling technique (PaCoRe, 2601.05593) where multiple
  reasoning "threads" run in parallel and cross-coordinate their
  intermediate conclusions, achieving further quality gains beyond
  sequential chain-of-thought at the same token budget.
```

## Multimodal and vision

```{glossary}
VLM
  Vision-Language Model. An LLM extended with a vision encoder so it
  can process image or video inputs alongside text. Representative
  open models (2025-2026): Qwen2.5-VL, InternVL3, Phi-4-Multimodal,
  LLaVA-OneVision.

vision encoder
  The component of a VLM that maps a raw image into patch embeddings
  that the LLM backbone can attend to. Earlier systems used CLIP
  (ViT-L/14); the 2025 standard shifted to SigLIP 2.

SigLIP
  Sigmoid Loss for Language-Image Pre-training. A vision encoder from
  Google that replaces the softmax over the full batch with per-pair
  sigmoid, enabling larger batch sizes and better multilingual and
  localisation performance. SigLIP 2 (Feb 2025) became the default
  vision backbone for open VLMs (Qwen2.5-VL, PaliGemma 2).

cross-modal adapter
  A lightweight projection (linear or MLP) that aligns vision encoder
  patch embeddings into the LLM token embedding space. Also called a
  visual projector or connector module.

VLA
  Vision-Language-Action model. Extends a VLM with an action decoder
  for robot control — the model ingests camera observations and
  language instructions and predicts motor actions. Examples: NVIDIA
  Groot N1, Physical Intelligence π0.
```

## Serving infrastructure

```{glossary}
NVIDIA Dynamo
  An open-source distributed inference serving framework (announced
  GTC March 2025) designed for the disaggregated prefill/decode
  pattern. Supports multiple backends (vLLM, TensorRT-LLM, SGLang)
  and includes a Smart Router, SLA-based Planner, and integration with
  NIXL for GPU-to-GPU KV cache transfer at wire speed.

NIXL
  NVIDIA Inference Transfer Library. A point-to-point library for
  transferring KV cache tensors between GPUs over RDMA (InfiniBand /
  RoCE), TCP, NVMe-oF, or S3. Open-sourced alongside Dynamo at GTC
  2025. Replaces the `multiprocessing.SharedMemory` approach used in
  the DistServe prototype (see
  {doc}`notebooks/05_serving/10_disaggregated_serving_distserve`).

LMCache
  A community KV-cache sharing layer that integrates with vLLM V1 to
  bring production-grade prefill/decode disaggregation without
  requiring NVIDIA hardware-specific NIXL (though NIXL is optional for
  maximum bandwidth).

NVLink
  NVIDIA's high-bandwidth chip-to-chip interconnect. NVLink 4 (H100)
  provides 900 GB/s bidirectional; NVLink 5 (Blackwell GB200) reaches
  1.8 TB/s. Critical for tensor-parallel within a node.

Blackwell
  NVIDIA's 2025 GPU generation (GB200, B100, B200). Key additions:
  FP4 tensor cores (2× Hopper INT8 throughput), NVLink 5, and a
  10 TB/s chip-to-chip NVLink interconnect in the GB200 NVL72 rack
  configuration.
```
