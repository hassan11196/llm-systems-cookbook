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
  132 on H100). Each SM runs many warps concurrently and has its own
  L1 cache and shared memory. First introduced in
  {doc}`notebooks/07_gpu/01_gpu_architecture_tour`.

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

INT8 / INT4
  8- or 4-bit integer weights. Used for post-training weight
  quantization (GPTQ, AWQ).

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

LoRA
  Low-Rank Adaptation. Freeze the base model; learn rank-r update
  matrices `B·A` added to selected linears. Reduces trainable params
  100-1000×.

QLoRA
  LoRA on top of a 4-bit-quantized base model. Enables fine-tuning
  65B-class models on one consumer GPU.

DPO
  Direct Preference Optimization. Learns from preference pairs without
  an explicit reward model, by reweighting the base policy's log-probs.

RLHF
  Reinforcement Learning from Human Feedback. Classic 3-stage recipe:
  SFT → reward model → PPO. DPO and GRPO are simpler alternatives.
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

MCP
  Model Context Protocol. An open standard for exposing tools and
  data sources to LLM clients over JSON-RPC. Covered in
  {doc}`notebooks/04_agents/05_mcp_server_client`.

DSPy
  A framework that compiles high-level program-like agents into
  optimised prompt+weights pairs. Covered in
  {doc}`notebooks/04_agents/04_dspy_3_miprov2`.
```
