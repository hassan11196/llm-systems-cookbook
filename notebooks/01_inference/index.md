# Inference engines

```{admonition} What you'll learn in this part
:class: tip

- Why decoding is memory-bound and prefill is compute-bound — and
  what that means for every optimisation that follows.
- The KV-cache byte formula and how paging reduces waste from 80 %
  to under 5 %.
- Scheduler designs: continuous batching, radix prefix cache,
  SARATHI chunked prefill, disaggregated prefill/decode.
- Speculative decoding, Medusa heads, EAGLE tree speculation.
- FlashAttention-2 drop-in for any transformer layer.

For H100/Hopper context: **{term}`FlashAttention-3`** (arXiv 2407.08608)
extends these ideas with async TMA pipelining and FP8 support, reaching
740 TFLOPs/s — 2× faster than FA2 on the same chip. The 2024–2026
**{term}`reasoning model`** wave (o1, o3, DeepSeek-R1, GPT-5.5 Thinking)
introduces a new axis — **{term}`test-time compute`** /
**{term}`inference-time scaling`** — where spending more generation tokens
trades compute for accuracy; see the glossary for orientation.
GPT-5.5 Thinking (May 2026) is the first unified model to auto-route
between fast and extended chain-of-thought at inference, retiring the
standalone o-series numbering.

On the production side, **vLLM's Model Runner V2 (MRV2)** — enabled via
`VLLM_USE_V2_MODEL_RUNNER=1` in vLLM ≥ 0.20 — delivers 56% higher
throughput on GB200 via GPU-native Triton kernels and zero-CPU-sync
speculative decoding. **vLLM v0.22.0** (May 29, 2026) extends MRV2 with
an experimental Rust frontend (DP Supervisor), 28.9% latency reduction
via Cutlass FP8 kernels, and multi-tier KV offload (CPU/filesystem/
Mooncake disk); **v0.22.1** (June 5, 2026) adds JetBrains Mellum v2
support and patches DeepSeek-V4 initialization. **{term}`SGLang`**
v0.5.12.post1 (May 26, 2026) adds HiCache hierarchical KV + SSD offload,
EAGLE-3 speculative decoding, and fixes a DeepSeek V4 accuracy regression
on B200/B300; SGLang v0.5 on GB300 NVL72 demonstrated 25× the inference
performance of the H100 baseline (Feb 2026).
```


## Key terms used in this part

- **{term}`prefill`** and **{term}`decode`** are the two inference
  phases; most optimizations here target one phase more than the other.
- **{term}`KV cache`** and **{term}`PagedAttention`** are the memory
  primitives behind modern LLM serving.
- **{term}`continuous batching`**, **{term}`chunked prefill`**, and
  **{term}`disaggregated prefill/decode`** are scheduler/system design
  patterns used to improve throughput and tail latency.
- **{term}`speculative decoding`** uses a draft/verify pattern to reduce
  target-model decode work.

## Reading order

Prerequisites: Part I (GPU architecture tour + roofline).

1. `01_autoregressive_decoding_kv_cache` — the KV cache is memoised
   Fibonacci, with numbers.
2. `02_attention_roofline` — attention from scratch + its
   arithmetic intensity.
3. `03_pagedattention_block_allocator` — vLLM's block allocator in
   pure Python.
4. `04_continuous_batching_orca` — iteration-level scheduling.
5. `05_flashattention2_triton` — FA2 kernel wrapped in an
   `nn.Module`. **Ampere+ for kernel execution**; falls back on
   CPU.
6. `06_radix_prefix_cache` — SGLang's radix tree.
7. `07_speculative_decoding` — the rejection rule and its
   closed-form speedup.
8. `08_medusa_eagle_tree_speculation` — tree verification of draft
   candidates.
9. `09_sarathi_chunked_prefill` — co-schedule decodes with prefill
   chunks.
10. `10_disaggregated_prefill_decode` — KV handoff between separate
    GPU pools.

```{seealso}
Part III turns the compute/memory tradeoffs here into serving-level
economics: goodput, SLO attainment, autoscaling.

{term}`NVIDIA Dynamo` is the 2025 production successor to the
hand-rolled disaggregated shm approach in chapter 10; the notebook
documents the Dynamo upgrade path in its exercises.
```

```{admonition} Coming in v0.3
:class: note

**11 — Inference-time compute scaling** will implement best-of-N
with a process reward model, MCTS-style tree search over reasoning
steps, and the "wait" budget-forcing trick from S1 (2501.10921).
This is the dominant 2025-2026 quality-scaling axis for reasoning
models like DeepSeek-R1, GPT-5.5 Thinking, and QwQ.
```

## Recent developments (2025–2026)

- **EAGLE-3** (arXiv 2503.01840): multi-layer hidden-state fusion for draft candidates; surpasses EAGLE-2 on long-context tasks. Referenced as a stretch goal in `08_medusa_eagle_tree_speculation`.
- **QuantSpec** (arXiv 2502.10424, Apple ML Research): combines self-speculative decoding with a hierarchical quantized KV cache, targeting both latency and memory simultaneously — a natural v0.2 addition to the speculative decoding notebooks.
- **IndexCache** (2025): reuses token-level attention indices across transformer layers and requests, cutting compute 15-25% on conversational workloads with no measurable quality loss.
- **Llama 4 Scout 10M-token context**: demonstrates practical very-long-context inference; stretch goal for `06_radix_prefix_cache` (cache hit-rate analysis at 1M+ token prefixes).
