# Inference engines

```{admonition} What you'll learn in this part
:class: tip

- Why decoding is memory-bound and prefill is compute-bound, and
  what that means for every optimisation that follows.
- The KV-cache byte formula and how paging reduces waste from 80 %
  to under 5 %.
- Scheduler designs: continuous batching, radix prefix cache,
  SARATHI chunked prefill, disaggregated prefill/decode.
- Speculative decoding, Medusa heads, EAGLE tree speculation.
- FlashAttention-2 drop-in for any transformer layer.

For H100/Hopper context: **{term}`FlashAttention-3`** (arXiv 2407.08608)
extends these ideas with async TMA pipelining and FP8 support, reaching
740 TFLOPs/s, 2× faster than FA2 on the same chip. The 2024-2026
**{term}`reasoning model`** wave (o1, o3, DeepSeek-R1, GPT-5.5 Thinking)
introduces a new axis (**{term}`test-time compute`** /
**{term}`inference-time scaling`**) where spending more generation tokens
trades compute for accuracy; see the glossary for orientation.
GPT-5.5 Thinking (May 2026) is the first unified model to auto-route
between fast and extended chain-of-thought at inference, retiring the
standalone o-series numbering.

On the production side, **vLLM's Model Runner V2 (MRV2)**, enabled via
`VLLM_USE_V2_MODEL_RUNNER=1` in vLLM ≥ 0.20, delivers 56% higher
throughput on GB200 via GPU-native Triton kernels and zero-CPU-sync
speculative decoding. **{term}`SGLang`** is a competitive or
leading alternative for MoE models, structured outputs, and speculative
decoding workloads; SGLang v0.5 on GB300 NVL72 demonstrated 25× the
inference performance of the H100 baseline (Feb 2026).
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

1. `01_autoregressive_decoding_kv_cache`: the KV cache is memoised
   Fibonacci, with numbers.
2. `02_attention_roofline`: attention from scratch + its
   arithmetic intensity.
3. `03_pagedattention_block_allocator`: vLLM's block allocator in
   pure Python.
4. `04_continuous_batching_orca`: iteration-level scheduling.
5. `05_flashattention2_triton`: FA2 kernel wrapped in an
   `nn.Module`. **Ampere+ for kernel execution**; falls back on
   CPU.
6. `06_radix_prefix_cache`: SGLang's radix tree.
7. `07_speculative_decoding`: the rejection rule and its
   closed-form speedup.
8. `08_medusa_eagle_tree_speculation`: tree verification of draft
   candidates.
9. `09_sarathi_chunked_prefill`: co-schedule decodes with prefill
   chunks.
10. `10_disaggregated_prefill_decode`: KV handoff between separate
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

**11: Inference-time compute scaling** will implement best-of-N
with a process reward model, MCTS-style tree search over reasoning
steps, and the "wait" budget-forcing trick from S1 (2501.10921).
This is the dominant 2025-2026 quality-scaling axis for reasoning
models like DeepSeek-R1, GPT-5.5 Thinking, and QwQ.
```

## Recent developments (2025-2026)

- **EAGLE-3** (arXiv 2503.01840): multi-layer hidden-state fusion for draft candidates; surpasses EAGLE-2 on long-context tasks. Referenced as a stretch goal in `08_medusa_eagle_tree_speculation`.
- **QuantSpec** (arXiv 2502.10424, Apple ML Research): combines self-speculative decoding with a hierarchical quantized KV cache, targeting both latency and memory simultaneously. It is a natural v0.2 addition to the speculative decoding notebooks.
- **IndexCache** (2025): reuses token-level attention indices across transformer layers and requests, cutting compute 15-25% on conversational workloads with no measurable quality loss.
- **Llama 4 Scout 10M-token context**: demonstrates practical very-long-context inference; stretch goal for `06_radix_prefix_cache` (cache hit-rate analysis at 1M+ token prefixes).
- **Mercury 2** (Inception): a production diffusion-based LLM that generates tokens in parallel rather than auto-regressively, reaching speeds above 1,000 tokens/sec. It contradicts the assumption that all LLM inference is sequential, and is relevant to the roofline analysis in chapters 1-2 as an alternative compute model.
- **Inference now ~2/3 of AI compute**: production inference has grown from roughly 1/3 of AI compute in 2023 to ~2/3 in 2026. Inference cost has fallen roughly 1,000× since late 2022, making large-scale agentic loops economically viable. Baseten's $1.5B funding round (a $13B valuation) confirms open-source model serving is a major capital destination.
- **Cloudflare Infire**: Cloudflare's custom inference engine that runs LLMs across multiple GPUs more efficiently, reduces memory usage, and starts models more quickly. It ships alongside **Unweight**, a weight compression system that reduces LLM model sizes 15-22% without measurable accuracy loss.
- **Mixed-precision KV-cache quantization**: 2026 research (e.g. PM-KVQ, ICLR 2026) moves past uniform INT8/FP8 KV quantization toward progressive, per-layer/per-token bit allocation, spending more bits where retrieval accuracy is most sensitive. vLLM's `--kv-cache-dtype fp8` now runs the QK/ScoreV attention matmuls themselves in FP8, not just cache storage — directly relevant to the KV cache mechanics in `02_kv_cache_from_scratch` and the roofline analysis in `03_attention_roofline`.
