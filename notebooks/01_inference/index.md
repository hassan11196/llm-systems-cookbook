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
```
