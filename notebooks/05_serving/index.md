# Serving and scaling

```{admonition} What you'll learn in this part
:class: tip

- KV-cache variants (MHA → GQA → MLA) and when each is worth the
  quality tradeoff.
- KV compression (StreamingLLM, H2O, SnapKV) and 2-bit KV
  quantisation (KIVI).
- Weight quantisation: GPTQ + AWQ; SmoothQuant rescaling; FP8 and
  NF4 numeric formats; QuaRot / SpinQuant rotations.
- Batching strategies side by side (static / dynamic / continuous /
  chunked-prefill).
- Mixture-of-experts routing, capacity factors, and load-balance
  loss.
- DistServe-style goodput-optimised disaggregation and
  Prometheus-shaped observability + SLO-driven autoscaling.
- **NVIDIA Dynamo** (GTC 2025): production disaggregated serving with
  KV-aware routing, NIXL KV transfer, and SLO-driven GPU rebalancing.
```


## Key terms used in this part

- **{term}`MHA`**, **{term}`GQA`**, and **{term}`MLA`** are attention
  variants with different KV-cache memory footprints.
- **{term}`KV quantization`** and **{term}`KV eviction`** reduce KV memory
  using different tradeoffs (numeric precision vs token retention).
- **{term}`continuous batching`** and **{term}`chunked prefill`** are
  scheduler patterns for higher goodput under mixed workloads.
- **{term}`MoE`** and **{term}`expert parallel`** describe sparse expert routing at scale.
- **{term}`SLO`** is the service reliability target that autoscaling is designed to satisfy.
- **{term}`FP8`** is now a first-class production format on H100; the 2025
  NVIDIA **{term}`Blackwell / GB200`** generation adds **{term}`NV-FP4`**
  tensor cores and ~8 TB/s HBM3e bandwidth on the B200.
- **{term}`SGLang`** and **{term}`NVIDIA Dynamo`** are the two new
  production serving runtimes that complement vLLM in 2025–2026.

## Reading order

Prerequisites: Part I (roofline) and Part II (KV cache, PagedAttention).

1. `01_roofline_analysis` (cross-ref in Part I) — the LLM-flavoured
   roofline.
2. `02_kv_cache_variants_mha_gqa_mla` — three attention shapes,
   one module.
3. `03_kv_compression_streamingllm_h2o_snapkv` — token-drop
   policies.
4. `04_2bit_kv_quantization_kivi` — per-channel (K) and per-token (V)
   asymmetric 2-bit.
5. `05_gptq_awq_weight_quant` — activation-aware 4-bit weights.
6. `06_smoothquant_fp8_nf4` — three weight/activation formats
   compared on the same layer.
7. `07_quarot_spinquant_rotations` — Hadamard and learned rotations.
8. `08_batching_strategies` — four schedulers, one workload.
9. `09_moe_expert_parallelism` — router, capacity, aux loss.
10. `10_disaggregated_serving_distserve` — goodput sweep over
    prefill/decode ratios.
11. `11_serving_observability_slo_autoscaler` — metrics + control
    loop.
```
