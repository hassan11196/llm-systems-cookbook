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
  NVIDIA **{term}`Blackwell / GB200`** generation adds **NV-FP4**
  tensor cores and ~8 TB/s HBM3e bandwidth on the B200.
- **{term}`SGLang`** and **{term}`NVIDIA Dynamo`** are the two new
  production serving runtimes that complement vLLM in 2025 to 2026.
- **vLLM Model Runner V2 (MRV2)** (`VLLM_USE_V2_MODEL_RUNNER=1`, vLLM
  ≥ 0.20): GPU-native Triton ops replace the CPU PyTorch prep path,
  delivering 56% more throughput on GB200 and eliminating CPU-GPU sync
  during speculative decoding.
- **{term}`Vera Rubin / Rubin GPU`** (H2 2026): NVIDIA's next platform
  targeting 5× Blackwell inference throughput at 10× lower cost per
  token; Rubin CPX variant optimised for massive-context inference.

```{admonition} Coming in v0.3
:class: note

**12: BitNet and sub-2-bit weight quantization** will implement a
from-scratch `BitLinear` with ternary {-1, 0, +1} weights using
`absmean` activation scaling, benchmark PPL vs INT4/INT8, and
demonstrate `bitnet.cpp` CPU-native inference achieving 15× better
energy efficiency than FP16. See {term}`BitNet` in the glossary.

NVIDIA **Dynamo** (GTC March 2025) and **NIXL** (see glossary) are
the production successor to the pure-Python `SharedMemory` approach
in chapter 10. The disaggregated serving notebook now documents the
Dynamo/NIXL upgrade path for multi-node production deployments.
**FP4** inference (Blackwell GB200) will be added as a hardware-gated
extension to the quantization notebooks once cc 10.0 hardware is
available in Colab.
```

## Reading order

Prerequisites: Part I (roofline) and Part II (KV cache, PagedAttention).

1. `01_roofline_analysis` (cross-ref in Part I): the LLM-flavoured
   roofline.
2. `02_kv_cache_variants_mha_gqa_mla`: three attention shapes,
   one module.
3. `03_kv_compression_streamingllm_h2o_snapkv`: token-drop
   policies.
4. `04_2bit_kv_quantization_kivi`: per-channel (K) and per-token (V)
   asymmetric 2-bit.
5. `05_gptq_awq_weight_quant`: activation-aware 4-bit weights.
6. `06_smoothquant_fp8_nf4`: three weight/activation formats
   compared on the same layer.
7. `07_quarot_spinquant_rotations`: Hadamard and learned rotations.
8. `08_batching_strategies`: four schedulers, one workload.
9. `09_moe_expert_parallelism`: router, capacity, aux loss.
10. `10_disaggregated_serving_distserve`: goodput sweep over
    prefill/decode ratios.
11. `11_serving_observability_slo_autoscaler`: metrics + control
    loop.

## Serving ecosystem (mid-2026)

Three open-source engines and one external KV layer account for most production deployments:

- **vLLM v0.20+** (V2 engine): async-first scheduler, Prometheus metrics, FP8 KV cache, multi-lora, NVIDIA Dynamo integration. Model Runner V2 delivers ~56% throughput improvement on GB200 via GPU-native Triton kernels and async scheduling. It is the common choice for most workloads. HuggingFace TGI officially entered maintenance mode in 2025.
- **SGLang v0.5+**: RadixAttention (shared prefix caching) + XGrammar-2 for ~80× faster grammar compilation and ~3× faster constrained decoding vs vLLM on structured-output workloads. Benchmarks show 3.1× throughput vs vLLM on DeepSeek-V3 traffic, with the largest gains when requests share long common prefixes (system prompts, RAG context).
- **TensorRT-LLM**: highest raw throughput on H100/H200 when compiled, but requires a compile step and custom kernels for new models. It is suited to high-scale inference at fixed model versions.
- **PegaFlow** (Novita AI, May 2026): Rust-core external KV cache storage engine that offloads GPU KV state to host memory or SSD and shares it across nodes via RDMA. Integrates with vLLM and SGLang as a drop-in KV connector with built-in Prometheus metrics. Enables effective KV capacity beyond GPU VRAM and cross-node prefix-cache sharing.

FP8 weight + KV cache + continuous batching + speculative decoding on H100 delivers 5-8× better cost-efficiency than naive FP16 with static batching (empirical from 2025 serving comparisons). The B200's native FP4 (9000 TFLOPS) is the newest format, with 1.3-1.6× throughput improvement over FP8 for 7-8B models. **NVIDIA Vera Rubin** (H2 2026) targets 5× Blackwell inference throughput at 10× lower token cost with 288 GB HBM4 and 50 PFLOPS FP4. The Rubin CPX variant is designed for massive-context workloads. The **Vera Rubin DSX AI Factory** reference design and the **Omniverse DSX Blueprint** reached general availability in July 2026, packaging the platform into a rack-to-datacenter build/simulate/operate workflow for continuously-running inference deployments.

KV cache memory, not compute, is the dominant cost lever at long context: production guidance for 2026 stacks it as paged attention (the memory-management substrate this track's notebooks build from scratch) + prefix caching (RadixAttention above) + attention-layer compression (MQA/GQA/MLA) + KV-cache quantization (INT8/FP8, and increasingly mixed-precision schemes like PM-KVQ that assign more bits to the layers/tokens that need them). vLLM's `--kv-cache-dtype fp8` flag now runs the full QK/ScoreV attention matmuls in FP8, not just the cache storage.

**Cloudflare Infire** is a custom inference engine that distributes LLM execution across multiple GPUs more efficiently than standard serving stacks, reducing memory usage and cold-start time. Cloudflare also released **Unweight**, a weight compression system that shrinks LLM sizes 15 to 22% without accuracy loss. It is distinct from quantization: it preserves floating-point precision while reducing parameter count via structured pruning. Production inference now represents roughly two-thirds of all AI compute spend, and open-source model serving infrastructure has become a major VC target (Baseten raised $1.5B at $13B valuation).

```{seealso}
Related parts of the cookbook:

- [Part I — Foundations](../07_gpu/index.md) derives the roofline model these serving decisions optimise against.
- [Part II — Inference engines](../01_inference/index.md) builds the KV cache, PagedAttention, and batching primitives this part scales to production.
- [Part VIII — Production patterns](../08_production/index.md) runs multi-provider routing, fallbacks, and GPU cost modelling against real APIs.
- The [glossary](../../glossary.md) defines every serving term used here (MHA/GQA/MLA, FP8, MoE, SLO, NVIDIA Dynamo).
```
