# 07 — GPU programming

Eight notebooks on kernel-level LLM engineering. Starts with an architecture
tour and peak-bandwidth/TFLOPs measurement, then Triton basics (softmax, tiled
matmul), a from-scratch FlashAttention-2 kernel, fused RoPE + RMSNorm,
torch.compile internals, Nsight profiling (or torch.profiler fallback), and
JAX sharding for distributed array layouts.

| NN | Notebook | Hardware | Runtime | Papers |
|---:|---|---|---:|---|
| 01 | GPU architecture tour | T4 | 10 min | Williams CACM 2009 |
| 02 | Triton 101 — softmax | T4 | 12 min | Tillet 2019 |
| 03 | Triton tiled matmul | T4 | 15 min | Tillet 2019 |
| 04 | Triton FlashAttention-2 | Ampere+ | 25 min | 2307.08691 |
| 05 | fused RoPE + RMSNorm | T4 | 12 min | 2104.09864, 1910.07467 |
| 06 | torch.compile deep dive | T4 | 18 min | — |
| 07 | Nsight profiling | local GPU / Colab Pro | 20 min | — |
| 08 | JAX sharding pipeline | T4 or TPU | 20 min | — |

Notebook 04 and 07 gate on hardware via `hardware_check(min_cc=(8, 0))` and
`torch.cuda.get_device_name()`; notebook 07 provides a `torch.profiler`
fallback that records a subset of the Nsight analysis.
