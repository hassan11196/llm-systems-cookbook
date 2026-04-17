# 05 — Serving and scaling

Eleven notebooks on the throughput-and-latency side of LLM deployment. Starts
with roofline analysis and KV-cache variants (MHA / GQA / MLA), proceeds
through compression (StreamingLLM, H2O, SnapKV) and quantisation (GPTQ, AWQ,
SmoothQuant, FP8, rotation methods), then batching, MoE expert parallelism,
disaggregated serving, and an SLO-driven autoscaler with observability.

| NN | Notebook | Hardware | Runtime | Focus |
|---:|---|---|---:|---|
| 01 | roofline analysis | CPU / T4 | 12 min | arithmetic intensity, bandwidth-bound vs compute-bound |
| 02 | KV cache variants (MHA / GQA / MLA) | T4 | 18 min | grouped-query / latent attention memory savings |
| 03 | KV compression (StreamingLLM / H2O / SnapKV) | T4 | 20 min | attention-score pruning |
| 04 | 2-bit KV quantization (KIVI) | T4 | 18 min | per-channel 2-bit KV with outlier handling |
| 05 | GPTQ + AWQ weight quant | T4 | 25 min | compensated-error vs activation-aware |
| 06 | SmoothQuant, FP8, NF4 | Ada+ for FP8 | 25 min | activation-weight rebalancing, FP8 simulation |
| 07 | QuaRot + SpinQuant rotations | T4 | 20 min | Hadamard rotations for outlier-free quant |
| 08 | batching strategies | CPU / T4 | 15 min | static, dynamic, continuous, chunked-prefill |
| 09 | MoE expert parallelism | 2×T4 sim | 22 min | all-to-all, capacity factor, load balancing |
| 10 | disaggregated serving (DistServe) | T4 | 25 min | prefill/decode split performance |
| 11 | serving observability + SLO autoscaler | CPU | 18 min | Prometheus metrics, SLO-driven scaling |
