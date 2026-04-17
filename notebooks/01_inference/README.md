# 01 — Inference engines

Ten notebooks on single-GPU decoding mechanics and multi-process serving
architecture. Progresses from KV-cache anatomy through attention roofline,
PagedAttention's block allocator, continuous-batching schedulers, FlashAttention-2
kernels, prefix-cache reuse, and speculative/tree decoding to disaggregated
prefill/decode workers sharing KV over POSIX shared memory.

| NN | Notebook | Hardware | Runtime | Papers |
|---:|---|---|---:|---|
| 01 | autoregressive decoding and KV-cache anatomy | CPU / T4 | 20 min | 2309.06180 |
| 02 | attention from scratch and roofline | 1×GPU | 15 min | 2205.14135, Williams 2009 |
| 03 | PagedAttention block allocator | CPU | 15 min | 2309.06180 |
| 04 | continuous batching (Orca) | CPU | 15 min | Orca OSDI'22 |
| 05 | FlashAttention-2 in Triton | Ampere+ GPU | 25 min | 2205.14135, 2307.08691, 2407.08608 |
| 06 | RadixAttention prefix cache | T4 | 20 min | 2312.07104 |
| 07 | speculative decoding | 1×GPU | 25 min | 2211.17192, 2302.01318 |
| 08 | Medusa + EAGLE tree speculation | T4 / L4 | 30 min | 2401.10774, 2401.15077, 2503.01840 |
| 09 | SARATHI-Serve chunked prefill | 1×GPU sim | 20 min | 2308.16369, 2403.02310 |
| 10 | disaggregated prefill/decode serving | T4 | 25 min | 2401.09670, 2407.00079, 2311.18677 |

Prerequisites within the track: `02_attention_roofline` precedes
`05_flashattention2_triton`; the `01`–`04` sequence is the recommended reading
order.
