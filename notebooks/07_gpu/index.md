# Foundations — GPU programming and the roofline

```{admonition} What you'll learn in this part
:class: tip

- Read a GPU's compute and memory-bandwidth ceilings from its
  datasheet and measure them empirically.
- Write your first Triton kernels (softmax, tiled matmul,
  FlashAttention-2) and evaluate each one against its theoretical
  peak.
- Fuse RoPE with RMSNorm to halve HBM traffic.
- Use `torch.compile` + Nsight / `torch.profiler` to find graph
  breaks and kernel bottlenecks.
- Translate to JAX for distributed arrays and automatic sharding.
```

Every later part of the book — inference, serving, training — ends up
on a graph whose axes are arithmetic intensity and throughput. This
part builds the axes.


## Key terms used in this part

- **{term}`HBM`** bandwidth and **{term}`TFLOPs`** ceilings define the hardware limits.
- **{term}`arithmetic intensity`** and **{term}`roofline`** are the primary analysis tools.
- **{term}`kernel`** launch and memory movement costs explain why fusion matters.
- **{term}`FlashAttention`**, **{term}`RoPE`**, and **{term}`RMSNorm`** are
  recurring primitives reused throughout later tracks.

## Reading order

1. `01_gpu_architecture_tour` — device discovery + peak-bandwidth
   and peak-TFLOPs microbenchmarks.
2. `02_triton_101_softmax` — first Triton kernel.
3. `03_triton_tiled_matmul` — grouped-order tiled matmul, target 70 %
   of cuBLAS.
4. `04_triton_flashattention` — FA2 forward with online softmax.
5. `05_fused_rope_rmsnorm` — position-dependent rotation + variance
   normalisation fused into two kernels.
6. `06_torch_compile_deep_dive` — TorchDynamo + Inductor, graph
   breaks, reduce-overhead mode.
7. `07_nsight_profiling` — NVTX annotations + `torch.profiler`
   fallback.
8. `08_jax_sharding_pipeline` — distributed arrays, 1-D mesh,
   PartitionSpec.
9. `05_serving/01_roofline_analysis` (cross-reference) — closes the
   foundations arc by applying the roofline to LLM serving
   workloads.

```{seealso}
Companion reading: **Part II** uses the ridge intensity numbers from
this part to classify every inference workload; **Part III** uses the
same math to reason about quantisation and KV compression.
```
