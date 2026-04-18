# Training and fine-tuning

```{admonition} What you'll learn in this part
:class: tip

- Memory-reduction tactics that fit training into consumer GPUs:
  bf16, gradient accumulation, activation checkpointing.
- DDP vs FSDP2 on a CPU gloo process group — identical final loss,
  very different per-rank memory.
- Single-GPU parameter-efficient fine-tuning (LoRA, QLoRA).
- Preference optimisation (DPO, GRPO) from first principles.
```

## Reading order

Prerequisites: Part I (roofline) and Part II (KV cache).

1. `01_mixed_precision_accum_checkpointing` — four-way ablation
   (fp32, bf16, bf16+accum=4, bf16+accum+checkpoint).
2. `02_ddp_vs_fsdp2` — fork-context multiprocessing over gloo;
   both converge to identical loss.

```{note}
Six more chapters (tensor parallel, pipeline parallel, LoRA, QLoRA,
DPO, GRPO) land in a follow-up PR. This index page will grow as
they arrive.
```
