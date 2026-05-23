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


## Key terms used in this part

- **{term}`mixed precision`** (often **{term}`BF16`**) is the default way
  to reduce training memory and increase throughput.
- **{term}`gradient accumulation`** and **{term}`activation checkpointing`**
  are the two first memory-saving levers used in this track.
- **{term}`DDP`** and **{term}`FSDP`** are two distributed-training
  strategies with different memory/communication tradeoffs.
- **{term}`LoRA`** and **{term}`QLoRA`** are parameter-efficient fine-tuning methods.
- **{term}`GRPO`** is the group-relative RL algorithm behind DeepSeek-R1;
  **{term}`DAPO`** (ByteDance, 2025) is a variant that removes the KL
  penalty and clips per-token, achieving faster convergence on math
  reasoning without a reference model.

## Reading order

Prerequisites: Part I (roofline) and Part II (KV cache).

1. `01_mixed_precision_accum_checkpointing` — four-way ablation
   (fp32, bf16, bf16+accum=4, bf16+accum+checkpoint).
2. `02_ddp_vs_fsdp2` — fork-context multiprocessing over gloo;
   both converge to identical loss.

```{note}
v0.1 of this book ships 2 of the 8 planned training-track notebooks.
The remaining six are fully specified in
[`CURRICULUM_SPEC.md`](../../CURRICULUM_SPEC.md) and in active v0.2
development:

- **03** Tensor parallel from scratch (ColumnParallelLinear / RowParallelLinear)
- **04** Pipeline parallelism — GPipe and 1F1B schedules
- **05** LoRA + **DoRA** (weight-decomposed LoRA) vs PEFT
- **06** QLoRA NF4 fine-tune on Llama-3.2-1B
- **07** DPO preference tuning + **ORPO** (no reference model)
- **08** GRPO DeepSeek-R1-style reasoning RL (GSM8K reward shaping)

This index will grow as each notebook lands. Contributions welcome —
see [`CONTRIBUTING.md`](../../CONTRIBUTING.md).
```
