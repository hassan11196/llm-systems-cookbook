# 03 — Training and fine-tuning

Eight notebooks on distributed training and modern fine-tuning recipes.
Memory-reduction tactics first (mixed precision, activation checkpointing,
FSDP2), then parameter-efficient tuning (LoRA, QLoRA), then preference
optimisation (DPO, GRPO).

| NN | Notebook | Hardware | Runtime | Papers |
|---:|---|---|---:|---|
| 01 | mixed precision + grad accum + checkpointing | T4 | 8–12 min | — |
| 02 | DDP vs FSDP2 | 2×L4 or CPU gloo | 10–15 min | 1910.02054, 2304.11277 |
| 03 | tensor parallel from scratch | CPU / T4 | 5–8 min | 1909.08053 |
| 04 | pipeline parallelism — GPipe + 1F1B | CPU / T4 | 6–10 min | 1811.06965, 1806.03377 |
| 05 | LoRA from scratch vs PEFT | T4 | 15–20 min | 2106.09685 |
| 06 | QLoRA NF4 fine-tune | T4 | 18–25 min | 2305.14314 |
| 07 | DPO preference tuning | T4 | 20–30 min | 2305.18290 |
| 08 | GRPO DeepSeek-R1-style | T4 | 25–40 min | 2501.12948, 2402.03300 |

FSDP2 notebooks use the new `torch.distributed.fsdp.fully_shard` API, not
legacy FSDP1. A CPU gloo fallback path is provided so the DDP/FSDP2 notebook
runs without multiple GPUs.
