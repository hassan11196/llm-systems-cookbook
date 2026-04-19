# LLM Systems Cookbook

```{image} https://img.shields.io/badge/book-llm--systems--cookbook-2ea44f?style=for-the-badge
:alt: LLM Systems Cookbook
```

**A hands-on curriculum for modern LLM systems engineering.** 56 Jupyter
notebooks that teach inference, RAG, training, agents, serving,
evaluation, and GPU programming - each by reimplementing the core
technique from first principles, comparing it to a production tool, and
self-scoring the result with numerical checks.

Written for a computer-science undergraduate who wants to go from
"I know what softmax is" to "I can reason about LLM serving economics,"
with no prior deep-learning background assumed.

```{admonition} How to read this book
:class: tip

1. Click the **rocket 🚀 button** at the top-right of any chapter and
   choose *Colab* to run it end-to-end on a free T4 - no install.
2. Start with the **Foundations** part if you're new to GPU
   programming; otherwise jump directly into the track that interests
   you.
3. Each chapter follows the same six-step shape: motivation →
   reference paper → first-principles warm-up → real implementation
   → deterministic scoring → exercises + further reading.
4. The repo source is at
   [hassan11196/llm-systems-cookbook](https://github.com/hassan11196/llm-systems-cookbook).
```

## What's inside

```{grid} 1 1 2 2
:gutter: 3
:class-container: full-width

:::{grid-item-card} 🏗️ Foundations
:link: notebooks/07_gpu/index
:link-type: doc

GPU architecture, roofline analysis, Triton 101 and tiled matmul,
FlashAttention-2, fused RoPE + RMSNorm, torch.compile, Nsight
profiling, JAX sharding.

*9 chapters · mostly GPU, two CPU-friendly*
:::

:::{grid-item-card} ⚡ Inference engines
:link: notebooks/01_inference/index
:link-type: doc

KV cache, attention roofline, PagedAttention, continuous batching,
FA2-in-layer, radix prefix cache, speculative + tree decoding,
SARATHI chunked prefill, disaggregated serving.

*10 chapters · one Ampere-only*
:::

:::{grid-item-card} 📈 Serving and scaling
:link: notebooks/05_serving/index
:link-type: doc

KV variants (MHA/GQA/MLA), compression (StreamingLLM/H2O/SnapKV),
KIVI, GPTQ/AWQ, SmoothQuant/FP8/NF4, QuaRot/SpinQuant, batching,
MoE, DistServe, observability + autoscaler.

*11 chapters · CPU-safe*
:::

:::{grid-item-card} 🔍 Retrieval-augmented generation
:link: notebooks/02_rag/index
:link-type: doc

Chunking strategies, FAISS indices, BM25/SPLADE/RRF, ColBERTv2
late interaction, two-stage reranking, HyDE, RAPTOR, GraphRAG,
RAGAS.

*9 chapters · CPU-safe*
:::

:::{grid-item-card} 🤖 Agent frameworks
:link: notebooks/04_agents/index
:link-type: doc

ReAct from scratch, structured outputs, LangGraph state machines,
DSPy/MIPROv2, MCP server+client, AutoGen vs CrewAI,
τ-bench/SWE-bench evaluation.

*7 chapters · CPU-safe*
:::

:::{grid-item-card} 🎯 Evaluation
:link: notebooks/06_eval/index
:link-type: doc

Perplexity, MMLU + calibration, HumanEval pass@k, LLM-as-judge bias,
Arena Elo + Bradley-Terry, NIAH/RULER, contamination detection,
lm-eval vs Inspect AI.

*8 chapters · CPU-safe*
:::

:::{grid-item-card} 🔧 Training and fine-tuning
:link: notebooks/03_training/index
:link-type: doc

Mixed precision + gradient accumulation + checkpointing,
DDP vs FSDP2 (+ tensor parallel, pipeline parallel, LoRA, QLoRA,
DPO, GRPO - landing soon).

*2 of 8 chapters today · rest land in the next PR*
:::

:::{grid-item-card} 📚 Reference
:link: CURRICULUM_SPEC
:link-type: doc

The full 61-notebook specification with per-chapter scoring
thresholds, paper citations, and prerequisite DAG.

*If a chapter and the spec disagree, the chapter is authoritative.*
:::
```

## Prerequisites

- **Programming.** Comfortable reading Python; a little PyTorch helps
  but isn't required.
- **Math.** High-school algebra; the notebooks explain every ML-
  specific equation the first time it appears.
- **Computer architecture.** Helpful to have seen cache hierarchy and
  memory bandwidth concepts once; the GPU architecture tour
  re-introduces them from first principles.
- **Hardware.** Free Colab T4 is enough for 53 of 56 chapters. Three
  chapters (FA2 Triton kernels, Nsight profiling) note their
  requirements in their chapter header.

## Citation

If you use this cookbook in teaching or research, please cite:

```bibtex
@misc{llm_systems_cookbook,
  author  = {Ahmed, Muhammad Hassan},
  title   = {The LLM Systems Cookbook},
  year    = {2026},
  url     = {https://github.com/hassan11196/llm-systems-cookbook},
}
```

## Acknowledgements

Paper authors cited in each chapter. The notebook style draws from
[Project Pythia cookbooks](https://projectpythia.org/), the
[EECS 245 Jupyter Book](https://notes.eecs245.org/), and the
[IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).
Scoring-harness pattern inspired by the `pytest` + `nbmake`
community. MIT-licensed; contributions welcome.
