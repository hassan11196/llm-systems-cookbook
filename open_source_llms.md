# The open-source LLM landscape

A tour of the open-weight model families that matter in 2026, how they
differ architecturally, and how to actually serve them. The goal here
isn't a leaderboard — benchmarks rot in weeks — but a mental map that
lets you pick the right model, estimate its memory footprint, and stand
up a throughput-reasonable endpoint in a single afternoon.

Everything below assumes you've read
[`07_gpu/01_gpu_architecture_tour`](notebooks/07_gpu/01_gpu_architecture_tour.ipynb)
and skimmed [`05_serving/01_roofline_analysis`](notebooks/05_serving/01_roofline_analysis.ipynb);
we lean on bandwidth- vs. compute-bound vocabulary throughout.

## Why open weights

Three practical reasons open-weight models have become the default for
systems work:

1. **Reproducibility.** You get a stable checkpoint, a tokenizer, and a
   known config — no silent behaviour drift behind an API.
2. **Cost predictability.** Once you've sized the deployment, cost per
   million tokens is a function of GPU hours, not a vendor's price
   sheet. For sustained workloads > ~1 req/s this usually wins.
3. **Systems research.** Most techniques in this book (PagedAttention,
   speculative decoding, quantisation, MoE routing) require model
   internals. Closed endpoints let you poke at `n+1` tokens, not the
   KV cache.

The trade-off is that you own the operations: serving, autoscaling,
safety classifiers, observability — all the things Parts III and VII
of this book exist to teach.

## The families worth knowing

The table below is the shortlist we reach for in the notebooks. Sizes
are parameter counts; "active" applies to Mixture-of-Experts (MoE)
models, where only a fraction of parameters runs per token.

| Family                | Release | Sizes (params)            | Architecture            | Context  | License               |
|-----------------------|---------|---------------------------|-------------------------|----------|-----------------------|
| **Llama 4**           | Meta    | Scout 109B / 17B active, Maverick 400B / 17B active | MoE, multimodal         | 10M / 1M | Llama 4 Community     |
| **Llama 3.3**         | Meta    | 70B                       | Dense                   | 128k     | Llama 3.3 Community   |
| **Qwen3**             | Alibaba | 0.6B–235B (dense + MoE)   | Dense + MoE             | 128k     | Apache 2.0 (most)     |
| **DeepSeek-V3 / R1**  | DeepSeek| 671B / 37B active         | MoE, MLA attention      | 128k     | DeepSeek License      |
| **gpt-oss**           | OpenAI  | 120B / 5.1B active, 20B / 3.6B active | MoE, MXFP4 native       | 128k     | Apache 2.0            |
| **Mistral / Mixtral** | Mistral | 7B, 22B, Mixtral 8×22B    | Dense + MoE             | 32k–128k | Apache 2.0 (most)     |
| **Gemma 3**           | Google  | 1B, 4B, 12B, 27B          | Dense, multimodal 4B+   | 128k     | Gemma Terms           |
| **Kimi K2**           | Moonshot| 1T / 32B active           | MoE                     | 128k     | Modified MIT          |
| **MiniMax-M1 / -01**  | MiniMax | 456B / 46B active         | MoE + lightning attn    | 1M–4M    | MiniMax License       |
| **Phi-4**             | Microsoft| 14B                      | Dense                   | 16k      | MIT                   |
| **SmolLM2**           | HF      | 135M / 360M / 1.7B        | Dense                   | 8k       | Apache 2.0            |

Three structural patterns to internalise:

**Dense vs. MoE.** Dense models activate every parameter per token and
scale memory linearly with size. MoE models (DeepSeek-V3, gpt-oss,
Llama 4, Kimi K2, MiniMax) activate only a handful of experts per
token, so the *compute* per token scales with the active subset (tens
of billions) while the *memory* scales with the total (hundreds of
billions to a trillion). This makes MoEs cheap to run per token but
expensive to host — you still need to keep the whole model resident.
See [`05_serving/09_moe_expert_parallelism`](notebooks/05_serving/09_moe_expert_parallelism.ipynb).

**Attention variants.** Vanilla multi-head attention (MHA) is
memory-bound at decode time because the KV cache grows as
`2 · L · H · D · T`. Grouped-query attention (GQA) shrinks `H` for
K/V, multi-query (MQA) collapses it to 1, and multi-head latent
attention (MLA, DeepSeek-V3) compresses K/V into a shared latent
space — roughly a 7× reduction over MHA at equal quality.
MiniMax's "lightning attention" mixes linear and softmax attention
layers to get sub-quadratic scaling up to 4M tokens. We reimplement
these side-by-side in
[`05_serving/02_kv_cache_variants_mha_gqa_mla`](notebooks/05_serving/02_kv_cache_variants_mha_gqa_mla.ipynb).

**Native quantisation.** gpt-oss ships with MXFP4 weights by default,
so the 120B MoE fits on a single H100 80GB. Most other families ship
BF16 checkpoints and rely on post-hoc GPTQ/AWQ/FP8 quantisation
(Part III, chapters 5–7).

## Picking a model

A rough decision tree that matches how we pick throughout the book:

- **Laptop / CPU-only:** SmolLM2-135M or -360M. Good enough for
  perplexity demos, chunking, retrieval, and most agent wiring.
- **Single T4 (Colab free):** Qwen2.5-0.5B-Instruct, Gemma 3 1B,
  Llama-3.2-1B-Instruct. Anything larger OOMs during kv-cache growth
  past ~2k tokens.
- **Single H100 80GB:** Llama 3.3 70B at FP8, Qwen3-32B, gpt-oss-20B,
  Mistral Small 24B, Phi-4 14B in BF16. Comfortable batch sizes.
- **Single H100 80GB, MoE:** gpt-oss-120B (MXFP4 native) fits with
  headroom. DeepSeek-V3 and Kimi K2 do **not** fit and need ≥ 8×H100
  or H200/B200.
- **Multi-node:** DeepSeek-V3, Kimi K2, MiniMax-M1, Llama 4 Maverick.
  Expect tensor-parallel × expert-parallel sharding; see
  [`07_gpu/08_jax_sharding_pipeline`](notebooks/07_gpu/08_jax_sharding_pipeline.ipynb).

A reliable sizing heuristic: **parameters × precision-bytes × 1.2**
for weights + KV cache at the batch sizes you care about. A 70B model
in FP8 is ≈ 84 GB weight-resident, which is why FP8 is the sweet spot
on H100 80GB.

## Serving stacks at a glance

You almost never want to write a serving loop from scratch. The four
stacks we exercise in Part III:

| Stack       | Strengths                                       | When we use it                          |
|-------------|-------------------------------------------------|-----------------------------------------|
| **vLLM**    | PagedAttention, continuous batching, prefix cache, broad model support | Default for Parts II–III notebooks      |
| **SGLang**  | RadixAttention prefix sharing, structured generation, fast for agentic workloads | Structured-output and agent chapters    |
| **TGI**     | Rust core, tight HF Hub integration, production-grade observability | Any time HF ecosystem is the constraint |
| **llama.cpp** | GGUF quantisation, runs anywhere (incl. M-series, CPU) | Laptop demos, edge-sized models         |

## Running them

The three examples below assume an H100 with CUDA 12.4+, `vllm==0.19.*`,
and `transformers==4.46.3` (matching this cookbook's `pyproject.toml`).
You can run any of them on a smaller GPU by swapping to a smaller model
from the same family — the serving code is identical.

### Offline batch inference with vLLM

The minimal "hello, LLM" on an H100. One process, one GPU, no network.

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    max_model_len=8192,
)
params = SamplingParams(temperature=0.7, max_tokens=256)

prompts = [
    "Explain KV caching in two sentences.",
    "Why does FP8 inference usually beat INT8 on H100?",
]
for out in llm.generate(prompts, params):
    print(out.outputs[0].text)
```

On an H100 80GB this hits ~2200 tok/s aggregate at batch 32 in BF16.
Flip `dtype="fp8"` (with a pre-quantised checkpoint) and you get
roughly 1.8× — the standard H100 FP8 tensor-core multiplier once the
model is memory-bandwidth bound.

### OpenAI-compatible server with vLLM

For everything downstream, serve the model behind the OpenAI API
schema so the rest of your stack (LangGraph, DSPy, evaluation harness)
doesn't care whether it's hitting a local model or a vendor:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --quantization fp8 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --enable-prefix-caching
```

Then, from Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Describe MoE routing in one paragraph."}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Key flags worth knowing:

- `--enable-prefix-caching` — reuses KV for repeated system prompts;
  often a 3–5× TTFT win on agent workloads.
- `--tensor-parallel-size N` — splits the model across N GPUs; see
  [`05_serving/10_disaggregated_serving_distserve`](notebooks/05_serving/10_disaggregated_serving_distserve.ipynb).
- `--quantization fp8|awq|gptq` — applies the matching kernel path.

### Plain HuggingFace for dev loops

When you need to poke at logits or patch a layer, `transformers`
beats any serving stack:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "google/gemma-3-4b-it"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(
    name, torch_dtype=torch.bfloat16, device_map="cuda"
)

inputs = tok("What is PagedAttention?", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=128)
print(tok.decode(out[0], skip_special_tokens=True))
```

This is what cell 3 of most notebooks in this book looks like.

## H100 reference numbers

Rough steady-state decode throughput on a single H100 80GB SXM5 with
vLLM 0.19, `--enable-prefix-caching`, BF16 unless noted. Numbers are
from our own runs in March 2026; treat them as ballpark, not gospel.

| Model                         | Precision | Batch | Throughput (tok/s) | TTFT (ms, 512 ctx) |
|-------------------------------|-----------|-------|--------------------|---------------------|
| Qwen2.5-7B-Instruct           | BF16      | 32    | ~2,200             | ~70                 |
| Qwen2.5-7B-Instruct           | FP8       | 32    | ~4,000             | ~55                 |
| Llama-3.1-8B-Instruct         | BF16      | 32    | ~2,050             | ~75                 |
| Mistral Small 24B             | FP8       | 16    | ~1,100             | ~110                |
| Qwen3-32B                     | FP8       | 16    | ~820               | ~140                |
| Llama-3.3-70B-Instruct        | FP8       | 8     | ~380               | ~230                |
| gpt-oss-20B                   | MXFP4     | 32    | ~3,100             | ~65                 |
| gpt-oss-120B                  | MXFP4     | 8     | ~420               | ~210                |

Two patterns to remember:

- **Throughput scales with batch until KV cache saturates VRAM.** The
  knee is at roughly `(GPU_mem − weight_bytes) / (per_token_kv_bytes
  × seq_len)`. Past that point you either shrink context, add PagedAttention-
  style fragmentation recovery, or go multi-GPU.
- **FP8 is a ~1.8× decode win on H100, MXFP4 about ~2.4× on Hopper.**
  Both are lossless enough for most chat workloads when the
  quantisation recipe is sound; see
  [`05_serving/05_gptq_awq_weight_quant`](notebooks/05_serving/05_gptq_awq_weight_quant.ipynb).

## Where to go next

Each family maps onto a chapter that unpacks one of its design
choices:

- **MoE routing** — [`05_serving/09_moe_expert_parallelism`](notebooks/05_serving/09_moe_expert_parallelism.ipynb)
- **MLA attention (DeepSeek-V3)** — [`05_serving/02_kv_cache_variants_mha_gqa_mla`](notebooks/05_serving/02_kv_cache_variants_mha_gqa_mla.ipynb)
- **MXFP4 / FP8 quantisation (gpt-oss, Llama)** — [`05_serving/05_gptq_awq_weight_quant`](notebooks/05_serving/05_gptq_awq_weight_quant.ipynb),
  [`05_serving/06_smoothquant_fp8_nf4`](notebooks/05_serving/06_smoothquant_fp8_nf4.ipynb)
- **Long-context (MiniMax, Llama 4)** — [`06_eval/06_long_context_niah_ruler`](notebooks/06_eval/06_long_context_niah_ruler.ipynb)
- **Disaggregated prefill/decode** — [`01_inference/10_disaggregated_prefill_decode`](notebooks/01_inference/10_disaggregated_prefill_decode.ipynb),
  [`05_serving/10_disaggregated_serving_distserve`](notebooks/05_serving/10_disaggregated_serving_distserve.ipynb)

The short version: open-weight LLMs have converged on a small set of
tricks (GQA/MLA, MoE, FP8/MXFP4, prefix caching, continuous batching),
and every one of those tricks has a chapter in this book.
