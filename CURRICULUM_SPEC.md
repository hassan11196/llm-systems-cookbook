# A 61-notebook curriculum for LLM systems engineering

> **v0.1 status:** this spec describes the full 61-notebook design. The
> repository currently ships **55 of 61**; the six gaps are all in the
> training track (notebooks `03_training/03..08`: tensor parallel,
> pipeline parallel, LoRA, QLoRA, DPO, GRPO), planned for v0.2. Every
> other track is complete.

This report delivers **61 fully-specified Jupyter notebooks** organized across seven engineering areas of modern LLM systems: inference engines, retrieval-augmented generation, training and fine-tuning, agent frameworks, serving and scaling, evaluation, and GPU programming. Every notebook includes exact 2026 package pins, a demo model drawn from a Colab-T4-compatible shortlist (SmolLM2-135M/360M, Qwen2.5-0.5B, Llama-3.2-1B, Phi-3.5-mini), a 12-20 cell outline with fully type-hinted function signatures, 4-6 numerical scoring checks with concrete thresholds (e.g., "Triton matmul ≥ 70% cuBLAS TFLOPs at 4096² FP16"), expected runtime outputs, and 2-3 stretch goals. The design is lab-style throughout - **no interview framing** - and every notebook instantiates a common `Scorer` that emits `scores/NN.json` for CI aggregation. Difficulty progresses within each track from single-GPU toy kernels to multi-process disaggregated serving; shared utilities (`_utils.py`, `scoring/harness.py`) keep the surface area small. The specs below are written so an implementer can produce each notebook directly without re-researching APIs.

## Curriculum architecture and shared scaffolding

The repo is organized as `notebooks/{01_inference, 02_rag, 03_training, 04_agents, 05_serving, 06_eval, 07_gpu}/NN_slug.ipynb`. Three cross-cutting conventions anchor the curriculum. First, every notebook's cell 2 instantiates `s = Scorer("AA_NN_slug")` from a shared `scoring/harness` package exposing `s.check(name: str, predicate: Callable[[], bool])` and `s.summary()`; the final cell writes a JSON artifact used by a CI runner to aggregate pass/fail across all 61 notebooks. Second, every user-defined function uses **full PEP-604 type hints** with `from __future__ import annotations` at the top; Triton kernels use `tl.constexpr` for compile-time args per convention. Third, hardware gating lives in a `hardware_check()` helper that reads `torch.cuda.get_device_capability()` and either falls back (e.g., Nsight-free profiler path, FP8 block-wise simulation) or raises a friendly `SystemExit`.

**Model shortlist** (all Colab-T4 feasible in fp16): `HuggingFaceTB/SmolLM2-135M`, `SmolLM2-360M-Instruct`, `Qwen/Qwen2.5-0.5B[-Instruct]`, `Qwen/Qwen2.5-1.5B`, `Qwen/Qwen2.5-Coder-0.5B-Instruct`, `meta-llama/Llama-3.2-1B[-Instruct]`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `microsoft/Phi-3.5-mini-instruct`. Larger models (Qwen2.5-3B judge, etc.) are invoked only where strictly needed and marked as such.

**Framework version pins (last validated April 2026; pins in `pyproject.toml` and `environment.yml` are authoritative and may drift as the ecosystem moves):** `torch==2.6.*` (2.9 for FSDP2 notebooks), `triton==3.2.*`, `transformers==4.46.3-4.56.*`, `vllm==0.19.*` (V1 engine), `sentence-transformers==5.4.1`, `faiss-cpu==1.13.2`, `peft==0.13.*`, `trl==0.24.*`, `bitsandbytes==0.44.*`, `llmcompressor==0.10.*`, `langgraph==1.1.*`, `dspy-ai==3.0.*`, `autogen-agentchat==0.4.*`, `crewai==0.80.*`, `mcp==1.2.*`, `lm-eval==0.4.11`, `inspect-ai==0.3.199`, `lighteval==0.11.0`, `ragas==0.2.14` (API-stable), `jax==0.5.*`.

**Progression and prerequisites.** Within each track, notebooks form a DAG: GPU track gates Triton kernels before FlashAttention; inference track requires `02_attention_roofline` before `05_flashattention2_triton`; training gates DDP before FSDP2; serving requires roofline before disaggregation. Cross-track edges are minimal by design - the radix-cache notebook (inference-06) assumes nothing from RAG; MoE (serving-09) is independent of training.

## Inference engines - 10 notebooks (notebooks/01_inference)

**01 Autoregressive decoding and KV-cache anatomy** (CPU/T4, 20 min). Implements `kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch, dtype_bytes=2) -> int = 2·L·H·D·T·B·bytes`; runs `generate_no_cache` vs `generate_with_cache` using HF `DynamicCache`; sweeps context {128..2048}; verifies measured `max_memory_allocated` within 5% of formula and cache gives ≥5× tokens/sec. Stretch: int8 KV (~2× memory reduction), `torch.compile` fullgraph. **Papers:** 2309.06180.

**02 Attention from scratch and roofline** (1×GPU, 15 min). `naive_attention` matches `F.scaled_dot_product_attention` to 1e-3; derives `attention_flops`, `attention_bytes`, `arithmetic_intensity`; microbenchmarks peak FLOPs and BW; plots roofline with ridge in (5, 200) FLOPs/byte; classifies prefill@N=1024 as compute-bound and decode@N=1 as memory-bound; demonstrates empirical O(N²) memory growth (ratio >12 for 4× N). **Papers:** 2205.14135, Williams CACM 2009.

**03 PagedAttention block allocator** (CPU-only, 15 min). Pure-Python `BlockAllocator(num_blocks, block_size=16)`, `PagedKVManager` with `translate(seq_id, logical_pos) -> (physical_block, offset)`, `fork_sequence` and `copy_on_write` with ref counting. Tests verify 100 random-length sequences waste <5% aggregate and <block_size each; paged saves ≥10× over `max_seq_len=4096` contiguous. Stretch: swapping-to-CPU with LRU, preemption-vs-swap recovery. **Papers:** 2309.06180.

**04 Continuous batching scheduler (Orca)** (CPU, 15 min). `simpy`-based simulator comparing `StaticBatcher` vs `ContinuousBatcher` on 500 Poisson arrivals (λ=30 req/s, log-normal lengths). Calibrated `step_latency(batch_size, total_tokens) = 0.005 + 0.0008·B + 0.00002·T`. Checks continuous throughput ≥2.5× static; TTFT p99 ≥40% lower; selective-attention beats padded by ≥1.5×; throughput sweet-spot at batch ∈ {16,32,64}. **Papers:** Orca OSDI'22.

**05 FlashAttention-2 in Triton** (1×GPU Ampere+, 25 min). Kernel `_fa2_fwd` with `BLOCK_M, BLOCK_N` constexprs parallelizing over Q-row tiles (FA2's key change); causal and non-causal variants match SDPA to 1e-3 max-abs; memory grows ≤20× when N grows 16× (linear); ≥3× naive at N=2048; within 1.5× of cuDNN SDPA. Stretch: backward with recomputation, varlen with `cu_seqlens`. **Papers:** 2205.14135, 2307.08691, 2407.08608.

**06 RadixAttention prefix cache** (T4, 20 min). `RadixCache.match_prefix(tokens) -> (matched_blocks, node)` with edge splitting; `insert`, `evict` (LRU over ref_count=0 leaves); `lpm_schedule` (longest-prefix-match scheduler). Benchmark: 200 multi-turn conversations with shared 128-token system prompt → hit-rate ≥0.8, LPM beats FIFO ≥1.3× under bounded cache, TTFT ≥3× lower than no-cache baseline. **Papers:** 2312.07104.

**07 Speculative decoding** (1×GPU, 25 min). `speculative_generate` with γ=4 draft tokens, rejection rule `min(1, p(x)/q(x))` with residual sampling from `normalize(relu(p-q))`. Checks: tokenizers match, total-variation distance of unigram distribution <0.05 vs target-only (distributional equivalence), acceptance α≥0.6 on 20 prompts, wall-clock ≥1.5× at α>0.6, closed-form `E[tok/step] = (1-α^(γ+1))/(1-α)` within 15% of empirical. **Papers:** 2211.17192, 2302.01318.

**08 Medusa + EAGLE tree speculation** (T4/L4, 30 min). Train 4 Medusa heads (500 steps on ultrachat_200k slice, ≥30% loss reduction); static tree `[4,3,2,2]` with 88 nodes (4 + 12 + 24 + 48) truncated to 64; `verify_tree` in a single forward with ancestor mask; dynamic pruning (EAGLE-2 style) with threshold 0.03. Tree accepts ≥1.3× more tokens/step than linear; wall-clock ≥1.7× speedup. Stretch: EAGLE-3 multi-layer fusion. **Papers:** 2401.10774, 2401.15077, 2503.01840.

**09 SARATHI-Serve chunked prefill** (1×GPU sim, 20 min). Fits latency `a + b·P + c·D + d·max(P,D)` on real forwards (R²≥0.9). `ChunkedPrefillScheduler(chunk_size=1024, budget=1536)` with decode-maximal fill. On 300 mixed requests (20% long prefills): TPOT p99 cut ≥30%, TTFT p99 within 1.2×, throughput within 5%; sweet-spot at chunk ∈ {1024, 1536}. **Papers:** 2308.16369, 2403.02310.

**10 Disaggregated prefill/decode serving** (T4, 25 min). Two processes via `multiprocessing`; KV serialized into `SharedMemory` blocks; `serialize_kv_to_shm` with `KVHandoff(req_id, shm_name, layers, num_kv_heads, head_dim, seq_len, dtype)`; decode worker attaches and deserializes. Greedy outputs bitwise-match collocated; KV bytes match formula ±1 KB; shm bandwidth ≥1 GB/s; on skewed workload TTFT p99 and TPOT p99 each improve ≥20%; transfer overhead <15% of TTFT. **Papers:** 2401.09670, 2407.00079, 2311.18677.

## Retrieval-augmented generation - 9 notebooks (notebooks/02_rag)

**01 Chunking strategies** (T4, 18 min) compares fixed, recursive, semantic (embedding-distance breakpoint at 95th percentile), and late chunking (full-doc encode with Jina v2, mean-pool token spans) on BEIR/scifact 300-dev. Recall@10 thresholds: fixed ≥0.72, recursive ≥0.78, **semantic ≥0.82**, late ≥0.80. Model: `BAAI/bge-small-en-v1.5`.

**02 FAISS dense retrieval** (T4, 22 min) embeds 10K Wikipedia mini-corpus with bge-small, sweeps `IndexFlatIP`, `IndexIVFPQ(nlist=256, M=48, nbits=8)`, `IndexHNSWFlat(M=32, efConstruction=200)`. Target: HNSW@efSearch=64 recall@10 ≥0.95 within 2× flat latency; IVF-PQ memory ≤25% of flat; Pareto plot of recall-vs-latency. Stretch: matryoshka dims on nomic-embed.

**03 BM25, SPLADE, and RRF hybrid** (T4, 20 min). Custom BM25 with k1=1.2, b=0.75 verified against `rank_bm25` to 1e-4; SPLADE-v3 encoding `w = log(1+ReLU(logits))` max-pooled, nnz ∈ [80, 300]/doc; RRF `Σ 1/(60+rank_i(d))`. Hybrid RRF NDCG@10 ≥ max(BM25, dense) + 0.02 on BEIR/scifact.

**04 ColBERTv2 late interaction** (T4, 24 min). Via `pylate` 1.1.7. `maxsim(q_emb, d_emb) = (q @ d.T).max(dim=1).values.sum()`. On scifact dev: Recall@10 ≥0.82, MRR@10 ≥ bge-small + 0.03. Storage blow-up bounded ≤50×. Stretch: PLAID centroid prefiltering.

**05 Two-stage reranking** (T4, 18 min). Bi-encoder top-100 (bge-small) → cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-v2-m3`, jina v2). On BEIR/fiqa dev: MiniLM uplift ≥0.05 NDCG@10, **bge-v2-m3 uplift ≥0.07**; `top_n=100` saturates within 0.01 of 200; MiniLM p50 ≤80 ms on T4.

**06 HyDE and query rewriting** (T4, 22 min). HyDE generates hypothetical answer with Qwen2.5-0.5B, embeds, retrieves. Multi-query (4 paraphrases + RRF union) and decomposition (max 3 sub-queries). On HotpotQA dev 500: HyDE Recall@10 ≥ baseline+0.03, decomposition ≥ baseline+0.04 on 2-hop subset. **Papers:** 2212.10496.

**07 RAPTOR hierarchical** (T4, 28 min). UMAP(10d) → BIC-selected GMM → summarize with Qwen2.5-0.5B → recurse up to 3 levels. On NarrativeQA 20-doc / 150-query subset, thematic-query Recall@5 ≥ flat + 0.05; collapsed-RAPTOR F1 ≥ flat on full set. Tree depth ∈ [2, 4]. **Papers:** 2401.18059.

**08 GraphRAG with Leiden** (T4, 30 min). LLM entity+relation JSON extraction on 500-doc AG-News slice → NetworkX graph (|V|≥500, |E|≥1000) → `leidenalg.find_partition` (modularity >0.3, 5-25 communities) → community summaries → map-reduce QA. **GraphRAG ≥60% on global-sensing queries vs <30% naive RAG**; GraphRAG not worse than naive on factoid by more than 10 pp. **Papers:** 2404.16130.

**09 RAGAS evaluation harness** (T4, 20 min). 50-query SQuAD subset; good pipeline (retrieve+ground) vs bad pipeline (ignore context). Three-run mean±std on `faithfulness, answer_relevancy, context_precision, context_recall`. Good faithfulness ≥0.80, bad <0.50, good > bad on all four metrics; judge variance std ≤0.08; 8-gram contamination histogram reported. **Papers:** 2309.15217.

## Training and fine-tuning - 8 notebooks (notebooks/03_training)

**01 Mixed precision + gradient accumulation + checkpointing** (T4, 8-12 min). Ablation: fp32 → bf16 (≥30% peak memory reduction) → bf16+accum=4 (loss within 2% of non-accum) → bf16+accum+activation checkpointing (≥30% further reduction, ≤25% step-time overhead). Uses `torch.utils.checkpoint(block, use_reentrant=False)`. Mini-GPT2 from SmolLM2 config.

**02 DDP vs FSDP2** (2×L4 or CPU gloo fallback, 10-15 min). Uses **`torch.distributed.fsdp.fully_shard`** (not legacy FSDP1): `fully_shard(layer, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16))`. Verifies `param.placements == (Shard(0),)` on ≥90% of wrapped params; FSDP2 memory ≤ (1/N + 0.1)·DDP; final loss within 1%. **Papers:** 1910.02054, 2304.11277.

**03 Tensor parallel from scratch** (CPU/T4, 5-8 min). Hand-rolled `ColumnParallelLinear` (split out-dim, optional all-gather) and `RowParallelLinear` (split in-dim, simulate all-reduce via `torch.stack(shards,0).sum(0)`). Applied to QKV (column) + output (row) and up/down SwiGLU. Output matches single-GPU to **1e-5** for TP ∈ {2, 4, 8}. **Papers:** 1909.08053.

**04 Pipeline parallelism - GPipe + 1F1B** (CPU/T4, 6-10 min). 4 stages as sequential nn.Modules. `gpipe_schedule` and `one_f_one_b_schedule` with warmup (P−1 forwards) / steady-state / cooldown. Empirical bubble matches theoretical `(P−1)/(M+P−1) + 0.02` for 1F1B and `(P−1)/M ± 0.05` for GPipe; gradients identical to 1e-4. **Papers:** 1811.06965, 1806.03377.

**05 LoRA from scratch vs PEFT** (T4, 15-20 min). `ManualLoRALinear` with `B∈R^{out×r}`, `A∈R^{r×in}`, scaling `α/r`; inject into `q_proj, v_proj` of Qwen2.5-0.5B; fine-tune on Alpaca-mini (1K). Manual matches `peft.LoraConfig` loss curve within 1% after 100 steps; trainable ≤1% of params; adapter ≤10 MB. **Papers:** 2106.09685.

**06 QLoRA NF4 fine-tune** (T4, 18-25 min). `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)` on Llama-3.2-1B; TRL `SFTTrainer` with `optim="paged_adamw_8bit"`. **Fits in ≤10 GB VRAM**; eval PPL within 0.5 of BF16-LoRA. **Papers:** 2305.14314.

**07 DPO preference tuning** (T4, 20-30 min). `DPOTrainer` on `trl-lib/ultrafeedback_binarized` 1K with β∈{0.1,0.3,0.5}. `rewards/margin > 0` monotone non-decreasing by step 50; `rewards/accuracies ≥ 0.65`; **win-rate vs SFT ≥55%** by LLM-judge on 50 prompts. Reference-free ablation via `loss_type="simpo"`. **Papers:** 2305.18290.

**08 GRPO DeepSeek-R1-style** (T4, 25-40 min). `GRPOTrainer(reward_funcs=[correctness_reward, format_reward], args=GRPOConfig(num_generations=4, beta=0.04, ...))` on GSM8K. System prompt forces `<reasoning>...</reasoning><answer>...</answer>`. **Mean reward ≥2× baseline over 100 steps**; format fraction ≥0.8 by step 50; eval accuracy up ≥5 pp; within-group std >0 on ≥80% of steps. **Papers:** 2501.12948, 2402.03300.

## Agent frameworks - 7 notebooks (notebooks/04_agents)

**01 ReAct from scratch** (CPU/T4, 6 min). Regex parser `r"Action:\s*(\w+)\[(.*?)\]"`; three typed tools (`calculator` via safe AST eval, `wiki_search` stub over 25 facts, `get_datetime`); 20-task bench (8 arithmetic, 8 lookup, 4 temporal). **Success rate ≥0.70**, parse errors ≤0.10, avg steps ≤5.0. Served via Ollama phi3.5 or vLLM Qwen2.5-0.5B. **Papers:** 2210.03629.

**02 Structured outputs three ways** (T4, 8 min). `Person` Pydantic v2 model; compares (a) prompt-only JSON, (b) Pydantic validator+retry, (c) Outlines `generate.json` FSM-constrained. **Outlines validity=1.00, compliance≥0.98**; prompt-only <0.95; Pydantic-retry lift ≥0.05 over prompt-only; Outlines latency ≤3× baseline. **Papers:** 2307.09702.

**03 LangGraph state machines** (CPU, 7 min). Typed `StateGraph` with `Annotated[list[dict], operator.add]` reducer; 4 nodes (supervisor/researcher/writer/critic); `add_conditional_edges` routes on `state["route"]`; `InMemorySaver` checkpointer; termination via `iteration >= 4`. Routes ≥9/10 test inputs; cycles terminate within 6 iters.

**04 DSPy 3.0 + MIPROv2** (T4, 12 min). `FactualQA` signature; `Predict` baseline → `ChainOfThought` (+0.03 lift) → `BootstrapFewShot` → **`MIPROv2(auto="light")` +≥0.10 absolute over baseline** on 10-item trivia_qa test. Inspects optimized instructions and demos; saves/reloads `.json` with identical deterministic score. **Papers:** 2310.03714, 2406.11695.

**05 MCP server/client** (CPU, 5 min). `%%writefile toy_server.py` implementing `@server.list_tools`/`@server.call_tool` with `echo(text)`, `add(a,b)`, and `notes://today` resource via stdio transport. 100-call benchmark: **median latency <100 ms, p95 <250 ms**; unicode round-trip byte-exact; LLM succeeds on ≥2/3 end-to-end prompts with tool-call schema translation.

**06 AutoGen 0.4 vs CrewAI** (CPU/T4, 10 min). Identical supervisor/worker/critic trio on 10 arithmetic word problems with programmatic checker. AutoGen 0.4: `SelectorGroupChat` with `TextMentionTermination("APPROVED") | MaxMessageTermination(12)`. CrewAI: `Process.hierarchical` with `manager_llm`. **Pass@1 ≥0.60** in at least one framework; both ≥0.40; critic flags ≥0.80 of 5 planted errors. **Papers:** 2308.08155.

**07 Agent evaluation suite** (L4, 18 min). Synthetic τ-bench-retail (10 episodes, Pydantic `RetailDB` with 5 tools and LLM-driven `UserSim`) plus 3 real SWE-bench-Lite instances (clone repo at `base_commit`, agent emits diff, `git apply` + pytest subset). **Pass@1 within 0.05 of reference on 3 SWE instances**; τ-retail success ≥0.40; unauthorized-action rate ≤0.10. Rich scorecard printed and saved to `artifacts/eval_report.json`. **Papers:** 2310.06770, 2406.12045, 2311.12983.

## Serving and scaling - 11 notebooks (notebooks/05_serving)

**01 Roofline analysis** (T4, 12 min). Measures peak TFLOP/s (GEMM sweep) and HBM BW (large copy) within 30% and 25% of vendor spec respectively. Computes AI for MLP, prefill-attn, decode-attn using formulas. Ridge point in (5, 200) FLOPs/byte; decode correctly classified memory-bound at all tested shapes.

**02 KV cache variants MHA/MQA/GQA/MLA** (T4, 14 min). Four `nn.Module` variants with identical APIs. Analytic `kv_bytes_per_token`: **MHA=4096, GQA(8)=512, MQA=64, MLA≈70** exact match. GQA reduces to MHA numerically with `n_kv_heads=n_heads`; MLA decode ≥3× faster than MHA at 8K context. Implements MLA's absorbed-matmul trick. **Papers:** 2305.13245, 1911.02150, 2405.04434.

**03 KV compression - StreamingLLM, H2O, SnapKV** (T4, 18 min). Monkey-patches attention via `attn_implementation="eager"` to hook probabilities. StreamingLLM keeps first 4 sinks + last 1020 tokens; H2O top-256 heavy hitters + 256 recent; SnapKV uses 32-token obs window to rank prompt tokens. PPL within 1.0 at 1024-budget; heavy-hitter Jaccard ≥0.7 across adjacent windows; ≥1.8× tokens/s at 8K prompt. **Papers:** 2309.17453, 2306.14048, 2404.14469.

**04 2-bit KV quantization (KIVI)** (T4, 16 min). Per-channel K (group_size=128), per-token V, residual window=32 FP16. **KIVI-2bit PPL within 0.5 of FP16** on WikiText-2; KV memory ≥6.5× reduction; per-channel K beats per-token K by ≥0.3 PPL at 2 bits; decode ≥1.5× at 8K context. **Papers:** 2402.02750.

**05 GPTQ + AWQ weight quant** (L4, 25 min). Hand-rolled GPTQ: `build_hessian(x) = 2·XXᵀ/n + 0.01·mean(diag)·I`; Cholesky-based column update loop; verify reconstruction MSE < 0.6× RTN. Production path via `gptqmodel.quantize(bits=4, group_size=128, desc_act=True)` and `autoawq`. **PPL within 0.5 of FP16**; AWQ int4 ≥2× FP16 throughput via vLLM V1 `quantization="awq_marlin"` at batch=8. **Papers:** 2210.17323, 2306.00978.

**06 SmoothQuant + FP8 + NF4** (L4, 22 min). `smooth_scale(act_max, weight_max, α) = max(|X|)^α / max(|W|)^(1-α)`, α-sweep {0.3, 0.5, 0.7, 0.85}. Absorbs `1/s` into preceding LN/RMSNorm. **SmoothQuant W8A8 ΔPPL <0.3** on SmolLM2-135M (naive W8A8 > 2.0 as control). NF4 double-quant reduces overhead ≥0.3 bits/param. FP8 block-wise simulation ΔPPL <0.2; real FP8 gated on `capability >= (8,9)`. **Papers:** 2211.10438, 2305.14314, 2412.19437.

**07 QuaRot + SpinQuant rotations** (T4, 18 min). Walsh-Hadamard `R` via Sylvester construction (scaled 1/√n); fuses into embed/unembed/norms using computational invariance. **Max activation magnitude reduced ≥5×**; rotation-only (no quant) ΔPPL <0.05 (invariance verified); W4A4 with Hadamard ≤ W4A4 without rotation − 8.0 PPL; 200-step Cayley-SGD on R1 adds ≥0.3 further. **Papers:** 2404.00456, 2405.16406.

**08 Batching strategies** (T4, 15 min). `simpy` simulator for Static, Dynamic (token budget), Continuous, and Chunked-Prefill schedulers. Calibrated execution model `prefill_ms ≈ a+b·tokens`, `decode_ms ≈ c+d·batch+e·kv_tokens`. **Continuous ≥3× static throughput; chunked p99 TPOT ≤0.7× unchunked**; sim vs real vLLM V1 TTFT RMSE ≤30%.

**09 MoE with expert parallelism** (CPU/T4, 14 min). 8 experts, d=256, top-2, 1 shared expert. `AuxLossMoE` vs `AuxFreeMoE` (DeepSeek-V3 bias update `b_i -= γ·(load_i - mean_load)`, γ=1e-3). **Load CoV <0.15** with aux-free after 500 steps; proxy loss within 2% of aux-loss; 4-rank gloo all-to-all runs cleanly; EPLB replicas reduce max-expert load ≥25%. **Papers:** 2401.06066, 2412.19437.

**10 Disaggregated serving (DistServe)** (T4, 18 min). Prefill and decode as separate spawned processes with ZMQ control + `multiprocessing.Queue` for `torch.save`-serialized KV blobs. Load sweep RPS ∈ {1..16}: **disagg TTFT p99 ≤ colocated at RPS≥4**; TPOT p99 ≤0.7× at RPS≥8; transfer overhead <20% of TTFT; 2P:1D improves goodput ≥20% over 1P:1D on prompt-heavy. Maps to NVIDIA Dynamo's Smart Router + NIXL + Planner. **Papers:** 2401.09670, 2407.00079, 2311.18677.

**11 Serving observability + SLO + autoscaler** (T4, 20 min). Launches vLLM V1 with Prometheus; minimal Prom-text parser extracts `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds`, `vllm:gpu_cache_usage_perc`. `SLO(ttft_p95_s=0.6, tpot_p95_s=0.05)`; `Autoscaler` with hysteresis (scale up if p95 TTFT > 1.1·SLO for 30s; down if <0.7·SLO for 60s). 5-minute sinusoidal burst workload: **attainment ≥95%**, scale-down within 120s, per-replica goodput up ≥40% during burst.

## Evaluation - 8 notebooks (notebooks/06_eval)

**01 Perplexity from scratch** (T4, 18 min). Naive non-overlapping `ppl_naive` then sliding-window `ppl_sliding(stride)` with `-100` masking of overlap. Synthetic uniform test: PPL ≈ vocab size within 1%. SmolLM2-360M on WikiText-2 within **±10%** of published (~10-12); monotone `ppl(stride=128) < ppl(stride=2048)` across all models.

**02 MMLU harness + calibration** (T4, 25 min). Both `score_loglikelihood` (length-normalized log-probs over {A,B,C,D}) and `score_generation` (regex `\b([A-J])\b`) with exact harness 5-shot prompt. 300 questions stratified × 15 subjects. Qwen2.5-0.5B-Instruct within **±3 pp** of leaderboard (~47%); MMLU-Pro drop ≥10 pp; ECE <0.20 with reliability diagram. **Papers:** 2009.03300, 2406.01574, 1706.04599.

**03 HumanEval with unbiased pass@k** (T4, 35 min). `pass_at_k(n, c, k) = 1 − C(n-c, k)/C(n, k)` with numerical stability; `unsafe_execute` sandbox via `multiprocess.Process` with a hard wall-clock timeout. `resource.setrlimit(RLIMIT_AS)` is an optional hardening step documented in the exercises - the shipped notebook relies on process isolation + timeout rather than seccomp/rlimit, and warns readers that running LLM-generated code is still a trust-the-source operation. n=20 samples × T ∈ {0.2, 0.8}. Unit test: n=20,c=5,k=1 → 0.25 ± 0.01. **Qwen2.5-Coder-0.5B pass@1 at T=0.2 ≥ 0.20**; T=0.8 pass@10 ≥ T=0.2 pass@10; infinite-loop killed within timeout+1s. **Papers:** 2107.03374.

**04 LLM-as-judge bias** (T4, 25 min). 20 MT-Bench prompts, two candidates, both orders (AB/BA). `position_flip_rate` on non-tie→different-non-tie; length-bias Spearman ρ with bootstrap 95% CI; self-preference via second judge. **Raw flip rate >0.15; after randomization+CoT <0.08**; binomial test for self-preference. **Papers:** 2306.05685, 2305.17926.

**05 Arena Elo + Bradley-Terry** (CPU, 6 min). Synthetic 1000 battles with known θ. BT-MLE via `LogisticRegression(fit_intercept=False)` with ±1 design matrix; recovery RMSE <0.10; bootstrap CI width <0.20; online Elo Spearman rank >0.9 vs MLE; √n CI shrinkage verified (width@1000/width@100 ∈ [0.25, 0.45]). **Papers:** 2403.04132.

**06 Long context NIAH + RULER** (L4, 30 min). Token-accurate needle insertion at `depth_pct ∈ {0.1,...,0.9}`; grid over context ∈ {1K..16K} × 3 seeds = 75 runs. RULER MK-2 (two needles) and MV (4 values/key) procedurally generated. **Phi-3.5-mini NIAH ≥0.8 at 4K/8K**; RULER avg ≥0.7 at 4K; hit-rate monotone non-increasing past window. **Papers:** 2404.06654, 2307.03172.

**07 Contamination detection** (T4 or CPU-135M fallback, 20 min). Brief LoRA fine-tune on 100 "seen" WikiText docs (3 epochs, r=8) vs 100 held-out. `ngram_set` + containment; `min_k_prob(k_pct=0.20)` averaging lowest-20% token log-probs. **Min-K% AUC ≥0.70 finetuned; ≈0.50 base** (null control); n-gram AUC ≥0.90 (trivial verbatim); Oren exchangeability p<0.05 on one memorized doc. **Papers:** 2310.17623, 2310.16789.

**08 lm-eval + Inspect AI** (T4, 35 min). Both frameworks on ARC-Easy 25-shot and HellaSwag 10-shot (limit=500). `inspect_ai.eval(tasks=[...], model="hf/Qwen/Qwen2.5-0.5B-Instruct")`. **Reproduces leaderboard within ±2 pp**; cross-framework gap ≤0.02 absolute. Lighteval command strings as third cross-reference. Documents divergence sources (whitespace, length norm, letter extraction).

## GPU programming - 8 notebooks (notebooks/07_gpu)

**01 GPU architecture tour** (T4, 12 min). STREAM-style `copy_kernel` sweeping 1MB-1GB; **achievable BW ≥80% of vendor peak** for ≥64MB buffers. Coalesced vs strided-gather: **stride=1 ≥5× stride-32**. Theoretical occupancy via `active_warps = min(max, regs_per_SM/regs/32, smem_per_SM/smem·blocks)` agrees with CUDA runtime API within 1 block/SM. **Papers:** PMPP Ch. 4-5.

**02 Triton 101 softmax** (T4, 10 min). Vector-add (within 5% of `torch.add`); naive softmax (one row per program); online softmax with running `(m, ℓ)` via Milakov-Gimelshein recurrence; `@triton.autotune` sweeping `BLOCK_N ∈ {128..2048}, num_warps ∈ {2,4,8}, num_stages ∈ {2,3,4}`. **Online ≥2× naive at N≥16384**, matches torch.softmax at small N. **Papers:** 1805.02867.

**03 Triton tiled matmul** (A10, 15 min). Blocked GEMM with `tl.dot`; group-M swizzling:
```python
group_id = pid // (GROUP_M * num_pid_n)
first_pid_m = group_id * GROUP_M
pid_m = first_pid_m + (pid % min(num_pid_m - first_pid_m, GROUP_M))
pid_n = (pid % (GROUP_M * num_pid_n)) // min(...)
```
Autotune `BLOCK_M/N/K ∈ {64..256}`, `GROUP_M ∈ {4,8}`. **≥70% of cuBLAS TFLOPs at 4096² FP16**; Group-M ≥1.2× non-grouped. **Papers:** Simon Boehm matmul blog.

**04 Triton FlashAttention-2** (A100, 15 min). Outer loop over Q tiles, inner over K/V tiles; online softmax `O_new = O_old·exp(m_old-m_new) + P_new·V_j`; causal mask via `tl.where(m_idx ≥ n_idx, S, -inf)` with early-exit `if n_start*BLOCK_N > (pid_m+1)*BLOCK_M: break`. **Matches SDPA to 1e-3**; memory linear in N (slope ≤2×); ≥0.7× SDPA TFLOPs at N=4096; causal ≥1.7× non-causal. **Papers:** 2205.14135, 2307.08691.

**05 Fused RoPE + RMSNorm** (T4, 8 min). Standalone kernels, then fused `rmsnorm_rope_kernel` that reads x once. On prefill (B=4, S=2048): **≥1.5× unfused eager**; on decode (B=1, S=1): **≥2×** due to launch-overhead amortization; ≥70% HBM BW. Matches HF Llama reference to 1e-3 FP16. **Papers:** 1910.07467, 2104.09864.

**06 torch.compile deep dive** (A10, 10 min). `MiniBlock` (RMSNorm → QKV → SDPA → residual → SwiGLU). Three modes (`default`, `reduce-overhead`, `max-autotune`); **max-autotune ≥1.3× eager**. `torch._dynamo.explain`: inject `.item()` break, observe reason, fix with `torch.where`; final **≤1 graph break**. Dumps Inductor `output_code.py` and annotates a fused `@triton.jit` kernel. Dynamic shapes keep single graph across S ∈ {512, 1024, 2048}.

**07 Nsight profiling** (local GPU or Colab Pro, 15 min). `nsys profile --trace=cuda,nvtx,osrt --stats=true` around 64-token decode on SmolLM2-135M with NVTX ranges. `ncu --kernel-name "regex:(gemm|flash|rms)" --set full`. Expected `sm__throughput`: **60-90% for GEMM, 40-70% for attention, 20-40% for elementwise/norm**; decode aggregate <50% (memory-bound), prefill >65% (compute-bound). Falls back to `torch.profiler` on free Colab.

**08 JAX sharding and pipeline** (CPU simulated 8 devices or 2+ GPU, 10 min). `Mesh(devices.reshape(4,2), ("data","model"))`. Megatron-style TP: W1 column-shard (`P(None,"model")`), W2 row-shard (`P("model",None)`), `shard_map` with `jax.lax.psum(y_local, "model")`. Output matches single-device to **1e-5**. Pipeline via `jax.lax.scan` over microbatches with `jax.lax.ppermute`; P=4, M=8 → **theoretical bubble 3/11 ≈ 0.273; measured within 5% absolute**.

## Implementation notes and coverage gaps

Several topics are intentionally absent and worth flagging for a v2. **Multimodal** (vision encoders, speech, audio/video tokenization) gets no notebook; a natural addition is a `08_multimodal/` track with CLIP dual-encoder retrieval, LLaVA-style vision-language fine-tune, and Whisper streaming inference. **Safety and red-teaming** (jailbreak evals, constitutional AI, watermarking) is also absent - a candidate extension mirrors the eval track. **Embedding training** (contrastive loss, hard negative mining, Matryoshka) is referenced via stretch goals but not taught directly.

Two spec-level risks deserve attention. First, the **2026 version pin stack is not fully mutually compatible**: sentence-transformers 5.4.1 + pylate 1.1.7 have not been validated against `transformers` 5.x, so RAG notebooks pin `transformers==4.46.3`, while some training notebooks want 4.56.x for newer TRL features - expect one environment per track rather than one unified env. Second, **RAGAS 0.4.3 (Jan 2026)** broke the 0.2.x metric API used in notebook 02_rag/09; the spec pins 0.2.14 deliberately and flags the upgrade as a stretch goal. Third, **FSDP2's `fully_shard`** has no `auto_wrap_policy` / `use_orig_params` kwargs - implementers porting from FSDP1 tutorials will hit errors if they pass these.

The 61 notebooks are **runnable end-to-end on a free Colab T4** with two exceptions: FlashAttention notebooks (inference-05, gpu-04) require Ampere+, and Nsight profiling (gpu-07) needs a local GPU or Colab Pro (the spec includes a `torch.profiler` fallback path). Notebook 05_serving/06 gates FP8 on `torch.cuda.get_device_capability() >= (8,9)` and simulates with block-wise `torch.float8_e4m3fn` casts otherwise. Total estimated cumulative runtime across all 61 notebooks on a single L4 is approximately **6-8 hours of wall-clock compute** plus model-download time.

## Conclusion

The 61 specifications form a production-ready blueprint for an LLM systems curriculum that is simultaneously **deep** (each notebook reproduces a named paper's mechanism from scratch, then validates against a production tool) and **broad** (the seven tracks cover the full stack from CUDA warps to agent SLOs). Three design decisions differentiate it from typical "LLM from scratch" curricula: every scoring check is **numerical and machine-checkable** (no human-graded essays), every notebook has **two paths** - pedagogical from-scratch and production toolchain - enabling direct comparison, and the **hardware bar is deliberately low** (T4-compatible demo models throughout) so the curriculum is accessible without a datacenter. The primary risk is environment fragility given 2026-era version pins across 40+ packages; the mitigation is one virtualenv per track rather than a monolithic requirements file, and explicit stretch goals for users who want bleeding-edge variants (RAGAS 0.4, Transformers 5, FSDP HYBRID_SHARD, EAGLE-3, Microsoft Agent Framework).