# Production patterns

Real LLM code that runs against real APIs (Anthropic, OpenAI, Ollama,
local vLLM). No regex stubs, no rule-based "policies" pretending to be
models. Each notebook works in two modes:

- **LIVE**: when an API key is set, hits the real provider and shows
  fresh numbers.
- **Replay**: without keys, loads recorded responses from
  `_fixtures/` so the notebook still runs end-to-end on a fresh Colab.

Recorded fixtures are regenerated with `python scripts/refresh_fixtures.py`
on a real run; the responses you see in replay mode are real responses,
just from a previous run.

## Reading order

1. `01_claude_sdk_prompt_caching`: `cache_control` on a long system
   prompt; measured cache hit rate, $ saved, latency drop.
2. `02_litellm_router_fallbacks`: multi-provider routing with cost
   and latency tracking; fallback when the primary provider 429s.
3. `03_tool_use_agent`: native Anthropic tool use, parallel tool
   calls, no parser. Compared against the regex-parser approach
   from `04_agents/01`.
4. `04_structured_outputs_real`: head-to-head compliance/latency
   for Anthropic tool-use, Outlines + Qwen2.5, Instructor, BAML.
5. `05_hybrid_rag_production`: BGE-M3 dense + BM25 + RRF + reranker
   → Claude with citations on a 1k-doc corpus.
6. `06_mcp_real_server`: an MCP server (stdio transport, `mcp` SDK)
   that Claude Code / Cursor can connect to.
7. `07_dspy_miprov2_optimizer`: DSPy 3 program for classification;
   MIPROv2 optimises against held-out accuracy.
8. `08_inspect_ai_eval_harness`: Inspect AI task + scorer + solver
   on a real benchmark.
9. `09_gpu_providers_pricing_and_model_fit`: practical reference,
   GPU types in production, on-demand and spot pricing across ten
   cloud providers, vRAM math, and a calculator that maps a model
   size to the smallest cluster that holds it.

## Models

Defaults: `claude-sonnet-4-6` (Anthropic), `gpt-5.5` (OpenAI; GPT-5.5
Instant is the current ChatGPT default; GPT-5.5 Thinking is the unified
successor to the o-series for reasoning tasks),
`Qwen/Qwen2.5-1.5B-Instruct` (local via Ollama or vLLM). Override with
the `MODEL_*` env vars listed at the top of each notebook.

The current frontier tier (late August 2026): **Claude Fable 5**
(`claude-fable-5`, GA July 1, 2026, $10/$50 per M tokens) is the
leading publicly available model on SWE-bench Pro (80%, August 29
snapshot) and long-context agentic tasks — Anthropic's gated **Claude
Mythos 5** security-research preview tops the raw leaderboard at
80.3%, but it's invite-only via Project Glasswing, not a production
option; **Claude Opus 5**
(`claude-opus-5`, July 24, 2026, $5/$25 per M tokens) holds Opus 4.8's
price while more than doubling its Frontier-Bench v0.1 agentic-coding
score (43.3% vs. 21.1%), ahead of GPT-5.6 Sol and Claude Fable 5 on that
eval, and carries the freshest (May 2026) knowledge cutoff in Anthropic's
lineup; **Claude Sonnet 5** (`claude-sonnet-5`, June 30, 2026, 63.2%
SWE-bench Pro) is the balanced-tier option one step down. **GPT-5.6 Sol**
(OpenAI, July 9, 2026, $5/$30 per M tokens) edges out Fable 5 on the
Artificial Analysis Coding Agent Index at under half the output tokens,
and OpenAI cut its API and credit pricing by over 20% for the next
three months starting August 21, 2026;
**Grok 4.5** (xAI, July 8, 2026) undercuts Opus-class pricing by over 60%
while landing fourth on the Artificial Analysis Intelligence Index;
xAI followed with **Grok 4.6** (built for long-running agents and
deeper coding, 500K-token context, $2/$0.50/$6 per M tokens
input/cached/output), now available in GitHub Copilot and Cursor. The
next step up, **Grok 4.7** (2.1T parameters), has slipped past its
original within-weeks target: xAI added SpaceX company data to its
training run in mid-August and now points to early September.
Google's flagship **Gemini 3.5 Pro** remains delayed — it missed its
fourth target date and its latest rumored August 12 date too, with
reporting pointing to coding-performance shortfalls and a disappointing
training-data refresh, and remains unreleased as of August 2026 with no
new date given; in its place, **Gemini 3.6 Flash** (July 21, 2026, the
new Gemini default, 58.7% SWE-bench Pro, $1.50/$7.50 per M tokens), the
low-latency **Gemini 3.5 Flash-Lite**, and the newer **Gemini 3.7 Flash**
(August 13, 2026) cover the Google tier for production use today. On the
open-weight side, **GLM-5.2** (Z.ai, 744B MoE, 62.1% SWE-bench Pro; a
faster, cheaper **GLM-5.2 Turbo** variant shipped August 17, 2026),
the smaller **GLM-5.3-Flash** (320B-A18B MoE, natively multimodal,
1M-token context, shipped August 26, 2026 at roughly a tenth of
GLM-5.2's price), **Kimi K3** (Moonshot AI, 2.8T MoE, full 594 GB
weights shipped July 27, 2026), and the **DeepSeek-V4-Flash-0731**
refresh (July 31, 2026) now
edge into frontier-tier territory, making self-hosted deployment a
credible alternative to the closed-API tier for cost-sensitive teams
willing to run their own serving stack. **Qwen3.8-Max** (Alibaba,
shipped August 3, 2026, 2.4T MoE, ~95B active) beats GPT-5.6 Sol Max and
Claude Fable 5 on OSWorld-Verified computer use, with open weights
following August 12 — the first Qwen3.x release to break from the
closed-only pattern. **Sakana Fugu-Ultra v1.1** takes a different
approach entirely: a multi-model orchestration system (not a single
trained network) at $5 input / $30 output per M tokens, now leading the
GPQA-Diamond leaderboard at 95.5%. Meta's **Muse Spark 1.2** (shipped
August 6, 2026; multimodal, agentic-workflow focused) is a closed
US-only preview priced below GPT-5.6 Luna. **Note on cost modeling:**
Anthropic cancelled the planned September 1 increase to $3/$15 and made
Claude Sonnet 5's introductory $2/$10-per-M pricing permanent (August 11,
2026); Sonnet 5's newer tokenizer can still produce up to ~35% more
tokens for the same input than Sonnet 4.6's, so budget for a somewhat
larger effective cost than the flat per-token price alone implies. For
cost-sensitive production use via API, `claude-haiku-4-5-20251001` and
`gpt-5.5-instant` remain the default latency-optimised choices.
