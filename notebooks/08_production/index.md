---
html_meta:
  description: "Production LLM patterns: Anthropic SDK prompt caching, LiteLLM multi-provider routing, tool use, structured outputs, hybrid RAG with citations, MCP server, DSPy MIPROv2, Inspect AI eval harness, and GPU cost modeling — with real API fixtures for replay mode."
---

# Production patterns

Real LLM code that runs against real APIs (Anthropic, OpenAI, Ollama,
local vLLM). No regex stubs, no rule-based "policies" pretending to be
models. Each notebook works in two modes:

- **LIVE** — when an API key is set, hits the real provider and shows
  fresh numbers.
- **Replay** — without keys, loads recorded responses from
  `_fixtures/` so the notebook still runs end-to-end on a fresh Colab.

Recorded fixtures are regenerated with `python scripts/refresh_fixtures.py`
on a real run; the responses you see in replay mode are real responses,
just from a previous run.

## Reading order

1. `01_claude_sdk_prompt_caching` — `cache_control` on a long system
   prompt; measured cache hit rate, $ saved, latency drop.
2. `02_litellm_router_fallbacks` — multi-provider routing with cost
   and latency tracking; fallback when the primary provider 429s.
3. `03_tool_use_agent` — native Anthropic tool use, parallel tool
   calls, no parser. Compared against the regex-parser approach
   from `04_agents/01`.
4. `04_structured_outputs_real` — head-to-head compliance/latency
   for Anthropic tool-use, Outlines + Qwen2.5, Instructor, BAML.
5. `05_hybrid_rag_production` — BGE-M3 dense + BM25 + RRF + reranker
   → Claude with citations on a 1k-doc corpus.
6. `06_mcp_real_server` — an MCP server (stdio transport, `mcp` SDK)
   that Claude Code / Cursor can connect to.
7. `07_dspy_miprov2_optimizer` — DSPy 3 program for classification;
   MIPROv2 optimises against held-out accuracy.
8. `08_inspect_ai_eval_harness` — Inspect AI task + scorer + solver
   on a real benchmark.
9. `09_gpu_providers_pricing_and_model_fit` — practical reference:
   GPU types in production, on-demand and spot pricing across ten
   cloud providers, vRAM math, and a calculator that maps a model
   size to the smallest cluster that holds it.

## Models

Defaults: `claude-sonnet-4-6` (Anthropic), `gpt-5.5` (OpenAI; GPT-5.5
Instant is the current ChatGPT default; GPT-5.5 Thinking is the unified
successor to the o-series for reasoning tasks),
`Qwen/Qwen2.5-1.5B-Instruct` (local via Ollama or vLLM). Override with
the `MODEL_*` env vars listed at the top of each notebook.

The current frontier tier (mid-2026): **Claude Fable 5** (`claude-fable-5`,
$10/$50 per M tokens) leads on SWE-bench Pro (80.3%) and long-context
agentic tasks; **Gemini 3.5 Pro** (limited Vertex AI preview, 2M-token
context window with Deep Think reasoning) targets enterprise document
workloads; **GPT-5.6** is the next OpenAI release in preview. For
cost-sensitive production use, `claude-haiku-4-5-20251001` and
`gpt-5.5-instant` remain the default latency-optimised choices.
