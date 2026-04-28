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

## Models

Defaults: `claude-sonnet-4-6` (Anthropic), `gpt-5` (OpenAI),
`Qwen/Qwen2.5-1.5B-Instruct` (local via Ollama or vLLM). Override with
the `MODEL_*` env vars listed at the top of each notebook.
