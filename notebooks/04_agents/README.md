# 04 — Agent frameworks

Seven notebooks covering the full agent stack: a from-scratch ReAct loop,
three approaches to structured outputs, LangGraph state machines, DSPy
prompt optimisation, the MCP tool protocol, multi-agent orchestration, and a
combined reasoning + code-execution evaluation harness.

| NN | Notebook | Hardware | Runtime | Papers |
|---:|---|---|---:|---|
| 01 | ReAct from scratch | CPU / T4 | 6 min | 2210.03629 |
| 02 | structured outputs three ways | T4 | 8 min | 2307.09702 |
| 03 | LangGraph state machines | CPU | 10 min | — |
| 04 | DSPy 3 + MIPROv2 | T4 | 18 min | 2310.03714, 2406.11695 |
| 05 | MCP server + client | CPU | 8 min | Anthropic MCP spec |
| 06 | AutoGen 0.4 vs CrewAI | T4 | 15 min | — |
| 07 | agent evaluation suite | T4 | 22 min | τ-bench, SWE-bench Verified |

Agent notebooks default to a locally-hosted Qwen2.5-0.5B-Instruct via vLLM or
Ollama. API routes (OpenAI, Anthropic) are gated on ``OPENAI_API_KEY`` /
``ANTHROPIC_API_KEY`` and never required for scoring.
