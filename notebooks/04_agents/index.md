# Agent frameworks

```{admonition} What you'll learn in this part
:class: tip

- ReAct's parser + tool-call loop from scratch.
- Three strategies for structured LLM outputs (prompt-only,
  Pydantic validate+retry, FSM-constrained).
- State-machine-shaped agents (LangGraph-style 50-line clone).
- DSPy's signature + MIPROv2 prompt optimisation.
- The Model Context Protocol as a 2-file server + client.
- Conversation-driven (AutoGen / AG2 / Microsoft Agent Framework) vs
  role-driven (CrewAI) multi-agent idioms.
- Evaluating agents with τ-bench / SWE-bench-shaped benchmarks.

The primitives here — tools, state graphs, handoffs, guardrails —
are the same building blocks used by OpenAI Agents SDK (handoff
chains), Google ADK (supervisor hierarchies), Pydantic AI
(dependency-injected agents), and smolagents (code-as-actions).
The 2025 A2A protocol (Agent-to-Agent) adds horizontal discovery
across agent boundaries; see the glossary for an overview.
**Microsoft Agent Framework 1.0** unified AutoGen 0.4
and Semantic Kernel into one production SDK; the patterns in notebook
06 apply directly to it.
```


## Key terms used in this part

- **{term}`ReAct`** is the baseline loop pattern used to teach tool-using agents.
- **{term}`structured outputs`** means constraining model outputs to a
  schema instead of free-form text.
- **FSM (Finite-State Machine)** constraints are one way to enforce valid
  output structure during decoding. **{term}`XGrammar`** (v2)
  is the current production-grade FSM engine, delivering ~3× faster
  constrained decoding via its Structural Tag protocol.
- **{term}`MCP`** exposes tools/data over JSON-RPC so clients can use them uniformly.
- **{term}`DSPy`** frames prompts/pipelines as optimizable programs.
- **{term}`A2A`** (Agent-to-Agent Protocol, Google ADK 2025) is the open standard
  for cross-framework inter-agent delegation and discovery.
- **{term}`handoff`** and **{term}`guardrail`** are the two new first-class
  primitives introduced by the OpenAI Agents SDK.
- **{term}`Microsoft Agent Framework`** is the 2026 merger of AutoGen
  and Semantic Kernel into a single production-ready agent SDK.

## Ecosystem snapshot (mid-2026)

The agent framework landscape has consolidated around a few dominant patterns:

- **LangGraph** (v1.4): now the most-starred agent framework on GitHub. First-class checkpointing, durable execution, and human-in-the-loop approvals make it the default for production stateful agents.
- **AutoGen / AG2**: Microsoft's AutoGen v0.4 ships streaming and event-driven architecture; the community maintains the proven v0.2 lineage as `ag2ai/ag2` with typed tools and dependency injection. Both run on the same core `SelectorGroupChat` / `GroupChatManager` API.
- **CrewAI** (v1.12): added agent skills, hierarchical memory isolation, and native support for OpenRouter, DeepSeek, Ollama, and vLLM providers.
- **Google ADK v1.0** (stable 2026): hierarchical agent tree where a root agent delegates to sub-agents; v1.0 ships stable releases in Python, Go, Java, and TypeScript. Introduces the **A2A (Agent-to-Agent) protocol v1.0** (in production at 150+ organisations as of 2026) so a LangGraph or CrewAI agent can be invoked by an ADK agent without bespoke adapters. Announced at Google Cloud Next 2026 alongside Workspace Studio (no-code agent builder) and managed MCP servers via Apigee.
- **Pydantic AI**: lightweight, type-safe alternative gaining traction for simple agents where full LangGraph state management is overkill.
- **MCP 2026 RC**: stateless protocol core via Streamable HTTP; SEP-2549 TTL/cache-scope on resource reads; long-running Tasks extension; MCP Apps for server-rendered UIs; OAuth/OIDC authorization hardening. The RC is live at `modelcontextprotocol.io`; production SDKs targeting the RC are expected ahead of the final specification.

## Reading order

No mandatory prerequisites — CPU-only.

1. `01_react_from_scratch` — three-line agent loop + regex parser +
   three tools.
2. `02_structured_outputs_three_ways` — flaky-LLM simulator; prompt
   vs validate+retry vs FSM.
3. `03_langgraph_state_machines` — StateGraph clone with
   conditional edges.
4. `04_dspy_3_miprov2` — 3×3 (instruction, demo) grid; MIPROv2 as
   5-sample random search.
5. `05_mcp_server_client` — JSON-RPC 2.0 tool server + synchronous
   client.
6. `06_autogen_0_4_vs_crewai` — draft/critique/revise pipeline two
   ways (AutoGen/AG2 and CrewAI).
7. `07_agent_evaluation_suite` — success rate + trajectory
   efficiency + code-patch success.
