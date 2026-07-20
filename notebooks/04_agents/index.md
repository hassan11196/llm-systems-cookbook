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

The primitives here (tools, state graphs, handoffs, guardrails)
are the same building blocks used by OpenAI Agents SDK (handoff
chains), Google ADK (supervisor hierarchies), Pydantic AI
(dependency-injected agents), and smolagents (code-as-actions).
The 2025 A2A protocol (Agent-to-Agent) adds horizontal discovery
across agent boundaries; see the glossary for an overview.
**Microsoft Agent Framework 1.0** (April 2026) unified AutoGen 0.4
and Semantic Kernel into one production SDK; the patterns in notebook
06 apply directly to it.
```


## Key terms used in this part

- **{term}`ReAct`** is the baseline loop pattern used to teach tool-using agents.
- **{term}`structured outputs`** means constraining model outputs to a
  schema instead of free-form text.
- **FSM (Finite-State Machine)** constraints are one way to enforce valid
  output structure during decoding. **{term}`XGrammar`** v2 is the
  current production-grade FSM engine. It runs constrained
  decoding about 3× faster via its Structural Tag protocol.
- **{term}`MCP`** exposes tools/data over JSON-RPC so clients can use them uniformly.
- **{term}`DSPy`** frames prompts/pipelines as optimizable programs.
- **{term}`A2A`** (Agent-to-Agent Protocol, Google ADK 2025) is the open standard
  for cross-framework inter-agent delegation and discovery.
- **{term}`handoff`** and **{term}`guardrail`** are the two new first-class
  primitives introduced by the OpenAI Agents SDK.
- **{term}`Microsoft Agent Framework`** is the April 2026 merger of AutoGen
  and Semantic Kernel into a single production-ready agent SDK.

## Ecosystem snapshot (mid-2026)

The agent framework landscape has consolidated around a few common patterns:

- **LangGraph v1.0** (stable): now the most-starred agent framework on GitHub, used in production by companies including Uber, LinkedIn, and Klarna. The 1.0 release adds durable state (automatic execution persistence), built-in human-in-the-loop approvals, native sandboxing, sub-agents, first-class MCP support, and a distributed runtime via the CLI. It passed CrewAI in GitHub stars during early 2026.
- **AutoGen / AG2**: Microsoft's AutoGen v0.4 ships streaming and event-driven architecture; the community maintains the proven v0.2 lineage as `ag2ai/ag2` with typed tools and dependency injection. AG2 Beta (`autogen.beta`) is a ground-up redesign with multi-provider LLM support and first-class testing. Both run on the same core `SelectorGroupChat` / `GroupChatManager` API.
- **CrewAI** (v1.14.6, June 11): added agent skills, hierarchical memory isolation, Qdrant Edge memory backend, native support for OpenRouter, DeepSeek, Ollama, vLLM, Cerebras, and Dashscope providers, plus pluggable memory/knowledge/RAG/flow backends, a Chat API, and native Snowflake Cortex support.
- **Google ADK v1.0** (stable): hierarchical agent tree where a root agent delegates to sub-agents; ships stable releases in Python, Go, Java, and TypeScript. Introduces the **A2A (Agent-to-Agent) protocol v1.0** (in production at 150+ organisations) so a LangGraph or CrewAI agent can be invoked by an ADK agent without bespoke adapters. Announced at Google Cloud Next 2026 alongside Workspace Studio (no-code agent builder) and managed MCP servers via Apigee.
- **Pydantic AI**: a lightweight, type-safe alternative used for simple agents where full LangGraph state management is more than needed. **Pydantic AI V2** (June 23) is a harness-first redesign that makes capabilities (tool access, memory, sandboxing) a declared core primitive instead of manual wiring.
- **LlamaIndex Workflows 1.0** (June 22): event-driven orchestration for RAG and agent pipelines, promoted out of the LlamaIndex core into a standalone workflow engine.
- **Claude Agent SDK**: added hierarchical subagent spawning (up to 3 levels deep), fallback model chains, and a community MCP tool marketplace.
- **MCP (Model Context Protocol)**: the **2026-07-28** specification (release candidate; final text ships July 28, 2026) is the largest revision since launch — it removes the `Mcp-Session-Id` protocol session so any server instance can handle any request, drops the `initialize`/`initialized` handshake entirely, and rewrites authorization around standard OAuth/OIDC RFCs instead of bespoke wiring. A new **extensions framework** (reverse-DNS-namespaced, independently versioned) is how the **Tasks extension** (long-running async tool calls via `tasks/get`, `tasks/update`, `tasks/cancel`) and **MCP Apps** (server-rendered interactive UIs in sandboxed iframes) now ship. The **Enterprise-Managed Authorization** extension already reached stable status, adopted by Anthropic, Microsoft, and Okta. Beta SDKs with 2026-07-28 support now ship for Python, TypeScript, Go, and C#, ahead of the July 28 final spec. MCP is now the de-facto standard for AI tool integration; X (formerly Twitter) shipped a hosted MCP server for its platform API in July 2026. Enterprise adoption requires SSO-integrated auth, audit trails, and gateway behavior.

## Reading order

No mandatory prerequisites. CPU-only.

1. `01_react_from_scratch`: three-line agent loop, regex parser, and
   three tools.
2. `02_structured_outputs_three_ways`: flaky-LLM simulator; prompt
   vs validate+retry vs FSM.
3. `03_langgraph_state_machines`: StateGraph clone with
   conditional edges.
4. `04_dspy_3_miprov2`: 3×3 (instruction, demo) grid; MIPROv2 as
   5-sample random search.
5. `05_mcp_server_client`: JSON-RPC 2.0 tool server and synchronous
   client.
6. `06_autogen_0_4_vs_crewai`: draft/critique/revise pipeline two
   ways (AutoGen/AG2 and CrewAI).
7. `07_agent_evaluation_suite`: success rate, trajectory
   efficiency, and code-patch success.
