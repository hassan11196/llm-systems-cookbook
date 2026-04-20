# Agent frameworks

```{admonition} What you'll learn in this part
:class: tip

- ReAct's parser + tool-call loop from scratch.
- Three strategies for structured LLM outputs (prompt-only,
  Pydantic validate+retry, FSM-constrained).
- State-machine-shaped agents (LangGraph-style 50-line clone).
- DSPy's signature + MIPROv2 prompt optimisation.
- The Model Context Protocol as a 2-file server + client.
- Conversation-driven (AutoGen) vs role-driven (CrewAI) multi-agent
  idioms.
- Evaluating agents with τ-bench / SWE-bench-shaped benchmarks.
```


## Key terms used in this part

- **{term}`ReAct`** is the baseline loop pattern used to teach tool-using agents.
- **{term}`structured outputs`** means constraining model outputs to a
  schema instead of free-form text.
- **FSM (Finite-State Machine)** constraints are one way to enforce valid
  output structure during decoding.
- **{term}`MCP`** exposes tools/data over JSON-RPC so clients can use them uniformly.
- **{term}`DSPy`** frames prompts/pipelines as optimizable programs.

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
   ways.
7. `07_agent_evaluation_suite` — success rate + trajectory
   efficiency + code-patch success.
```
