"""Idempotent fixture refresher for production-track notebooks.

Re-records the JSON fixtures under ``notebooks/08_production/_fixtures/``
and ``notebooks/04_agents/_fixtures/`` from real API calls so each
notebook's REPLAY mode reflects current model behaviour.

Cheap, single-API fixtures (caching, litellm router, native tool-use,
hybrid RAG, ReAct traces, MCP round-trip) are refreshed inline.
Expensive fixtures whose canonical source is the notebook itself
(structured-outputs head-to-head with 800 calls, DSPy MIPROv2 compile,
multi-model Inspect AI eval) print a single-line "to regenerate, run
notebooks/08_production/<NN>.ipynb in LIVE mode" instruction; reproducing
their data here would just duplicate the notebook.

Usage::

    python scripts/refresh_fixtures.py --list           # registry
    python scripts/refresh_fixtures.py --only caching   # one fixture
    python scripts/refresh_fixtures.py                  # all available
    python scripts/refresh_fixtures.py --dry-run        # plan only

Required environment per fixture:

    caching, tool_use, react_traces, rag    ANTHROPIC_API_KEY
    litellm                                  ANTHROPIC_API_KEY + OPENAI_API_KEY
    mcp                                      none (uses subprocess + ``mcp`` package)

Models default to ``claude-sonnet-4-6`` / ``claude-haiku-4-5`` matching
the recorded fixtures; override via ``--model-anthropic``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROD = REPO / "notebooks/08_production/_fixtures"
AGENTS = REPO / "notebooks/04_agents/_fixtures"

DEFAULT_SONNET = "claude-sonnet-4-6"
DEFAULT_HAIKU = "claude-haiku-4-5"
DEFAULT_GPT4O_MINI = "gpt-4o-mini"


# --- helpers -----------------------------------------------------------------


def _require_env(*names: str) -> list[str]:
    missing = [n for n in names if not os.environ.get(n)]
    return missing


def _atomic_write_json(path: Path, data: dict, *, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would write {path.relative_to(REPO)} ({_size(data):,} bytes)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"  wrote {path.relative_to(REPO)}  ({path.stat().st_size:,} bytes)")


def _size(data: dict) -> int:
    return len(json.dumps(data, indent=2, ensure_ascii=False))


def _skip(name: str, reason: str) -> None:
    print(f"  [skip] {name}: {reason}")


# --- 01_caching --------------------------------------------------------------


CACHING_QUESTIONS = [
    "Which notebook covers PagedAttention?",
    "What is the prerequisite for the FlashAttention2 Triton notebook?",
    "Which track has the most notebooks in v0.1?",
    "Name the three KV cache compression methods covered in track 05.",
    "Which evaluation notebook teaches Bradley-Terry?",
]


def refresh_caching(*, model: str, dry_run: bool) -> None:
    name = "caching"
    if missing := _require_env("ANTHROPIC_API_KEY"):
        _skip(name, f"missing env: {', '.join(missing)}")
        return
    try:
        import anthropic
    except ImportError:
        _skip(name, "pip install anthropic")
        return

    doc = (REPO / "CURRICULUM_SPEC.md").read_text()[:25_000]
    client = anthropic.Anthropic()

    def call(question: str, *, use_cache: bool) -> dict:
        block: dict = {"type": "text", "text": doc}
        if use_cache:
            block["cache_control"] = {"type": "ephemeral"}
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=[block],
            messages=[{"role": "user", "content": question}],
        )
        latency = time.perf_counter() - t0
        u = resp.usage
        return {
            "question": question,
            "text": resp.content[0].text,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "latency_s": round(latency, 2),
        }

    print(f"  [{name}] running 5 uncached + 5 cached calls against {model}...")
    no_cache = [call(q, use_cache=False) for q in CACHING_QUESTIONS]
    cached = [call(q, use_cache=True) for q in CACHING_QUESTIONS]

    out = {
        "_comment": (
            "Recorded by scripts/refresh_fixtures.py. "
            f"Model {model}. Doc: first 25k chars of CURRICULUM_SPEC.md. "
            "Token counts from response.usage; latency from time.perf_counter."
        ),
        "model": model,
        "doc_tokens": no_cache[0]["input_tokens"] - 23,  # rough
        "no_cache": no_cache,
        "cached": cached,
    }
    _atomic_write_json(PROD / "01_caching.json", out, dry_run=dry_run)


# --- 02_litellm --------------------------------------------------------------


LITELLM_TICKETS = [
    "My production database is down and we have customers calling, please escalate immediately",
    "Hi, what's your refund policy for monthly plans?",
    "CONGRATULATIONS you have won a $1000 GIFT CARD click here to claim",
    "Login button does nothing on Firefox 127, console shows 401 from /api/auth",
    "Could you send me an invoice copy for March 2026?",
    "URGENT: payment system rejecting all cards since 02:14 UTC, blocking checkout",
    "Need help understanding the difference between Pro and Team plans",
    "Click https://bit.ly/2x to verify your account or be deleted",
    "Cannot generate a new API key, the dashboard hangs after I click Generate",
    "Just wanted to say I love the new dashboard redesign, great work team",
    "PCI compliance failure: card data appearing in plaintext logs, see ticket #4421",
    "How do I change my account email address from the settings page?",
    "Free Bitcoin generator earn 5 BTC per day no investment needed",
    "Webhook deliveries failing for 90 minutes with 500 errors from your end",
    "Question: do you offer educational discounts for students?",
    "DDoS in progress against our endpoint, can you confirm if it is on your side?",
    "Reset password email never arrives, tried 5 times over the past hour",
    "EARN $$$ working from home apply now NO experience required",
    "Got billed twice for the same month, transactions on April 12 and April 13",
    "Quick question - is there an SDK for Rust on the roadmap?",
]
LITELLM_TRUTH = [
    "urgent",
    "normal",
    "spam",
    "urgent",
    "normal",
    "urgent",
    "normal",
    "spam",
    "urgent",
    "normal",
    "urgent",
    "normal",
    "spam",
    "urgent",
    "normal",
    "urgent",
    "normal",
    "spam",
    "urgent",
    "normal",
]
LITELLM_LABELS = ["urgent", "normal", "spam"]
LITELLM_SYSTEM = (
    "Classify the support ticket as exactly one of: urgent, normal, spam. "
    "Reply with the single word and nothing else."
)


def refresh_litellm(
    *, model_anthropic_sonnet: str, model_anthropic_haiku: str, model_openai: str, dry_run: bool
) -> None:
    name = "litellm"
    if missing := _require_env("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        _skip(name, f"missing env: {', '.join(missing)}")
        return
    try:
        import litellm
    except ImportError:
        _skip(name, "pip install litellm")
        return

    def call(model: str, ticket: str) -> dict:
        t0 = time.perf_counter()
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": LITELLM_SYSTEM},
                {"role": "user", "content": ticket},
            ],
            max_tokens=8,
            temperature=0,
        )
        latency = time.perf_counter() - t0
        label = resp.choices[0].message.content.strip().lower()
        return {
            "label": label if label in LITELLM_LABELS else "normal",
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "cost_usd": round(litellm.completion_cost(completion_response=resp), 7),
            "latency_s": round(latency, 2),
        }

    sonnet_id = f"anthropic/{model_anthropic_sonnet}"
    haiku_id = f"anthropic/{model_anthropic_haiku}"
    gpt_id = f"openai/{model_openai}"

    print(f"  [{name}] 20 tickets × 3 providers = 60 calls...")
    sonnet_results = [call(sonnet_id, t) for t in LITELLM_TICKETS]
    haiku_results = [call(haiku_id, t) for t in LITELLM_TICKETS]
    gpt_results = [call(gpt_id, t) for t in LITELLM_TICKETS]

    print(f"  [{name}] router with primary={model_anthropic_sonnet}, fallback={model_openai}...")
    print(
        f"  [{name}] (LIVE refresh of fallback_run with injected 529s is non-trivial; reusing the existing"
    )
    print(
        f"  [{name}]  fallback_run sub-object from the on-disk fixture so the router demo stays runnable.)"
    )

    existing_path = PROD / "02_litellm.json"
    fallback_run = (
        json.loads(existing_path.read_text())["fallback_run"] if existing_path.exists() else {}
    )

    out = {
        "_comment": (
            "Recorded by scripts/refresh_fixtures.py. "
            "Each strategy classifies the same 20 support-ticket prompts. "
            "cost_usd from litellm.completion_cost(); latency from time.perf_counter."
        ),
        "labels": LITELLM_LABELS,
        "tickets": LITELLM_TICKETS,
        "ground_truth": LITELLM_TRUTH,
        "sonnet_only": {"model": model_anthropic_sonnet, "results": sonnet_results},
        "haiku_only": {"model": model_anthropic_haiku, "results": haiku_results},
        "gpt4o_mini": {"model": model_openai, "results": gpt_results},
        "fallback_run": fallback_run,
    }
    _atomic_write_json(PROD / "02_litellm.json", out, dry_run=dry_run)


# --- 03_tool_use -------------------------------------------------------------


TOOL_USE_TASKS = [
    ("Compute 12 + 30", "42"),
    ("What is 7 * 8?", "56"),
    ("Calculate (15 + 5) / 4", "5"),
    ("Compute 2 ** 10", "1024"),
    ("Evaluate 100 - 37", "63"),
    ("What is 1234 + 5678?", "6912"),
    ("Compute 9 * 9 - 1", "80"),
    ("Calculate 144 / 12", "12"),
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the tallest mountain?", "Everest"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the speed of light?", "299,792,458"),
    ("Who painted the Mona Lisa?", "Leonardo"),
    ("How many continents are there?", "Seven"),
    ("What is the current year?", "2026"),
    ("What is the current month?", "April"),
    ("What is the date tomorrow?", "2026-04-18"),
    ("What is the date yesterday?", "2026-04-16"),
]


def refresh_tool_use(*, model: str, dry_run: bool) -> None:
    name = "tool_use"
    if missing := _require_env("ANTHROPIC_API_KEY"):
        _skip(name, f"missing env: {', '.join(missing)}")
        return
    try:
        import anthropic
    except ImportError:
        _skip(name, "pip install anthropic")
        return

    tools = [
        {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression.",
            "input_schema": {
                "type": "object",
                "required": ["expression"],
                "properties": {"expression": {"type": "string"}},
            },
        },
        {
            "name": "wiki_lookup",
            "description": "Look up a short fact.",
            "input_schema": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
        {
            "name": "get_date",
            "description": "Get today, tomorrow, year, month, etc.",
            "input_schema": {
                "type": "object",
                "required": ["spec"],
                "properties": {
                    "spec": {
                        "type": "string",
                        "enum": ["today", "tomorrow", "yesterday", "year", "month", "day"],
                    }
                },
            },
        },
    ]

    client = anthropic.Anthropic()

    def native_run(task: str) -> dict:
        messages: list = [{"role": "user", "content": task}]
        used_tool, used_input = None, None
        n_tool_calls = 0
        in_tok = out_tok = 0
        t0 = time.perf_counter()
        while True:
            resp = client.messages.create(
                model=model,
                max_tokens=512,
                tools=tools,
                messages=messages,
            )
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            if resp.stop_reason == "end_turn":
                final = "".join(b.text for b in resp.content if b.type == "text")
                latency = time.perf_counter() - t0
                return {
                    "tool": used_tool,
                    "tool_input": used_input,
                    "final": final,
                    "n_tool_calls": n_tool_calls,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "cost_usd": round((in_tok * 3 + out_tok * 15) / 1e6, 6),
                    "latency_s": round(latency, 2),
                }
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                n_tool_calls += 1
                used_tool, used_input = block.name, dict(block.input)
                # The script doesn't actually run the tools — it records what
                # the model asked for. The notebook's tool functions do the
                # execution on replay.
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": "OK"})
            messages.append({"role": "user", "content": results})

    print(f"  [{name}] 20 native tool_use runs against {model}...")
    native = [native_run(task) for task, _ in TOOL_USE_TASKS]

    print(f"  [{name}] prompted_react sub-object regenerated by running")
    print(f"  [{name}] notebooks/08_production/03_tool_use_agent.ipynb in LIVE mode")
    print(f"  [{name}] with PROMPTED_REACT=1 — preserving existing block.")

    existing_path = PROD / "03_tool_use.json"
    prompted_react = (
        json.loads(existing_path.read_text())["prompted_react"] if existing_path.exists() else []
    )

    out = {
        "_comment": "Recorded by scripts/refresh_fixtures.py. native_tool_use is fresh; prompted_react preserved from the on-disk fixture.",
        "model": model,
        "tasks": [list(t) for t in TOOL_USE_TASKS],
        "native_tool_use": native,
        "prompted_react": prompted_react,
    }
    _atomic_write_json(PROD / "03_tool_use.json", out, dry_run=dry_run)


# --- 05_rag ------------------------------------------------------------------


RAG_PASSAGES = [
    "Spectrum is a code-search platform that indexes Git repositories and exposes a low-latency search API. It supports 14 languages out of the box, including Python, TypeScript, Rust, and Go.",
    "The free tier of Spectrum allows up to 5,000 indexed files and 100 queries per minute. Teams looking for higher quotas can upgrade to the Pro plan at $29 per developer per month.",
    "All search traffic to Spectrum is encrypted with TLS 1.3, and source code is encrypted at rest using AES-256-GCM. Customer keys are managed through AWS KMS or GCP Cloud KMS depending on the deployment region.",
    "Authentication is handled via OAuth 2.0 with GitHub, GitLab, and Bitbucket as identity providers. Personal access tokens are also supported for CI/CD pipelines that cannot use OAuth.",
    "Spectrum offers webhook integrations for Slack, Microsoft Teams, and Discord. Webhooks fire on indexing completion, configurable saved-search hits, and access-control violations.",
    "The query language supports regex, structural patterns (e.g. function definitions in Python), and language-aware tokens like 'sym:funcName' to match symbol identifiers regardless of casing convention.",
    "Latency targets: p50 query latency is under 80 ms for repositories up to 10 GB; p99 stays under 350 ms. Latency degrades gracefully on cold-start indexes during the first minute after deploy.",
    "Customer support is available via email for the free plan and includes a 4-hour response SLA on the Pro plan. Enterprise customers receive a dedicated Slack Connect channel and a 1-hour SLA.",
    "Spectrum's data residency program lets enterprise customers pin all indexed code and search logs to one of seven regions: us-east, us-west, eu-west, eu-central, ap-southeast, ap-northeast, sa-east.",
    "The CLI ships as a single static binary for Linux, macOS, and Windows. It is published on Homebrew, apt, and Chocolatey, and a Docker image is available at ghcr.io/spectrum-search/cli.",
    "Spectrum's hosted API is rate-limited per organisation: 600 requests per minute by default, raised to 6,000 rpm for Pro and 60,000 rpm for Enterprise plans. Rate-limit headers follow the GitHub convention.",
    "On-prem deployment is supported via a Helm chart that runs on any CNCF-certified Kubernetes 1.28+ cluster. The chart provisions PostgreSQL, OpenSearch, and an internal MinIO instance for object storage.",
]
RAG_QUERIES = [
    "What is the per-developer monthly price of the Pro plan?",
    "Which languages does Spectrum support out of the box?",
    "What is the p99 query latency target?",
    "How is data encrypted at rest?",
    "What are the rate-limit defaults for the Pro plan?",
]
RAG_TRUTH = [2, 1, 7, 3, 11]
RAG_SYSTEM = (
    "Answer the question using ONLY the provided passages. "
    "Cite passages inline like [3] for passage 3. If no passage answers, "
    "say 'no answer in context'."
)


def refresh_rag(*, model: str, dry_run: bool) -> None:
    name = "rag"
    if missing := _require_env("ANTHROPIC_API_KEY"):
        _skip(name, f"missing env: {', '.join(missing)}")
        return
    try:
        import anthropic
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        _skip(name, f"missing import: {e.name} (pip install anthropic sentence-transformers)")
        return
    import re

    print(f"  [{name}] encoding 12 passages + 5 queries with bge-small-en-v1.5...")
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    p_emb = enc.encode(RAG_PASSAGES, normalize_embeddings=True)
    q_emb = enc.encode(RAG_QUERIES, normalize_embeddings=True)
    dense_scores = [[round(float(s), 3) for s in row] for row in (q_emb @ p_emb.T)]

    # Replicate the notebook's BM25 + RRF to pick top-3 per query for the LLM.
    import math
    from collections import Counter

    _TOK = re.compile(r"[a-z0-9]+")

    def toks(s):
        return _TOK.findall(s.lower())

    docs = [toks(p) for p in RAG_PASSAGES]
    avgdl = sum(len(d) for d in docs) / len(docs)
    df: dict[str, int] = Counter()
    for d in docs:
        for w in set(d):
            df[w] += 1
    N = len(docs)

    def bm25(q):
        out = [0.0] * N
        for w in toks(q):
            if w not in df:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            for i, d in enumerate(docs):
                tf = d.count(w)
                if tf == 0:
                    continue
                out[i] += idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * len(d) / avgdl))
        return out

    bm25_scores = [bm25(q) for q in RAG_QUERIES]

    def rrf(score_lists, k=60):
        n = len(score_lists[0])
        fused = [0.0] * n
        for scs in score_lists:
            ranks = sorted(range(n), key=lambda i: -scs[i])
            for r, idx in enumerate(ranks):
                fused[idx] += 1.0 / (k + r + 1)
        return fused

    top_ks = []
    for i in range(len(RAG_QUERIES)):
        fused = rrf([bm25_scores[i], dense_scores[i]])
        top_ks.append(sorted(range(len(fused)), key=lambda j: -fused[j])[:3])

    print(f"  [{name}] running 5 cited-answer calls against {model}...")
    client = anthropic.Anthropic()
    answers: dict[str, dict] = {}
    for q, top in zip(RAG_QUERIES, top_ks, strict=True):
        ctx = "\n".join(f"[{i + 1}] {RAG_PASSAGES[i]}" for i in top)
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=RAG_SYSTEM,
            messages=[{"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", text)]
        answers[q] = {"answer": text, "citations": cites}

    out = {
        "_comment": (
            "Recorded by scripts/refresh_fixtures.py. "
            f"Dense scores from bge-small-en-v1.5; cited answers from {model}."
        ),
        "model": model,
        "passages": RAG_PASSAGES,
        "queries": RAG_QUERIES,
        "expected_passage_ids": RAG_TRUTH,
        "dense_scores": dense_scores,
        "answers": answers,
    }
    _atomic_write_json(PROD / "05_rag.json", out, dry_run=dry_run)


# --- 06_mcp ------------------------------------------------------------------


def refresh_mcp(*, dry_run: bool) -> None:
    name = "mcp"
    try:
        import mcp  # noqa: F401
    except ImportError:
        _skip(name, "pip install 'mcp>=1.2'")
        return

    server_path = REPO / "scripts/_lab_mcp_server.py"
    if not server_path.exists():
        _skip(
            name,
            f"server source missing at {server_path.relative_to(REPO)} "
            "(generated by 08_production/06_mcp_real_server.ipynb's first run)",
        )
        return

    print(f"  [{name}] spawning {server_path.relative_to(REPO)} via stdio_client...")

    async def drive():
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(command="python", args=[str(server_path)])
        async with stdio_client(params) as (read, write), ClientSession(read, write) as sess:
            init = await sess.initialize()
            tools_resp = await sess.list_tools()
            tool_calls = [
                {"name": "calculator", "arguments": {"expression": "2 ** 10 + 5"}},
                {"name": "wiki_lookup", "arguments": {"query": "capital of france"}},
                {"name": "get_date", "arguments": {"spec": "today"}},
            ]
            results = []
            for tc in tool_calls:
                resp = await sess.call_tool(tc["name"], tc["arguments"])
                text = next((c.text for c in resp.content if c.type == "text"), "")
                results.append({**tc, "result": text})
            return {
                "init": {
                    "name": init.serverInfo.name,
                    "version": init.serverInfo.version,
                    "protocolVersion": init.protocolVersion,
                },
                "tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                    for t in tools_resp.tools
                ],
                "calls": results,
            }

    session = asyncio.run(drive())

    out = {
        "_comment": (
            "Recorded by scripts/refresh_fixtures.py against a real mcp.Server "
            "spawned via stdio. tools_list is session.list_tools(); tool_calls are "
            "the 'name(args) -> result' triples driven by mcp.client.ClientSession."
        ),
        "server_name": session["init"]["name"],
        "server_version": session["init"]["version"],
        "protocol_version": session["init"]["protocolVersion"],
        "tools_list": session["tools"],
        "tool_calls": session["calls"],
        "claude_code_config_snippet": {
            "mcpServers": {
                session["init"]["name"]: {
                    "command": "python",
                    "args": [str(server_path)],
                }
            }
        },
    }
    _atomic_write_json(PROD / "06_mcp.json", out, dry_run=dry_run)


# --- 04_agents/_fixtures/01_react_traces ------------------------------------


REACT_SYSTEM = (
    "You are a ReAct agent. For each step, respond with 'Thought:' then either "
    "'Action: tool_name[arg]' (tools: calculator, wiki_search, get_datetime) or "
    "'Final Answer: <answer>'. Keep replies short."
)


def refresh_react_traces(*, model: str, dry_run: bool) -> None:
    name = "react_traces"
    if missing := _require_env("ANTHROPIC_API_KEY"):
        _skip(name, f"missing env: {', '.join(missing)}")
        return
    try:
        import anthropic
    except ImportError:
        _skip(name, "pip install anthropic")
        return

    client = anthropic.Anthropic()

    def trace_for(task: str, expected: str) -> dict:
        scratch = ""
        rounds: list[str] = []
        in_tok = out_tok = 0
        t0 = time.perf_counter()
        for _ in range(4):  # safety cap
            resp = client.messages.create(
                model=model,
                max_tokens=200,
                system=REACT_SYSTEM,
                messages=[{"role": "user", "content": task + "\n" + scratch}],
            )
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            text = "".join(b.text for b in resp.content if b.type == "text")
            rounds.append(text)
            if "Final Answer" in text:
                break
            # Mock the observation so the next round has something to react to.
            scratch += f"\n{text}\nObservation: {expected}\n"
        latency = time.perf_counter() - t0
        return {
            "task": task,
            "expected": expected,
            "rounds": rounds,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost_usd": round((in_tok * 0.8 + out_tok * 4.0) / 1e6, 7),  # haiku rates
            "latency_s": round(latency, 2),
        }

    print(f"  [{name}] 20 ReAct trajectories against {model}...")
    trajs = [trace_for(t, e) for t, e in TOOL_USE_TASKS]

    out = {
        "_comment": (
            "Recorded by scripts/refresh_fixtures.py. Per-task assistant outputs "
            f"from {model} with the ReAct system prompt below. The notebook "
            "replays these strings; the parser, tools, and loop run for real."
        ),
        "model": model,
        "system_prompt": REACT_SYSTEM,
        "trajectories": trajs,
    }
    _atomic_write_json(AGENTS / "01_react_traces.json", out, dry_run=dry_run)


# --- non-trivial fixtures: redirect to notebook ------------------------------


_NOTEBOOK_OWNED = {
    "structured": (
        "notebooks/08_production/04_structured_outputs_real.ipynb",
        "200 prompts × 4 strategies; ~$0.40 of API spend",
    ),
    "dspy": (
        "notebooks/08_production/07_dspy_miprov2_optimizer.ipynb",
        "MIPROv2 light compile (~120 LM calls / ~$0.014)",
    ),
    "inspect": (
        "notebooks/08_production/08_inspect_ai_eval_harness.ipynb",
        "Inspect AI eval across two models",
    ),
}


def refresh_notebook_owned(name: str, *, dry_run: bool) -> None:
    nb, why = _NOTEBOOK_OWNED[name]
    print(f"  [{name}] regenerate by running {nb} in LIVE mode")
    print(f"  [{name}] ({why}). Save the resulting fixture by hand; this script")
    print(f"  [{name}] does not duplicate the notebook's data-collection logic.")


# --- registry + CLI ----------------------------------------------------------


def build_registry(args: argparse.Namespace) -> dict[str, Callable[[], None]]:
    return {
        "caching": lambda: refresh_caching(model=args.model_anthropic, dry_run=args.dry_run),
        "litellm": lambda: refresh_litellm(
            model_anthropic_sonnet=args.model_anthropic,
            model_anthropic_haiku=args.model_anthropic_haiku,
            model_openai=args.model_openai,
            dry_run=args.dry_run,
        ),
        "tool_use": lambda: refresh_tool_use(model=args.model_anthropic, dry_run=args.dry_run),
        "rag": lambda: refresh_rag(model=args.model_anthropic, dry_run=args.dry_run),
        "mcp": lambda: refresh_mcp(dry_run=args.dry_run),
        "react_traces": lambda: refresh_react_traces(
            model=args.model_anthropic_haiku, dry_run=args.dry_run
        ),
        "structured": lambda: refresh_notebook_owned("structured", dry_run=args.dry_run),
        "dspy": lambda: refresh_notebook_owned("dspy", dry_run=args.dry_run),
        "inspect": lambda: refresh_notebook_owned("inspect", dry_run=args.dry_run),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="list known fixture names and exit")
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="refresh only the named fixture (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be written; do not call APIs or write files",
    )
    parser.add_argument(
        "--model-anthropic",
        default=DEFAULT_SONNET,
        help=f"Anthropic Sonnet-class model (default: {DEFAULT_SONNET})",
    )
    parser.add_argument(
        "--model-anthropic-haiku",
        default=DEFAULT_HAIKU,
        help=f"Anthropic Haiku-class model (default: {DEFAULT_HAIKU})",
    )
    parser.add_argument(
        "--model-openai",
        default=DEFAULT_GPT4O_MINI,
        help=f"OpenAI model id (default: {DEFAULT_GPT4O_MINI})",
    )
    args = parser.parse_args(argv)

    registry = build_registry(args)
    names = args.only or list(registry.keys())

    unknown = [n for n in names if n not in registry]
    if unknown:
        print(f"unknown fixture(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(registry)}", file=sys.stderr)
        return 2

    if args.list:
        print("Available fixtures (run with --only NAME to refresh a single one):\n")
        for n in registry:
            owner = "notebook-owned" if n in _NOTEBOOK_OWNED else "scripted"
            print(f"  {n:<14}  ({owner})")
        return 0

    print(
        f"refresh_fixtures.py  ({'DRY-RUN' if args.dry_run else 'LIVE'}, {len(names)} fixture(s))"
    )
    for name in names:
        print(f"\n{name}")
        try:
            registry[name]()
        except Exception as e:  # noqa: BLE001 — surface every failure with context
            print(f"  [error] {name}: {type(e).__name__}: {e}", file=sys.stderr)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
