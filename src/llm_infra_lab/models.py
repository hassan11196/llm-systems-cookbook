"""Model shortlist used across the curriculum.

Every notebook that needs a demo LLM should pick one from this registry
rather than hardcoding a HuggingFace id. If a model is deprecated or moved,
one edit here fixes every notebook.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    params_m: int
    family: str
    dtype: str = "bfloat16"
    context_length: int = 2048
    chat: bool = False
    role: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "smollm2-135m": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-135M",
        params_m=135,
        family="smollm2",
        context_length=2048,
        role="smallest base model; KV cache and perplexity baseline",
    ),
    "smollm2-360m-instruct": ModelSpec(
        hf_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        params_m=360,
        family="smollm2",
        context_length=2048,
        chat=True,
        role="instruct baseline for eval tracks",
    ),
    "qwen2.5-0.5b": ModelSpec(
        hf_id="Qwen/Qwen2.5-0.5B",
        params_m=500,
        family="qwen2.5",
        context_length=32768,
        role="small base model for speculative decoding and DPO",
    ),
    "qwen2.5-0.5b-instruct": ModelSpec(
        hf_id="Qwen/Qwen2.5-0.5B-Instruct",
        params_m=500,
        family="qwen2.5",
        context_length=32768,
        chat=True,
        role="small instruct model for agents and RAG",
    ),
    "qwen2.5-1.5b": ModelSpec(
        hf_id="Qwen/Qwen2.5-1.5B",
        params_m=1500,
        family="qwen2.5",
        context_length=32768,
        role="larger experiments where 0.5B is too small",
    ),
    "qwen2.5-coder-0.5b-instruct": ModelSpec(
        hf_id="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        params_m=500,
        family="qwen2.5-coder",
        context_length=32768,
        chat=True,
        role="HumanEval code-generation notebook",
    ),
    "llama-3.2-1b": ModelSpec(
        hf_id="meta-llama/Llama-3.2-1B",
        params_m=1230,
        family="llama-3.2",
        context_length=131072,
        role="QLoRA and FSDP2 base model",
    ),
    "llama-3.2-1b-instruct": ModelSpec(
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        params_m=1230,
        family="llama-3.2",
        context_length=131072,
        chat=True,
        role="instruct variant for SFT/DPO pipelines",
    ),
    "tinyllama-1.1b-chat": ModelSpec(
        hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        params_m=1100,
        family="llama-1",
        context_length=2048,
        chat=True,
        role="chat-optimised small baseline",
    ),
    "phi-3.5-mini-instruct": ModelSpec(
        hf_id="microsoft/Phi-3.5-mini-instruct",
        params_m=3800,
        family="phi-3.5",
        context_length=131072,
        chat=True,
        role="compact instruct model for agents and eval judges",
    ),
}


def get(key: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``key``.

    Raises ``KeyError`` with a helpful message if the key is unknown.
    """

    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown model key {key!r}; available: {known}")
    return REGISTRY[key]
