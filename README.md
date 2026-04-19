# LLM Systems Cookbook

[![Book](https://img.shields.io/badge/Read-The%20Book-2ea44f?style=for-the-badge)](https://hassan11196.github.io/llm-systems-cookbook/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hassan11196/llm-systems-cookbook/blob/main/intro.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A hands-on curriculum for modern LLM systems engineering.** 61 Jupyter
notebooks covering inference, retrieval, training, agents, serving,
evaluation, and GPU programming. Each notebook reimplements a core
technique from first principles, compares against a production tool,
and self-scores with numerical checks.

Target audience: a computer-science undergrad who wants to go from
"I know what softmax is" to "I can reason about LLM serving
economics." No prior deep-learning background assumed.

## ▶️ Run in Colab - no install

Every notebook has an Open-in-Colab badge at the top. Landing page:

👉 **[hassan11196.github.io/llm-systems-cookbook](https://hassan11196.github.io/llm-systems-cookbook/)**

Start with [`07_gpu/01_gpu_architecture_tour`](notebooks/07_gpu/01_gpu_architecture_tour.ipynb)
if you're new to GPU programming; otherwise jump to the track you
want from the book's landing page.

## The seven parts

| Part | Chapters | Theme |
|---|---|---|
| **I · Foundations** | [9](notebooks/07_gpu/index.md) | GPU architecture, Triton, roofline |
| **II · Inference engines** | [10](notebooks/01_inference/index.md) | KV cache, PagedAttention, speculative, SARATHI |
| **III · Serving & scaling** | [11](notebooks/05_serving/index.md) | KV variants/compression, quantisation, MoE, DistServe |
| **IV · Training** | [2 of 8](notebooks/03_training/index.md) | Mixed precision, FSDP2; more landing soon |
| **V · Retrieval-augmented generation** | [9](notebooks/02_rag/index.md) | Chunking, indices, hybrid, RAPTOR, GraphRAG, RAGAS |
| **VI · Agent frameworks** | [7](notebooks/04_agents/index.md) | ReAct, structured outputs, LangGraph, DSPy, MCP |
| **VII · Evaluation** | [8](notebooks/06_eval/index.md) | Perplexity, MMLU, HumanEval, Arena, NIAH, contamination |

Full 61-notebook spec: [`CURRICULUM_SPEC.md`](CURRICULUM_SPEC.md).

## Run it locally

```bash
# Book build (jupyter-book + sphinx-book-theme + myst-nb)
pip install -e ".[book,dev]"
make book          # builds into _build/html/
open _build/html/index.html

# Or run notebooks directly (pick only the tracks you need)
pip install -e ".[inference,serving,gpu]"     # GPU-bound parts
pip install -e ".[rag,eval,agents,dev]"       # fully CPU-safe
jupyter lab notebooks/
python -m scoring.run_all                      # aggregate scores/*.json
```

## Hardware

Most notebooks run on a free Colab T4. Three require more:

- `01_inference/05_flashattention2_triton.ipynb` - Ampere+ GPU
- `07_gpu/04_triton_flashattention.ipynb` - Ampere+ GPU
- `07_gpu/07_nsight_profiling.ipynb` - local GPU or Colab Pro

Each chapter declares its requirements in its header cell.

## Layout

```
llm-systems-cookbook/
├── intro.md                      # book landing page
├── _toc.yml                      # seven-part table of contents
├── _config.yml                   # Jupyter Book config (launch buttons, theme)
├── environment.yml               # Binder / conda-forge reproducible env
├── CITATION.cff                  # academic citation metadata
├── CURRICULUM_SPEC.md            # per-chapter specification
├── notebooks/                    # the 61 chapters, grouped by track
│   ├── 01_inference/index.md
│   ├── 02_rag/index.md
│   ├── …
├── src/llm_systems_cookbook/     # shared helpers (hardware_check, seed, etc.)
├── scoring/                      # Scorer harness + aggregator + unit tests
├── scripts/                      # fetch_data.py, warm_cache.py
└── .github/workflows/            # CI (lint + CPU-safe notebook execution + Pages deploy)
```

## Citation

```bibtex
@misc{llm_systems_cookbook,
  author  = {Ahmed, Muhammad Hassan},
  title   = {The LLM Systems Cookbook},
  year    = {2026},
  url     = {https://github.com/hassan11196/llm-systems-cookbook},
}
```

## Acknowledgements

Format and structure inspired by
[Project Pythia cookbooks](https://projectpythia.org/),
[EECS 245 notes](https://notes.eecs245.org/), and
[IRSA tutorials](https://caltech-ipac.github.io/irsa-tutorials/).

## License

MIT.
