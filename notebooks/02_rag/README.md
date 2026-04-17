# 02 — Retrieval-augmented generation

Nine notebooks that build retrieval pipelines from first principles. Each
notebook compares a technique against a plain baseline on a small BEIR slice
or a 10K-doc Wikipedia mini-corpus and reports the recall/NDCG lift the
technique is supposed to deliver.

| NN | Notebook | Hardware | Runtime | Focus |
|---:|---|---|---:|---|
| 01 | chunking strategies | T4 | 18 min | fixed / recursive / semantic / late chunking |
| 02 | FAISS dense retrieval | T4 | 22 min | Flat vs IVF-PQ vs HNSW Pareto |
| 03 | BM25, SPLADE, and RRF hybrid | T4 | 20 min | sparse + dense fusion |
| 04 | ColBERTv2 late interaction | T4 | 24 min | token-level MaxSim via pylate |
| 05 | two-stage reranking | T4 | 18 min | bi-encoder → cross-encoder |
| 06 | HyDE and query rewriting | T4 | 22 min | hypothetical-answer retrieval; multi-query; decomposition |
| 07 | RAPTOR hierarchical | T4 | 28 min | recursive GMM summarisation |
| 08 | GraphRAG with Leiden | T4 | 30 min | entity graph + community summaries |
| 09 | RAGAS evaluation harness | T4 | 20 min | faithfulness / answer relevancy / context precision+recall |

This track pins `transformers==4.46.3` (required by `sentence-transformers==5.4.1`
and `pylate==1.1.7`) and `ragas==0.2.14` (0.4.x broke the metric API).
Use a dedicated venv with `.[rag,eval]` to avoid clashing with training deps.
