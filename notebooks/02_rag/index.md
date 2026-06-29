# Retrieval-augmented generation

```{admonition} What you'll learn in this part
:class: tip

- Chunking strategies that preserve semantic coherence.
- Dense retrieval (FAISS flat / IVF-PQ / HNSW) vs sparse (BM25,
  SPLADE) vs their RRF hybrid.
- ColBERTv2 late-interaction scoring and its storage tradeoff.
- Reranking, HyDE, multi-query rewriting.
- Hierarchical retrieval (RAPTOR) and graph-based retrieval
  (GraphRAG + Leiden communities).
- RAGAS evaluation metrics implemented from scratch.

These building blocks are the foundation of agentic RAG systems,
where an agent controls the retrieval loop, issues follow-up queries,
and applies corrective retrieval when initial results are poor.
See the glossary entries for **{term}`agentic RAG`** and
**{term}`corrective RAG`** for an overview of how these chapters
compose into a full agentic pipeline.
```


## Key terms used in this part

- **{term}`dense retrieval`** retrieves by vector similarity; **{term}`sparse retrieval`** retrieves by lexical overlap.
- **{term}`BM25`**, **{term}`SPLADE`**, and **{term}`RRF`** are baseline building blocks for hybrid retrieval systems.
- **{term}`ColBERT`** is a late-interaction retriever that balances quality and serving cost.
- **{term}`HyDE`** and **{term}`reranking`** are common techniques for improving top-k relevance.
- **{term}`agentic RAG`** and **{term}`corrective RAG`** are the dominant production patterns in 2025-2026.

## Reading order

No mandatory prerequisites. This part is self-contained and CPU-safe.

1. `01_chunking_strategies`: fixed / recursive / semantic / late
   chunking on a tiny synthetic corpus.
2. `02_faiss_dense_retrieval`: flat, IVF-PQ-style, and tiny HNSW
   indices from scratch.
3. `03_bm25_splade_rrf_hybrid`: BM25 + SPLADE stub + RRF fusion.
4. `04_colbertv2_late_interaction`: MaxSim over stubbed per-token
   embeddings.
5. `05_two_stage_reranking`: bi-encoder → cross-encoder
   rerank-top-N.
6. `06_hyde_query_rewriting`: hypothetical answer embedding +
   multi-query RRF.
7. `07_raptor_hierarchical`: k-means + nearest-to-centroid
   summarisation tree.
8. `08_graphrag_leiden`: entity extraction + community detection
   + summary-level retrieval.
9. `09_ragas_evaluation`: faithfulness, answer relevancy, context
   precision/recall.
