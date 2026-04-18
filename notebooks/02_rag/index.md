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
```

## Reading order

No mandatory prerequisites — this part is self-contained and CPU-safe.

1. `01_chunking_strategies` — fixed / recursive / semantic / late
   chunking on a tiny synthetic corpus.
2. `02_faiss_dense_retrieval` — flat, IVF-PQ-style, and tiny HNSW
   indices from scratch.
3. `03_bm25_splade_rrf_hybrid` — BM25 + SPLADE stub + RRF fusion.
4. `04_colbertv2_late_interaction` — MaxSim over stubbed per-token
   embeddings.
5. `05_two_stage_reranking` — bi-encoder → cross-encoder
   rerank-top-N.
6. `06_hyde_query_rewriting` — hypothetical answer embedding +
   multi-query RRF.
7. `07_raptor_hierarchical` — k-means + nearest-to-centroid
   summarisation tree.
8. `08_graphrag_leiden` — entity extraction + community detection
   + summary-level retrieval.
9. `09_ragas_evaluation` — faithfulness, answer relevancy, context
   precision/recall.
```
