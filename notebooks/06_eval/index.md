# Evaluation

```{admonition} What you'll learn in this part
:class: tip

- Perplexity three ways (sliding window / non-overlapping /
  single pass) and bits-per-character as a Shannon interpretation.
- Multiple-choice scoring with calibration and Expected Calibration
  Error.
- Chen et al. 2021 unbiased pass@k estimator for code benchmarks.
- Position, verbosity, and self-enhancement bias in
  LLM-as-judge setups; position-swap mitigation.
- Arena-style ratings via online Elo and Bradley-Terry.
- Long-context stress tests (NIAH, RULER composite).
- Contamination detection (n-gram overlap, Min-K Prob).
- Reconciling numbers across `lm-eval` and Inspect AI.
```


## Key terms used in this part

- **{term}`perplexity`** is the core language-modeling metric.
- **{term}`calibration`** asks whether confidence estimates match empirical correctness.
- **Expected Calibration Error (ECE)** summarizes calibration gap across confidence bins.
- **{term}`pass@k`** is the standard code-generation success metric.
- **{term}`LLM-as-judge`**, **{term}`Elo`**, and **{term}`Bradley-Terry`** are core concepts for preference-based evaluation.
- **{term}`GPQA`** and **{term}`HLE`** are the 2025 frontier benchmarks that
  replaced MMLU as the discriminating tests for top-tier models.
- **{term}`LiveCodeBench`** is a contamination-resistant alternative to
  HumanEval for coding evaluation.
- **{term}`ARC-AGI`** is a reasoning benchmark whose hard-problem score
  became a proxy for AGI progress in 2024-2025.
- **{term}`SWE-bench`** and its live-updatable successor **SWE-bench Live**
  measure real GitHub issue resolution; **{term}`Terminal-Bench`** 2.0
  (January 2026) extends agentic evaluation to multi-step CLI workflows.

## Reading order

No mandatory prerequisites — all CPU-safe, all from-scratch.

1. `01_perplexity_from_scratch` — the canonical metric derived and
   implemented three ways.
2. `02_mmlu_harness_calibration` — logit-based multiple-choice +
   ECE.
3. `03_humaneval_unbiased_pass_k` — sandboxed candidate execution +
   unbiased estimator.
4. `04_llm_as_judge_bias` — position and verbosity bias, quantified.
5. `05_arena_elo_bradley_terry` — pairwise preferences to rankings.
6. `06_long_context_niah_ruler` — decay model + RULER composite.
7. `07_contamination_detection` — n-gram overlap + Min-K Prob.
8. `08_lm_eval_inspect_ai` — cross-framework reconciliation on a
   synthetic task.

## Benchmark landscape (mid-2026)

The evaluation frontier has shifted since 2024. Several previously-challenging benchmarks are now saturated:

| Benchmark | 2024 SOTA | 2026 SOTA | Status |
|---|---|---|---|
| MMLU | ~86% | 88–94% | Saturated |
| HumanEval | ~95% | ~99% | Saturated |
| GSM8K | ~97% | ~99% | Saturated |
| GPQA-Diamond | ~70% | 91–94% | Active frontier |
| ARC-AGI-2 | ~5% | ~60–65% | Active frontier |
| AIME 2025 | — | 91–94% | Active frontier |
| SWE-bench Verified | ~45% | ~77–80% | Active frontier |

The notebooks in this part teach the **mechanics** of evaluation on un-saturated tasks; the numerical thresholds in the scoring checks target 2026-accessible open-weight models (Qwen2.5-0.5B, Phi-3.5-mini, SmolLM2-360M) that still show non-trivial variance on MMLU and HumanEval, making them pedagogically useful even as frontier models approach ceiling.
