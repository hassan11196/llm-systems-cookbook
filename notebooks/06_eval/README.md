# 06 — Evaluation

Eight notebooks on rigorous LLM measurement: perplexity on an unambiguous
tokenisation, MMLU with calibration, unbiased pass@k on HumanEval, judge-bias
analysis, Bradley-Terry-based Arena Elo, long-context NIAH/RULER, contamination
detection, and a cross-framework reproducibility check between lm-eval and
Inspect AI.

| NN | Notebook | Hardware | Runtime | Focus |
|---:|---|---|---:|---|
| 01 | perplexity from scratch | CPU / T4 | 10 min | sliding-window PPL vs non-overlapping vs lm-eval |
| 02 | MMLU harness + calibration | T4 | 15 min | logit-normalised scoring, ECE |
| 03 | HumanEval unbiased pass@k | T4 | 18 min | Chen et al. unbiased estimator |
| 04 | LLM-as-judge bias | T4 | 15 min | position, verbosity, self-enhancement |
| 05 | Arena Elo (Bradley-Terry) | CPU | 10 min | pairwise ranking from preference data |
| 06 | long-context NIAH / RULER | T4 | 20 min | needle-in-a-haystack at multiple depths |
| 07 | contamination detection | CPU | 12 min | n-gram overlap, min-k-prob |
| 08 | lm-eval + Inspect AI reproducibility | T4 | 20 min | cross-framework agreement |
