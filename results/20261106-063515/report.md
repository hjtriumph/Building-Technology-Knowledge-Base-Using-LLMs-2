# Benchmark report

Run: 20261106-063515
Rows: 810
Answerable quality means exclude unanswerable cases; refusal rates are reported separately.

## Overall

| Method | Questions | Rows | Completeness | Relevance | Faithfulness | Citation P | Key points | Latency p50 s | Latency p95 s | Cost CNY | Refusal (unans) | False refusal | Error rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llm | 22 | 66 | 0.053 | 0.106 | 0.000 | 0.000 | 0.361 | 3.020 | 4.612 | 0.172 | 1.000 | 0.879 | 0.000 |
| llm_structured | 22 | 66 | 0.083 | 0.258 | 0.000 | 0.000 | 0.373 | 3.928 | 4.743 | 0.247 | 1.000 | 0.545 | 0.000 |
| rag_0shot | 22 | 66 | 0.833 | 0.886 | 0.996 | 0.662 | 0.814 | 8.358 | 10.037 | 0.615 | 0.917 | 0.061 | 0.000 |
| fewshot_1 | 22 | 66 | 0.864 | 0.894 | 0.985 | 0.658 | 0.821 | 8.310 | 9.907 | 0.625 | 1.000 | 0.076 | 0.000 |
| fewshot_3 | 22 | 66 | 0.833 | 0.871 | 0.978 | 0.663 | 0.822 | 7.842 | 9.813 | 0.682 | 1.000 | 0.061 | 0.000 |
| fewshot_5 | 22 | 66 | 0.909 | 0.932 | 0.965 | 0.684 | 0.848 | 7.520 | 9.566 | 0.760 | 1.000 | 0.030 | 0.000 |
| fewshot_no_cot_3 | 22 | 66 | 0.705 | 0.864 | 0.971 | 0.000 | 0.697 | 3.794 | 4.874 | 0.385 | 1.000 | 0.091 | 0.000 |
| rag_k18 | 22 | 66 | 0.924 | 0.962 | 0.998 | 0.547 | 0.887 | 9.218 | 11.278 | 1.209 | 0.792 | 0.000 | 0.000 |
| agent | 22 | 66 | 0.811 | 0.894 | 0.993 | 0.554 | 0.762 | 13.338 | 15.618 | 1.556 | 1.000 | 0.076 | 0.000 |

## Paired comparisons (completeness, question-level means)

| Baseline | Compare | Metric | Pairs | Mean diff | CI low | CI high | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rag_0shot | fewshot_5 | completeness | 22 | 0.076 | 0.000 | 0.189 | 0.088 |
| rag_0shot | rag_k18 | completeness | 22 | 0.091 | 0.008 | 0.189 | 0.025 |
| rag_0shot | agent | completeness | 22 | -0.023 | -0.189 | 0.144 | 0.829 |
| rag_k18 | agent | completeness | 22 | -0.114 | -0.273 | 0.015 | 0.126 |

## By question type

| Method | Type | Questions | Rows | Completeness | Faithfulness | Refusal | Latency p50 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm | method | 10 | 30 | 0.117 | 0.000 | 0.733 | 3.391 |
| llm | multi_hop | 6 | 18 | 0.000 | 0.000 | 1.000 | 2.689 |
| llm | numeric | 5 | 15 | 0.000 | 0.000 | 1.000 | 2.942 |
| llm | single_hop | 1 | 3 | 0.000 | 0.000 | 1.000 | 2.739 |
| llm | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 1.708 |
| llm_structured | method | 10 | 30 | 0.183 | 0.000 | 0.367 | 4.155 |
| llm_structured | multi_hop | 6 | 18 | 0.000 | 0.000 | 0.611 | 3.350 |
| llm_structured | numeric | 5 | 15 | 0.000 | 0.000 | 0.800 | 3.929 |
| llm_structured | single_hop | 1 | 3 | 0.000 | 0.000 | 0.667 | 4.544 |
| llm_structured | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 2.334 |
| rag_0shot | method | 10 | 30 | 0.750 | 0.993 | 0.100 | 9.214 |
| rag_0shot | multi_hop | 6 | 18 | 0.806 | 1.000 | 0.056 | 7.742 |
| rag_0shot | numeric | 5 | 15 | 1.000 | 0.994 | 0.000 | 8.057 |
| rag_0shot | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 7.180 |
| rag_0shot | unanswerable | 0 | 24 | 0.000 | 0.000 | 0.917 | 3.037 |
| fewshot_1 | method | 10 | 30 | 0.800 | 0.995 | 0.100 | 8.617 |
| fewshot_1 | multi_hop | 6 | 18 | 0.833 | 0.977 | 0.111 | 8.058 |
| fewshot_1 | numeric | 5 | 15 | 1.000 | 0.973 | 0.000 | 8.327 |
| fewshot_1 | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 6.210 |
| fewshot_1 | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 2.706 |
| fewshot_3 | method | 10 | 30 | 0.800 | 0.971 | 0.100 | 8.280 |
| fewshot_3 | multi_hop | 6 | 18 | 0.722 | 0.978 | 0.056 | 7.644 |
| fewshot_3 | numeric | 5 | 15 | 1.000 | 1.000 | 0.000 | 7.767 |
| fewshot_3 | single_hop | 1 | 3 | 1.000 | 0.944 | 0.000 | 6.272 |
| fewshot_3 | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 2.543 |
| fewshot_5 | method | 10 | 30 | 0.800 | 0.979 | 0.067 | 8.488 |
| fewshot_5 | multi_hop | 6 | 18 | 1.000 | 0.934 | 0.000 | 7.348 |
| fewshot_5 | numeric | 5 | 15 | 1.000 | 0.967 | 0.000 | 7.502 |
| fewshot_5 | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 6.061 |
| fewshot_5 | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 2.571 |
| fewshot_no_cot_3 | method | 10 | 30 | 0.617 | 1.000 | 0.100 | 3.874 |
| fewshot_no_cot_3 | multi_hop | 6 | 18 | 0.778 | 0.883 | 0.167 | 3.942 |
| fewshot_no_cot_3 | numeric | 5 | 15 | 0.733 | 1.000 | 0.000 | 3.375 |
| fewshot_no_cot_3 | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 0.643 |
| fewshot_no_cot_3 | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 0.923 |
| rag_k18 | method | 10 | 30 | 0.933 | 1.000 | 0.000 | 9.699 |
| rag_k18 | multi_hop | 6 | 18 | 0.833 | 0.993 | 0.000 | 9.079 |
| rag_k18 | numeric | 5 | 15 | 1.000 | 1.000 | 0.000 | 8.785 |
| rag_k18 | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 7.760 |
| rag_k18 | unanswerable | 0 | 24 | 0.000 | 0.000 | 0.792 | 3.159 |
| agent | method | 10 | 30 | 0.833 | 0.995 | 0.100 | 14.035 |
| agent | multi_hop | 6 | 18 | 0.917 | 0.989 | 0.056 | 12.210 |
| agent | numeric | 5 | 15 | 0.600 | 0.994 | 0.067 | 13.717 |
| agent | single_hop | 1 | 3 | 1.000 | 1.000 | 0.000 | 12.528 |
| agent | unanswerable | 0 | 24 | 0.000 | 0.000 | 1.000 | 7.148 |

## Notes

- Generation temperature: 0.2; judge temperature: 0.0; retrieval: MMR, k=6, fetch_k=20.
- Quality metrics average the three repeats per question first, then bootstrap over questions (n≈30), not rows.
- Paired comparisons use question-level completeness and a paired bootstrap against the stated baseline.
- Faithfulness and citation precision exclude refused answers. Closed-book arms have no retrieved context, so faithfulness is not applicable (N/A).
- Latency is reported as median and p95 because one request in the raw data hung for several hours and would distort the mean.
- LLM judge and the evaluated model are both DeepSeek; use the human spot-check table for calibration.
- Costs are estimates from token counts and configured rates, not provider invoices.
