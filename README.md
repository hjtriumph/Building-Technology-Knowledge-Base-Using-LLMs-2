# Building Technology Knowledge Base with Agentic RAG

A retrieval-augmented question answering system over building-technology literature, with a reproducible evaluation harness covering retrieval, answer quality, faithfulness, citations, abstention, latency, and cost.

The project started as a master's thesis prototype and was rebuilt around one question: **how do we know the system works, and where does it fail?** The corpus, benchmark, and demo all run on the same pipeline.

## Corpus

15 English papers, 238 pages, 1,410 chunks:

- 8 papers from the original thesis corpus (heating/cooling load forecasting, occupancy, SHAP, imputation, Mixture of Experts).
- 7 papers from the Hermann-Rietschel-Institut (HRI) at TU Berlin (operating-room ventilation, district heating economics, UVC vs HEPA air purification, heat pump load shifting, gas-turbine SHAP, ED-LSTM heat demand forecasting, LLM-based building energy modeling).

The 7 added papers act as distractors for questions anchored in the original 8. No test question can be answered from a few-shot example.

## Architecture

```text
PDF -> loader -> chunking (1000/150) -> bge-m3 embeddings -> Chroma
                                                               |
question -> bge-m3 query -> MMR (k=6, fetch_k=20) -> [C1..C6] context
                                                               |
                         prompt (system + optional few-shot + context)
                                                               |
                              DeepSeek -> structured answer + [C#] citations
```

- `src/ingest.py`: stable chunk ids (`sha1`), reset/upsert, metadata, `storage/ingest_report.json`.
- `src/pipelines.py`: four answer paths (`llm`, `rag`, `few-shot`, `agent`) with token usage and cost.
- `src/citations.py`: `[C#]` labels mapped back to source/page/chunk.
- `src/agent_graph.py`: ReAct-style agent with step and total-chunk budgets.
- `eval/`: retrieval metrics, LLM judges, benchmark runner, reports.
- `api/` + `web/`: FastAPI + single-page demo.

## Setup

```bash
conda activate chain_1_fixed
pip install -r requirements.txt
```

Create `.env` from `.env.example`. The main settings:

```env
DEEPSEEK_API_KEY=...
EMBEDDING_MODEL=BAAI/bge-m3
K=6
FETCH_K=20
SEARCH_TYPE=mmr
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

`bge-m3` is multilingual (~2 GB); it lets the demo accept Chinese questions against the English corpus.

## Build the index

```bash
python -m src.ingest --reset
```

Expected report: 15 documents, 238 pages, 1,410 chunks, 0 duplicate ids. Re-running without `--reset` is idempotent: chunk ids are content-derived and written with upsert.

**Known issue (Windows + non-ASCII path).** Chroma 1.5.8 silently fails to persist the HNSW segment when the persist path contains Chinese characters and the default `hnsw:sync_threshold=1000` is crossed mid-write. The collection is created with `hnsw:sync_threshold=10000` and `hnsw:batch_size=1000`, which avoids the broken incremental sync and is covered by `tests/test_ingest_persistence.py`. If the corpus grows beyond 10,000 chunks, re-check this workaround.

## Ask questions

```bash
python -m src.cli --mode rag --q "Does adding occupancy data improve heating load forecasting?" --json --show-sources
python -m src.cli --mode fewshot --fewshot-n 3 --q "How does the MoE gating mechanism combine experts?"
python -m src.cli --mode agent --q "Compare the occupancy study and the ED-LSTM study."
```

Modes: `llm`, `llm_structured`, `rag`, `fewshot`, `fewshot_no_cot`, `agent`.

## Demo

```bash
python -m uvicorn api.main:app --port 8000
```

Open `http://127.0.0.1:8000/`. The page supports Chinese and English questions, mode switching, citation cards, and per-request latency/token/cost.

## Benchmark

```bash
python -m eval.retrieval_eval --check          # retrieval only
python -m eval.runner --num-cases 30 --repeats 3   # full run (about ¥6 at 2026 rates)
python -m eval.runner --smoke                  # 2 cases, 1 repeat
```

Case composition: 5 numeric, 1 single-hop, 6 multi-hop, 10 method, 8 unanswerable. The 6 method questions from the original plan grew to 10 because the added HRI papers naturally supply method questions; the original 10 thesis questions are preserved. Quality means use the 22 answerable questions (all non-unanswerable cases).

Nine arms, each question repeated three times:

| Arm | What it isolates |
|---|---|
| `llm` | closed-book, short answer format |
| `llm_structured` | closed-book, same five-section format as RAG |
| `rag_0shot` | retrieval + five-section prompt (main baseline) |
| `fewshot_1/3/5` | same retrieval, 1/3/5 leak-free examples |
| `fewshot_no_cot_3` | same 3 examples, short answers, no reasoning |
| `rag_k18` | single-pass retrieval with 18 chunks (equal-context control for agent) |
| `agent` | ReAct, max 4 steps, max 18 chunks |

Metrics:

- Retrieval: Recall@1/3/6/10, MRR@10, nDCG@10, context precision/recall.
- Generation: completeness, answer relevance, key-point coverage.
- Faithfulness: claim support against retrieved context (refused answers excluded).
- Citation precision: cited `[C#]` labels whose snippet supports the claim.
- Abstention: correct refusal on unanswerable questions, false refusal on answerable ones.
- Cost and latency: token usage -> CNY, median and p95 latency.

## Results (30 questions x 3 repeats, 810 rows, 0 errors)

Full run: `results/20261106-063515/` (`summary.csv`, `by_type.csv`, `report.md`).

### Retrieval

| Search | Recall@6 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| MMR (k=6, fetch_k=20) | 0.371 | 0.530 | 0.272 | 0.312 |
| Similarity (k=10) | 0.462 | 0.670 | 0.309 | 0.380 |

MMR underperformed similarity at this corpus size. The generation benchmark keeps the pre-registered MMR/k=6 configuration so the two experiments are not mixed.

### Generation

| Arm | Completeness | Relevance | Faithfulness | Citation P | Key points | Latency p50 | Cost ¥ | Refusal (unans.) | False refusal |
|---|---|---|---|---|---|---|---|---|---|
| llm | 0.053 | 0.106 | N/A | N/A | 0.362 | 3.0s | 0.17 | 1.000 | 0.879 |
| llm_structured | 0.083 | 0.258 | N/A | N/A | 0.373 | 3.9s | 0.25 | 1.000 | 0.545 |
| rag_0shot | 0.833 | 0.886 | 0.996 | 0.662 | 0.814 | 8.4s | 0.62 | 0.917 | 0.061 |
| fewshot_1 | 0.864 | 0.894 | 0.986 | 0.647 | 0.821 | 8.3s | 0.62 | 1.000 | 0.076 |
| fewshot_3 | 0.833 | 0.871 | 0.978 | 0.658 | 0.822 | 7.8s | 0.68 | 1.000 | 0.061 |
| fewshot_5 | 0.909 | 0.932 | 0.964 | 0.674 | 0.848 | 7.5s | 0.76 | 1.000 | 0.030 |
| fewshot_no_cot_3 | 0.705 | 0.864 | 0.971 | 0.000 | 0.697 | 3.8s | 0.39 | 1.000 | 0.091 |
| rag_k18 | 0.924 | 0.962 | 0.998 | 0.547 | 0.887 | 9.2s | 1.21 | 0.792 | 0.000 |
| agent | 0.811 | 0.894 | 0.994 | 0.550 | 0.762 | 13.3s | 1.56 | 1.000 | 0.076 |

Quality metrics average the three repeats per question first, then bootstrap over the 22 answerable questions. Completeness 95% CIs and paired comparisons are in `summary.csv` and `paired.csv`; latency uses median/p95 because one request in the raw data hung for 2.6 hours.

### Paired comparisons (completeness, question-level means)

| Baseline | Compare | Mean diff | 95% CI | p |
|---|---|---|---|---|
| rag_0shot | fewshot_5 | +0.076 | 0.000 to 0.189 | 0.088 |
| rag_0shot | rag_k18 | +0.091 | 0.008 to 0.189 | 0.025 |
| rag_0shot | agent | -0.023 | -0.189 to 0.144 | 0.829 |
| rag_k18 | agent | -0.114 | -0.273 to 0.015 | 0.126 |

`rag_k18` is the only arm with a significant completeness gain over `rag_0shot`. The 5-shot gain and the agent difference are not significant at n=22.

### What the numbers say

1. **Retrieval is the dominant factor.** Closed-book arms score 0.05-0.08 completeness and refuse 55-88% of answerable questions. RAG arms score 0.83+.
2. **More context helps quality but hurts citation and abstention.** `rag_k18` has the best completeness (0.924; paired gain +0.091 over 0-shot, p=0.025) and key-point coverage (0.887), but citation precision falls from 0.662 to 0.547 and correct refusal from 0.917 to 0.792.
3. **Few-shot gains are modest and non-monotonic.** At k=6, 5 examples reach 0.909 completeness, but the paired gain over 0-shot is +0.076 (p=0.088), and 3 examples (0.833) do not beat 0-shot (0.833). The earlier "0.13 -> 0.90" result was contaminated by example/test overlap and a straw-man baseline.
4. **CoT costs latency.** `fewshot_no_cot_3` answers in 3.8s vs 7.8s, at 0.705 vs 0.833 completeness and zero explicit citations.
5. **The agent helps multi-hop but hurts numeric questions.** By type: multi-hop 0.917 vs 0.806 for `rag_0shot`; numeric 0.600 vs 1.000. The overall difference from `rag_0shot` is not significant (p=0.829), and `rag_k18` beats the agent overall (0.924 vs 0.811) at lower cost and latency, so the agent's value is not established at this scale.
6. **Total benchmark cost: ¥6.25 for 810 rows**, well under the ¥150 budget.

## Benchmark visualization

```bash
python -m eval.visualize --run results/20261106-063515
```

Writes `results/20261106-063515/visuals/`:

- `radar.png` — quality profile of the four main RAG arms
- `by_type.png` — completeness by question type
- `paired_ci.png` — paired bootstrap confidence intervals
- `quality_latency.png` and `quality_cost_latency.png` — quality vs latency vs cost
- `abstention.png` — correct refusal vs false refusal
- `retrieval.png` — MMR vs similarity recall
- `dashboard.html` — one-page dashboard with embedded images and all tables

The dashboard is a single self-contained HTML file; open it in a browser. The interview deck also reads the latest run's `visuals/` directly, so re-running the benchmark and this command updates both.

## Human spot-check

25 samples are exported to `results/20261106-063515/human_review.csv` with a hidden key in `human_review_key.json`. A local rating form avoids editing the CSV by hand:

```bash
python -m eval.human_form --csv results/20261106-063515/human_review.csv
```

Open the generated `human_review.html` in a browser, click 0 / 0.5 / 1 for completeness and faithfulness, then click 导出 CSV. Progress is kept in browser storage. Rate completeness against the reference answer; rate faithfulness against the cited snippets. Leave faithfulness blank for refusals and both fields blank for unanswerable questions.

Then run agreement against the exported file:

```bash
python -m eval.human_check agreement --human <path-to>/human_review_filled.csv --key results/20261106-063515/human_review_key.json
```

The output reports exact agreement, +/-0.25 agreement, and quadratic weighted kappa. Results for the 25 sampled answers (`human_agreement.json`):

| Dimension | Comparable n | Exact | Within 0.25 | Weighted kappa |
|---|---|---|---|---|
| Completeness | 21 | 1.000 | 1.000 | 1.000 |
| Faithfulness | 17 | 0.824 | 0.882 | -0.055 |

Completeness agreed perfectly: 15 samples rated 1 and 6 rated 0 by both the human and the judge. Faithfulness had high raw agreement but near-constant ratings (15 of 17 human ratings were 1.0), so kappa is unstable and not informative; the two disagreements (s004, s011) are human 0 vs judge 1. The 4 unanswerable samples and the closed-book arms have no judge faithfulness score and are excluded.

This validates the completeness judge on the sampled set; faithfulness should still be read as a relative ranking, not a calibrated absolute.

## Limitations

- 15 papers in one domain; this is a vertical knowledge-base demo, not a general building-technology corpus.
- The judge and the evaluated model are both DeepSeek, so self-preference bias is possible. Human calibration found perfect completeness agreement (n=21) but low-variance faithfulness ratings, so faithfulness is reported as a relative ranking.
- Retrieval recall is modest (MMR Recall@6 = 0.371); hybrid BM25 + dense retrieval or a cross-encoder reranker is the next experiment, not a claim.
- Citation precision is 0.55-0.67, meaning some `[C#]` references do not fully support the attached claim.
- One request hung for 2.6 hours before the request timeout was added. Requests are now capped at 120s; the old row remains in raw data, so latency is reported as median/p95.

## Repository layout

```text
data/docs/            # 15 source PDFs (not committed)
storage/chroma/       # vector store + ingest_report.json (not committed)
src/                  # pipeline, ingestion, prompts, citations, agent
eval/                 # cases, metrics, judges, runner, report, human check
api/ + web/           # demo
tests/                # unit and regression tests
results/              # benchmark outputs (raw.jsonl ignored)
docs/superpowers/     # design and implementation plan
```
