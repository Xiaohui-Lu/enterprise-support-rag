# Experiment design

**Support assistant on WixQA: when do retrieval, agentic search and long context pay off?**

Draft v0.2. Corpus numbers were measured in Step 1 ([reports/corpus_stats.md](../reports/corpus_stats.md)); token counts use tiktoken as an approximation.

## 1. Goal

Build a support assistant that answers customer questions from a 6,221-article help center, and measure, with the model held constant, when each answering strategy is worth what it costs. Ship the winners behind a router as a production service.

The headline result is a chart of cost per question against % correct, not a single accuracy number.

## 2. Questions this project answers

| ID | Question | Answered in |
|---|---|---|
| RQ1 | How much does retrieval add over what the model already knows? | Step 3 |
| RQ2 | Which retrieval setup finds the right articles: keyword, embeddings, hybrid, hybrid + reranker? Whole articles or sections? | Steps 2, 4 |
| RQ3 | Which answering strategy gets the most questions right per dollar and per second? | Step 6 |
| RQ4 | Does the answer to RQ3 change by question type (single- vs multi-article, expert-written vs simulated)? | Step 6 |
| RQ5 | How fast does accuracy fall as look-alike articles are added to the context? | Step 6 |
| RQ6 | Can a router keep most of the best strategy's accuracy at a fraction of its cost? | Step 7 |

## 3. Data

Source: [`Wix/WixQA`](https://huggingface.co/datasets/Wix/WixQA) on Hugging Face, MIT license. The knowledge base (KB) is a snapshot of the Wix Help Center taken on 2024-12-02. Step 1 pins the dataset revision so every result can be reproduced.

| Set | Rows | Notes |
|---|---|---|
| KB corpus | 6,221 | 4,109 regular articles, 2,049 feature requests, 63 known issues |
| ExpertWritten | 200 | Real support tickets; long step-by-step answers (mean ≈1,020 characters); 52 need 2–3 articles |
| Simulated | 200 | Distilled from real support chats; short answers (mean ≈260 characters); 27 need 2–3 articles |
| Synthetic | 6,221 | LLM-generated, one question per article |

Article length: median 251 tokens, mean 470, p99 2,677, longest 10,614. Only 156 articles exceed 2,000 tokens. The whole KB is about 2.9M tokens.

What this means for the design:

- **Long context can't hold the whole KB.** About 2.9M tokens is beyond any context window, so the long-context strategy means many retrieved articles in one prompt, not the entire help center.
- **Most articles are short.** A whole article may work as the retrieval unit; splitting into sections matters mainly for the 156 articles over 2,000 tokens. Step 4 tests both instead of assuming. Nearly half the articles (2,892) have two or more HTML section headings to split on.
- **A third of the KB is feature requests, but they rarely answer real questions.** Only 10 of the 400 real questions have one among their gold articles, so for real traffic they're mostly look-alike noise. They stay in the index, since those 10 questions need them. Step 4 found that down-weighting them couldn't clear the bar on dev even in its best case, so it wasn't built (D-040). They also make up a third of the synthetic questions: one more way synthetic differs from real traffic.
- **Only 79 of the 400 real questions need more than one article.** That's where agentic search should help most, and it's a small group: results on it will have wide error bars (§8).
- **Synthetic questions are easier than real ones.** They're generated from the article and reuse its wording, while real users describe symptoms ("my payout hasn't arrived"). Retrieval scores on synthetic will look better than on real tickets.

## 4. Splits

| Split | Contents | Used for |
|---|---|---|
| Dev | 50 real (25 ExpertWritten + 25 Simulated, stratified by single/multi-article) + 500 synthetic | Tuning, prompt work, reading failures, judge validation |
| Test | The other 350 real questions | Milestone runs only. Locked. |
| Synthetic reference | The other 5,721 synthetic | Retrieval metrics only (no LLM calls, so free) |

Rules:

- Never tune on test failures. If anything changes after a test run, it gets logged in the results notes.
- The dataset has no question IDs, so an ID is the config prefix plus the row index at the pinned revision (`ew-017`, `sim-142`, `syn-3021`). Split files are committed and never regenerated.

## 5. Answering strategies

Every strategy uses the same generator model and the same output format: an answer with cited article IDs, or a decline ("I couldn't find this in the help center").

| ID | Strategy | What the model sees | LLM calls |
|---|---|---|---|
| S0 | Closed-book | The question only | 1 |
| S1 | Single-pass RAG | Top-k results from the best retriever (RQ2) | 1 |
| S2 | Agentic search | Tools: `search(query)` returns titles and snippets; `read(article_id)` returns full text; `answer(...)` ends the loop. Capped at about 6 steps and a token budget | Up to ~7 |
| S3 | Long context | Top-K retrieved full articles, K ∈ {20, 50, 100} | 1 |
| S* | Oracle context | Exactly the gold articles | 1 |

- **S0** is a baseline and a contamination check: newer models may have seen Wix's public help pages, or WixQA itself. It is never a production route.
- **S\*** isn't a real strategy. It's the ceiling for answer generation when retrieval is perfect. If S1 is far below S\*, retrieval is the bottleneck; if S\* is low too, the generator or the prompt is. This is the most useful diagnostic in the project. It isn't a strict ceiling: on dev, S1 on dense search scored above it, because ten related articles can answer more completely than the gold ones alone (D-042).

## 6. Metrics

**Retrieval** (no LLM calls; retrieved sections are mapped back to their article first):

- Recall@k for k ∈ {1, 5, 10, 20}: share of gold articles found in the top k.
- All-gold@k: a question counts only if every gold article is in the top k. This is what multi-article questions need.
- MRR (mean reciprocal rank): how high the first gold article ranks.

**Answers:**

- Correctness (headline): the judge compares the answer with the expert reference and grades it correct, partially correct or incorrect. Report % correct, with partial separately.
- Groundedness: every claim is supported by the cited articles. This catches answers that are right from the model's memory rather than the docs.
- Citation hit rate: share of cited articles that are gold.
- Wrong-answer rate and decline rate, reported separately. Every WixQA question is answerable, so a decline is a miss, but for a support team it's a cheaper miss than a confident wrong answer.
- Token F1 against the reference: a weak metric, included only because the WixQA paper reports it.

**Cost and speed** (per question):

- Input, cached-input and output tokens; number of LLM and tool calls; dollars from the provider's price list at run time.
- End-to-end latency at p50 and p95; time to first token for streamed answers.

## 7. The judge

- A stronger model than the generator, ideally from a different provider, since models tend to favor their own writing.
- Validated before use: hand-label about 80 dev answers across strategies with the same rubric, then measure agreement with the judge (% agreement and Cohen's kappa). Below a kappa of about 0.6, fix the rubric before any test run.
- The judge model version and prompt are pinned. Changing either means older runs are no longer comparable.

## 8. Statistics

- Every run stores per-question results, not just averages.
- 95% confidence intervals by bootstrap (resample the questions 10,000 times).
- Strategies are compared with paired tests on the same questions: a paired bootstrap of the difference, or McNemar's test on correct/incorrect.
- Expected precision: on 350 test questions a single accuracy is about ±5 points; on the ≈70 multi-article test questions, about ±12. Paired differences are tighter. Subgroup results are reported as directional.

## 9. Context-rot test (RQ5)

- 60 test questions, stratified by single/multi-article.
- Context = the gold article(s) + N distractor articles, N ∈ {0, 10, 50, 200}, which is ≈1K to ≈95K tokens by tiktoken's count. Step 6 checks the count with Claude's tokenizer, because Claude Haiku 5.5 charges more per token above 100K.
- Two distractor types: **hard** (top-ranked non-gold articles from the retriever, which look relevant but aren't) and **random** (any other articles).
- Gold articles go at a random position, recorded per question, so position effects ("lost in the middle") can be checked.
- Same model, prompt and judge as the main runs.
- Output: % correct against context size, one line per distractor type. If the hard line falls faster, similarity hurts more than length alone, as [Chroma's 2025 context-rot study](https://research.trychroma.com/context-rot) found, here checked on enterprise support data.

## 10. Router (RQ6)

The router chooses between S1 (fast) and S2 (agentic). S3 joins only if Step 6 finds a question type where it wins.

| Router | How it decides | Tuned on |
|---|---|---|
| R1 Rules | Question length, troubleshooting phrases, several products named | Dev |
| R2 Cascade | Run S1; escalate to S2 if the top retrieval score is low or S1 declines | Dev (thresholds) |
| R3 Classifier (optional) | A small model predicts the route from the question; label = cheapest strategy that got it right | 5-fold cross-validation over the 400 real questions |
| Oracle | Hindsight: the cheapest strategy that got each question right | Nothing; it's the ceiling |

Output: cost per question against % correct for always-S1, always-S2, R1–R3 and the oracle.

Skip condition: if S2 doesn't beat S1 on any question type, there is nothing to route. Ship S1 and write that up.

## 11. Reproducibility and cost

- Each run records its config (strategy, models, k, prompt version, git commit, dataset revision), per-question outputs, scores, tokens and latency.
- Every LLM call is cached by model, prompt and parameters, so re-running a report is free and deterministic.
- Large eval runs use the provider's discounted batch API. Latency is measured separately on a sample of real-time calls, since batch timings mean nothing.
- Iterate on dev with a cheap model; run on test only at milestones.

Rough input tokens for one full test run (350 questions), before caching and discounts:

| Run | Per question | Total |
|---|---|---|
| S0 closed-book | ≈0.3K | ≈0.1M |
| S1 single-pass RAG | ≈4.5K | ≈1.6M |
| S2 agentic (history is resent every step; prompt caching cuts the cost) | ≈25K | ≈9M |
| S3 long context, K = 20 / 50 / 100 | ≈11K / 29K / 57K | ≈34M |
| S* oracle | ≈1.5K | ≈0.5M |
| Judge, 7 runs | ≈2.5K per answer | ≈6M |
| **Total** | | **≈51M** |
| Context-rot test (run once) | | ≈18M |

Long context is about two-thirds of the bill. These estimates assumed 570 tokens per article; Step 1 measured 470, so the long-context rows are roughly 20% high. Step 3 turns these into dollars once the models are chosen.

## 12. Roadmap

Each step ends with something runnable and a commit. ✅ marks a checkpoint where the project is complete and shippable as it stands.

| Step | What gets built | Decisions to defend | Status |
|---|---|---|---|
| 0 | This design and the decision log | Benchmark, thesis, no framework, splits, judge, statistics | Done |
| 1 | Repo, pinned data download, corpus stats, frozen splits | Python tooling, repo layout, data format | Done |
| 2 | Retrieval eval harness, BM25 baseline, run records, LLM call cache | Metric definitions, run-record schema | Done |
| 3 | Generation and judge: S0, S1 on BM25, S*; judge validated on hand labels | LLM provider and models, judge rubric, cost tracking | Done |
| ✅ M1 | Public repo; README with S0/S1/S* results and confidence intervals | | Done |
| 4 | Postgres + pgvector, embeddings, hybrid fusion, reranker, chunking (RQ2) | Postgres vs a vector DB, keyword engine, embedding model, reranker | Done |
| 5 | Agentic search S2 with a hand-written tool loop | Tool design, step budget, stop conditions, prompt caching | |
| 6 | Long context S3, context-rot test, full comparison (RQ3–RQ5) | K sweep, distractor design, cost guardrails | |
| ✅ M2 | Cost-vs-correctness and context-rot charts in the README | | |
| 7 | Router: cascade (plus optional classifier) against the oracle (RQ6) | Signals, thresholds, cascade vs classifier | |
| 8 | Service: FastAPI with streaming, React chat UI with citations, Redis cache and rate limit | FastAPI vs Django, Redis vs RabbitMQ (or neither), SSE vs WebSockets | |
| 9 | Deployment: Docker, Helm on k3s, OpenTelemetry tracing, Prometheus/Grafana | k3s vs managed Kubernetes vs PaaS, what to trace | |
| 10 | CI eval gate on pull requests, final README | Gate size and thresholds, handling noisy evals | |
| ✅ M3 | Live demo, architecture diagram, write-up | | |

## 13. Risks

| Risk | Mitigation |
|---|---|
| The model already knows the answers (contamination) | S0 measures it; report retrieval's lift over S0, not only absolute scores |
| The judge is biased or drifts | Validate against hand labels; pin judge version and prompt |
| Small subgroups give noisy results | Paired tests; directional wording; no claims from overlapping intervals |
| API costs run over | Call cache, batch API, dev-first iteration, price check before every large run |
| Scope creep | Every milestone ships on its own; ideas outside the roadmap go to the "Later" list (§14) |

## 14. Later

Ideas outside the roadmap. Each gets its own decision before it's scheduled.

| Idea | Why | When | Notes |
|---|---|---|---|
| Second knowledge base: [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) (D-014) | Multi-source, internal-style data with unanswerable questions; shows which findings transfer | After M3 | New loader behind the D-013 boundary; its own dev/test split; re-measure chunking and router thresholds, and validate the judge again (see `porting.md`). Confirm the dataset license first; [HERB](https://huggingface.co/datasets/Salesforce/HERB) is the fallback |
| "Bring your own docs" loader | Point the service at a folder of PDFs or Markdown files | After M3 | No gold labels, so it's for demos, not results (see `porting.md`) |

## References

- Cohen et al., [WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation](https://arxiv.org/abs/2505.08643), 2025
- Hong, Troynikov, Huber, [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot), Chroma, 2025
- Jeong et al., [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity](https://arxiv.org/abs/2403.14403), 2024
