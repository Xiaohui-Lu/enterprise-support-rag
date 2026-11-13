# Enterprise support RAG

A support assistant for a 6,221-article help center, built to answer one question with data: **when do retrieval, agentic search and long context each pay off?** Closed-book answering, single-pass RAG, agentic search and long-context prompting run on the [WixQA](https://huggingface.co/datasets/Wix/WixQA) benchmark with the model held constant, and are compared on correctness, cost and latency.

> Work in progress: Steps 0 to 5 are done, and the last milestone reached is M1. See the [roadmap](docs/experiment-design.md#12-roadmap).

## Results so far (M1)

The first milestone answers RQ1: how much does retrieval add over what the model already knows? Each of the 350 real questions in the held-out test set was answered once by Claude Haiku 5.5 and graded by Claude Sonnet 5.5 against WixQA's expert-written reference answer.

| Strategy | What the model sees | % correct [95% CI] | Cost per answer | Latency p50 | Input tokens |
|---|---|---:|---:|---:|---:|
| S0, closed book | The question only | 33.4% [28.6, 38.3] | $0.00036 | 4.2 s | 138 |
| S1, single-pass RAG | BM25's top 10 articles | 64.0% [59.1, 68.9] | $0.00177 | 3.8 s | 14,201 |
| S\*, oracle context | Exactly the gold articles | 74.6% [69.7, 78.9] | $0.00043 | 2.7 s | 1,785 |

Paired on the same questions:

- Retrieval adds **30.6 points** over closed book [+23.7, +37.1].
- Perfect retrieval would add another **10.6 points** [+4.9, +16.3]. S\* got 73 questions right that S1 missed; S1 got 36 that S\* missed.
- All three differences hold after Holm's correction (p < 0.001). Full tables: [reports/answers_test.md](reports/answers_test.md).

What this shows:

- **Retrieval pays off, and plain BM25 gets most of the way.** It closes about three-quarters of the gap between closed book and the gold articles (30.6 of 41.1 points).
- **Retrieval still leaves accuracy and money on the table.** A gold article is in BM25's top 10 for only 57% of test questions, and S1 sends 8 times the input tokens of S\* for a worse answer. Better retrieval (Step 4) has a measured target: up to about 10 points, at fewer tokens.
- **The generator sets a ceiling too.** Even with the gold articles, a quarter of answers aren't graded correct: 48 are partial, 34 are declines and 7 are wrong. Better retrieval can't fix those.
- **The dev set couldn't see the retrieval gap.** On the 50 dev questions, S1 and S\* looked level (76% against 78%, +2 [−14, +18]). The test set is 7 times larger, so its intervals are about √7 ≈ 2.6 times narrower, and the gap shows. S1 also fell from 76% on dev to 64%. Part of that is the questions: a gold article was in the top 10 for 66% of dev questions, against 57% of test ones. Part is that k=10 was picked on dev, which flatters dev.

### How it was measured

- **One generator for every strategy:** Claude Haiku 5.5 at medium effort, with the same prompt and answer format: numbered articles, inline citations, and an exact decline when the answer isn't there (D-023).
- **Grading:** Claude Sonnet 5.5 at high effort compares each answer with the reference and grades it correct, partial or incorrect. Only correct counts, and a decline counts as incorrect (D-025).
- **The judge was validated before use** against 80 labels on dev answers: Cohen's kappa 0.82 on the three grades and 0.92 on correct vs not (D-031). 65 labels are the unanimous votes of a panel of four Claude configurations; a person labeled the 15 items the panel split on (D-028). See [reports/judge_validation.md](reports/judge_validation.md).
- **Statistics:** 95% intervals come from a bootstrap over questions. Comparisons use a paired bootstrap and McNemar's exact test, with Holm's adjustment for three comparisons (D-032).
- **Tuning happened on dev only.** The generator's effort and S1's k were chosen on 50 dev questions (D-033). The test set runs only at a named milestone and from a clean commit, enforced in code. These numbers are the first and only test run, from commit `eadbd6a`, and nothing was changed after seeing them (D-034).
- **Cost:** the M1 run cost about $4.60: 1,050 answers in real time, so latency is real, and 1,050 grades through the batch API.

### Caveats

- The 65 panel labels haven't been audited by hand yet. A random audit is planned before the final write-up (D-028).
- The judge is weakest where people disagree too. On the 15 borderline items, it matched the human label on only 8 (kappa 0.33 on correct vs not).
- The judge also leans lenient. On the 80 labels, it graded 3 not-correct answers correct and never the reverse (55% correct against the labels' 51%). Absolute rates may run a few points high; differences between strategies are affected less.
- Each strategy was run once, and the generator isn't deterministic. The intervals cover which questions were drawn, not which answers the model happened to write.
- "Correct" means "agrees with WixQA's reference answer", and a reference can itself be incomplete.

## Quick start

Needs [uv](https://docs.astral.sh/uv/), which installs the right Python version by itself.

```bash
uv sync                # create the environment from uv.lock
uv run srag download   # WixQA at the pinned revision, into data/raw/
uv run srag stats      # corpus statistics, into reports/corpus_stats.md
uv run pytest
```

Answering and grading call the Anthropic API, with `ANTHROPIC_API_KEY` in a git-ignored `.env`. Every call is cached in `cache/`, so a rerun costs nothing. On dev:

```bash
uv run srag answer --strategy s1                            # 50 dev questions, about $0.09
uv run srag judge runs/<answer run> --effort high --batch   # grades it through the batch API
uv run srag report-answers runs/<judge runs>                # one judge run per strategy
```

Dense retrieval can also search Postgres with pgvector. Copy `.env.example` to `.env`, then run `docker compose up -d` and `uv run srag db-load`, and add `--store postgres` to `srag eval-retrieval`.

The dev/test split lives in `data/splits/splits.json`. It was generated once with `srag splits` and is never regenerated, so the test set can't drift. The test set runs only with `--split test --milestone M1` (or M2, M3) from a clean commit.

## Docs

- [Experiment design](docs/experiment-design.md): research questions, splits, strategies, metrics, roadmap
- [Decision log](docs/decisions.md): what was chosen, what else was considered, and why
- [Porting to a new dataset](docs/porting.md): what to redo, including how many labels
- Reports: [test answers (M1)](reports/answers_test.md), [dev answers](reports/answers_dev_real.md), [judge validation](reports/judge_validation.md), tuning of [effort](reports/tuning_effort.md) and [k](reports/tuning_k.md), [BM25 retrieval](reports/retrieval_bm25.md), [dense retrieval](reports/retrieval_dense.md), [hybrid retrieval](reports/retrieval_hybrid.md), [reranking](reports/retrieval_rerank.md), [sections](reports/retrieval_sections.md), [reranking on sections](reports/retrieval_rerank_sections.md), [Postgres and HNSW](reports/retrieval_pgvector.md), [S1 on dense search](reports/tuning_s1_reading.md), [S2 on dev](reports/answers_s2_dev.md), [S3 on dev](reports/answers_s3_dev.md), [context-rot pilot](reports/context_rot_dev.md), [corpus statistics](reports/corpus_stats.md)

## Data

WixQA is by Wix.com AI Research, released under the MIT license: Cohen et al., [WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation](https://arxiv.org/abs/2505.08643), 2025. The raw files aren't stored in this repo; `srag download` fetches them at the commit pinned in `src/support_rag/config.py`.

## License

MIT
