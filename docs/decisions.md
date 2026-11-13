# Decision log

Short records of the choices that shape this project: what was decided, what else was considered, and why. Entries are added at the step where each decision is made. A decision that's later reversed gets a new entry rather than an edit.

Each entry has the same parts: context, decision, alternatives, why, trade-offs, and **In short**, the two-or-three-sentence version you can say out loud.

---

## D-001 · Benchmark: WixQA (Step 0)

**Context.** Measuring a RAG system needs ground truth for both halves: which documents should be retrieved, and what a correct answer looks like.

**Decision.** Use WixQA: 6,221 Wix Help Center articles, 400 real support questions (200 expert-written, 200 simulated) and 6,221 synthetic ones, each linked to the articles that answer it.

**Alternatives.**
- Scrape some product docs and write my own questions: no independent ground truth, and questions I write myself would drift toward what my system handles well.
- Wikipedia-based QA sets (Natural Questions, HotpotQA): plenty of data, but open-domain trivia, not enterprise support.

**Why.**
- Real tickets with expert-written answers, grounded in a published KB snapshot.
- Every question lists its gold articles, so retrieval and answer quality can be scored separately.
- The answers are procedural and multi-step, which is what enterprise support actually looks like.
- MIT license.

**Trade-offs.** The KB is a December 2024 snapshot, so the assistant answers from that snapshot, not today's site. Newer models may have seen the public help pages; the closed-book run (S0) measures how much. Only 400 real questions, so every number needs a confidence interval (D-006).

**In short.** "I needed ground truth for retrieval and for answers. WixQA has real support tickets with expert answers and the exact help-center snapshot they were written against, so I could score each half separately."

---

## D-002 · Thesis: a controlled comparison, not just a RAG app (Step 0)

**Context.** A single RAG pipeline with one accuracy number doesn't answer the question teams actually argue about: when is retrieval, agentic search or long context worth its cost?

**Decision.** Run closed-book, single-pass RAG, agentic search and long context, plus an oracle ceiling, with the same generator model. Compare correctness, cost and latency, and ship the winners behind a router.

**Alternatives.**
- Build one RAG pipeline and tune it: simpler, but it produces one number with nothing to compare against.
- Compare models instead of strategies: mostly measures which vendor is ahead this month.

**Why.** Holding the model constant isolates the effect of the strategy. The result is a decision (what to run for which question) rather than a demo.

**Trade-offs.** More runs, more API cost, more code. Mitigated by caching every call, iterating on dev, and running the test set only at milestones.

**In short.** "Instead of one more RAG demo, I measured when retrieval, agentic search and long context each pay off, with the model held constant, and used the results to decide what the service runs."

---

## D-003 · No LangChain or LlamaIndex (Step 0)

**Context.** Orchestration frameworks provide retrievers, agents and prompt templates out of the box.

**Decision.** Call the model providers' SDKs directly through a thin internal interface. Write the retrieval pipeline, the agent loop and the evaluation in this repo. Use libraries for well-defined pieces: BM25, embedding models, the Postgres driver.

**Alternatives.** LangChain / LangGraph, LlamaIndex, Haystack.

**Why.**
- The comparison counts every token and call. Hidden prompts, retries and extra calls inside a framework would distort the cost and latency numbers.
- The strategies must differ only in strategy, not in framework defaults, or the comparison isn't fair.
- The agent loop is small (on the order of 150 lines), so owning it is cheap, and owned code is easier to debug and to explain.

**Trade-offs.** More code to write. Retries, timeouts and tool-call parsing are ours to handle, and there are no ready-made connectors. Revisit if the system needs many data-source integrations, where a framework's connectors start to pay off.

**In short.** "Frameworks are great for prototypes, but this project is a measurement. I needed exact control over prompts and token counts, and identical plumbing across strategies so the comparison is fair. The agent loop is small, so owning it was cheap."

---

## D-004 · Splits: tune on dev, lock the test set (Step 0)

**Context.** There are 400 real questions and 6,221 synthetic ones. Synthetic questions are generated from the articles and reuse their wording, so they're easier than real tickets.

**Decision.** Dev = 50 real questions (stratified by dataset and by single/multi-article) plus 500 synthetic. Test = the other 350 real questions, run only at milestones. Question IDs and split files are committed.

**Alternatives.**
- All 400 real questions as test, tuning on synthetic only: a slightly tighter test (±4.8 instead of ±5.1 points), but tuning would never see real phrasing, and peeking at test failures becomes tempting.
- A 50/50 dev/test split: more tuning data, but a much weaker test (about ±7 points).

**Why.** Real failures can be studied on dev without contaminating the test set, and synthetic questions provide volume for retrieval tuning.

**Trade-offs.** With only 50 real dev questions, tuning decisions are noisy; each one gets confirmed at the next test run.

**In short.** "Synthetic questions reuse the article's wording, so they're easier than real tickets. I tuned on synthetic plus 50 real questions and locked the other 350 as a test set I only ran at milestones."

---

## D-005 · LLM judge, validated against hand labels (Step 0)

**Context.** The expert answers are long, multi-step procedures. A correct answer can be worded completely differently.

**Decision.** A stronger model, ideally from a different provider than the generator, grades correctness and groundedness against the expert answer. Before it's used on the test set, I hand-label about 80 dev answers and require reasonable agreement (Cohen's kappa of about 0.6 or higher).

**Alternatives.**
- Token overlap only (F1, BLEU, ROUGE): cheap and deterministic, but it can't tell a correct paraphrase from a wrong answer.
- Human grading only: the most trustworthy, but it doesn't scale and can't run in CI.
- An eval framework (Ragas, DeepEval) as-is: quick to adopt, but its metrics still need validating, and its prompts change between versions.

**Why.** A judge scales and can run in CI, but it's a model, so it gets measured like one.

**Trade-offs.** Judge calls cost money, the judge version must be pinned, and the hand labeling takes a couple of hours.

**In short.** "Token overlap can't tell a correct paraphrase from a wrong answer, so I used an LLM judge, but I treated it as a model to validate. I labeled about 80 answers myself and checked agreement before trusting it on the test set."

---

## D-006 · Confidence intervals and paired tests (Step 0)

**Context.** With 350 test questions, a single accuracy figure is accurate to about ±5 points; on the roughly 70 multi-article questions, about ±12.

**Decision.** Store per-question results for every run. Report bootstrap 95% confidence intervals, and compare strategies with paired tests on the same questions.

**Alternatives.** Report averages only: common, and misleading at this sample size.

**Why.** A 3-point "improvement" can be noise. Paired tests use the fact that every strategy answers the same questions, which makes real differences easier to detect.

**Trade-offs.** More analysis code, and some results will honestly say "no detectable difference".

**In short.** "With 350 questions, a 3-point gain can be noise, so every number in the README has a confidence interval, and strategy comparisons use paired tests on the same questions."

---

## D-007 · An oracle-context baseline (Step 0)

**Context.** When an answer is wrong, either retrieval failed (the right article never reached the model) or generation failed (the model had it and still got it wrong).

**Decision.** Include S*, a run where the model is given exactly the gold articles.

**Why.** S* is the ceiling for generation when retrieval is perfect. The gap between S* and a real pipeline is retrieval's share of the errors; the gap between S* and 100% is generation's. That tells you where the next week of work should go.

**Trade-offs.** One extra run, and a cheap one (about 1.5K input tokens per question).

**In short.** "Before tuning anything, I measured the ceiling: the model with perfect retrieval. The gap between that and my pipeline showed how much of the error was retrieval's fault."

---

## D-008 · Evaluate what you ship (Step 0)

**Context.** It's common to evaluate a notebook pipeline and then deploy different code.

**Decision.** The eval harness imports the same retrieval and generation code the service runs, and from Step 4 it uses the same database. There is no separate eval pipeline.

**Why.** Otherwise the README numbers describe a system nobody deployed, and the CI eval gate (Step 10) wouldn't protect production.

**Trade-offs.** The code needs clean interfaces from the start (question in; answer, citations and usage out), which takes some upfront design.

**In short.** "My eval harness calls the production code paths, so the numbers in the README describe the deployed system, and the CI gate tests the same code."

---

## D-009 · Python tooling: uv (Step 1)

**Context.** The project needs a pinned Python version, an isolated environment and identical installs on a Windows laptop, in Linux containers and in CI.

**Decision.** uv manages the Python version (`.python-version`), the virtual environment, the dependencies (`pyproject.toml`) and a lockfile (`uv.lock`, committed).

**Alternatives.**
- pip + venv + requirements.txt: universal, but there's no lockfile without extra tools, and the Python version is managed separately.
- Poetry: a mature lockfile workflow, but much slower.
- conda: built for heavy scientific and native stacks, which this project doesn't need.

**Why.** One fast tool replaces pyenv, venv, pip and pip-tools. The lockfile is cross-platform, so Windows, the Docker image and CI install exactly the same versions. `pyproject.toml` stays standard, so switching tools later is cheap.

**Trade-offs.** It's newer than pip and Poetry, so some teams haven't adopted it yet.

**In short.** "uv gives me one tool for the Python version, the environment and a cross-platform lockfile, so my laptop, the Docker image and CI install exactly the same versions."

---

## D-010 · Repo layout: one package, src layout, one CLI (Step 1)

**Context.** The same code will serve the experiments, the eval harness, the API service and CI (D-008).

**Decision.** All code lives in one installable package, `src/support_rag/`. Every step is a CLI command (`srag download`, `srag stats`, `srag splits`), not a notebook or a loose script. Tests in `tests/` import the installed package.

**Alternatives.**
- Notebooks: fast for exploring, but hard to test, review, diff and reuse in a service.
- Loose scripts in a flat folder: fine at first, but imports get messy as the project grows, and tests can silently import from the working directory instead of the installed package.

**Why.** The src layout makes tests run against the package as installed, which is how it will run in the Docker image. A CLI gives every step a reproducible command, which is also what CI will call.

**Trade-offs.** A little more setup than a script, and the package must be installed (`uv sync` does that).

**In short.** "Everything is one installable package with a CLI, so the experiments, the eval harness, the API and CI all run the same code through the same commands."

---

## D-011 · Data: pinned and untouched raw files, committed splits (Step 1)

**Context.** Every number must be reproducible from a fresh clone, by anyone, months from now.

**Decision.**
- Download the four WixQA files at a pinned Hugging Face commit (`d662dc4`) with `huggingface_hub`.
- Keep the raw files exactly as published (JSONL) in `data/raw/`, ignored by git.
- Commit what's small and derived: the split file (lists of question IDs) and the corpus report.
- Load records into plain dataclasses with the standard library; no dataframe library yet.

**Alternatives.**
- Always download the latest version: results could change silently if the dataset is updated.
- Commit the raw data (about 60 MB), or version it with Git LFS or DVC: unnecessary for a public dataset that can be re-downloaded exactly by commit hash.
- The `datasets` library: one-line loading, but a heavy dependency with its own caching and naming conventions (the dataset viewer shows these `test.jsonl` files as a "train" split), for what is four JSONL files.
- Regenerating the split from the seed when needed: that only works if the code and Python's random module never change.

**Why.** A commit hash plus committed question IDs makes every result reproducible from a fresh clone. Untouched raw files mean every transformation is visible in code rather than hidden in a preprocessed file.

**Trade-offs.** A fresh clone needs a one-time download of about 60 MB. Moving to a newer dataset revision is a deliberate change that invalidates earlier results.

**In short.** "The dataset is pinned to a Hugging Face commit and the split is committed as lists of question IDs, so anyone can reproduce my numbers from a fresh clone, and the test set can't drift."

---

## D-012 · Measure before preprocessing: whole-article baseline, chunking in Step 4 (Step 2)

**Context.** A typical RAG pipeline goes ingest → parse → clean → chunk → index → evaluate. WixQA ships the KB already extracted: plain text plus the original HTML for each article. The text has small artifacts: 5,066 of 6,221 articles have sentences glued together where HTML blocks were joined ("...to ADI.If you want..."), and 5,112 contain non-breaking spaces. The 33 articles with HTML-like tags in their text are embed-code examples, which are content, not noise.

**Decision.**
- Step 2 builds the retrieval harness and a BM25 baseline on whole articles, using the dataset's plain text as published.
- No separate cleaning stage. The glued sentences are handled by the BM25 tokenizer, which splits on non-letters ("ADI.If" → "adi", "if").
- Chunking moves to Step 4 as a measured variant (RQ2: whole articles or sections). The harness maps retrieved sections back to their article before scoring, so the same metrics apply to both.

**Alternatives.**
- Build the full pipeline first, with cleaning and fixed-size chunks: looks complete early, but chunk size, overlap and cleaning rules get chosen before they can be measured, and the baseline mixes two choices.
- Clean the text into a preprocessed copy: hides a transformation outside the code, against D-011, and fixes a problem only for readers that have it.

**Why.**
- Change one thing at a time. If the baseline used chunks, a later "embeddings on whole articles" run would change the retriever and the retrieval unit together, and the gain couldn't be attributed to either.
- The data allows it: the median article is 251 tokens, and only 156 exceed 2,000, so whole articles are a reasonable unit, and chunking mainly matters for the long tail.
- Whether text is dirty depends on its reader: the tokenizer fixes the glued sentences for BM25, and an LLM reads them fine.

**Trade-offs.** The first baseline may score a little below what chunking could give. And because WixQA is clean, this project doesn't demonstrate ingestion from messy sources (PDFs, wikis, ticket systems); it demonstrates measurement.

**In short.** "WixQA ships a clean, pinned snapshot, so ingestion was just a download. I built the measuring harness and a whole-article baseline first, and treated chunking as a Step 4 experiment, so every preprocessing choice had a measured effect instead of a guess."

---

## D-013 · Other knowledge bases: a concrete WixQA loader behind plain records, no pluggable interface yet (Step 2)

**Context.** The pipeline should be reusable on another knowledge base later (a company wiki, PDF manuals, another help center), but WixQA is the only one planned.

**Decision.**
- Keep `wixqa.py` concrete. The retrieval, evaluation and answering code takes plain `Article` and `Question` records and never imports the WixQA loader.
- The record types currently live in `wixqa.py`; they move to a neutral module when Step 2's retriever first needs them.
- Logic that depends on WixQA-specific fields, such as `article_type` (feature requests, known issues), stays optional.
- Extract a loader interface when a second knowledge base is actually added.

**Alternatives.** A generic knowledge-base interface now, with the source chosen by config: looks extensible, but an interface designed from one example is a guess about what the second source needs (PDF parsing, incremental sync, permissions), and it's code nothing calls.

**Why.**
- The boundary costs nothing and gives most of the portability.
- What a new knowledge base really costs is a new loader, a new eval set and re-tuned parameters. No interface can provide those in advance.

**Trade-offs.** Adding a second knowledge base means a small refactor to extract the interface. The boundary is a convention, kept by review rather than enforced by code.

**In short.** "I kept the WixQA loader concrete, but everything downstream depends only on plain document and question records. I'd extract a loader interface when a second knowledge base actually shows up, because an interface designed from one example is a guess."

---

## D-014 · Stay on WixQA; add EnterpriseRAG-Bench as a second knowledge base after M3 (Step 2)

**Context.** WixQA is clean, single-source data (D-012). EnterpriseRAG-Bench (Onyx and UC Berkeley, May 2026) simulates a fictional company's internal knowledge across nine sources (Slack, Gmail, Linear, Google Drive, HubSpot, meeting transcripts, GitHub, Jira, Confluence): about 512K documents and 500 questions with gold documents, gold answers and unanswerable cases.

**Decision.** Continue on WixQA through M3. Then add EnterpriseRAG-Bench as a second knowledge base: a new loader behind the D-013 boundary, its own dev/test split, and choices re-measured on its data (chunking, router thresholds). HERB (Salesforce AI Research, about 39K artifacts) is the fallback if EnterpriseRAG-Bench's dataset license doesn't allow publishing results.

**Alternatives.**
- Switch now: redoes Steps 0–1 (D-001, D-004, D-011) and needs indexing infrastructure for 512K documents before the first retriever exists. Its questions and gold labels are synthetic, and the authors expect to revise the gold sets.
- WixQA only: fastest, but leaves the single-source gap and gives no evidence that the findings transfer.

**Why.**
- WixQA's real tickets with expert answers give the most trustworthy measurements, so the harness and strategies are built and validated there first.
- A second knowledge base is most informative once there are results to compare: the same harness on different data shows which findings transfer and which were properties of WixQA.
- EnterpriseRAG-Bench adds what WixQA lacks: several source types, conflicting information, and unanswerable questions, which make the decline rate meaningful (on WixQA every decline is a miss). Its company is fictional, so models can't have memorized it.
- The dataset is new; by then its license and versioned gold labels should be settled.

**Trade-offs.** The multi-source work comes last, so it's the first thing cut if time runs short. Its documents are JSON exports, not live connectors, so it shows heterogeneous data rather than connector engineering.

**In short.** "I built and validated the harness on WixQA because it has real support tickets with expert answers. Then I ported it to EnterpriseRAG-Bench, a multi-source enterprise corpus, to see which findings transferred and which were properties of the first dataset."

---

## D-015 · BM25 as the retrieval baseline (Step 2)

**Context.** Every retrieval improvement in Step 4 (embeddings, hybrid fusion, a reranker, chunking) needs a reference point. Without one, "recall@5 is 0.7" says nothing about whether the extra cost bought anything.

**Decision.** The first retriever is BM25 over whole articles (D-012): keyword matching that weights rare words more (IDF), gives diminishing credit for repeated words (TF saturation, `k1`) and discounts long articles (length normalization, `b`). It runs in memory, with no model, API or database.

**Alternatives.**
- Embeddings first: they handle paraphrases, but they mean choosing a model, paying for it and storing vectors before there's a harness to show whether they help. Those are Step 4 decisions, made by measurement (RQ2).
- TF-IDF: simpler, but BM25 is essentially TF-IDF plus saturation and length normalization, at the same cost and usually better.
- A random ranking: a true floor, but it tells you nothing useful.

**Why.**
- Cheap, fast and deterministic: indexing 6,221 articles takes seconds, and the same query always gives the same ranking.
- A long-standing strong baseline: in the BEIR benchmark (2021), many neural retrievers failed to beat it on unfamiliar domains.
- Its failure mode is easy to explain. BM25 matches exact words, so it misses paraphrases: a user who asks how to "rename web links" never uses the words in "Changing Your Page URL". The gap between BM25 and embeddings on real questions measures exactly that.

**Trade-offs.** Expect a large gap between synthetic and real questions: synthetic questions reuse the article's wording, real users describe symptoms in their own words. Synthetic scores are optimistic; headline retrieval claims come from real dev questions.

**In short.** "I started with BM25 because it's cheap, deterministic and a hard baseline to beat. Every later retriever had to beat it on real questions. Its blind spot is paraphrase, which is exactly what embeddings were meant to fix, so the gap between them measures what embeddings bought."

---

## D-016 · BM25 implementation: bm25s for scoring, our own tokenizer (Step 2)

**Context.** D-015 chose BM25. It still needs a library, and a tokenizer that decides what counts as a word, which decides what can match what.

**Decision.**
- Scoring: `bm25s`, in memory, at its defaults (the Lucene variant, `k1 = 1.5`, `b = 0.75`), untuned.
- Tokenizer: our own function, used for both articles and questions. Lowercase; split on anything that isn't a letter or digit (so error codes like "404" survive); drop Lucene's 33-word English stopword list (the one Elasticsearch's English analyzer uses); stem with Snowball (`PyStemmer`).
- Stemming is on by default, and Step 2 measures it on and off on dev, everything else fixed (an ablation). The tokenizer settings go into every run record.
- Index `contents` only: it already starts with the title.

**Alternatives.**
- `rank_bm25`: the classic small library, but no release since February 2022, and it scores each query by looping over every document in Python.
- Our own BM25: about 40 lines, but easy to get subtly wrong (there are several IDF variants), and D-003 says to use libraries for well-defined pieces.
- Elasticsearch or OpenSearch: the industry-standard engine, but a Java server for 6,221 documents, before there's a need for one.
- Postgres full-text search: its ranking (`ts_rank`) isn't BM25. It's a Step 4 decision.
- The library's built-in tokenizer: less code, but then tokens change if the library changes, and the tokenizer is harder to test and record.

**Why.**
- `bm25s` precomputes every (token, article) score at index time, so a query just adds up a few rows of a sparse matrix. That suits a frozen snapshot queried thousands of times. It's the read/write trade-off of a database index: reads get cheaper, changes get expensive, because adding one document shifts the document count, the IDF values and the average length for every stored score. A collection that changed constantly would be better served by an engine that scores at query time.
- Owning the tokenizer means one tested function on both sides, recorded settings, and tokens that don't change when the library does.
- Stemming catches word forms ("connecting" / "connect", "domains" / "domain"), which real users vary more than synthetic questions do. It can also merge unrelated words (Snowball maps "animation" and "animal" to `anim`), so it's measured rather than assumed.

**Trade-offs.** No incremental updates: adding articles means rebuilding the index, which takes seconds and is fine for a help center adding a few articles a day. Default `k1` and `b` may leave a little recall on the table; tuning them is possible later, on dev only. Tokenization can't fix vocabulary mismatch ("discount" vs "promotion"); that's for embeddings in Step 4.

**In short.** "I used bm25s, which precomputes scores at index time, because my corpus is a frozen snapshot queried thousands of times: the same trade-off as a database index. The tokenizer is my own function, shared by articles and queries and recorded in every run, and I measured stemming instead of assuming it helps."

---

## D-017 · Retrieval metrics: recall@k, all-gold@k and MRR@20, averaged per question (Step 2)

**Context.** A retriever returns a ranked list; each question has one to three gold articles. Whether gold at rank 23 is good or bad depends on how many articles the generator will see, and multi-article questions need every gold article, not just one.

**Decision.**
- Recall@k: the share of a question's gold articles in the top k.
- All-gold@k: 1 if every gold article is in the top k, else 0.
- Reciprocal rank: 1 / the rank of the first gold article, 0 if none is retrieved. Averaged over questions, that's MRR.
- k ∈ {1, 5, 10, 20}. Each question is retrieved once at depth 20, and every metric comes from that list, so MRR is MRR@20.
- Every metric is computed per question and then averaged over questions (macro), never pooled over gold articles.

**Alternatives.**
- Pooling over gold articles (micro): gives a three-article question three times the weight of a one-article question.
- Precision@k: with one to three gold articles, precision@20 can't exceed 0.15, so it says little.
- nDCG: built for graded relevance ("very" vs "somewhat" relevant). WixQA's labels are binary, and recall plus MRR already cover coverage and rank.
- MRR over the full list: a gold article at rank 23 would add 0.04 instead of 0, a negligible difference, and it's beyond what any strategy reads.
- Mean rank of all gold articles: mixes "how high" and "how complete" into one number; recall and all-gold measure completeness separately.

**Why.**
- Each k is a context size a strategy actually uses: 1 asks whether the top hit is right, 5–10 is the range S1 would get, and 20 is where S3 starts.
- All-gold@k is what multi-article answers need. Recall gives partial credit; all-gold doesn't.
- Per-question scores match the user (one question at a time) and the statistics (D-006 bootstraps over questions).

**Trade-offs.** MRR ignores everything after the first gold article, by design. Metrics only know the gold labels: another article that also answers the question counts as a miss, which is why answers get judged separately (D-005).

**In short.** "I report recall@k at the context sizes my strategies actually use, all-gold@k because multi-article questions need every article, and MRR for how high the first useful article lands. Everything is scored per question and then averaged, so each question counts once and the bootstrap can resample questions."

---

## D-018 · Run records: plain files, per-question JSONL, git state recorded (Step 2)

**Context.** A number like "recall@5 = 0.62" is only worth something if you can say what produced it, on which questions, and what happened question by question. Every strategy, from retrieval to agentic search, will need the same record.

**Decision.** Each run writes a directory, `runs/<timestamp>_<name>/`, with:
- `config.json`: what produced it (retriever and its settings, tokenizer settings, retrieval depth, split, dataset revision, git commit and whether the working tree had uncommitted changes, library versions).
- `questions.jsonl`: one line per question (ID, subset, single or multi-article, gold IDs, the top 20 IDs and scores, its metrics, latency), appended as each question finishes.
- `summary.json`: per-question averages, overall and by group.

`runs/` is git-ignored. Committed reports name the run IDs and git commit they came from.

**Alternatives.**
- Averages only: enough to report a number, not to defend one. No paired tests, no confidence intervals, no failure analysis.
- An experiment tracker (MLflow, Weights & Biases): the industry tools, with a comparison UI, but a server or account and a dependency for a few dozen runs, and another layer between the code and the numbers (D-003).
- Postgres: queryable, but there's no database until Step 4. Runs can be loaded into it later.
- One JSON file per run: simpler, but only valid once the run finishes.
- Committing runs: a retrieval run is free to regenerate and a synthetic-reference run is a few MB, so the report is committed instead. Revisited in Step 3, when runs cost money.

**Why.**
- Per-question results make the statistics possible (D-006): a paired comparison of two runs on the same questions, bootstrap confidence intervals, and reading the questions that failed. They also allow new breakdowns and new metrics (recall@3 from the saved top 20) without re-running.
- JSONL appends one finished question at a time, so a long LLM run that crashes keeps what it finished and can resume.
- The git commit plus the uncommitted-changes flag says whether a run can be reproduced from its commit. Test-set runs will refuse to start with uncommitted changes.

**Trade-offs.** No dashboard: comparisons are small reports we write. Run directories accumulate locally and need occasional cleanup.

**In short.** "Every run saves its config, git commit and per-question results, not just averages. Averages are for reporting; per-question results are what let me run paired tests, put confidence intervals on differences and read the failures."

---

## D-019 · Keep stemming on: no measurable effect on real questions, a loss on synthetic (Step 2)

**Context.** D-016 made stemming the default and said Step 2 would measure it. The ablation ran BM25 on the dev split (50 real and 500 synthetic questions) with stemming on and off, everything else fixed. Results are in [reports/retrieval_bm25.md](../reports/retrieval_bm25.md).

**What we measured.** Question by question, reciprocal rank with stemming on vs off:

| Questions | Stemming helped | Stemming hurt | No change | Sign test |
|---|---:|---:|---:|---|
| Real (dev) | 8 | 11 | 31 | p ≈ 0.65: no detectable difference |
| Synthetic (dev) | 38 | 79 | 383 | p < 0.001: a real loss |

Recall@5 on real questions: 2 helped, 2 hurt. The losses come from over-stemming: "locales" appears in 4 articles (IDF 7.2), but its stem `local` appears in 232 (IDF 3.3), so the question's most useful word lost more than half its weight.

**Decision.** Keep stemming on for the BM25 baseline. Keep the `stem` option and record it in every run, so it can be re-tested.

**Alternatives.** Turn stemming off: supported by the only significant result, and one dependency fewer.

**Why.**
- Synthetic questions copy the article's wording, so they reward exact word forms and aren't the traffic the service answers. Headline retrieval claims come from real questions (D-015).
- On real questions there's no measurable difference either way, and real users do vary word forms ("connecting", "domains"), so the standard default stays.
- The test set can't settle it: tuning on test is not allowed (D-004).

**Trade-offs.** This keeps a setting that measurably hurts synthetic retrieval, so synthetic dev scores are lower than they could be (recall@5 0.900 vs 0.914, MRR@20 0.781 vs 0.807). With 50 real dev questions, a small real effect in either direction would go undetected. Revisit in Step 4, where hybrid search changes how much BM25's exact matching matters.

**In short.** "I measured stemming on and off. It clearly hurt on synthetic questions, which copy the article's wording, and made no measurable difference on real ones. Since synthetic questions aren't the traffic I serve, I kept the standard default, kept the flag in every run record, and planned to re-test it with hybrid search."

---

## D-020 · LLM response cache: SQLite, keyed by a hash of the full request (Step 2)

**Context.** From Step 3, every answer and every judge grade is a paid API call, and the same request can return different wording each time, even at temperature 0. Reports get re-run, metrics get recomputed, and the budget is $25.

**Decision.**
- A client-side cache of complete responses. The key is the SHA-256 of the request as canonical JSON (sorted keys): model, system prompt, messages and every parameter. Any change is a new key and a real call.
- Each entry stores the response plus the original call's token usage and latency. Cost reports use the stored usage; latency is reported from real calls only.
- Only successful, complete responses are stored. The first response for a key is kept, so replays are stable.
- Storage: one SQLite file (`cache/llm_calls.sqlite`, git-ignored), through the standard library's `sqlite3`.

**Alternatives.**
- No cache: every re-run pays again, and results drift between runs.
- Rely on the provider's prompt caching: a different thing. It reuses a processed prompt prefix on the provider's side, but the model still runs, produces a new answer and is billed (at a discount for the cached tokens). It's for Step 5's agent loop, not for replaying runs.
- One JSON file per call: readable, but thousands of small files, in a folder OneDrive syncs.
- `diskcache`: ready-made, but a dependency for about 40 lines of code.
- Redis: fast, but a server just to run evals. Redis comes in Step 8, for the live service, where caching has different rules (expiry, per-user privacy).

**Why.**
- Re-running a report or recomputing a metric costs nothing, and gives identical results.
- Dev iteration gets cheaper: only the requests that changed are paid for. The test set still runs only at milestones (D-004).
- SQLite is in the standard library, writes safely if a run crashes, keeps everything in one file, and can be queried ("how many tokens went to the judge?").

**Trade-offs.** A binary file that needs a tool to read. A cache hit replays an answer from the model as it was when first called; if the provider changes a model behind the same ID, only new calls see it. The cache must never be used to report latency.

**In short.** "Every LLM call goes through a cache keyed by a hash of the full request, so re-running a report is free and gives the same answers, and changing a single word of a prompt makes a real call. It's different from the provider's prompt caching, which still runs the model. Each entry keeps the original tokens and latency, so a cache hit never looks like a free, instant call."

---

## D-021 · Models: Claude Haiku 5.5 generates, Claude Sonnet 5.5 judges (Step 3)

**Context.** The thesis holds the generator constant and compares strategies, so the generator has to be the same in every run and affordable for the most expensive strategy. The judge has to be stronger than the generator (D-005). The budget is $100 of API credit.

**Decision.** `claude-haiku-5-5` ($0.10 / $0.50 per million input / output tokens) generates every answer. `claude-sonnet-5-5` ($2 / $10) judges every answer. Both are pinned by model ID and recorded in every run.

**Alternatives.**
- A stronger generator (Sonnet 5.5 or Opus 5.5): one long-context run at K = 100 (≈ 57K input tokens × 350 questions) would cost ≈ $40 or ≈ $80 in input alone, against ≈ $2 with Haiku.
- Opus 5.5 as judge: the strongest, at twice Sonnet's price. Kept as the upgrade if Sonnet fails validation.
- Haiku 5.5 as judge: very cheap, but a model grading its own answers with no extra capability.
- A judge from another provider (D-005's ideal): avoids same-family bias, but needs a second SDK, key and bill. Kept as the fallback if validation fails.

**Why.**
- The thesis measures differences between strategies, which don't need the strongest generator, and a small, fast model is realistic for a high-volume support assistant.
- Sonnet is clearly stronger than Haiku at a mid-range price.
- Self-preference (judges favor answers from their own model family) is measured, not assumed: hand labels are compared with the judge per strategy (D-005). A bias that is the same for every strategy cancels out of the comparisons; one that depends on the strategy doesn't, which is why the check is per strategy.

**Trade-offs.** Results describe a small model: a stronger one might know more without retrieval (S0) or cope better with long, noisy contexts (S3). Same-family judging stays a risk the validation can only bound. The judge is about 95% of the cost of a judged run.

**In short.** "The generator is held constant, so I chose the cheapest capable model: Haiku 5.5 made a 100-article long-context run cost $2 instead of $40. The judge is Sonnet 5.5, stronger than the generator. It's the same model family, so I validated it against my own labels per strategy, because a bias that's the same for every strategy cancels out of the comparisons, and one that isn't doesn't."

---

## D-022 · Bulk runs go through the batch API; latency is measured on real-time calls (Step 3)

**Context.** About 95% of a judged run's cost is the judge (Sonnet's price per token is 20× Haiku's, and two thirds of each judge call is output). Evaluation runs don't need answers within seconds.

**Decision.** Large generation and judging runs use the Message Batches API (half price, results within hours). Latency is measured separately, on a small sample of real-time calls, because batch timings say nothing about the service. Batch results are written to the LLM cache (D-020) like any other successful call.

**Alternatives.**
- Real-time calls for everything: results in minutes, at twice the price.
- Cut the judge's output (lower effort, a shorter verdict): possibly free, possibly at the cost of accuracy. It's an empirical question, answered by comparing the judge's agreement with the hand labels at each effort level, not assumed.

**Why.** Half price with no effect on quality; the only cost is waiting, which doesn't matter for evaluation.

**Trade-offs.** Slower iteration on large runs, so dev iteration stays small and real-time. Two code paths for calling the model (real-time and batch); both go through the same request builder and cache, so the requests are identical.

**In short.** "The judge was about 95% of the cost, so bulk runs went through the batch API at half price, since nobody waits on an eval. Latency came from a separate sample of real-time calls, because batch timings mean nothing for a live service."

---

## D-023 · Answer format: numbered articles, inline [n] citations, an exact NOT_FOUND decline (Step 3)

**Context.** Every strategy must return the same format: an answer that cites the articles it used, or a decline (design section 5). The citations feed the citation hit rate and the groundedness check; the declines feed the decline rate. WixQA article IDs are 64 hexadecimal characters.

**Decision.**
- Articles in the prompt are numbered [1] to [n]; the model cites by number, and the code maps numbers back to article IDs.
- The answer is plain text with inline citations like [2] or [1, 3]. A decline is a reply of exactly `NOT_FOUND`.
- Parse problems are measured, not assumed away: citations to numbers outside the prompt are counted as invalid, and every run reports them.
- S0 (closed book) sees no articles, so any citation it makes is invalid, and groundedness doesn't apply to it.

**Alternatives.**
- Citing the real article IDs: a single wrong character in a 64-character ID makes a citation point nowhere.
- Structured output (a JSON schema the API enforces): guaranteed to parse with an exact decline flag, but streaming the answer to a chat UI would mean extracting a JSON string as it arrives, so the service in Step 8 would want a different format and the eval would measure something nobody ships (D-008).
- The API's citations feature: precise character spans, but only for articles attached as documents (not S0, not the agent's tool results), and it can't be combined with structured output.

**Why.** Short numbers are copied reliably, and an out-of-range number is easy to detect. Inline markers stream to a chat UI with no extra machinery and become clickable citations as they arrive, so the format evaluated is the format shipped.

**Trade-offs.** The format isn't enforced: the model may paraphrase a decline or cite oddly. Both are counted on dev, and if they're common the format is revisited. A reply that mentions `NOT_FOUND` inside other text counts as an answer, not a decline.

**In short.** "Articles are numbered in the prompt and the model cites them as [2], because models mangle 64-character IDs and an out-of-range number is easy to catch. I chose inline citations over enforced JSON because the service streams answers to a chat UI, and I wanted the eval to measure the format I'd ship. The cost is that the format isn't guaranteed, so I count invalid citations in every run."

---

## D-024 · Thinking and effort: the judge's effort chosen by agreement with hand labels; the generator starts at its default (Step 3)

**Context.** Current Claude models think before answering. Thinking tokens are billed as output even when their text isn't returned, and they come before the first visible word. A thinking budget can't be set directly; `effort` (`low` to `max`) controls how much the model thinks. Defaults: `medium` on Haiku 5.5, `high` on Sonnet 5.5. Haiku can turn thinking off (at effort `high` or below); Sonnet can't. Sampling parameters such as `temperature` were removed from the API on 2026-10-21.

**Decision.**
- The judge's effort is chosen by measurement: grade the same hand-labeled answers at `medium` and `high`, and keep the cheaper setting unless its agreement with the labels (Cohen's kappa) is clearly worse.
- The generator starts at Haiku's default, `medium`, recorded in every run. Its effort, and S1's k, are tuned on dev only after the judge is validated.
- The effort is part of every request, so it's part of the cache key and the run record.

**Alternatives.**
- Judge at its default (`high`) without measuring: possibly paying for thinking that doesn't improve agreement.
- Judge at `low` to save money: decided by measurement, not assumed either way.
- Tune the generator's effort now: its answers can only be graded by the judge, which isn't validated yet, so this would tune against an unchecked instrument.
- Generator with thinking off: the fastest first word and the cheapest, but possibly worse at answers that combine several articles. A candidate in the later tuning.

**Why.** The judge has ground truth to measure against (the hand labels), so its setting can be chosen by evidence. The generator's quality can only be measured through the judge, so it waits for the judge. Any fixed generator setting keeps the strategy comparison fair, because it's the same for every strategy.

**Trade-offs.** The generator runs at an untuned setting until the judge is validated. Thinking volume, and so output cost and time to first word, is unknown until the pilot measures it.

**In short.** "Thinking tokens are billed as output and delay the first word, and effort is the only control. I chose the judge's effort by measuring its agreement with my own labels at two settings, and only tuned the generator after the judge was validated, because the generator's answers can only be graded by the judge."

**Outcome (D-031).** The judge runs at `high`. `medium` was cheaper by only $0.0003 per call, but it agreed less with the labels and scored the closed-book strategy 11 points too high.

---

## D-025 · The judge's rubric: three grades, reference-guided, blind, with JSON output (Step 3)

**Context.** Expert answers are long procedures, and a correct answer can take a different but valid route. In the pilot, Haiku gave the expert's Editor method word for word but a different second method; without a rule, graders (the judge or a human) decide such cases inconsistently, and agreement drops for reasons unrelated to the judge.

**Decision.**
- Correctness, against the expert reference: **correct** (following the answer resolves the customer's issue, consistent with the reference; one complete valid method is enough, extra accurate detail is fine), **partially correct** (the right direction, but a needed step is missing or wrong), **incorrect** (wrong, contradicts the reference, or doesn't address the question).
- Groundedness, in the same call: is every claim supported by the articles the answer cites (yes / partly / no; n/a when nothing is cited, as in S0)?
- A `NOT_FOUND` decline isn't sent to the judge: it's incorrect for correctness (every WixQA question is answerable) and counted separately as a decline.
- The judge is blind: it sees the question, the reference, the answer and the cited articles, never the strategy, model or run.
- Output is JSON enforced by the API (`reasoning`, then `correctness`, then `grounded`). The reasoning comes first, and is kept for reading disagreements during validation.

**Alternatives.**
- Correct / incorrect only: easiest to agree on, but hides "almost right", which matters for support.
- A 1–5 score: finer, but LLM judges and humans are poorly calibrated on fine scales, so agreement drops.
- Free-text verdicts: readable, but parsing them is fragile. Unlike answers (D-023), a verdict is never streamed to a user, so guaranteed parsing wins.
- Telling the judge the strategy: invites a bias that depends on the strategy, which doesn't cancel out of the comparisons (D-021).
- Groundedness in a separate call: cleaner separation of the two grades, at twice the judge calls.

**Why.** Three grades separate "fixes the problem" from "gets close" without the poor calibration of a fine scale. Rules for the hard cases make the judge and the hand labels consistent. Skipping declines saves money with no loss of information.

**Trade-offs.** "Partially correct" remains the hardest grade to agree on. Blinding can't hide everything: citation markers reveal that an answer came from a strategy with articles, which is why agreement is checked per strategy. Groundedness adds the cited articles to every judge call (about +$0.003 each).

**In short.** "The judge grades correct, partially correct or incorrect against the expert's answer, with written rules for hard cases like a valid alternative route, plus whether the answer's claims are supported by the articles it cites. It never sees which strategy wrote the answer, declines don't cost a judge call, and its verdict is JSON the API enforces, because nobody streams a verdict."

---

## D-026 · No automatic refusal fallback for the judge (Step 3)

**Context.** Sonnet 5.5's safety classifiers can occasionally decline a request (`stop_reason: "refusal"`). The API can re-run a refused request on another model automatically, and Anthropic's guidance is to turn that on by default for Sonnet 5.5.

**Decision.** The judge runs without the fallback. A refused or cut-off verdict is recorded as a judge failure, counted in every judge run's summary, and looked at by hand; it's never replaced by another model's grade. The production chatbot (Step 8) will use the fallback.

**Alternatives.** Fallback on for the judge: no manual step, but some verdicts would silently come from a grader that was never validated against the hand labels.

**Why.** The judge is a measuring instrument: pinned by model and prompt (D-005) so every verdict is comparable. A missing verdict is visible and countable; a verdict from a different grader is neither. For a customer waiting on an answer the trade-off flips: another model's answer beats no answer.

**Trade-offs.** Refusals need manual handling. They're expected to be very rare for grading help-center answers, and the count in each summary shows whether that holds.

**In short.** "The judge has no automatic fallback, because a verdict from an unvalidated model would silently change my measuring instrument; a refusal is counted and checked by hand instead. The chatbot does use the fallback, because there a customer is better served by another model's answer than by none."

---

## D-027 · Hand labels: drawn by whole question, labeled blind before any judge grade, with a reserve for a revised rubric (Step 3)

**Context.** The judge is trusted only if it agrees with hand labels (D-005: Cohen's kappa of about 0.6 or more). The labels are the ground truth for that check, so the way they're collected decides whether the check means anything. If the judge falls short, the rubric is revised by reading the disagreements, and the revised judge then needs a check of its own.

**Decision.**
- `srag label-sample` draws the sample once from the three full dev_real answer runs (S0, S1, S*) and commits it to `data/labels/sample.json` with the answer texts, like the frozen split.
- The sample takes whole questions: questions are shuffled (seed 20261102) and taken until the sample holds 80 answers. That gives 28 questions and 80 answers (S0 28, S1 25, S* 27). The other 21 questions (56 answers) are a reserve that no label touches.
- Declines aren't sampled, because they never reach the judge (D-025). Question ew-060 is excluded, because its pilot grade was discussed while writing the rubric.
- `srag label` shows one answer at a time, in random order, with the question, the expert reference and the judge's own rubric text. It never shows the strategy, the run or any judge grade. Each label is appended to `data/labels/labels.jsonl` with a timestamp, so labeling can stop and resume.
- Correctness is labeled for all 80 answers. Groundedness is labeled for 20 answers picked at random among those with citations, since it means reading the articles, and its agreement is reported as a small sample.
- Labels come first: the judge's validation run starts only after the last label, which the timestamps show.
- Judge-v1 is measured on these labels. If it falls short, the rubric is revised by reading the disagreements, and the revised judge is measured on fresh labels from the reserve, never re-measured on these.

**Alternatives.**
- Sampling single answers instead of whole questions: the same counts, but the reserve would share questions with the labeled set, so a rubric fix aimed at one answer to a question would also help on the reserve's answers to it.
- Labeling test answers: the test set is for M1 results only (D-004).
- Labeling with the judge's verdicts visible: faster, but every label would be anchored on the judge's grade, so agreement would be inflated.
- Re-measuring a revised rubric on the same labels: it is sure to look better, because it was fitted to them.
- Groundedness on all 80: about an hour more of reading articles, for a secondary metric.
- Picking the groundedness items by hand: this tends to favor short, easy answers, which would bias the sample.

**Why.** It's the dev/test principle (D-004) applied to the judge: what you tune on can't also be what you measure on. The first measurement of judge-v1 is honest because its rubric was committed before any label existed. Drawing whole questions keeps the reserve clean, for the same reason the dev/test split is drawn by question.

**Trade-offs.**
- The labeler sees all of a question's answers, so grading one can be colored by having read another; the judge sees each answer alone.
- With 25 to 28 answers per strategy, kappa per strategy has wide error bars: it shows a strategy where the judge is clearly off, not small differences.
- There's one labeler, so there's no human-to-human agreement to set a ceiling for the judge.
- The reserve is smaller than the sample (56 answers), so a second round would have wider error bars.

**In short.** "Before seeing any judge grade, I hand-labeled 80 dev answers, blind to the strategy, and measured the judge against them. The sample was drawn by whole questions, and the other dev questions were held in reserve: if the rubric had to change, I'd validate it on fresh labels from the reserve, because a rubric fixed on my labels will always agree better with those same labels."

**Outcome (D-031).** Judge-v1 passed, so the rubric wasn't revised and the reserve questions weren't needed.

**Update (D-028).** A panel of four Claude configurations now pre-labels the 80 answers; hand labels go to the items the panel splits on, plus a random audit. The sample, the blinding and the reserve are unchanged.

---

## D-028 · A panel of four Claude configurations pre-labels the sample; a person settles its splits (Step 3)

**Context.** Hand-labeling all 80 sample answers (D-027) takes about two hours. Stronger models can pre-label them, but a panel's agreement isn't a person's: if the panel shares a blind spot with the judge, a unanimous vote raises the judge's kappa exactly where the judge is wrong. Example: an answer gives a plausible menu path that isn't in the reference. A person can't confirm the path exists, but Claude models that "remember" a similar menu from shared training data may all accept it as a valid alternative route.

**Decision.**
- Four configurations grade the 80 sample items: Claude Opus 5.5 at high and at max effort, and Claude Fable 5.1 at high and at max effort (`srag panel`).
- They use the judge's own request (D-025) with only the model, the effort and the output limit changed, so they see exactly what the judge sees and never the strategy. The output limit is 20,000 tokens instead of 16,000, because Opus at max effort thought for up to 14,140 tokens on a pilot item. Bulk calls go through the batch API (D-022). As with the judge, there's no refusal fallback (D-026).
- If all four agree on an item, it takes that grade. Items they split on are labeled by hand, blind to their votes. So is a random audit of about 12 unanimous items, which measures how often a unanimous panel is wrong.
- The printed panel summary shows aggregates only: how many items are unanimous or split, and which members disagree. That way the split items can still be labeled blind. Whether and how much to hand-label is decided after seeing those aggregates.

**Alternatives.**
- Hand-labeling all 80 (D-027 as first written): the strongest ground truth, at about two hours of labeling.
- A panel with no hand labels: no labeling time, but the judge would then be validated against stronger Claude models, not a person, and blind spots shared by Claude models would go unmeasured.
- Majority vote instead of unanimity: fewer items to label, but three votes from two models could outvote the one dissent that marks a hard item.
- Sonnet 5.5 on the panel: cheap, but it's the judge, so its mistakes would become part of the ground truth it's measured against. Haiku 5.5 wrote the answers, so it stays off too.
- A model from another provider: the most independent member, but outside the Anthropic-only setup (D-021).
- Claude Code sessions or subagents as labelers: no API cost, but their input isn't exactly the judge's. Claude Code adds its own instructions, and subagents in this repo could read each item's strategy. They would also grade all 80 items in one context and leave no run record.

**Why.**
- Two models give two opinions. Running each at two efforts shows whether a grade holds when the model thinks longer, which flags borderline items.
- Requiring unanimity sends every doubtful item to a person.
- Reusing the judge's request makes "the panel saw what the judge saw" true by construction, not by instruction.

**Trade-offs.**
- All the models involved are Claude, so the panel can share the judge's blind spots; the audit measures that, on a small sample.
- It cost $11.29, where hand labels cost nothing: $1.39 for a 5-item pilot, $9.48 for the batch, and $0.42 for retrying in real time a call that Opus at max effort had cut off at its output limit. A batch result that comes back incomplete is retried in real time, which covers temporary errors but pays twice for a cut-off.
- The final labels mix panel grades (easier items) and hand labels (harder items), so kappa is also reported on the hand-labeled items alone.

**In short.** "Four Claude configurations, two models at two efforts each, pre-label the sample through the judge's own prompt, so they see exactly what the judge sees. Items they all agree on take the panel's grade; the items they split on are labeled by hand, blind, plus a random audit of the unanimous ones, because models from one family can share a blind spot with the judge, and only a human check measures that. The judge's own model stays off the panel, so it can't grade itself into the ground truth."

**Update (2026-11-02).**
- The panel settled 65 of 80 items and didn't settle 15. One of the 15 is unsettled because Opus at max effort hit its output limit twice.
- The rule is `panel.consensus`: an item gets a label only when all four members graded it and agreed. A failed call counts as a split. The votes and labels are committed in `data/labels/panel.json`.
- The 15 are hand-labeled blind (`srag label --adjudicate`), with groundedness asked where the panel split on it too.
- The random audit of unanimous items is deferred to the end of the project, by choice, together with the 2 groundedness splits outside the 15. Until then the 65 panel labels are unaudited, so the judge's agreement is reported separately on them and on the 15 hand labels.
- Without audit items mixed in, the labeler knows every item shown is one the panel split on.

---

## D-029 · Real-time LLM calls stream (Step 3)

**Context.** Calls were non-streaming: one request, one complete response. The Anthropic SDK refuses a non-streaming call whose `max_tokens` allows more than about 10 minutes of generation (above about 21,300 tokens), because a long silent HTTP request can time out. At max effort, the panel's Opus thought for up to its full 20,000-token limit on one item, so that ceiling was close to binding.

**Decision.** `LLMClient` calls `messages.stream()` and collects the whole message with `get_final_message()`. Everything downstream is unchanged: the same response JSON, the same cache entry, and latency still measured end to end. Batch calls aren't affected (D-022). Output limits stay where they are, so cached calls stay valid; they can now be raised when a run needs it.

**Alternatives.**
- Stay non-streaming and keep `max_tokens` under the SDK's limit: the simplest code, but it caps how long a model may think, and long calls risk HTTP timeouts.
- Stream only calls with a large `max_tokens`: two code paths for one job.
- Stream and also record time to first token: that's the latency the chatbot's users will feel (Step 8), but it needs another field in the cache and the run records. It's added when latency becomes a result.

**Why.** One code path, with no ceiling on output and no long silent HTTP wait. It's also the path the service will use, since the chatbot streams its answers (D-008: evaluate what you ship).

**Trade-offs.** Slightly more involved client code and test fakes. Streaming doesn't change what the model generates or what it costs. Time to first token isn't recorded yet.

**In short.** "Every real-time call streams, so a model that thinks for a long time never hits the SDK's non-streaming limit or an HTTP timeout, and evaluation uses the same path the chatbot will. The whole message is collected before it's scored, so outputs, the cache and costs don't change."

---

## D-030 · Judge agreement: unweighted Cohen's kappa, on the three grades and on correct vs not (Step 3)

**Context.** The judge is trusted only if it agrees with the labels (D-005). Raw percent agreement flatters a judge when one grade dominates: if 70% of answers are correct, a judge that always says "correct" agrees 70% of the time and has learned nothing. The grades are also ordered (incorrect, partial, correct), and the headline metric only counts "correct".

**Decision.**
- Report percent agreement and Cohen's kappa: (p_o − p_e) / (1 − p_e), where p_o is observed agreement and p_e is the agreement expected by chance from each rater's own grade frequencies.
- Two kappas: one on the three grades, and one on correct vs not correct (partial and incorrect merged), which is the decision behind % correct.
- Both kappas unweighted: every disagreement counts the same.
- Reported overall, per strategy, and per label source (hand labels on the panel's splits, panel labels on its unanimous items), with the judge failures counted separately. The bar is 0.6 (D-005).
- Example: 10 answers labeled 7 correct, 2 partial, 1 incorrect. A judge that swaps one correct and one partial agrees on 8 of 10 (80%), but chance alone gives 54%, so kappa is (0.80 − 0.54) / (1 − 0.54) = 0.57, below the bar. A judge that always says "correct" agrees 70% and scores exactly 0.

**Alternatives.**
- Percent agreement only: easy to read, but it can't tell a judge from a rubber stamp.
- Weighted kappa (linear or quadratic): gives partial credit when correct and partial are confused, since the grades are ordered. But the weights are a choice our three grades don't justify (they aren't evenly spaced), and they'd hide exactly the confusion that moves % correct.
- Kappa on the three grades only: misses whether the judge gets the decision the headline metric uses right.
- Krippendorff's alpha: handles many raters and missing labels. Here there are exactly two raters (the judge and the labels) on complete data, where it adds nothing over kappa.

**Why.** Kappa is the standard chance-corrected agreement for two raters, and it's what D-005 committed to. The correct-vs-not kappa checks the decision the results actually depend on. Splitting by strategy and label source shows where agreement is weak: on closed-book answers, and on the hard items the panel split on.

**Trade-offs.** Kappa depends on how often each grade occurs, so it isn't comparable across groups with very different grade mixes, and it's undefined when both raters use a single grade. Each strategy has only 25 to 28 items, so per-strategy kappas are rough until the statistics step adds confidence intervals (D-006).

**In short.** "I measured the judge with Cohen's kappa, not raw agreement, because when most answers are correct, a judge that always says 'correct' agrees most of the time and has learned nothing. Kappa subtracts the agreement you'd get by chance. I report it on the three grades and on correct versus not, since that's the decision behind my headline metric, and per strategy, against a bar of 0.6."

---

## D-031 · The judge is validated: judge-v1 at high effort, trusted for % correct (Step 3)

**Context.** Sonnet 5.5 graded all dev answers with rubric judge-v1, at `medium` and at `high` effort. On the 80 labeled sample items (15 hand labels, 65 unanimous panel labels; D-027, D-028), it was compared with the labels by agreement and Cohen's kappa (D-030). Results are in `reports/judge_validation.md`.

| Judge effort | Agreement | Kappa, 3 grades | Kappa, correct vs not | Cost per call |
|---|---:|---:|---:|---:|
| medium | 85.0% | 0.73 | 0.85 | $0.0034 |
| high | 90.0% | 0.82 | 0.92 | $0.0037 |

**Decision.**
- Judge-v1 at `high` effort is the judge from here on. It clears the 0.6 bar overall and on correct vs not for every strategy, so the rubric isn't revised and the reserve questions aren't used.
- The headline metric is % correct. Splitting the rest into partial vs incorrect isn't reliable for S0 (kappa 0.58 on three grades at `high`), so S0's partial count isn't interpreted.

**Alternatives.**
- `medium` effort: $0.0003 cheaper per call, but it scored S0 at 25% correct where the labels say 14%, while leaving S1 about right. That's a strategy-dependent bias, which doesn't cancel out of comparisons (D-021). Its kappa on S0 was 0.49.
- Revise the rubric for partial vs incorrect on closed-book answers: it would need fresh labels from the reserve (D-027), for a distinction the headline metric doesn't use.

**Why.** D-024's rule keeps the cheaper effort unless it's clearly worse. `medium` was clearly worse where it matters: on the closed-book strategy, in a direction that would flatter it against the others.

**Trade-offs and caveats.**
- The overall kappa is carried by clear-cut items. On the panel's 65 unanimous items, kappa is 0.97. That's likely optimistic, because the labels come from the same model family and aren't audited yet (D-028).
- On the 15 hand-labeled items, where the panel split, kappa is 0.22 (three grades) and 0.33 (correct vs not): close to chance. About 1 answer in 5 is borderline, and there the judge's grade is close to a coin flip.
- At `high`, the judge's % correct on the sample is 4 points above the labels on S0, equal on S1, and 7 points above on S*. That's 1 to 2 items each, within noise, but small gaps between strategies need confidence intervals (D-006) before they mean anything.

**In short.** "I validated the judge against 80 labels: kappa 0.82 on three grades and 0.92 on correct versus not, at high effort, well above my 0.6 bar. Medium effort was barely cheaper but scored the closed-book strategy 11 points too high, a bias that would have distorted the comparison, so I kept high. The honest caveat: on the hardest answers, where even four strong models disagreed, the judge is close to chance, so small differences between strategies need confidence intervals before they mean anything."

---

## D-032 · Statistics: bootstrap intervals over questions, paired comparisons with McNemar's test, Holm across comparisons (Step 3)

**Context.** Strategies are compared on 50 dev questions now and about 350 test questions at M1. At these sizes, one accuracy figure is uncertain by about ±14 points (dev) or ±5 (test), and every strategy answers the same questions, which a comparison should use (D-006).

**Decision.**
- Each strategy's % correct gets a 95% percentile bootstrap interval: resample the questions with replacement 10,000 times (seed fixed), recompute, and keep the middle 95%.
- Two strategies are compared question by question. The difference gets a paired bootstrap interval (resampling questions, which is the same as resampling the per-question differences). It also gets McNemar's exact test: only the questions where exactly one strategy is right count, and under "no difference" each is a coin flip. That's the sign test used for stemming in Step 2 (D-019).
- The three comparisons (S1 vs S0, S* vs S1, S* vs S0) get Holm-adjusted p-values.
- Declines count as incorrect. A judge failure drops the question from that strategy's figures and from any comparison that includes it.
- `srag report-answers` writes `reports/answers_<split>.md`.

**Alternatives.**
- Wilson or normal-approximation intervals: exact enough for a single proportion and needs no resampling. But the bootstrap treats every metric the same way (% correct, cost, MRR, a difference), so one method covers the whole project.
- BCa bootstrap: better coverage for skewed statistics, but more complex, for little gain on proportions.
- Unpaired comparisons (two separate intervals, or a two-proportion test): they ignore that both strategies saw the same questions, so they're wider. Easy and hard questions affect both strategies, and pairing cancels that out.
- A t-test on the 0/1 results: it assumes roughly normal data, which per-question 0/1 results aren't.
- No correction for multiple comparisons: with three tests, the chance of at least one false "significant" result is about three times higher than the stated level.

**Why.** The question is the unit that varies, so it's what gets resampled. Pairing turns "is A's rate different from B's?" into "on the same questions, how often does A win where B loses?", which is the more sensitive question. McNemar's test is exact at any sample size and easy to explain.

**Trade-offs.**
- The percentile bootstrap is slightly too narrow on small samples with rates near 0% or 100%.
- The intervals cover sampling noise only, not the judge's own errors (D-031). A difference within a few points of zero should still be read with the judge's caveats in mind.
- Holm is conservative when the comparisons are related, as these are.

**In short.** "Every rate has a 95% bootstrap interval over questions, and strategies are compared on the same questions: a paired bootstrap for the size of the difference, and McNemar's exact test, which only looks at questions where one strategy is right and the other wrong. With three comparisons I adjust the p-values with Holm. On 50 dev questions, after tuning, retrieval beats closed book by 52 points, with an interval of 36 to 68."

---

## D-033 · Tuning effort and k on dev: the default stays unless a candidate is clearly better (Step 3)

**Context.** The generator has run at Haiku's default effort (`medium`), and S1 at k=5, until the judge was validated (D-024, D-031). Dev has 50 real questions. On that sample, one rate is uncertain by about ±14 points, so only large differences can be detected. Trying several settings makes the best one look better than it is: if three settings are truly equal, the one that measures highest is typically several points high by luck (the winner's curse).

**Decision (written and committed before the tuning runs).**
- Effort: `low` and `high` against `medium`, on all three strategies, since one effort applies to all of them and keeps the comparison fair (D-024). The decision uses the pooled paired difference: per question, the candidate's result minus the default's, averaged over the strategies, with a bootstrap interval over questions.
- k for S1: 1, 3 and 10 against 5, at the effort chosen above.
- A candidate replaces the default only if it's clearly better: its paired 95% interval lies entirely above zero. Otherwise the default stays. A cheaper setting that's within noise is noted for retesting on more questions; Step 4 re-tunes retrieval anyway.
- The judge stays at judge-v1, `high` (D-031). Answers are generated in real time, so latency can be compared; judging goes through the batch API.
- `srag report-tuning` writes `reports/tuning_<name>.md`.

**Alternatives.**
- Take the setting with the best number: simple, but on 50 questions that's partly picking luck, and the gain shrinks on the test set.
- Tune on the 500 synthetic dev questions: ten times more questions, but easier ones, with wording taken from the articles, a distribution the judge wasn't validated on and the product won't see.
- A full grid of effort × k (12 settings): catches interactions, but gives luck more chances to produce a false winner.
- Tune on the test set: never (D-004).

**Why.** With few questions and several candidates, the default needs protection against luck, and a rule written before seeing results can't be bent to fit them. Pooling over strategies uses all 150 answers for the effort decision while keeping the question as the unit, since three answers to one question aren't independent.

**Trade-offs.** A real but small improvement, a few points, can't be detected on 50 questions and will be missed. Keeping the default when nothing wins is conservative, not a finding that the settings are equal. The runs cost about $2.

**In short.** "I tuned the generator's effort and the number of retrieved articles on dev, with the rule fixed before any run: the default stays unless a candidate's paired 95% interval is entirely above zero. On 50 questions, picking the best-looking of several settings mostly picks luck, so a clear win was the only kind I'd accept."

**Outcome (2026-11-03).** Results are in `reports/tuning_effort.md` and `reports/tuning_k.md`.

- **Effort stays `medium`.** Pooled over the three strategies, `high` was +4.0 points [−0.7, +8.7] and `low` was +2.7 [−3.3, +9.3]; both intervals include zero.
  - Both low and high measuring above medium is itself a sign of noise: if thinking helped, the order would be low < medium < high.
  - `low` was about 28% cheaper and a second faster on S0, with no sign of being worse. It's the leading candidate for the chatbot (Step 8), where latency matters, and gets retested there.
- **S1 moves to k=10** (`DEFAULT_K` in `answer_eval.py`).
  - k=10 against k=5: +22 points [+10, +34], with 12 questions where only k=10 was right and 1 where only k=5 was (Holm p = 0.007).
  - k=3 measured −8 [−16, −2] (0 against 4; Holm p = 0.125) and k=1 −26 [−40, −14].
  - Cost per answer rises from $0.0010 to $0.0017.
  - BM25's recall explains part of the gain: a gold article is in the top 10 for 66% of real dev questions, against 53% in the top 5. Declines also fell from 6 to 2.
- **What it changes.**
  - At k=10, S1 is level with S* on dev (76% against 78%, +2 [−14, +18]; each beats the other on 7 or 8 questions).
  - The earlier reading that "retrieval is the bottleneck, about 24 points" came mostly from k=5 being too few articles.
  - About 1 in 5 answers is wrong even with the gold articles, so the generator, the prompt and the reference answers set the ceiling.
  - Step 4's better retrieval may pay off more as fewer tokens for the same accuracy than as higher accuracy.
  - k of 20 and above is S3's question (Step 6), so the grid wasn't extended past 10.
- **One more source of noise.** The generator isn't deterministic, so rerunning a setting changes some answers. The bootstrap covers which questions were drawn, not which answers the model happened to write. That's another reason not to read small gaps.

---

## D-034 · The test set runs only at a named milestone, from a clean commit, enforced in code (Step 3)

**Context.** Everything so far was tuned on dev: the judge's effort, the generator's effort and S1's k (D-031, D-033). The test set (350 real questions) gives the numbers the README reports. It's only trustworthy if no choice was made after seeing it (D-004), and if each reported number can be traced to the exact code that produced it.

**Decision.**
- `srag answer --split test` runs only with `--milestone M1` (or M2, M3), and only when the git working tree has no uncommitted changes. Otherwise it refuses. The milestone and the commit are recorded in the run.
- M1 answers the test set once with the tuned settings: S0, S1 at k=10 and S*, at effort `medium`. The answers are generated in real time, so latency is measured (D-022), and judged by judge-v1 at `high` through the batch API.
- The results are reported as they come out. No setting is changed in response to them; changes go through dev and wait for the next milestone.

**Alternatives.**
- The rule in CLAUDE.md only: it works until a debugging session runs `--split test` "just to check".
- Locking the test data away (a separate file only the milestone job can read): the strongest guard, but more machinery than a one-person project needs.
- Running the test set after every step: it would quietly become a second dev set, since every look tempts a change.

**Why.** A guard in code turns "never by accident" into a property of the tool, and the clean-commit requirement means every reported number points to the code that produced it. Committed reports leave an audit trail: any later test run would show in the git history.

**Trade-offs.** It can't stop a deliberate rerun after a change; it makes one visible. A milestone run is one sample of a nondeterministic generator, so rerunning it gives slightly different answers, and the intervals don't cover that (D-033).

**In short.** "The test set only runs at a named milestone, from a clean commit, and the tool refuses otherwise. Every setting was tuned on dev first, the test numbers are reported as they came out, and each one traces back to the commit that produced it."

**Outcome (2026-11-03): M1.** The test run was made once, from commit `eadbd6a`, and cost about $4.60. Results are in `reports/answers_test.md` and the README.

- **% correct on 350 test questions:** S0 33.4% [28.6, 38.3], S1 at k=10 64.0% [59.1, 68.9], S* 74.6% [69.7, 78.9].
- **Paired:** S1 beats S0 by +30.6 [+23.7, +37.1]. S* beats S1 by +10.6 [+4.9, +16.3], with 73 questions only S* got right against 36 only S1 did. All three hold after Holm (p < 0.001).
- **This revises D-033's reading of dev.** On dev, S1 looked level with S* (+2 [−14, +18]). That interval was wide enough to hide a 10-point gap, and the test set, 7 times larger, shows one.
  - So better retrieval (Step 4) has an accuracy target, not only a token-saving one.
  - A gold article is in BM25's top 10 for 57% of test questions, against 66% on dev, which explains part of S1's drop from 76% to 64%. Picking k=10 on dev flatters dev, too.
- **The generator's ceiling:** with the gold articles, 89 of 350 answers aren't correct: 48 are partial, 34 are declines and 7 are wrong.
- Nothing was changed in response. This entry and the README report the numbers as they came out.

---

## D-035 · Embeddings: local open-weight models first; Voyage is a measured upgrade (Step 4)

**Context.** BM25 misses paraphrases. A gold article is in its top 10 for 66% of real dev questions and 57% of test questions, and S* beats S1 by 10.6 points on test (D-034). Dense embeddings target that gap. An embedding model can run locally or be called through an API. Context length matters too: 33% of the articles are over 512 tokens and 12% are over 1,024, so a 512-token model would see only the start of a third of the knowledge base.

**Decision.**
- Local, open-weight models first, run with `sentence-transformers`: on the GPU to embed the knowledge base, on the GPU or CPU to embed questions. Two candidates, both Apache 2.0 with a 32K-token context:
  - Qwen3-Embedding-0.6B: 0.6B parameters, 1,024 dimensions, takes a task instruction with each query.
  - voyage-4-nano: 340M parameters. It shares an embedding space with Voyage's API models, so it keeps the upgrade path open.
- Weights are pinned to a Hugging Face revision, which every run records, like the dataset revision. Article vectors are computed once per model and revision and stored locally.
- Whole articles first (D-012). Chunking is its own experiment later in this step.
- Voyage's API is the upgrade, with the trigger fixed now: if the better local model doesn't clearly beat BM25 on real dev questions (recall@10, paired 95% interval entirely above zero), voyage-4 is measured before moving on to hybrid fusion. Its free allowance (200M tokens per model) covers the whole project, which needs about 20M.

**Alternatives.**
- Voyage first: top quality and effectively free here, but it needs an account and a key, adds a network call to every query, and the provider can retire the model.
- OpenAI or Cohere embeddings: the same trade-offs as Voyage. Voyage is the provider Anthropic's documentation points to.
- A 512-token model (e5, bge-small): smaller and faster, but blind to the end of a third of the articles.
- Larger local models (Qwen3-Embedding-4B or 8B): better on benchmarks, but the 4B model alone needs about 8 GB of the GPU's 11 GB in fp16, and neither could embed queries on the small server planned for Step 9.

**Why.**
- Free, reproducible from a pinned revision, and the data stays on the machine. No new account.
- Both fit the RTX 2080 Ti for indexing, and both are small enough to embed questions on a CPU when serving.
- A 32K context holds every article whole.
- Two candidates instead of one, so "why this model?" has a measured answer.

**Trade-offs.**
- PyTorch adds about 3 GB to the environment, and the service image will carry the model (Steps 8–9).
- The RTX 2080 Ti has no bfloat16 or FlashAttention 2, so the models run in fp16 or fp32 and may differ slightly from their reference setup.
- voyage-4-nano needs `trust_remote_code`: loading it runs a small Python file from its repository. The file is read before first use, and the revision is pinned.
- Hosted models score higher on public benchmarks, so this may leave a few points on the table. That's what the trigger checks.

**Public evidence (checked 2026-11-04).** [HAKARI-Bench](https://huggingface.co/datasets/hakari-bench/results/discussions/14), an independent benchmark, ran both models on 551 retrieval tasks (nDCG@10, the same weights as the pinned revision).
- voyage-4-nano leads overall, 0.634 to 0.598, and in 30 of its 35 task groups.
- Qwen3-Embedding-0.6B leads on the English MTEB v2 retrieval tasks, 0.637 to 0.616. That group is the closest to an English help center.
- Neither clearly wins for this corpus, so both stay and real dev questions decide. Note that nano was scored at its best size in bf16, while here it runs at 1,024 dimensions in fp32 (fp16 broke its vectors; see D-036's outcome).

**Outcome (2026-11-05).** Measured by D-036's rule, both models clearly beat BM25 on real dev questions, so the Voyage trigger doesn't fire. voyage-4-nano goes forward (recall@10 0.86, against 0.83 for Qwen and 0.66 for BM25).

**In short.** "I started with local open-weight embedding models: free, pinned by revision and fully reproducible, with a 32K context, so a third of the articles aren't cut off. I compared two of them on real dev questions, and decided in advance when I'd switch to a hosted API: if local embeddings didn't clearly beat BM25."

---

## D-036 · Choosing a retriever on dev: real questions decide by recall@10, and only a clear win replaces the default (Step 4)

**Context.** Step 4 makes a chain of retrieval choices: BM25 → dense (Qwen3-Embedding-0.6B or voyage-4-nano) → hybrid → reranker → sections → feature-request down-weighting. All of them are measured on dev, which has 50 real questions; the test set stays locked until M2 (D-034). BM25's recall@10 is 0.66 on the real dev questions, and for 14 of the 50 none of the gold articles is in its top 10. On the 500 synthetic dev questions it's 0.95. If the rule is written after the results, it's easy to pick the rule that fits them.

**Decision (written and committed before any dense retrieval run on dev).**
- The 50 real dev questions decide. The 500 synthetic dev questions are reported alongside, as a check for a broken retriever, not as a vote.
- The deciding metric is recall@10, because S1 gives the generator 10 articles (D-033). MRR@20, all-gold@10 and recall@5 are reported but don't decide.
- **Replacing the default.** A challenger replaces the default only if the paired 95% bootstrap interval of its recall@10 difference (challenger minus default, question by question) lies entirely above zero. Otherwise the simpler option stays. This is D-033's rule.
- The order is fixed: BM25 → dense → hybrid → reranker → sections → feature-request down-weighting. Each winner becomes the next default.
- **Choosing between two new options** (Qwen vs nano). Neither is a default, so the one with the higher real-dev recall@10 goes forward, even within noise. An exact tie goes to MRR@20. The report calls it "chosen", and "better" only if its interval clears zero.
- The Voyage trigger is unchanged (D-035): if the chosen model doesn't clearly beat BM25 by this rule, voyage-4 is measured before hybrid.
- The rule isn't changed after results are seen. A new question gets its own rule, written here before its runs. One is planned for the last sub-step: whether the better retriever lets S1 use fewer articles (k=5) at the same accuracy.

**Alternatives.**
- The higher number wins: on 50 questions that's partly luck (D-033's winner's curse).
- Synthetic dev decides: ten times the questions, but BM25 is already at 0.95, and the questions copy the articles' wording, which favors keyword search (D-019).
- The 5,721 synthetic reference questions: the same bias, more of it.
- A minimum gain on top of the interval, such as +5 points: an arbitrary threshold. At n = 50, any win that clears the interval is already at least 8 points.
- MRR or nDCG as the deciding metric: rank within the top 10 matters less than whether the gold article is there at all, since the generator reads all 10.

**Why.**
- Real questions are the traffic; synthetic ones measure an easier, different task.
- Recall@10 is about what the generator actually reads.
- A default that works is protected from a lucky challenger. Between two new options there's nothing to protect, so the higher score is the better bet.
- On 50 questions, what decides is how many questions flip each way. A challenger that breaks none needs to fix 4; one that breaks 4 needs to fix 12. Fixing 10 and breaking 5 (+10 points) isn't enough: its interval is [−4, +24].

**Trade-offs.**
- Gains of a few points can't be seen on 50 questions. Keeping the default means "not shown to be better", not "equal".
- Every choice reuses the same 50 questions, so dev numbers get more optimistic with each one. M2's test run is the honest check.
- Picking the better of two models and then comparing it with BM25 flatters it slightly. That only makes the Voyage trigger less likely to fire.

**In short.** "Every retrieval choice was decided on the 50 real dev questions by recall@10, which is what the generator actually reads, with the rule written before any run: a change replaces the default only if its paired 95% interval is entirely above zero. Synthetic questions copy the articles' wording, so they're reported but don't vote. Between two new embedding models, the higher score went forward, since neither had a position to protect."

**Outcome (2026-11-05): dense replaces BM25, with voyage-4-nano.** Results are in [reports/retrieval_dense.md](../reports/retrieval_dense.md).

| Real dev questions, recall@10 | Difference [95% CI] | Fixed / broken | Under the rule |
|---|---|---|---|
| Qwen3-Embedding-0.6B vs BM25 (0.83 vs 0.66) | +17 points [+3, +32] | 16 / 4 | Clearly better |
| voyage-4-nano vs BM25 (0.86 vs 0.66) | +20 points [+8, +33] | 16 / 2 | Clearly better |
| voyage-4-nano vs Qwen3-Embedding-0.6B | +3 points [−6, +12] | 4 / 2 | Within noise: nano is chosen, not shown to be better |

- **Dense becomes the default, using voyage-4-nano.** Both models clearly beat BM25. Nano scored higher, so it goes forward to hybrid. It's also the smaller model, which helps when questions are embedded on a CPU (Step 9).
- **The Voyage trigger doesn't fire** (D-035): the chosen model clearly beats BM25, so voyage-4 isn't measured.
- **Synthetic questions agree.** Dense beats BM25 there too (recall@10 0.998 against 0.95), even though those questions copy the articles' wording, which favors keyword search.
- **Recall@5 also rose,** from 0.53 to 0.78 with nano. That matters for the planned k=5 question.
- **Nano runs in fp32.** In fp16 it produced vectors that were zero or not finite, and a check added before this run refused them. Its reference setup is bf16, which this GPU doesn't have. Embedding the knowledge base took about 4 minutes, against under 3 for Qwen in fp16.

---

## D-037 · Hybrid fusion: RRF and a weighted score, one untuned setting each, over each retriever's top 100 (Step 4)

**Context.** voyage-4-nano is the default retriever (D-036). BM25 matches exact words and nano matches meaning, so combining them might catch what each misses. Their scores can't simply be added: BM25's have no upper limit, while cosine similarities here run from about 0 to 1. The room to improve is small. Nano misses a gold article on 8 of the 50 real dev questions, so under D-036 hybrid needs 4 fixes with none broken, or 7 with one broken. With two broken, a clear win is out of reach.

**Decision (written and committed before any hybrid run).**
- Hybrid combines BM25 (with stemming) and voyage-4-nano over whole articles. Each supplies its top 100.
- **RRF (Reciprocal Rank Fusion).** An article's score is the sum, over the lists it appears in, of 1 / (60 + rank), with rank starting at 1.
- **Weighted score.** Each list is scaled per question so its best score is 1 and its lowest is 0. The score is 0.5 × dense + 0.5 × BM25, and an article missing from a list gets 0 from it.
- Equal scores are ordered by article ID, so every run gives the same ranking.
- Neither k, the weight nor the depth is tuned.
- Between the two, D-036's rule for two new options applies: the higher real-dev recall@10 goes forward, and an exact tie goes to MRR@20.
- The winner then challenges dense under D-036: its paired 95% interval must lie entirely above zero. Otherwise dense stays.
- One question is planned for the reranker sub-step: whether its candidates come from dense alone or from both retrievers. Its rule gets written before those runs.

**Alternatives.**
- Tuning the weight or k on dev: a tuned weighted score [beat RRF in Bruch et al.](https://arxiv.org/abs/2210.11934), and needed little data to tune. But with 8 questions left to fix, a tuned value would fit a handful of questions, and those same questions would then judge it.
- RRF only: less code, but the weighted score is its main competitor, and measuring it is free.
- Adding raw scores: BM25's larger numbers would decide almost everything.
- Other scaling, such as z-scores: Bruch et al. found the weighted score generally isn't sensitive to the choice of scaling, so the simplest one is used.
- Learned fusion: needs far more than 50 questions.
- Each retriever's top 20: an article at rank 40 could never climb. The whole corpus: more weak matches, with no reason to expect a gain.

**Why.**
- RRF uses only ranks, so it needs no scaling, and it keeps working if sub-step 7 swaps BM25 for another keyword engine.
- A weighted score keeps how strong each match is, which RRF throws away, so the two can disagree (RRF favors agreement; weighted favors one strong signal).
- One setting each leaves luck the fewest chances to produce a false winner (D-033).

**Trade-offs.**
- This compares two untuned defaults, not each method at its best. A tuned weighted score might do better.
- RRF rewards agreement: an article at rank 30 in both lists scores 2/90 = 0.022, beating one at rank 1 in a single list at 1/61 = 0.016. On paraphrase questions, where BM25 can't see the gold article, that can push it out of the top 10.
- In the weighted score, an article missing from a list counts the same as that list's weakest article.
- Two retrievers in production mean more latency and one more index to maintain. If hybrid doesn't clearly win, the simpler system stays.

**In short.** "I measured hybrid two ways, RRF and a weighted score, one default setting each, because tuning on 50 questions mostly fits noise. Hybrid then had to clearly beat dense alone, under the rule I'd already fixed. With only 8 questions left to fix, I said before the run that it might not."

**Outcome (2026-11-05): dense stays the default; hybrid didn't clear the bar.** Results are in [reports/retrieval_hybrid.md](../reports/retrieval_hybrid.md).

| Real dev questions, recall@10 | Difference [95% CI] | Fixed / broken | Under the rule |
|---|---|---|---|
| RRF vs dense (0.79 vs 0.86) | −7 points [−18, +4] | 3 / 8 | Not clearly better |
| Weighted vs dense (0.83 vs 0.86) | −3 points [−12, +5] | 2 / 5 | Not clearly better |
| Weighted vs RRF | +4 points [−4, +12] | 3 / 1 | Weighted goes forward as the higher score |

- **Dense with voyage-4-nano stays the default.** The weighted score went forward and then didn't clear the bar. Both intervals include zero, so hybrid isn't shown to be worse either, but nothing supports the extra index and latency.
- **The risk named before the run is what happened.** On every question hybrid broke, under both methods, the gold article that dropped out was beyond BM25's top 20. For example, on `ew-000` nano had it at rank 2, and under RRF it fell out of the top 20.
- **The fixes were the expected ones.** They came where BM25 ranked the gold article well: rank 1 on `sim-184` and rank 5 on `sim-021`. RRF also fixed `sim-064` (BM25 rank 14).
- **Why equal say hurts here.** BM25 is much weaker than nano on these questions (0.66 against 0.86), and both methods give it the same say.
- **Synthetic questions lean the same way.** RRF is clearly lower than dense (−1.2 points [−2.2, −0.4]), and the weighted score is level.
- **No tuning after the fact.** A weight moved toward dense would only approach dense alone, and choosing it after seeing these results would fit those same few questions.
- **What this doesn't show.** It doesn't show that hybrid is bad in general. It's two untuned defaults on 50 questions and one knowledge base. BM25 can still contribute candidates for the reranker, which is the question planned above.

---

## D-038 · Reranking: a cross-encoder over dense's top 50, two local models, BM25 candidates reported only (Step 4)

**Context.** voyage-4-nano is the default retriever (D-037). Its vectors are made without seeing the question. A cross-encoder reads the question and an article together, one pair at a time, so it can weigh a single sentence against that question. Nothing can be computed ahead, and reading all 6,221 articles would take about 3 minutes per question on this GPU, so it only reads a shortlist and can only reorder what's on it. The room to improve at recall@10 is small. Of the 62 gold articles of the 50 real dev questions, 53 are in nano's top 10, 6 are at ranks 11–20 and 3 are beyond 20. That leaves 8 questions missing a gold article at recall@10, and 13 at recall@5.

**Decision (written and committed before any reranker run).**
- **Two local models,** both Apache 2.0, with no remote code, pinned to a Hugging Face revision:
  - Qwen3-Reranker-0.6B: 596M parameters, 32K-token context, so it reads every article whole.
  - ms-marco-MiniLM-L6-v2: 23M parameters, 512-token limit, so a third of the articles get cut.
- **Candidates.** Dense's top 50, whole articles (`contents`, the same text that was embedded). Each model uses its authors' default settings, including Qwen's default instruction. Nothing is tuned. Equal scores keep dense's order.
- **Between the two models,** D-036's rule for two new options applies: the higher real-dev recall@10 goes forward, and an exact tie goes to MRR@20. Speed is reported but doesn't vote.
- **Against the default,** the chosen model must clear D-036's bar: its paired 95% interval on real-dev recall@10 lies entirely above zero. Otherwise dense stays. On these 50 questions that takes 4 fixed with none broken, or 7 fixed with one broken. With two broken, it's out of reach, since only 8 questions have anything to fix.
- **BM25 candidates** (the question D-037 left open). The chosen model also runs once on dense's top 50 plus BM25's top 50, without duplicates, so up to 100 articles. This run is reported but can't replace the default. See Why for the arithmetic.
- **A path for recall@5.** If the chosen model misses the bar on recall@10 but its recall@5 interval against dense lies entirely above zero, dense stays the default and the reranker becomes a candidate for the k=5 question (D-036), under a rule written before those runs. If it clears neither, it's dropped.
- **Speed.** Runs record latency on the GPU. A small CPU timing is also taken, because Step 9's server may have no GPU.

**Alternatives.**
- bge-reranker-v2-m3: the usual baseline, but as large as Qwen (568M parameters), and its model card's examples cut documents at 512 or 1,024 tokens. It gives up both Qwen's reach and MiniLM's size.
- mxbai-rerank-base-v2: long-context and a similar size to Qwen, so it adds a third chance for luck without a different trade-off. The only benchmark I found for it is its authors'.
- jina-reranker-v3: a non-commercial license (CC BY-NC 4.0), and it needs remote code.
- Hosted rerankers (Voyage, Cohere): an account, a key and a network call on every question. They're the fallback if local models can't be served fast enough.
- A Claude model as the reranker: slower and more expensive per question.
- Top 20: can't reach the 3 gold articles beyond rank 20. Top 100: twice the reading per question for articles dense ranks that low.
- Tuning the instruction or depth on dev: with 8 questions to fix, a tuned value would fit a handful of questions, and those same questions would then judge it (D-037).
- Deciding this one by recall@5 or MRR, where rerankers usually shine: that picks the rule to suit the challenger. S1 reads 10 articles, so recall@10 still decides the default. The recall@5 path keeps the other benefit visible without moving the bar.
- Letting BM25 candidates challenge the reranker on dense: they can't win, as Why shows, so a verdict would add nothing.

**Why.**
- The two models span the real trade-off: reading everything against being small enough to run without a GPU. Public evidence doesn't settle it. I found no independent head-to-head, and the one leaderboard that ranks Qwen first marks its results as self-reported.
- A cross-encoder scores each pair on its own, so an article gets the same score whatever else is on the list. BM25's candidates can therefore only help if a gold article is missing from dense's list. At most 3 can be (`ew-127`, `sim-064` and one of `sim-016`'s three). Even if BM25 rescued all 3 and broke nothing, the interval would be [0, +11], which doesn't clear zero.
- Reordering within the top 10 doesn't change recall@10. Moving a gold article from rank 8 to rank 3 raises recall@5, but the generator already reads all 10. That only pays off if S1 then sends fewer articles, which is the k=5 question.

**Trade-offs.**
- The bar is high and recall@10 barely sees what rerankers do best, so a useful reranker can miss it. Keeping dense would then mean "not shown to be better at k=10", not "no use".
- Qwen may be too slow on a CPU to serve. If so, Step 9 has to choose between a GPU, a hosted reranker and the smaller model.
- MiniLM reads only the start of a third of the articles, and it was trained on short web passages, not help-center articles.
- A gain from BM25 candidates too small to see on 50 questions isn't proof there is none. Leaving them out keeps production to one index on that basis.
- This is one more choice made on the same 50 questions, so dev numbers keep getting more optimistic (D-036). M2's test run is the honest check.

**In short.** "A reranker reads the question and each candidate together, which is more accurate than comparing precomputed vectors but far too slow for the whole knowledge base, so it only gets dense's top 50. I measured a long-context model and a tiny one, under the rule I'd already fixed. I also worked out before running anything that BM25 candidates couldn't clear the bar: the reranker scores each pair on its own, so they could only rescue the 3 gold articles dense might have missed."

**Outcome (2026-11-06): dense stays the default, and the reranker is dropped.** Results are in [reports/retrieval_rerank.md](../reports/retrieval_rerank.md).

| Real dev questions, recall@10 | Difference [95% CI] | Better / worse | Under the rule |
|---|---|---|---|
| MiniLM vs dense (0.78 vs 0.86) | −8 points [−19, +2] | 3 / 7 | Not clearly better |
| Qwen vs dense (0.86 vs 0.86) | −0.3 points [−6, +6] | 2 / 2 | Not clearly better |
| Qwen vs MiniLM | +8 points [−2, +18] | 6 / 2 | Qwen goes forward as the higher score |
| Qwen with BM25 candidates vs Qwen on dense | 0 points | 0 / 0 | Reported only |

- **Dense with voyage-4-nano stays the default.** Qwen went forward and then didn't clear the bar.
- **The recall@5 path doesn't fire.** Qwen's recall@5 is +3 points [−8, +14] against dense (6 better, 4 worse), so under the rule above the reranker is dropped, and the k=5 question will use dense alone.
- **Qwen made the very top of the ranking worse.** On real questions, MRR@20 fell from 0.61 to 0.51 (−10 points [−19, −3]) and recall@1 from 0.40 to 0.25. On synthetic questions, MRR@20 fell from 0.93 to 0.75 (−18 points [−20, −15]) and recall@10 from 0.998 to 0.96 (−3.8 points [−5.6, −2.2]). MiniLM's recall@5 on real questions is clearly lower too (−16 points [−31, −2]).
- **That looked like a bug, so it was checked first. It isn't one.** On 25 synthetic questions where the gold article fell, scoring at batch size 8, at batch size 1 and with the raw code from Qwen's model card gave a correlation of at least 0.9998, with the gold article at the same rank every time. This also supports the assumption above: an article's score doesn't depend on what else is in its batch.
- **What Qwen favors.** Its top article is about half as long as dense's: a median of 1,200 characters against 2,248 on real questions, and 567 against 1,072 on synthetic ones. Real gold articles are long, with a median of 3,302. On synthetic questions, feature requests at rank 1 rise from 32% to 44%; on real questions they stay at 10%. This is what the rankings show, not a proven cause.
- **BM25 candidates rescued nothing.** Recall@10 and recall@5 on real questions are identical with or without them, and they add about 1.6 seconds per question. `sim-064`'s gold article was already in dense's top 50: Qwen ranked it 11th, and 15th with BM25's articles added. The other two stayed out of the top 20. Nothing here supports a keyword index, so production needs only the vector index.
- **Speed per question.** The GPU times include embedding the question.

| | Dense only | MiniLM | Qwen | Qwen with BM25 candidates |
|---|---:|---:|---:|---:|
| GPU, median | 28 ms | 123 ms | 2.7 s | 4.2 s |
| GPU, 95th percentile | 33 ms | 138 ms | 5.3 s | 8.2 s |
| CPU, 8 threads, reranking only | | 1.6 s | about 3 minutes | |

  The CPU timing is rough: 50 articles per question, partly filled at random, on 10 questions for MiniLM and 5 for Qwen (106 to 336 seconds). The slowest Qwen question on the GPU took 117 seconds, probably because a batch of long articles overflowed the GPU's memory; that wasn't investigated.
- **What this doesn't show.** It doesn't show that reranking is bad in general. These are two untuned models reading whole articles, and rerankers are more often used on passages. Qwen also kept its default "web search" instruction. Sections could behave differently, which would be a new question with its own rule.
- **One knock-on change.** The router's cascade (Step 7) planned to use the top reranker score as a signal. With no reranker, it now says the top retrieval score.

---

## D-039 · Sections: three ways to cut an article, scored by the best piece, with the rerankers retested on sections (Step 4)

**Context.** voyage-4-nano on whole articles is the default retriever (D-038). One vector has to stand for everything in an article, while a question is usually about one part of it. The 62 gold articles of the 50 real dev questions are long (median 707 tokens, against 251 for the corpus), and only 3 of them have no headings to cut at. Long ones also rank worse: of the 19 over 1,000 tokens, 3 are at rank 1 and 15 are in the top 10; of the 13 under 300 tokens, 5 are at rank 1 and all 13 are in the top 10. Those counts are small, and I saw them before writing this rule. All 9 gold articles outside the top 10 would be split, so sections can reach all 8 questions with something to fix. A reranker couldn't, because it only reorders a list. D-012 left this measurement for Step 4.

**Decision (written and committed before any section run).**
- **Three candidates, nothing tuned.** Each cuts every article into pieces, and each piece gets its own vector.
  - `h2`: cut at `h2` headings. 21,524 pieces, median 88 tokens.
  - `headings`: cut at every `h2`, `h3` and `h4` heading. 32,363 pieces, median 69 tokens.
  - `window`: 200 words with 40 words of overlap, ignoring headings. 17,027 pieces.
- **Cut positions come from the HTML, not from a text search.** Ignoring whitespace, the HTML's text equals `contents` minus the title in all 6,221 articles, so a heading's position is found by counting non-space characters. Searching `contents` for the heading's text would cut in the wrong place in 925 articles at `h2` level, because "In this article" lists repeat the headings. Heading pieces rejoin to exactly `contents`, so nothing is lost or cleaned (D-012): only the unit changes.
- **What gets embedded.** The article's title, then the piece. The first piece already starts with the title, so it isn't added twice. An article with no headings stays one piece with the same text as before.
- **An article's score is its best piece's score.** The metrics still count articles (D-017).
- **Tiny pieces stay.** Only 331 of the `headings` pieces (1%) are under 10 tokens, and merging them would add a threshold.
- **Between the three,** D-036's rule for new options applies: the higher real-dev recall@10 goes forward, and an exact tie goes to MRR@20.
- **Against the default,** the chosen variant must clear D-036's bar: its paired 95% interval on real-dev recall@10 lies entirely above zero. On these 50 questions that takes about 4 fixed with none broken, or 7 fixed with one broken. With two broken it's out of reach, since even all 8 fixed gives [0, +24].
- **A path for recall@5,** as in D-038. If the chosen variant misses the bar on recall@10 but its recall@5 interval against the default lies entirely above zero, the default stays and sections become a candidate for the k=5 question (D-036), under a rule written before those runs.
- **Titles, reported only.** The chosen variant also runs once without titles. That run can't replace the default.
- **The rerankers get one retest on sections.** Both D-038 models, with their settings unchanged, run after the verdict above, whichever way it goes. Each reads the chosen variant's top 50 articles, with each article represented by its best piece, title included. Between the two, the higher real-dev recall@10 goes forward, and an exact tie goes to MRR@20. It replaces the default of that moment only if it clears D-036's bar against it. The recall@5 path applies here too. If it clears neither, the reranker is dropped.
- **What S1 reads isn't decided here.** S1 keeps receiving whole articles, which are what these metrics measure. Sending only sections would change tokens and answers, not recall@10, so it needs paid answer runs and a rule of its own before the S1 sub-step.

**Alternatives.**
- Testing only one way to cut: `h2` and `headings` differ in a way that can't be settled in advance. 1,004 articles have an `FAQs` heading, and only `headings` gives each question in it its own vector, while `h2` avoids most of the near-copies (see Trade-offs).
- Other window sizes: size and overlap are two more settings, and 8 questions can't tune them (D-037). 200 words is a common starting size. The windows are a control for "did the headings help?", and a fallback for a future knowledge base without headings.
- LLM-written context for every piece (contextual retrieval): one paid call per piece, or 17,000 to 32,000 calls. It's worth considering only if a cheap title doesn't do enough.
- Semantic chunking, which cuts where neighboring sentences' embeddings differ: another threshold to tune, when this corpus already marks its sections.
- A mean or a sum over an article's pieces, instead of the best one: a mean penalizes a long article for its off-topic sections, and a sum rewards length itself.
- Cleaning glued sentences ("...to ADI.If you want...") while cutting: that changes the text and the unit at once, so a gain couldn't be attributed to either (D-012).
- Reranking the top 50 pieces instead of the top 50 articles: several pieces can come from one article, so the list could hold far fewer than 50 articles, and fewer than the 20 the metrics need.

**Why.**
- Headings are boundaries the authors chose, and using them adds no size or overlap to tune.
- The first piece and any article with no headings keep exactly the text they had, so the comparison with whole articles changes one thing.
- A piece like "To cancel your site plan:" doesn't say which product it's about, and identical FAQ sections appear in many articles. The title is the cheap way to give a piece that context. Running without it once shows what it adds.
- The retest is a fair new question. Qwen's pull toward short text should matter less when candidates are closer in length, though that's a hypothesis. MiniLM's 512-token cut would affect about 184 `headings` pieces or 653 `h2` pieces, against a third of the articles, and it's the only one of the two fast enough for a CPU.

**Trade-offs.**
- **Short articles can get crowded out.** A one-piece article keeps its score, but other articles now compete with their best piece, which tends to score higher than their whole-article vector. 2,418 articles (39%) stay in one piece at `headings`, including 1,998 of the 2,049 feature requests. Only 3 of the 62 real dev gold articles do, so real questions may gain from this while synthetic ones, a third of which target feature requests, may lose. That would describe this corpus's real traffic, not a general law.
- **Near-copies can flood the results.** At `headings`, 2,597 pieces share their exact text with another piece, and "How do I get my payouts?" appears in 129 articles. Titles make them differ only slightly, so a question on that topic could fill the top 20 with them. At `h2` it's 274 pieces, and the largest group is 17.
- **More vectors.** Up to 32,363 rows instead of 6,221, which affects memory and the pgvector sub-step, and an exact search takes longer.
- **Recall@10 can't see the token savings.** If sections only help the generator read less, this rule won't show it. That's the later S1 question.
- **More chances for luck.** Six options have been tried against the default on these 50 questions, and this adds up to five more, designed after seeing dense's ranks. Each bar is still strict, but only M2's test run can confirm a win.

**In short.** "A whole article gets one vector, but a question is usually about one part of it. So I cut articles at their headings, with fixed-size windows as a control, and scored each article by its best piece. The cuts come from the HTML and rejoin to the exact text, so only the unit changes. I fixed the decision rule first, gave the rerankers one retest on sections because their weaknesses on whole articles might not apply there, and kept 'what does the generator read' as a separate, paid question."

**Outcome (2026-11-06): dense on whole articles stays the default, sections narrowly miss the bar, and the reranker is dropped.** Results are in [reports/retrieval_sections.md](../reports/retrieval_sections.md) and [reports/retrieval_rerank_sections.md](../reports/retrieval_rerank_sections.md).

| Real dev questions, against whole articles (recall@10 0.86) | Recall@10 | Difference [95% CI] | Better / worse | MRR@20 |
|---|---:|---|---|---:|
| `h2` (goes forward on the tie-break) | 0.913 | +5.3 points [0, +12] | 3 / 0 | 0.657 |
| `headings` | 0.913 | +5.3 points [0, +12] | 3 / 0 | 0.651 |
| `window` | 0.913 | +5.3 points [−2, +14] | 4 / 1 | 0.618 |
| MiniLM on `h2`'s best sections | 0.740 | −12 points [−25, +1] | 4 / 11 | 0.486 |
| Qwen on `h2`'s best sections (goes forward over MiniLM) | 0.807 | −5.3 points [−17, +6] | 4 / 8 | 0.586 |

- **An exact three-way tie.** All three variants score 137/3 out of 50, by fixing different questions, so MRR@20 decided and `h2` went forward. Five of the 8 questions were fixed by at least one variant; `ew-127`, `sim-016` and `sim-064` by none.
- **`h2` misses by one question, and the rule stands.** Its interval's lower bound is exactly zero, so it doesn't lie entirely above zero. A bootstrap resample of 50 questions misses all 3 fixed questions (47/50)^50 ≈ 4.5% of the time, more than the 2.5% tail, so the lower bound stays at 0. With 4 fixed, that chance is about 1.5%. This means "not shown to be better on 50 questions", not "no use". The evidence is also thinner than 3 suggests: `sim-021` and `sim-027` share one gold article.
- **The recall@5 path doesn't fire.** `h2`'s recall@5 is −5 points [−14, +3] (3 better, 6 worse), so the k=5 question stays on whole articles. Sections pulled far-off gold articles into the top 10 while pushing others out of the top 5. MRR@20 is +4.7 points [−1, +11], within noise.
- **Titles lean helpful, but not clearly.** Without titles, `h2`'s recall@10 is 0.893 (−2 points [−6, 0] against `h2` with titles) and its MRR@20 is 0.600 (−5.8 points [−12, +0.4], 6 better, 13 worse).
- **Crowding out happened as predicted.** Gold articles left in one piece kept their scores to the 4 decimals recorded, in all 215 cases. Of the 211 on synthetic questions, 200 kept their rank, 11 fell and none rose. One real gold feature request stayed at 0.5105 and fell from rank 2 to rank 6. Feature requests' share of the top 10 fell from 17% to 8% on real questions and from 23% to 16% on synthetic ones. Top-10 articles also got longer on real questions, with a median of 2,805 characters against 1,614, so searching by section and then sending whole articles would raise S1's cost.
- **Near-copies didn't flood the results on dev.** Articles with a piece shared by 20 or more articles hold 1.2 of the top 20 on average with whole articles and 1.5 at `headings`. One question has 10 or more either way, and the most for any question rises from 10 to 14. This is a proxy, because run records don't keep which piece won.
- **The reranker retest fails too, so the reranker is dropped.** Qwen went forward (+7 points [−5, +19] over MiniLM), then cleared neither the bar nor the recall@5 path (−9 points [−23, +5]). Reranking also lowered `h2`'s own recall@10: MiniLM by 17 points [−29, −7], Qwen by 11 points [−22, 0].
- **The hypotheses held only in part.** Against Qwen on whole articles, real recall@1 rose from 0.25 to 0.39 and MRR@20 from 0.51 to 0.59, but recall@10 fell from 0.857 to 0.807. On synthetic questions MRR@20 barely moved (0.75 to 0.77), so length probably wasn't the main cause there. Lifting MiniLM's cut didn't help it either (0.777 to 0.740). One untested possibility: dense picks the one piece a reranker reads, so it can't help when that piece isn't the one that answers.
- **Speed per question** (GPU median, including embedding the question). Embedding the corpus took 201 to 220 seconds per variant, against 246 for whole articles. CPU speed wasn't measured, because the reranker was dropped.

| Dense, whole articles | `h2` | `headings` | `window` | MiniLM on `h2` | Qwen on `h2` |
|---:|---:|---:|---:|---:|---:|
| 28 ms | 34 ms | 36 ms | 33 ms | 115 ms | 829 ms |

- **Where these numbers come from.** The tables and intervals are in the two reports. The fixed-question lists, the crowding-out counts and the near-copy proxy came from one-off scripts over the run records, not from an `srag` command.
- **What this leaves open.** The pgvector sub-step indexes 6,221 article vectors. `h2` broke nothing at recall@10 and could send the generator far less text, so "S1 reads sections" is still worth a paid dev run, under a rule written before it.

---

## D-040 · Feature-request down-weighting: not built, because even its best case can't clear the bar (Step 4)

**Context.** A third of the knowledge base (2,049 articles) is feature requests, and only 10 of the 400 real questions have one as a gold article, 2 of them among the 50 real dev questions. With dense on whole articles, the default (D-039), feature requests fill 1.7 of the top 10 on real dev questions on average, and 35 of the 50 questions have at least one there. Down-weighting them is the last retrieval change in D-036's order.

**Decision.** Don't build it. Before writing a rule, I worked out the best case from the default's dev run (`20261105-161402_dense-voyage-4-nano_dev`). It can't clear D-036's bar, and it can't trigger D-038's recall@5 path either:

| Real dev questions, best case | Change [95% CI] | Fixed / broken |
|---|---|---|
| Recall@10 (now 0.86) | +2.7 points [0, +7] | 2 / 0 |
| Recall@5 (now 0.78) | +5 points [0, +11] | 3 / 0 |

- **How the best case is worked out.** Pushing feature requests down never reorders the other articles. So a regular gold article's best possible rank is 1 plus the regular articles above it, whatever the method or the penalty. The best case also assumes a scheme gentle enough to keep both gold feature requests where they are. Every real top 20 holds at least 11 regular articles, so nothing beyond rank 20 can reach the top 10. This was a one-off calculation over the run record, not an `srag` command.
- **The real worry moves to the S1 dev run.** That run will report how many wrong answers cite a feature request that isn't gold, since feature requests in the context might mislead the generator. If that's common, down-weighting comes back as an answer-level question, under a rule of its own. The check costs nothing extra.

**Alternatives.**
- A soft penalty, run anyway and reported only: free, but it adds code and a penalty value that 50 questions can't tune, for a result that can't change the default.
- Removing feature requests from the index: recall@10 changes by −0.3 points [−6, +6], with 2 fixed and 2 broken, because `ew-143` and `sim-194` have a feature request as gold. It would also drop the "not possible yet" answers that 10 of the 400 real questions rely on.
- A cap (say, at most 2 in the top 10) or a reserved slot: both only move feature requests, so the same best case limits them.
- A paid S1 run without feature requests now: about $0.40, and 50 questions are unlikely to show a small effect, which was M1's lesson (D-034).

**Why.**
- The arithmetic settles it without a run, as it did for BM25 candidates in D-038, because no run can beat its own best case.
- With 2 fixed questions, a bootstrap redraw of 50 misses both about 13% of the time, and with 3 fixed about 4.5%. Both are above the 2.5% tail, which is why neither interval starts above zero.
- It isn't free. Feature requests are short (median 350 characters, against 2,464 for regular articles), so replacing them with regular articles adds 21% more text to the top 10 on real questions, and S1 would cost more.

**Trade-offs.**
- A small real effect could exist and stay invisible on 50 questions. The best case for MRR@20 is clearly positive (+4.4 points [+1, +8]), but MRR doesn't decide (D-036).
- Removing feature requests outright would miss all 159 synthetic questions that target one. Synthetic questions don't vote, but production traffic could hold more "is this possible?" questions than WixQA's real sets.
- `h2` sections halved feature requests' share of the top 10 without targeting them (D-039), but sections aren't the default.

**In short.** "Before building feature-request down-weighting, I worked out its best case. Pushing feature requests down can't reorder the other articles, so a gold article's best rank is fixed by the regular articles above it. On dev that's 2 questions at most, which can't clear my bar, so I didn't build it. The real worry, feature requests misleading the generator, became a free check in the next answer run."

---

## D-041 · Postgres with pgvector for the vectors: exact search unless it's too slow, with HNSW measured (Step 4)

**Context.** Dense retrieval with voyage-4-nano on whole articles is the default (D-040). It runs in NumPy inside the evaluating process: 6,221 vectors of 1,024 numbers (25 MB), searched exactly in a few milliseconds. The service (Step 8) needs the vectors in a store it can query and update, next to the articles' text. The roadmap planned Postgres with pgvector. The keyword engine on the same roadmap row isn't needed (D-038).

**Decision (written and committed before any Postgres run).**
- **Store.** Postgres 17 with pgvector, in Docker from the pinned image `pgvector/pgvector:0.8.6-pg17`. The port is open only on this machine, and the password lives in `.env`. 0.8.7 came out a week ago; 0.8.6 has been out since August.
- **Schema.** An `articles` table with the text, the type and `position`, each article's row in the pinned file. A narrow `article_vectors` table holds 32-bit `vector(1024)` values kept in the row (`STORAGE PLAIN`), so a scan doesn't fetch each vector from a side table; Postgres otherwise moves values over about 2 KB out of the row. A `corpus` row records the model, its revision and the corpus fingerprint, and the retriever refuses to search vectors from another model or corpus.
- **Loading.** `srag db-load` copies the stored vectors (D-035), so it never runs the embedding model.
- **Distance.** Inner product (`<#>`). The vectors have length 1, so it ranks the same as cosine. Exact search sorts by score and then by `position`, so ties come out in NumPy's order.
- **Driver.** psycopg 3 with the pgvector package, and no ORM (D-003).
- **The plumbing check comes first.** Exact search in Postgres must return NumPy's top 20 for every dev question, in the same order, except where two scores differ only by rounding (by less than 1e-5). Any other difference is a bug to fix, not a result.
- **Exact stays unless it's too slow.** Exact search remains the method unless its 95th-percentile search time, the SQL alone, exceeds 50 ms over the 550 dev questions. An S1 answer takes about 4 seconds, so 50 ms is about 1% of that.
- **HNSW is measured either way.** With pgvector's defaults (16 links per vector, 64 candidates while building, 40 while searching), the report gives its overlap with exact search's top 10 and top 20, its recall@10 under D-036's comparison, its build time, index size and speed. If it's ever needed, it must keep a mean top-10 overlap of at least 99% with exact search, or its search candidates go up.
- **A projection.** Exact search time grows in line with the number of rows, so the report states the corpus size at which it would reach 50 ms.
- **Which code path evaluation uses (D-008).** The Postgres retriever is what the service and M2 use. NumPy stays as the reference for dev experiments, and the plumbing check shows the two agree.

**Alternatives.**
- A dedicated vector database (Qdrant, Pinecone, Weaviate): built for far larger scale, with compression and sharding, but it's a second system to run and keep in sync with the text. Pinecone is hosted, which adds an account and a network call to every question.
- Keeping NumPy in memory: enough for search at 6,221 vectors, but there are no updates without a restart, and every server process loads its own copy.
- IVFFlat instead of HNSW: it has to be trained on the data and needs a list count chosen. HNSW is pgvector's usual recommendation for the trade between recall and speed.
- Half-precision vectors (`halfvec`): half the storage, but they'd no longer match NumPy's exactly, and size isn't a problem here.
- asyncpg, which is async only, or SQLAlchemy, an ORM this doesn't need. psycopg has both a normal and an async interface, so FastAPI in Step 8 can use the same driver.
- Switching every evaluation to Postgres now: every dev run would then need Docker running.

**Why.**
- The case for Postgres is the service around search: one database for text, vectors, metadata and later run results, filters in plain SQL, transactions when articles change, and a database I already know how to run.
- Exact search is simpler and gives the same answer every time. An approximate index only pays off when exact search is too slow, and the bar says what "too slow" means before anything is measured.
- Comparing HNSW with exact search, not only with the gold articles, isolates the approximation. A missed article that isn't gold doesn't move recall, but it's still an error.

**Trade-offs.**
- Two retrievers compute the same thing, NumPy for dev and Postgres for the service, so they could drift apart. The plumbing check guards that now, and M2 runs on Postgres.
- Database tests are skipped unless a test database is set, so `uv run pytest` alone doesn't cover the SQL. Step 10's CI can run them against a database container.
- Keeping vectors in the row only works while a row fits in Postgres's 8 KB page. At 1,024 dimensions a vector takes about 4 KB, so this is fine here.
- The timings come from one machine with a warm cache. They're for comparing options, not a promise about the server.

**In short.** "I moved the vectors into Postgres with pgvector, because the service needs text, vectors and metadata in one database I already know how to run. Before trusting it, I checked that Postgres returns exactly what my NumPy search did. I set a latency bar before measuring: exact search stays unless it's slower than 50 ms, and an HNSW index is measured against exact search either way, so I know what the approximation costs."

**Outcome (2026-11-08): exact search in Postgres stays, and HNSW isn't needed at this size.** Results are in [reports/retrieval_pgvector.md](../reports/retrieval_pgvector.md).

| 550 dev questions | Search alone, median | Search alone, 95th percentile | Total per question, median | Top-10 overlap with exact | Same top 20, same order |
|---|---:|---:|---:|---:|---:|
| NumPy, exact | not recorded | not recorded | 28.0 ms | reference | reference |
| Postgres, exact | 7.4 ms | 8.2 ms | 34.1 ms | 1.000 | 550 of 550 |
| Postgres, HNSW | 2.3 ms | 2.6 ms | 28.5 ms | 0.994 | 490 of 550 |

- **The plumbing check passes.** Exact search in Postgres returned NumPy's top 20, in the same order, on all 550 dev questions, so every metric is identical.
- **Exact stays.** Its 95th-percentile search time is 8.2 ms, well under the 50 ms bar, and its plan is the full scan it should be (`Limit > Sort > Seq Scan`).
- **HNSW meets its 99% requirement, but it isn't needed.** Its top-10 overlap with exact search is 0.994 and its top-20 overlap is 0.991. Real recall@10 and MRR@20 are identical. It saves about 5 ms of a 4-second answer, took 1.0 second to build, and its index is 49 MB, as large as the table.
- **What an approximate miss looks like.** On `syn-4552`, the gold article is at rank 1 in exact search and absent from HNSW's top 20, so that question's recall@10 falls from 1 to 0. A 0.994 average hides a complete miss like that one. On real questions, the top-10 set differed on 1 of 50, and no metric moved.
- **A projection.** Exact search time grows in line with the number of rows, so it would reach 50 ms at about 38,000 articles (6,221 × 50 / 8.2). That assumes the table stays in memory. At this row size, Postgres's default 128 MB buffer cache holds about 16,000 articles, and beyond that it depends on the operating system's cache, so the real limit may be lower.
- **The table is twice the size of the vectors.** A 1,024-dimension vector takes 4,108 bytes, just over half of an 8 KB page, so each page holds one row and the table is 49 MB. pgvector's default for this type does keep long values out of the row (`typstorage` is `e`), but I didn't measure search with that default.
- **Two setup facts.** On this machine `localhost` tries IPv6 first, and nothing answers there because the port is open on IPv4 only, so each connection waited for its timeout. The connection strings use `127.0.0.1` instead. The image digest is `sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f`.
- **What these timings are.** Docker Desktop on Windows, one connection, a warm cache and 6,221 rows. They compare options on this machine and aren't a promise about the server.
- **What's still to do.** S1 still reads BM25's results (D-033). Moving it to dense search on Postgres is the next sub-step, and M2 will use that path.

---

## D-042 · S1 on dense search: what it buys, and two ways to read less that must show they aren't worse (Step 4)

**Context.** S1 still reads BM25's top 10 (D-033), while D-036 to D-041 made dense search with voyage-4-nano, on Postgres, the default retriever. Those choices used recall@10, so nothing yet says what dense search does to answers. This run also has to settle two questions left open earlier: whether the better retriever lets S1 read fewer articles (D-036), and whether S1 can read only sections (D-039). D-040 moved the worry about feature requests misleading the generator here too. On the 50 real dev questions:

| What S1 reads | Prompt text (tiktoken, mean) | Questions with a gold article in it |
|---|---:|---:|
| BM25, top 10 whole articles (today) | 8,908 | 36 |
| Dense, top 10 whole articles | 5,677 | 44 |
| Dense, top 5 whole articles | 2,865 | 41 |
| `h2` search, the best section of each of the top 10 articles | about 1,500 to 3,200 | 46 |

The last row's range assumes each best section is of average size at one end and the article's largest at the other, because run records don't keep which section won. "Gold in it" means the article was found there; its best section may not be the part that answers.

**Decision (written and committed before any dense S1 run).**
- **Three new runs, nothing tuned,** on the 50 real dev questions, at effort `medium` with prompt `answer-v1`, graded by judge-v1 at `high` through the batch API (D-031). The test set waits for M2.
  - Dense k=10, the new default: voyage-4-nano, exact search in Postgres, the top 10 whole articles.
  - k=5: the same search, the top 5 whole articles.
  - Sections: search `h2` sections (in NumPy, as D-041 allows for dev experiments), then send the top 10 articles, each as its title and best section.
  - The reference is the judged BM25 k=10 run from D-033 (`20261103-004215_judge-high_20261103-003456_s1-k10-medium_dev_real`).
- **S1 moves to dense search.** D-036 chose the retriever by recall@10, and the design has S1 use the best retriever. This run measures what that buys in answers, without re-deciding it on 50 questions. It's reversed only if dense's paired 95% interval against BM25 lies entirely below zero, and then I look for a bug first.
- **k=5 and sections have to show they aren't worse.** Each is compared with dense k=10, on % correct, question by question. A candidate replaces the default only if both hold:
  - its % correct is at least the default's, and
  - its paired 95% interval lies entirely above −10 points.
- **Anything that isn't cheaper keeps D-033's bar.** A candidate with no fewer mean input tokens than the default replaces it only if its interval lies entirely above zero.
- **Between two that pass,** the higher % correct goes forward, and an exact tie goes to fewer mean input tokens. If neither passes, dense k=10 stays.
- **The judge reads what the generator read.** A section run saves the sections it sent, and the judge grades groundedness against those, not the whole cited article. Correctness compares the answer with the reference either way. Runs with whole articles send the judge exactly what they sent before, so their cached verdicts stay valid.
- **The feature-request count, from D-040.** In the default run, count the not-correct answers (partial or incorrect) that cite at least one feature request that isn't gold. "Common" means 4 or more. Fewer can't clear D-036's bar even if every one of them were fixed, so below 4 feature requests stay as they are. At 4 or more, down-weighting comes back as an answer-level question with its own rule.
- **If sections win,** Postgres gets a table of section vectors before M2, under D-041's plumbing check. If they don't, nothing changes there.
- **`srag report-tuning`** writes all four arms to one report, with dense k=10 as the default, so BM25's row reads as "BM25 minus dense". BM25 isn't cheaper, so D-033's bar on that row is exactly the reversal test above.
- **Cost.** About $0.90 at most: roughly $0.14 for the answers and up to $0.75 for three judge runs.

**Alternatives.**
- D-033's bar (interval entirely above zero) for k=5 and sections: it asks whether they're better, and neither is expected to be. It would almost never fire, and "not shown better" would say nothing about whether they're safe to use.
- Accepting a candidate that merely scores the same: a tie can hide real losses. 2 fixed and 2 broken, and 6 fixed and 6 broken, are both a difference of zero, and only the first says the two arms mostly agree.
- A margin of −5: on 50 questions one broken question with nothing fixed already gives [−6, 0], so −5 would be nearly the same as D-033's bar.
- A margin of −15: passes candidates that flip 6 answers each way, and a 15-point loss is far too large to call "not worse".
- Searching whole articles, then picking the best section inside each: a two-stage path nobody has measured, and it gives up `h2`'s better retrieval.
- The top N pieces whatever article they came from: fewer distinct articles and an unmeasured recall (D-039 rejected the same idea for reranking).
- The best section plus its neighbors, or two sections per article: another setting that 50 questions can't tune (D-037).
- `h2` search with whole articles: it missed its bar (D-039) and costs more than whole-article search, so it has no case to answer.
- The 500 synthetic dev questions, for more power: the judge isn't validated on them, and they're easier (D-033).

**Why.**
- For an option that saves money, the useful question is whether it's worse, which is the reverse of the question D-033 asks. A non-inferiority margin turns "no sign of a difference" into a claim with a size.
- The margin says what 50 questions can rule out. It isn't a statement that losing 10 points is acceptable. At the sizes that decide this, 2 fixed and 2 broken gives [−8, +8] and passes, and 3 fixed and 1 broken gives [−4, +12] and passes, while 0 fixed and 2 broken gives [−10, 0] and 1 fixed and 3 broken gives [−12, +4], and neither does.
- It can pass. Changing only the effort flipped 3 to 4 of S1's 50 answers in D-033 (3 and 0 for `high`, 3 and 1 for `low`), though changing the context will probably flip more.
- The score clause reflects the prices. k=5 saves about $0.0005 per answer and about half a second. If a wrong answer costs one support ticket, even a small loss in correctness outweighs that, so a candidate that measures lower shouldn't replace the default however narrow its interval is.

**Trade-offs.**
- The margin is wide. Passing means "not shown to lose more than 10 points on 50 questions". Only a larger sample can narrow it, and the test set can't be used to choose.
- A sections win would change two things at once, the search unit and what gets read. That's accepted because reading the best section is what section search makes possible, and each change was already measured for recall (D-039).
- These are the same 50 questions used for every Step 4 choice, and dev has little room to show a gain from dense search: BM25 at k=10 scored 76% and S* 78% (D-033). M2 is the honest check (D-034).
- Each arm is one sample from a nondeterministic generator, and the intervals don't cover that (D-033).

**In short.** "S1 moved to dense search because the retriever was already chosen on recall, and the answer run measured what that bought. Reading fewer articles or only the best sections saves tokens, so I asked a different question for those: are they worse? I set a non-inferiority rule before running them. The option has to score at least as well as the default, and its paired interval has to stay above minus 10 points, which is about the tightest loss 50 questions can rule out."

**Outcome (2026-11-08): S1 moves to dense search at k=10 on whole articles, and neither cheaper option passes.** Results are in [reports/tuning_s1_reading.md](../reports/tuning_s1_reading.md).

| 50 real dev questions | % correct [95% CI] | Against dense k=10 [95% CI] | Only it right / only dense k=10 right | Input tokens | Cost per answer | Latency p50 |
|---|---:|---|---:|---:|---:|---:|
| **Dense, k=10 (the default)** | 90.0% [80, 98] | | | 8,758 | $0.00122 | 3.5 s |
| BM25, k=10 | 76.0% [64, 88] | −14 points [−26, −2] | 2 / 9 | 13,572 | $0.00169 | 3.6 s |
| Dense, k=5 | 76.0% [64, 88] | −14 points [−24, −6] | 0 / 7 | 4,507 | $0.00077 | 3.0 s |
| `h2` search, best section | 64.0% [50, 76] | −26 points [−40, −14] | 1 / 14 | 2,858 | $0.00060 | 3.0 s |

- **S1 moves to dense search.** The reversal test doesn't come close: dense is 14 points above BM25, with 35% fewer input tokens and 28% lower cost. Seven of the 9 questions only dense got right had no gold article in BM25's context. The paired interval is entirely above zero, while McNemar's p is 0.065, so the evidence is good but not overwhelming, and the move didn't depend on it.
- **k=5 fails both clauses.** It lost 7 questions and won none. On 5 of the 7, no gold article was in the top 5. It saves about $0.00045 per answer and half a second. It does match BM25 at k=10 (76% each, +0 [−14, +14]) on a third of the tokens, which is what the better retriever buys if tokens ever matter more than correctness.
- **Sections fail clearly.** They lost 14 questions and won 1. On 12 of the 14, every gold article was in the context, and 12 of the 14 were graded partial. Search found the right article, but its best-scoring section didn't hold the whole answer. Finding by section and reading by section are separate questions: D-039 showed sections can help find articles, and this shows reading only one piece loses content. Groundedness against what the model read is also lower, 42 yes and 6 partly, against 48 and 2. Postgres needs no table of section vectors.
- **Feature requests aren't common.** Of the default run's 5 not-correct answers, 2 cite a feature request that isn't gold, under the 4 this rule requires, so down-weighting stays unbuilt (D-040). Another 6 correct answers cite one.
- **A surprise: dense k=10 scores above S\*.** It beats S\*'s gold-only context by 12 points [+4, +22], 6 questions to 0. On those 6, S\* was partial 5 times and incorrect once.
  - All 6 questions with no gold article in dense's top 10 were answered correctly, and groundedness is 48 of 50, so the answers come from the retrieved articles rather than from memory.
  - So gold lists aren't the only route to a correct answer, and gold alone can be less complete than ten related articles. S\* is a useful diagnostic, not a strict ceiling.
  - I checked it isn't a bug: these were new paid calls, the judge doesn't see the strategy, and no question was right only for S\*.
- **Don't quote 90%.** These 50 questions chose k=10 and the retriever. BM25's dev score of 76% fell to 64% on the test set (D-034). M2 is the honest check, and it runs dense k=10 through Postgres.
- **More context may help further.** Dense k=10's lead over k=5 suggests it. Going past 10 is S3's question (Step 6).
- **Where these numbers come from.** The tables and intervals are in the report. The breakdowns of the lost questions and the comparison with S\* came from one-off scripts over the run records. The runs are `20261108-121330_s1-dense-k10-medium_dev_real`, `20261108-121653_s1-dense-k5-medium_dev_real` and `20261108-122004_s1-dense-h2-k10-medium_dev_real`, all from commit `6fc8110` with a clean tree.
- **Cost.** $0.74: $0.13 for the answers, including a two-question pilot, and $0.61 for judging.

---

## D-043 · S2: a hand-written loop with two tools, two limits and prompt caching (Step 5)

**Context.** S2 lets the model decide what to search and read before it answers. The design listed three tools (`search`, `read`, `answer`), about 6 steps and a token budget. Step 4 left S1 on dense search at k=10, with 90% on the 50 real dev questions (D-042). S2 should help most where one search isn't enough: 79 of the 400 real questions need more than one article. The generator stays Claude Haiku 5.5 at effort `medium`, so the only new variable is the model steering the search.

**Decision (written and committed before any S2 run).**
- **A hand-written loop** that calls `LLMClient` for every model turn, so each turn is cached and counted (D-003, D-020).
- **Two tools.**
  - `search(query)` runs the same search as S1: voyage-4-nano, exact search in Postgres. It returns the top 10, each as a number and the article's first 50 words, which start with its title. That's about 60 tokens per result.
  - `read(number)` returns the full article.
  - An article gets a number the first time it appears and keeps it for the rest of the question (D-023).
- **The loop ends when the model writes text,** in D-023's format: inline `[n]` citations or exactly `NOT_FOUND`. There is no `answer` tool, which changes the design.
- **Two limits.** A step is one model turn that asks for tools, however many tools it asks for. The limits are 6 steps, and a prompt of 30,000 tokens, a little over 3 times what S1 sends. When either is reached, the model gets the last tool results, a note telling it to answer from what it has read, and one more call with `tool_choice: none`. So a question makes at most 7 model calls.
- **Other stops.** A bad tool call (an unknown number, an empty query) goes back to the model as an error result, and the loop continues. A turn that ends in `max_tokens` or `refusal` ends the question, its tools aren't run, and the run records the stop reason.
- **Prompt caching** through the API's top-level `cache_control`, which moves the cache point to the end of each request. The 5-minute lifetime is enough, since steps take seconds. The report also gives the cost without caching, calculated from the same usage numbers, so no second run is needed.
- **The judge reads what the generator read (D-042).** For a cited article the judge gets the full text if the model called `read` on it, and otherwise only the snippet the model saw.
- **The prompt is `agent-v1`.** It uses D-023's citation and decline rules word for word. S0, S1 and S\* don't change.
- **The runs.** A 5-question pilot, whose traces get read before going further, then the 50 real dev questions in real time, graded by judge-v1 at `high` through the batch API (D-031). Each step depends on the one before, so the loop can't use the batch API.
- **The limits are tuned only if they bind.** They stay if 5 or fewer of the 50 questions hit one. If more do, one extra run doubles the limit that fired, and it replaces the default only if its paired 95% interval lies entirely above zero (D-033).
- **What dev can say.** S1 is at 90%, so S2 can win at most 5 questions, and it would need 4 fixed and 0 broken to clear D-036's bar. Dev shows whether the loop works and what it costs. S2 against S1 is decided on the test set at M2 (D-034).
- **Cost.** About $0.50: $0.10 to $0.20 for the answers, about $0.25 for judging, and a few cents for the pilot.

**Alternatives.**
- The SDK's tool runner: less code, but it's a beta helper that would call the API outside `LLMClient`, so turns wouldn't be cached or counted the same way. Writing the loop is also the point of this step.
- An `answer` tool: it gives the loop a definite end and a checked structure, but a tool's input is JSON, the format D-023 rejected because it streams badly to a chat UI. S2 would also differ from every other strategy.
- `search` returning titles only: too little to choose from, so more wasted reads.
- `search` returning full articles: close to S1, and every result is re-sent on every later step.
- `search` returning the best-matching passage: a better preview, but it needs a table of section vectors that D-042 decided not to build.
- A second, keyword search tool: BM25 added nothing on real questions (D-037, D-038).
- Running one search before the model starts: guarantees S2 sees what S1 sees and saves a step, but it takes the first decision away from the model, which is what's being measured.
- One tool call per step (`disable_parallel_tool_use`): easier to count, but more round trips, more latency and more re-sent history.
- A step cap alone: doesn't bound cost, since one step can read several long articles and the longest is about 10,600 tokens. A token budget alone: doesn't bound latency, and a model repeating short searches would run for a long time.
- Declining at a limit: simplest, but it throws away everything the model has read.
- Keeping tools on at the last call and asking the model to stop: keeps the message cache, but nothing guarantees it stops. `tool_choice: none` costs one cache miss on the messages, at most about $0.003.
- Explicit cache breakpoints: more bookkeeping for the same result. The 1-hour lifetime: writes cost twice the input price instead of 1.25 times, for steps that take seconds.

**Why.**
- The model only needs enough from a search result to decide what to read. Ten 60-token results cost about as much as one article, which keeps early steps cheap.
- A guardrail should bound the worst case and rarely fire. The worst case here is about a cent per question: 7 calls with a 30,000-token prompt, mostly read from the cache. Checking how often a limit fires is more informative on 50 questions than tuning it would be.
- Haiku's cache writes cost $0.125 per million tokens and reads $0.01, against $0.10 uncached, so anything read back once has paid for itself. With prompts of 1K, 2K, 4K, 5K and 7K tokens, the input bill goes from $0.0019 to $0.0010. It's about half and not 90%, because every new token is still written once and these loops are short.

**Trade-offs.**
- 6 steps and 30,000 tokens are reasoned guesses, not measured values. The dev run shows whether they bind.
- Nothing enforces the text format at the end of the loop. `parse_answer` counts bad citations, as it does for every strategy.
- A snippet can mislead: 439 articles open with an "Important" note rather than with what they're about.
- S2's latency includes its tools' time, while S1's doesn't include its one search. That difference is tens of milliseconds against seconds.
- Every Step 4 choice used these same 50 questions, and S1's 90% is flattered by that (D-042), which makes S2's gap look smaller than it may be.

**In short.** "S2 is a hand-written loop with two tools. Search returns numbered 50-word previews and read returns the full article, and the loop ends when the model writes its answer in the same format as every other strategy. I dropped the answer tool the design had, because tool input is JSON and streams badly. Two limits, 6 steps and 30,000 prompt tokens, bound the worst case at about a cent per question, and hitting one forces a final answer from what was read instead of throwing it away. Prompt caching halves the input cost, not 90%, because every token is still written once and the loops are short."

**Outcome (2026-11-13): the loop works, no limit binds, and S2 trails S1 by 6 points on dev, within noise.** Results are in [reports/answers_s2_dev.md](../reports/answers_s2_dev.md).

| 50 real dev questions | % correct [95% CI] | Against S1 [95% CI] | Only it right / only S1 right | Input tokens | Cost per answer | Latency p50 |
|---|---:|---|---:|---:|---:|---:|
| **S1, dense k=10 (the default)** | 90.0% [80, 98] | | | 8,758 | $0.00122 | 3.5 s |
| S2, agentic search | 84.0% [74, 94] | −6 points [−16, +2] | 1 / 4 | 8,972 | $0.00098 | 4.7 s |

- **The loop works.** All 50 questions ended with the model's own answer. No limit fired, no tool call failed, no citation was invalid, and one answer declined. In the 5-question pilot, the API accepted every re-sent turn, and the cache was read from the second call on.
- **The limits stay.** None of the 50 questions hit one, against D-043's threshold of more than 5, so 6 steps and 30,000 tokens aren't tuned. The loops are short: 43 questions took 2 steps (search, then read), 6 took 3 and 1 took 4, with 1.7 searches and 2.0 reads per question. The largest prompt averaged 4,889 tokens.
- **No verdict against S1, as planned.** The difference is −6 [−16, +2], McNemar p 0.375, and S1's 90% was tuned on these same questions (D-042). D-043 left this comparison to the test set at M2.
- **S2 searched better but read less, and neither changed a grade.** Its searches surfaced every gold article in 45 questions, against 42 for S1's top 10: 3 that S1 missed, and none that S1 had and S2 didn't. But it read every gold article in only 37. On the 9 questions involved (3 found, 6 seen but not fully read), both strategies were graded correct.
- **Why it lost 4.**
  - Two come down to how the answer was written. S2 read the gold article and was graded partial, once for leaving out a note about name changes and once for giving a different fix from the reference's.
  - The other two had no gold article in either strategy's context. S1 answered correctly from its 10 full articles. S2 read 2 and 4 articles, then answered for a different Wix product (`sim-064`), or said "yes, in some cases" where the answer is no (`sim-184`).
  - Its one win, `sim-113`, had the gold article in both contexts.
- **Caching is what makes S2 cheaper than S1.**
  - S2 sends about as many input tokens over its 3 calls as S1 does in one, but 45% of them are cache reads at a tenth of the price. So it costs 20% less per answer. Without caching it would have cost $0.00122, exactly S1's figure.
  - Caching cut the input cost by 27%, not the half the example above suggested. With 2 steps, most tokens arrive in the last tool result, so they're written once at 1.25 times the price and never read back.
- **Latency.**
  - S2 takes 1.2 s more than S1 at the median, for 3 model calls instead of 1. Its tools took 130 ms per question at the median.
  - The first uncached question of each S2 run took 27 to 30 s, nearly all of it in model calls, while S1's first questions took 3 to 5 s. The cause isn't known yet. It affects the maximum, not the median, and goes to Step 8, where latency is measured.
- **Where these numbers come from.** The tables are in the report. The search-and-read breakdown and the lost questions came from one-off scripts over the run records. The runs are `20261108-221507_s2-medium_dev_real` (the 5-question pilot) and `20261113-002731_s2-medium_dev_real`, both from commit `cd3117f` with a clean tree. The judge run is `20261113-003350_judge-high_20261113-002731_s2-medium_dev_real`.
- **Cost.** $0.26 against the $0.50 estimate: $0.05 for the answers, including the pilot, and $0.21 for judging.

---

## D-044 · S3: S1 with more articles, and a token cap below the 100K price line (Step 6)

**Context.**
- **What S3 is.** S3 tests the main alternative to precise retrieval: give the model a lot and let it find the answer. The help center is about 2.9M tokens by tiktoken's count and about 4.5M by Claude's, beyond Haiku 5.5's 1M-token window. So long context here means many retrieved articles in one prompt, and the design set K ∈ {20, 50, 100} (section 5).
- **The price line.** Haiku 5.5 bills a prompt over 100,000 tokens at the higher rate for the whole request: input goes from $0.10 to $0.50 per million tokens, and output from $0.50 to $2.50.
- **Claude's tokenizer counts more than tiktoken.** On 100 dev prompts (S1 dense k=10 and S\*), Claude's input tokens fit 143 + 1.536 × the articles' tiktoken count. On the 53 prompts over 3,000 tokens, the fit was off by −7.8% to +8.4%, median −0.4%. Those prompts were at most 21,591 tokens, so 90K is outside the fitted range.
- **Estimated prompt sizes on the 50 real dev questions,** with dense search's ranking:

| | Median | 95th percentile | Over 90K | Over 100K |
|---|---:|---:|---:|---:|
| K=20 | 17K | 30K | 0 | 0 |
| K=50 | 46K | 59K | 0 | 0 |
| K=100 | 94K | 107K | 38 | 12 |

  The run records keep each question's top 20 only, so ranks 21 to 100 are estimated at the mean of ranks 11 to 20 (625 tiktoken tokens per article, against 472 for the corpus).

**Decision (written and committed before any S3 run).**
- **S3 is S1 with a larger K, and nothing else changes:** the same dense search (voyage-4-nano, exact search in Postgres), the same `answer-v1` prompt, whole articles, best first. S3 against S1 then measures only "more articles".
- **K ∈ {20, 50, 100}, untuned.**
- **A token cap of 90,000.** Articles are added in rank order, and the context stops at the first article that would take the estimated prompt past 90,000 tokens.
  - The estimate is 1.54 × tiktoken's count of the system prompt, the question and each numbered article. It runs locally, so the cap adds no network call.
  - The 10% margin covers the worst error seen: a prompt estimated at 90,000 tokens and 8.4% low is still about 98,300.
  - The context is always the top n of the ranking. A lower-ranked article never takes the place of a better one that didn't fit.
  - If the top article alone were over the budget, the question fails with an error instead of getting an empty context, which would silently turn it into closed book. The longest article is about 16,300 tokens, so this can't happen here.
- **Every answer records** the estimate, how many articles were sent and whether the cap cut the list. The summary counts prompts over 100,000 tokens from the API's own usage, which is the check on the estimate. If any dev prompt goes over, the budget is lowered before M2, by enough that the worst error seen would have stayed under.
- **What dev can say.** S1 is at 90% on dev, so S3 can win at most 5 questions, and it would need 4 fixed and 0 broken to clear D-036's bar. Dev checks that S3 works, what the cap does, what each K costs and how S3 fails. No K is chosen. All three run on test at M2, because the cost-vs-correctness chart needs every point.
- **The judge doesn't change.** It reads only the articles an answer cites, so its cost doesn't grow with K.
- **The runs.** A 5-question pilot at K=100, the only K the cap affects, then the 50 real dev questions at K=20, 50 and 100 in real time, because latency on a 90K-token prompt is part of S3's cost. Each is graded by judge-v1 at `high` through the batch API (D-031).
- **Cost.** About $1.50: about $0.80 for the answers ($0.10, $0.25 and $0.46 for K=20, 50 and 100) and about $0.65 for judging 150 answers.

**Alternatives.**
- Paying the higher rate at K=100: exactly 100 articles every time, but 12 of 50 dev questions would cross the line, which roughly doubles K=100's cost (about $0.019 a question against $0.009). No service would ship that.
- Anthropic's token-counting endpoint: exact and free, so the margin could shrink to almost nothing. But it adds a network call to every question, with its own rate limit, to gain about 10% more articles at K=100 only.
- Truncating long articles so that 100 always fit: cuts text from the best-ranked articles too.
- Skipping an article that doesn't fit and trying smaller ones further down: more articles under the same budget, but the context stops being the top n, so K no longer means a rank cutoff.
- The best article last, next to the question: may help, since models use the ends of a long context best. But it changes two things against S1 at once, and the context-rot test measures position directly.
- Dropping K=100: no cap is needed at all, since K=50 stays under 70K, but the curve would stop at about 46K tokens and couldn't say anything about long context near the price line.
- Prompt caching: nothing to reuse, because each question's articles differ, and the system prompt is too short to cache.

**Why.**
- Changing one thing keeps the comparison clean: if S3 and S1 differ, it's because of how much the model read.
- The cap is where the money is. For a question whose top 100 articles come to 104,000 tokens, sending all 100 costs about $0.052 in input. Sending the 86 or so that fit under 90,000 costs about $0.009. The articles dropped are the lowest-ranked ones, the least likely to hold the answer.

**Trade-offs.**
- "K=100" mostly means "about 90,000 tokens of articles": the cap is expected to shorten about 38 of the 50 dev questions. Each run reports how many articles were actually sent.
- The ratio was fitted on prompts up to about 22K tokens and is used at 90K. The pilot checks it before the full runs.
- K=50 and K=100's estimates assume ranks 21 to 100 look like ranks 11 to 20. The runs record the real sizes.
- On 50 questions, S3's results can't separate the three K values from each other or from S1, unless the effect is large.

**In short.** "S3 is S1 with more articles: the same search, the same prompt, best first, so the only thing that changes is how much the model reads. Haiku charges five times as much for the whole request once a prompt passes 100,000 tokens, so S3 adds articles in rank order and stops before an estimated 90,000. The 10% margin covers the error in my local token estimate, and every run checks the real count against the line. For a question whose top 100 articles come to 104,000 tokens, that's about a cent instead of five."

**Outcome (2026-11-13): the cap held, and more articles bought nothing on dev: S3 scored 88% to 90% at every K, against S1's 90%.** Results are in [reports/answers_s3_dev.md](../reports/answers_s3_dev.md).

| 50 real dev questions | % correct [95% CI] | Against S1 [95% CI] | Only S3 right / only S1 right | All gold in context | Input tokens | Cost per answer | Latency p50 |
|---|---:|---|---:|---:|---:|---:|---:|
| **S1, dense k=10 (the default)** | 90.0% [80, 98] | | | 42 | 8,758 | $0.00122 | 3.5 s |
| S3, up to 20 | 88.0% [78, 96] | −2 points [−6, 0] | 0 / 1 | 47 | 18,258 | $0.00219 | 4.2 s |
| S3, up to 50 | 88.0% [78, 96] | −2 points [−8, +4] | 1 / 2 | 48 | 43,996 | $0.00479 | 4.0 s |
| S3, up to 100 | 90.0% [82, 98] | 0 points [−6, +6] | 1 / 1 | 49 | 76,830 | $0.00808 | 4.9 s |

- **The cap held.**
  - No prompt went over 100,000 tokens in the 150 answers or the pilot. The largest was 92,201.
  - The estimate was off by −7.1% to +4.6%, so the 10% margin stays for M2.
  - The cap cut fewer lists than estimated: 20 of 50 at K=100, not 38, and 1 at K=50. Articles ranked 21 to 100 turned out shorter than those ranked 11 to 20. The shortest K=100 context had 45 articles.
  - One question got the same 45 articles at K=50 and at K=100. The two prompts were identical, so the K=100 run replayed that answer from the cache.
- **More articles found more gold, but there was nothing left to fix.** Questions with every gold article in context rose from 42 for S1 to 47, 48 and 49. But S1 was already graded correct on all 8 questions where it lacked a gold article, so the extra gold changed no grade. S1's 5 partial answers all had the gold article in context: those are the generator's misses, not retrieval's.
- **The few flips come down to how the answer was written.**
  - K=20 lost `ew-171`, which S2 also lost: the answer told the customer to skip a troubleshooting step.
  - K=50 lost `ew-171` and `ew-110`, and won `sim-113`.
  - K=100 won `sim-113` and lost `sim-184`, which S2 also lost. The answer said "yes, in some parts" where the answer is no, and it didn't cite the gold article, ranked 19th.
  - Every S3 run declined `ew-059`, which S1 answered and was graded partial on.
- **Grounding slipped as the context grew.** Answers the judge graded only partly grounded: S1 2, K=20 0, K=50 3, K=100 6, out of 50. The judge's reasons were claims credited to the wrong article among many, or details that appear in no cited article. On 50 questions this is only a direction, worth watching at M2 and in the context-rot test.
- **Cost grows with K; latency hardly does.**
  - Per answer, S3 costs 1.8, 3.9 and 6.6 times what S1 does.
  - The median latency is 4.0 to 4.9 s against S1's 3.5 s, and the 95th percentile is 6.4 to 7.8 s. Haiku reads 77K tokens of articles in about a second more than 9K.
  - The 27-second first question appeared in the pilot only (`ew-000`, later replayed from the cache). The first questions of the K=20 and K=50 runs took 6 s, so it doesn't happen on every run. It stays with Step 8.
- **No verdict, as planned.** All three intervals include zero, and McNemar's p is 1.0 for each. All three K values run on test at M2.
- **The judge's cost did grow a little with K,** unlike what the decision expected: $0.0055, $0.0058 and $0.0061 per graded answer against S1's $0.0049. It reads only the cited articles, so 9 times the context cost up to 25% more to grade.
- **Where these numbers come from.**
  - The tables are in the report. The flips, declines and grounding reasons came from one-off scripts over the run records.
  - Answer runs, all from commit `c63d824` with a clean tree: `20261113-135341_s3-k100-medium_dev_real` (the 5-question pilot), `20261113-135507_s3-k20-medium_dev_real`, `20261113-135901_s3-k50-medium_dev_real` and `20261113-140307_s3-k100-medium_dev_real`.
  - Judge runs: `20261113-141009_judge-high_20261113-135507_s3-k20-medium_dev_real`, `20261113-141316_judge-high_20261113-135901_s3-k50-medium_dev_real` and `20261113-141522_judge-high_20261113-140307_s3-k100-medium_dev_real`.
- **Cost.** $1.59 against the $1.50 estimate: $0.74 for the answers, including the pilot, and $0.84 for judging.

---

## Coming up

Where each upcoming decision is leaning now. Each becomes a full entry at the step where it's made.

| Step | Decision | Leaning | Decided by |
|---|---|---|---|
| 8 | API framework: FastAPI vs Django | FastAPI: the service mostly waits on LLM calls and streams tokens, which suits async; typed request and response models and OpenAPI docs come free; Django's admin, ORM and templates aren't needed here | Workload shape |
| 8 | Redis vs RabbitMQ | Redis for response caching and rate limiting. No message broker: there's no service-to-service messaging, and eval runs are batch jobs | Actual need |
| 8 | Streaming: Server-Sent Events vs WebSockets | SSE: one-way, server to browser, over plain HTTP, which is all a chat answer needs | Actual need |
| 9 | Hosting: k3s on one VM vs managed Kubernetes vs a PaaS | k3s on a small VM: real Kubernetes at hobby cost | Cost |
| 9 | Observability: OpenTelemetry with the Grafana stack vs an LLM tracing tool (Langfuse, Phoenix) | Open | Step 9 |
