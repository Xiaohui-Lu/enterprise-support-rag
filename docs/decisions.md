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

---

## Coming up

Where each upcoming decision is leaning now. Each becomes a full entry at the step where it's made.

| Step | Decision | Leaning | Decided by |
|---|---|---|---|
| 4 | Vector store: Postgres + pgvector vs a dedicated vector DB (Qdrant, Pinecone) | pgvector: about 6K articles is small, and one database can hold text, vectors and run results | Scale and operating cost |
| 4 | Keyword search in production: Postgres full-text search vs a BM25 extension | Open. Postgres's built-in ranking isn't BM25 | Measured on dev by D-036's rule |
| 4 | Hybrid fusion, reranker, chunking, feature-request down-weighting | Open | Measured on dev by D-036's rule |
| 8 | API framework: FastAPI vs Django | FastAPI: the service mostly waits on LLM calls and streams tokens, which suits async; typed request and response models and OpenAPI docs come free; Django's admin, ORM and templates aren't needed here | Workload shape |
| 8 | Redis vs RabbitMQ | Redis for response caching and rate limiting. No message broker: there's no service-to-service messaging, and eval runs are batch jobs | Actual need |
| 8 | Streaming: Server-Sent Events vs WebSockets | SSE: one-way, server to browser, over plain HTTP, which is all a chat answer needs | Actual need |
| 9 | Hosting: k3s on one VM vs managed Kubernetes vs a PaaS | k3s on a small VM: real Kubernetes at hobby cost | Cost |
| 9 | Observability: OpenTelemetry with the Grafana stack vs an LLM tracing tool (Langfuse, Phoenix) | Open | Step 9 |
