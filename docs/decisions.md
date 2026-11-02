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

## Coming up

Where each upcoming decision is leaning now. Each becomes a full entry at the step where it's made.

| Step | Decision | Leaning | Decided by |
|---|---|---|---|
| 3 | LLM provider and models | Claude Haiku 5.5 as the generator, Claude Sonnet 5.5 as the judge; a second provider only if the judge disagrees with hand labels | Budget and dev results |
| 4 | Vector store: Postgres + pgvector vs a dedicated vector DB (Qdrant, Pinecone) | pgvector: about 6K articles is small, and one database can hold text, vectors and run results | Scale and operating cost |
| 4 | Keyword search in production: Postgres full-text search vs a BM25 extension | Open. Postgres's built-in ranking isn't BM25 | Measured on dev (RQ2) |
| 4 | Embedding model, reranker, chunking | Open | Measured on dev (RQ2) |
| 8 | API framework: FastAPI vs Django | FastAPI: the service mostly waits on LLM calls and streams tokens, which suits async; typed request and response models and OpenAPI docs come free; Django's admin, ORM and templates aren't needed here | Workload shape |
| 8 | Redis vs RabbitMQ | Redis for response caching and rate limiting. No message broker: there's no service-to-service messaging, and eval runs are batch jobs | Actual need |
| 8 | Streaming: Server-Sent Events vs WebSockets | SSE: one-way, server to browser, over plain HTTP, which is all a chat answer needs | Actual need |
| 9 | Hosting: k3s on one VM vs managed Kubernetes vs a PaaS | k3s on a small VM: real Kubernetes at hobby cost | Cost |
| 9 | Observability: OpenTelemetry with the Grafana stack vs an LLM tracing tool (Langfuse, Phoenix) | Open | Step 9 |
