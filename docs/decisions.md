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

## Coming up

Where each upcoming decision is leaning now. Each becomes a full entry at the step where it's made.

| Step | Decision | Leaning | Decided by |
|---|---|---|---|
| 2 | BM25 for the baseline | An in-memory library; no infrastructure yet | Simplicity |
| 3 | LLM provider and models | Claude Haiku 5.5 as the generator, Claude Sonnet 5.5 as the judge; a second provider only if the judge disagrees with hand labels | Budget and dev results |
| 4 | Vector store: Postgres + pgvector vs a dedicated vector DB (Qdrant, Pinecone) | pgvector: about 6K articles is small, and one database can hold text, vectors and run results | Scale and operating cost |
| 4 | Keyword search in production: Postgres full-text search vs a BM25 extension | Open. Postgres's built-in ranking isn't BM25 | Measured on dev (RQ2) |
| 4 | Embedding model, reranker, chunking | Open | Measured on dev (RQ2) |
| 8 | API framework: FastAPI vs Django | FastAPI: the service mostly waits on LLM calls and streams tokens, which suits async; typed request and response models and OpenAPI docs come free; Django's admin, ORM and templates aren't needed here | Workload shape |
| 8 | Redis vs RabbitMQ | Redis for response caching and rate limiting. No message broker: there's no service-to-service messaging, and eval runs are batch jobs | Actual need |
| 8 | Streaming: Server-Sent Events vs WebSockets | SSE: one-way, server to browser, over plain HTTP, which is all a chat answer needs | Actual need |
| 9 | Hosting: k3s on one VM vs managed Kubernetes vs a PaaS | k3s on a small VM: real Kubernetes at hobby cost | Cost |
| 9 | Observability: OpenTelemetry with the Grafana stack vs an LLM tracing tool (Langfuse, Phoenix) | Open | Step 9 |
