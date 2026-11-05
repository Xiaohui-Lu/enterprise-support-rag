# CLAUDE.md

A support assistant for a 6,221-article help center (the WixQA benchmark), built to measure when retrieval, agentic search and long context each pay off. The generator model is held constant, and strategies are compared on correctness, cost and latency.

Before starting a roadmap step, read its row in `docs/experiment-design.md` (section 12) and the related entries in `docs/decisions.md`.

## Commands

```bash
uv sync                # environment from uv.lock (Python 3.12)
uv run pytest          # tests; must pass before every commit
uv run ruff check .    # lint
uv run ruff format .   # format
uv run srag download   # WixQA at the pinned revision, into data/raw/
uv run srag stats      # regenerate reports/corpus_stats.md
uv run srag eval-retrieval --split dev           # BM25 on dev; add --no-stem for the ablation
uv run srag eval-retrieval --retriever dense --model qwen3-0.6b   # or voyage-4-nano (D-035)
uv run srag report-retrieval --name dense runs/<a> runs/<b>   # reports/retrieval_<name>.md, compared by D-036
uv run srag answer --strategy s1 --limit 10       # answers on dev_real (paid, cached); s0, s1 or s_star
uv run srag answer --strategy s1 --split test --milestone M1   # test set: milestones only, clean tree (D-034)
uv run srag judge runs/<answers> --effort high --batch   # grade an answer run (paid, cached)
uv run srag validate-judge runs/<judge runs>      # judge vs labels: kappa, to reports/judge_validation.md
uv run srag report-answers runs/<judge runs>      # % correct with CIs and paired tests, to reports/
uv run srag report-tuning <name> runs/<judge runs> # candidates vs the default on dev, to reports/
uv run srag label-sample runs/<s0> runs/<s1> runs/<s_star>   # draw the hand-label sample, once
uv run srag panel --limit 5                       # four-model panel pilot; --batch for the full run
uv run srag panel-labels runs/<4 panel runs>      # commit the panel's votes and labels, once
uv run srag label --adjudicate                    # hand-label only what the panel didn't settle
```

## Layout

- `src/support_rag/`: the only package. Every step is an `srag` subcommand in `cli.py`; no notebooks or loose scripts (D-010).
- `tests/`: pytest, run against the installed package. Tests never touch the network or `data/raw/`.
- `docs/`: `experiment-design.md` (questions, splits, strategies, metrics, roadmap), `decisions.md` (decision log) and `porting.md` (what to redo for a new dataset).
- `reports/`: generated reports, committed.
- `data/raw/`: the downloaded dataset, git-ignored. `data/splits/splits.json`: the frozen split, committed.
- `data/labels/`: the hand-label sample and labels for validating the judge, committed (D-027).
- `runs/`: run records (config, per-question results, summary), git-ignored; reports cite run IDs (D-018).
- `cache/`: the LLM response cache (SQLite), git-ignored (D-020).

## Rules

- Never regenerate or hand-edit `data/splits/splits.json` or `data/labels/`, and never tune prompts, thresholds or parameters on test-set results. Tune on dev; run on test only at milestones (D-004, D-011).
- Don't change `DATASET_REVISION` in `src/support_rag/config.py`. Every result depends on it.
- No LangChain, LlamaIndex or similar frameworks. Call provider SDKs through this repo's own thin interfaces, so every token and call can be counted (D-003).
- Evaluation calls the same code paths the service uses; there is no separate eval pipeline (D-008).
- Record significant choices in `docs/decisions.md` in the existing format, and keep the roadmap's Status column current.
- Secrets live in `.env`, which is git-ignored. Never commit, log or print API keys.

## Style

- Python 3.12 with type hints, frozen dataclasses for records, and the standard library first. Ruff settings are in `pyproject.toml` (line length 100).
- Developed on Windows, deployed to Linux: use `pathlib`, open text files with `encoding="utf-8"`, and keep LF line endings (see `.gitattributes`).
- Keep tests small and name them after the behavior they check, like `test_existing_split_is_never_overwritten`.
