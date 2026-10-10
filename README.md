# Enterprise support RAG

A support assistant for a 6,221-article help center, built to answer one question with data: **when do retrieval, agentic search and long context each pay off?** Closed-book answering, single-pass RAG, agentic search and long-context prompting run on the [WixQA](https://huggingface.co/datasets/Wix/WixQA) benchmark with the model held constant, and are compared on correctness, cost and latency.

> Work in progress: Step 1 of 10. See the [roadmap](docs/experiment-design.md#12-roadmap).

## Quick start

Needs [uv](https://docs.astral.sh/uv/), which installs the right Python version by itself.

```bash
uv sync                # create the environment from uv.lock
uv run srag download   # WixQA at the pinned revision, into data/raw/
uv run srag stats      # corpus statistics, into reports/corpus_stats.md
uv run pytest
```

The dev/test split lives in `data/splits/splits.json`. It was generated once with `srag splits` and is never regenerated, so the test set can't drift.

## Docs

- [Experiment design](docs/experiment-design.md): research questions, splits, strategies, metrics, roadmap
- [Decision log](docs/decisions.md): what was chosen, what else was considered, and why
- [Corpus statistics](reports/corpus_stats.md): what the data looks like, plus integrity checks

## Data

WixQA is by Wix.com AI Research, released under the MIT license: Cohen et al., [WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation](https://arxiv.org/abs/2505.08643), 2025. The raw files aren't stored in this repo; `srag download` fetches them at the commit pinned in `src/support_rag/config.py`.

## License

MIT
