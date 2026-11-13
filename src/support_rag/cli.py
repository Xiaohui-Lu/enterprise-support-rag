"""Command-line entry point: `srag <command>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="srag", description="Support assistant on WixQA.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download WixQA at the pinned revision into data/raw/")
    commands.add_parser("stats", help="write corpus statistics to reports/corpus_stats.md")
    splits = commands.add_parser("splits", help="create the frozen dev/test splits (run once)")
    splits.add_argument("--force", action="store_true", help="overwrite existing splits")
    evaluate = commands.add_parser(
        "eval-retrieval", help="run a retriever over a split and save a run record in runs/"
    )
    evaluate.add_argument(
        "--retriever", choices=["bm25", "dense", "hybrid", "rerank"], default="bm25"
    )
    evaluate.add_argument(
        "--model",
        choices=["qwen3-0.6b", "voyage-4-nano"],
        default="qwen3-0.6b",
        help="embedding model for --retriever dense, hybrid or rerank (D-035)",
    )
    evaluate.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="how --retriever hybrid merges its two rankings (D-037)",
    )
    evaluate.add_argument(
        "--reranker",
        choices=["qwen3-reranker-0.6b", "minilm-l6"],
        default="qwen3-reranker-0.6b",
        help="cross-encoder for --retriever rerank (D-038)",
    )
    evaluate.add_argument(
        "--candidates",
        choices=["dense", "both"],
        default="dense",
        help="what --retriever rerank reads: dense's top 50, or BM25's top 50 too (D-038)",
    )
    evaluate.add_argument(
        "--unit",
        choices=["article", "h2", "headings", "window"],
        default="article",
        help="what --retriever dense or rerank searches: whole articles or sections (D-039)",
    )
    evaluate.add_argument(
        "--no-title",
        action="store_true",
        help="sections: don't put the article's title in front of each one (D-039)",
    )
    evaluate.add_argument(
        "--store",
        choices=["numpy", "postgres"],
        default="numpy",
        help="where --retriever dense searches whole articles: in memory or Postgres (D-041)",
    )
    evaluate.add_argument(
        "--index",
        choices=["exact", "hnsw"],
        default="exact",
        help="--store postgres: scan every vector, or use the HNSW index (D-041)",
    )
    evaluate.add_argument(
        "--split",
        choices=["dev", "synthetic_reference"],
        default="dev",
        help="questions to evaluate (the test set runs only at milestones)",
    )
    evaluate.add_argument("--no-stem", action="store_true", help="BM25: turn off stemming")
    db_load = commands.add_parser(
        "db-load", help="load the articles and their stored vectors into Postgres (D-041)"
    )
    db_load.add_argument(
        "--model",
        choices=["qwen3-0.6b", "voyage-4-nano"],
        default="voyage-4-nano",
        help="whose stored vectors to load; the model itself isn't run",
    )
    db_load.add_argument("--no-hnsw", action="store_true", help="don't build the HNSW index")
    report = commands.add_parser(
        "report-retrieval",
        help="write reports/retrieval_<name>.md: each run vs every earlier one (D-036)",
    )
    report.add_argument("--name", default="bm25", help="report name, e.g. bm25 or dense")
    report.add_argument("runs", nargs="+", type=Path, help="run directories, earliest first")
    answer = commands.add_parser(
        "answer", help="answer a split with one strategy (paid API calls, cached)"
    )
    answer.add_argument("--strategy", choices=["s0", "s1", "s2", "s3", "s_star"], required=True)
    answer.add_argument(
        "--split", choices=["dev_real", "test"], default="dev_real", help="test: milestones only"
    )
    answer.add_argument(
        "--milestone", choices=["M1", "M2", "M3"], help="required for the test split (D-034)"
    )
    answer.add_argument(
        "--effort", choices=["low", "medium", "high"], help="default: the tuned value (D-033)"
    )
    answer.add_argument(
        "--k",
        type=int,
        help="articles retrieved: for S1, default the tuned value (D-033); required for S3 (D-044)",
    )
    answer.add_argument(
        "--retriever",
        choices=["bm25", "dense"],
        help="what S1 searches; default: dense, exact search in Postgres (D-042)",
    )
    answer.add_argument(
        "--unit",
        choices=["article", "h2"],
        help="what S1 reads: whole articles (default) or each one's best h2 section (D-042)",
    )
    answer.add_argument("--limit", type=int, help="answer only this many questions (a pilot)")
    judge = commands.add_parser("judge", help="grade an answer run with the judge (paid, cached)")
    judge.add_argument("answers_run", type=Path, help="the answer run directory to grade")
    judge.add_argument("--effort", choices=["low", "medium", "high"], required=True)
    judge.add_argument(
        "--batch", action="store_true", help="send uncached calls through the batch API"
    )
    judge.add_argument("--batch-id", help="collect a batch submitted earlier; never pays twice")
    report_answers = commands.add_parser(
        "report-answers", help="% correct per strategy with CIs and paired tests, to reports/"
    )
    report_answers.add_argument("--name", help="report name; default: the split")
    report_answers.add_argument(
        "runs", nargs="+", type=Path, help="one judge run per strategy, same judge settings"
    )
    report_tuning = commands.add_parser(
        "report-tuning", help="candidate settings vs the default on dev, to reports/"
    )
    report_tuning.add_argument("name", help="report name, e.g. effort or k")
    report_tuning.add_argument("--default-effort", help="the baseline's effort (default: tuned)")
    report_tuning.add_argument("--default-k", type=int, help="the baseline's k (default: tuned)")
    report_tuning.add_argument(
        "--default-retriever",
        choices=["bm25", "dense"],
        help="S1's baseline retriever, to tell arms at the same k apart (D-042)",
    )
    report_tuning.add_argument(
        "--margin", type=float, help="points a cheaper candidate must stay within (D-042)"
    )
    report_tuning.add_argument(
        "--feature-requests",
        action="store_true",
        help="count the answers citing a feature request that isn't gold (D-040)",
    )
    report_tuning.add_argument(
        "runs", nargs="+", type=Path, help="judge runs, including each strategy's default"
    )
    validate = commands.add_parser(
        "validate-judge", help="judge vs labels: agreement and kappa, to reports/"
    )
    validate.add_argument(
        "runs", nargs="+", type=Path, help="judge runs: one per strategy for each effort"
    )
    label_sample = commands.add_parser(
        "label-sample", help="draw the hand-labeling sample from answer runs (run once)"
    )
    label_sample.add_argument(
        "runs", nargs="+", type=Path, help="one full dev_real answer run per strategy"
    )
    label = commands.add_parser(
        "label", help="hand-label the sample, blind to strategy (resumable)"
    )
    label.add_argument(
        "--adjudicate", action="store_true", help="only the items the panel didn't settle"
    )
    panel = commands.add_parser(
        "panel", help="pre-label the sample with the four-model panel (paid, cached)"
    )
    panel.add_argument("--limit", type=int, help="grade only the first N sample items (a pilot)")
    panel.add_argument(
        "--batch", action="store_true", help="send uncached calls through the batch API"
    )
    panel.add_argument("--batch-id", help="collect a batch submitted earlier; never pays twice")
    panel_labels = commands.add_parser(
        "panel-labels", help="commit the panel's votes and labels from its four runs (once)"
    )
    panel_labels.add_argument("runs", nargs=4, type=Path, help="the four full panel runs")
    args = parser.parse_args(argv)

    try:
        run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        sys.exit(f"error: {error}")


def run(args: argparse.Namespace) -> None:
    # Imports sit inside each command so `srag --help` stays fast.
    match args.command:
        case "download":
            from support_rag.wixqa import download

            for path in download():
                print(f"downloaded {path}")
        case "stats":
            from support_rag.corpus_stats import write_report

            print(f"wrote {write_report()}")
        case "splits":
            from support_rag.splits import write_splits

            counts = write_splits(force=args.force)["counts"]
            print("wrote data/splits/splits.json")
            for name, value in counts.items():
                print(f"  {name}: {value}")
        case "eval-retrieval":
            from support_rag.retrieval_eval import (
                format_summary,
                run_bm25_eval,
                run_dense_eval,
                run_hybrid_eval,
                run_rerank_eval,
                run_section_rerank_eval,
            )
            from support_rag.runs import load_config, load_summary

            if args.unit != "article" and args.retriever not in ("dense", "rerank"):
                raise ValueError("--unit works only with --retriever dense or rerank")
            if args.unit != "article" and args.candidates != "dense":
                raise ValueError("--unit reranks dense's candidates only (D-039)")
            if args.no_title and (args.retriever != "dense" or args.unit == "article"):
                raise ValueError("--no-title works only with --retriever dense and sections")
            if args.store == "postgres" and (args.retriever != "dense" or args.unit != "article"):
                raise ValueError("--store postgres works only with --retriever dense on articles")
            if args.index != "exact" and args.store != "postgres":
                raise ValueError("--index works only with --store postgres")

            if args.retriever == "rerank" and args.unit != "article":
                run_dir = run_section_rerank_eval(
                    args.split, model=args.model, reranker=args.reranker, unit=args.unit
                )
            elif args.retriever == "rerank":
                run_dir = run_rerank_eval(
                    args.split,
                    model=args.model,
                    reranker=args.reranker,
                    candidates=args.candidates,
                )
            elif args.retriever == "hybrid":
                run_dir = run_hybrid_eval(args.split, model=args.model, fusion=args.fusion)
            elif args.retriever == "dense":
                run_dir = run_dense_eval(
                    args.split,
                    model=args.model,
                    unit=args.unit,
                    title=not args.no_title,
                    store=args.store,
                    index=args.index,
                )
            else:
                run_dir = run_bm25_eval(args.split, stem=not args.no_stem)
            print(f"wrote {run_dir}\n")
            print(format_summary(load_summary(run_dir)))
            if load_config(run_dir)["git"]["dirty"]:
                print(
                    "\nnote: uncommitted changes, so this run can't be reproduced from its commit"
                )
        case "db-load":
            from support_rag.config import database_url
            from support_rag.embeddings import MODELS, stored_article_vectors
            from support_rag.store import connect, load
            from support_rag.wixqa import load_articles

            articles = load_articles()
            model = MODELS[args.model]
            vectors = stored_article_vectors(articles, model)
            with connect(database_url()) as connection:
                loaded = load(connection, articles, vectors, model, hnsw=not args.no_hnsw)
            print(f"loaded {model.name} vectors into Postgres")
            for key, value in loaded.items():
                print(f"  {key}: {value}")
        case "report-retrieval":
            from support_rag.retrieval_report import write_report

            print(f"wrote {write_report(args.runs, name=args.name)}")
        case "answer":
            import json

            from support_rag.answer_eval import (
                DEFAULT_EFFORT,
                DEFAULT_K,
                DEFAULT_RETRIEVER,
                run_answers,
            )
            from support_rag.runs import load_summary

            if args.strategy != "s1" and (args.retriever or args.unit):
                raise ValueError("--retriever and --unit work only with --strategy s1")
            if args.strategy == "s3" and not args.k:
                raise ValueError("S3 needs --k: 20, 50 or 100 (D-044)")
            if args.k and args.strategy not in ("s1", "s3"):
                raise ValueError("--k works only with --strategy s1 or s3")
            retriever = args.retriever or DEFAULT_RETRIEVER
            unit = args.unit or "article"
            if retriever == "bm25" and unit != "article":
                raise ValueError("--unit h2 works only with --retriever dense (D-042)")

            run_dir = run_answers(
                args.strategy,
                args.split,
                effort=args.effort or DEFAULT_EFFORT,
                k=args.k or DEFAULT_K,
                retriever_name=retriever,
                unit=unit,
                limit=args.limit,
                milestone=args.milestone,
            )
            print(f"wrote {run_dir}\n")
            print(json.dumps(load_summary(run_dir), indent=2))
        case "judge":
            import json

            from support_rag.judge_eval import run_judge
            from support_rag.runs import load_summary

            run_dir = run_judge(
                args.answers_run, effort=args.effort, batch=args.batch, batch_id=args.batch_id
            )
            print(f"wrote {run_dir}\n")
            print(json.dumps(load_summary(run_dir), indent=2))
        case "report-answers":
            from support_rag.answer_report import write_report

            path = write_report(args.runs, name=args.name)
            print(path.read_text(encoding="utf-8"))
        case "report-tuning":
            from support_rag.answer_eval import DEFAULT_EFFORT, DEFAULT_K
            from support_rag.tuning import write_report

            feature_requests: frozenset[str] = frozenset()
            if args.feature_requests:
                from support_rag.wixqa import load_articles

                feature_requests = frozenset(
                    a.id for a in load_articles() if a.article_type == "feature_request"
                )
            path = write_report(
                args.name,
                args.runs,
                default_effort=args.default_effort or DEFAULT_EFFORT,
                default_k=args.default_k or DEFAULT_K,
                default_retriever=args.default_retriever,
                margin=args.margin,
                feature_requests=feature_requests,
            )
            print(path.read_text(encoding="utf-8"))
        case "validate-judge":
            from support_rag.judge_validation import final_labels, validate_judge, write_report
            from support_rag.labels import load_labels, load_sample
            from support_rag.panel import load_panel_labels

            labels = final_labels(load_panel_labels(), load_labels())
            path = write_report(validate_judge(args.runs, load_sample(), labels), labels)
            print(path.read_text(encoding="utf-8"))
        case "label-sample":
            from support_rag.labels import write_sample

            counts = write_sample(args.runs)["counts"]
            print("wrote data/labels/sample.json")
            for name, value in counts.items():
                print(f"  {name}: {value}")
        case "label":
            from support_rag.labels import adjudication_items, label_session, load_sample
            from support_rag.wixqa import load_articles, load_questions

            sample = load_sample()
            if args.adjudicate:
                from support_rag.panel import load_panel_labels

                items = adjudication_items(sample, load_panel_labels())
            else:
                items = sample["items"]
            questions = {
                q.id: q for subset in ("expertwritten", "simulated") for q in load_questions(subset)
            }
            label_session(items, questions, {a.id: a for a in load_articles()})
        case "panel-labels":
            from support_rag.panel import write_panel_labels

            counts = write_panel_labels(args.runs)["counts"]
            # Counts only; the file holds every vote, so don't open it before labeling.
            print("wrote data/labels/panel.json")
            for name, value in counts.items():
                print(f"  {name}: {value}")
        case "panel":
            import json

            from support_rag.labels import load_sample
            from support_rag.panel import format_agreement, panel_agreement, run_panel

            run_dirs, batch_summary = run_panel(
                limit=args.limit, batch=args.batch, batch_id=args.batch_id
            )
            if batch_summary:
                print(json.dumps(batch_summary, indent=2))
            for run_dir in run_dirs:
                print(f"wrote {run_dir}")
            # Aggregates only: the runs hold each item's grades, so the split items can
            # still be labeled blind (D-027, D-028).
            print(f"\n{format_agreement(panel_agreement(run_dirs, load_sample()))}")


if __name__ == "__main__":
    main()
