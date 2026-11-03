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
        "eval-retrieval", help="run BM25 over a split and save a run record in runs/"
    )
    evaluate.add_argument(
        "--split",
        choices=["dev", "synthetic_reference"],
        default="dev",
        help="questions to evaluate (the test set runs only at milestones)",
    )
    evaluate.add_argument("--no-stem", action="store_true", help="turn off stemming")
    report = commands.add_parser(
        "report-retrieval", help="write reports/retrieval_bm25.md from run records"
    )
    report.add_argument(
        "runs", nargs="+", type=Path, help="run directories; the first is the baseline"
    )
    answer = commands.add_parser(
        "answer", help="answer a split with one strategy (paid API calls, cached)"
    )
    answer.add_argument("--strategy", choices=["s0", "s1", "s_star"], required=True)
    answer.add_argument(
        "--split", choices=["dev_real"], default="dev_real", help="the test set runs only at M1"
    )
    answer.add_argument("--effort", choices=["low", "medium", "high"], default="medium")
    answer.add_argument("--k", type=int, default=5, help="articles retrieved for S1")
    answer.add_argument("--limit", type=int, help="answer only this many questions (a pilot)")
    judge = commands.add_parser("judge", help="grade an answer run with the judge (paid, cached)")
    judge.add_argument("answers_run", type=Path, help="the answer run directory to grade")
    judge.add_argument("--effort", choices=["low", "medium", "high"], required=True)
    label_sample = commands.add_parser(
        "label-sample", help="draw the hand-labeling sample from answer runs (run once)"
    )
    label_sample.add_argument(
        "runs", nargs="+", type=Path, help="one full dev_real answer run per strategy"
    )
    commands.add_parser("label", help="hand-label the sample, blind to strategy (resumable)")
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
            from support_rag.retrieval_eval import format_summary, run_bm25_eval
            from support_rag.runs import load_config, load_summary

            run_dir = run_bm25_eval(args.split, stem=not args.no_stem)
            print(f"wrote {run_dir}\n")
            print(format_summary(load_summary(run_dir)))
            if load_config(run_dir)["git"]["dirty"]:
                print(
                    "\nnote: uncommitted changes, so this run can't be reproduced from its commit"
                )
        case "report-retrieval":
            from support_rag.retrieval_report import write_report

            print(f"wrote {write_report(args.runs)}")
        case "answer":
            import json

            from support_rag.answer_eval import run_answers
            from support_rag.runs import load_summary

            run_dir = run_answers(
                args.strategy, args.split, effort=args.effort, k=args.k, limit=args.limit
            )
            print(f"wrote {run_dir}\n")
            print(json.dumps(load_summary(run_dir), indent=2))
        case "judge":
            import json

            from support_rag.judge_eval import run_judge
            from support_rag.runs import load_summary

            run_dir = run_judge(args.answers_run, effort=args.effort)
            print(f"wrote {run_dir}\n")
            print(json.dumps(load_summary(run_dir), indent=2))
        case "label-sample":
            from support_rag.labels import write_sample

            counts = write_sample(args.runs)["counts"]
            print("wrote data/labels/sample.json")
            for name, value in counts.items():
                print(f"  {name}: {value}")
        case "label":
            from support_rag.labels import label_session, load_sample
            from support_rag.wixqa import load_articles, load_questions

            sample = load_sample()
            questions = {
                q.id: q for subset in ("expertwritten", "simulated") for q in load_questions(subset)
            }
            label_session(sample, questions, {a.id: a for a in load_articles()})


if __name__ == "__main__":
    main()
