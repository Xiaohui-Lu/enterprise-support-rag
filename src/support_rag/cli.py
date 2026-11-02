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
    args = parser.parse_args(argv)

    try:
        run(args)
    except (FileExistsError, FileNotFoundError) as error:
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


if __name__ == "__main__":
    main()
