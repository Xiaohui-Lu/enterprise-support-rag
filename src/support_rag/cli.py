"""Command-line entry point: `srag <command>`."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="srag", description="Support assistant on WixQA.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="download WixQA at the pinned revision into data/raw/")
    commands.add_parser("stats", help="write corpus statistics to reports/corpus_stats.md")
    splits = commands.add_parser("splits", help="create the frozen dev/test splits (run once)")
    splits.add_argument("--force", action="store_true", help="overwrite existing splits")
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


if __name__ == "__main__":
    main()
