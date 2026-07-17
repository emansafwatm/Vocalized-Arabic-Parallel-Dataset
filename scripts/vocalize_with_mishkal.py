#!/usr/bin/env python3
"""Generate a fully vocalized Arabic column using Mishkal.

Input and output use UTF-8 tab-separated files. Existing columns are preserved and
an ``arabic_vocalized`` column is added. Processing failures are recorded rather
than silently discarded.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

LOGGER = logging.getLogger("mishkal_vocalizer")


@dataclass
class VocalizationStatistics:
    examined: int = 0
    vocalized: int = 0
    empty: int = 0
    errors: int = 0


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def load_vocalizer() -> Any:
    """Import Mishkal lazily so utility modules remain importable without it."""

    try:
        import mishkal.tashkeel
    except ImportError as exc:
        raise RuntimeError(
            "Mishkal is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return mishkal.tashkeel.TashkeelClass()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fully vocalize Arabic text with Mishkal.")
    parser.add_argument("--input", type=Path, required=True, help="Input UTF-8 TSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Output UTF-8 TSV file.")
    parser.add_argument("--arabic-column", default="text_ar")
    parser.add_argument("--output-column", default="arabic_vocalized")
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_paths(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")


def main() -> int:
    args = build_parser().parse_args()
    validate_paths(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors_path = args.errors or args.output.with_suffix(".errors.jsonl")
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    statistics = VocalizationStatistics()
    vocalizer = load_vocalizer()

    with args.input.open("r", encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination, \
         errors_path.open("w", encoding="utf-8") as error_file:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Input TSV has no header.")
        if args.arabic_column not in reader.fieldnames:
            raise KeyError(
                f"Column {args.arabic_column!r} not found. Available: {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        if args.output_column not in fieldnames:
            fieldnames.append(args.output_column)
        writer = csv.DictWriter(destination, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row_number, row in enumerate(tqdm(reader, desc="Vocalizing", unit="row"), start=2):
            if args.limit is not None and statistics.examined >= args.limit:
                break
            statistics.examined += 1
            arabic_text = normalize_text(row.get(args.arabic_column))
            if not arabic_text:
                statistics.empty += 1
                row[args.output_column] = ""
                writer.writerow(row)
                continue
            try:
                vocalized = normalize_text(vocalizer.tashkeel(arabic_text))
                row[args.output_column] = vocalized
                statistics.vocalized += 1
            except Exception as exc:
                statistics.errors += 1
                row[args.output_column] = ""
                error_file.write(
                    json.dumps(
                        {
                            "row_number": row_number,
                            "arabic_text": arabic_text,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            writer.writerow(row)

    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "statistics": asdict(statistics),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "mishkal": package_version("mishkal"),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Completed: %s", asdict(statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
