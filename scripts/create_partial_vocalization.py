#!/usr/bin/env python3
"""Create partially vocalized Arabic by removing word-final inflectional marks.

Internal diacritics are preserved. At the end of each Arabic word, the script
removes short vowels, tanween, and sukun while preserving a terminal shadda.
Punctuation attached to the word is handled without deleting the punctuation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

LOGGER = logging.getLogger("partial_vocalization")

# Arabic short vowels and case/ending marks. Shadda (U+0651) is intentionally
# excluded because it represents consonant gemination rather than i'rab.
FINAL_ENDING_MARKS = frozenset(
    {
        "\u064b",  # fathatan
        "\u064c",  # dammatan
        "\u064d",  # kasratan
        "\u064e",  # fatha
        "\u064f",  # damma
        "\u0650",  # kasra
        "\u0652",  # sukun
    }
)
ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063A\u0641-\u064A\u066E-\u06D3\u06FA-\u06FF]")


@dataclass
class PartialStatistics:
    examined: int = 0
    converted: int = 0
    empty: int = 0


def remove_word_final_endings(text: str) -> str:
    """Remove final inflectional marks from each whitespace-delimited token.

    The function searches backward through punctuation and combining marks to find
    the last Arabic letter in each token. Only ending marks attached after that
    letter are removed; internal marks and punctuation are preserved.
    """

    if not text:
        return text

    converted_tokens: list[str] = []
    for token in text.split(" "):
        if not token:
            converted_tokens.append(token)
            continue

        chars = list(token)
        last_arabic_index = -1
        for index in range(len(chars) - 1, -1, -1):
            if ARABIC_LETTER_RE.fullmatch(chars[index]):
                last_arabic_index = index
                break

        if last_arabic_index == -1:
            converted_tokens.append(token)
            continue

        prefix = chars[: last_arabic_index + 1]
        suffix = chars[last_arabic_index + 1 :]
        filtered_suffix = [char for char in suffix if char not in FINAL_ENDING_MARKS]

        # Fathatan is commonly encoded before a final supporting alif, as in
        # "مَرْحَبًا" (beh + fathatan + alif). Remove that mark while preserving
        # the alif and any following punctuation.
        if (
            prefix
            and prefix[-1] in {"ا", "ى"}
            and len(prefix) >= 2
            and prefix[-2] == "\u064b"
        ):
            del prefix[-2]

        converted_tokens.append("".join(prefix + filtered_suffix))

    return " ".join(converted_tokens)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove word-final Arabic case/ending marks from a TSV column."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocalized-column", default="arabic_vocalized")
    parser.add_argument("--output-column", default="arabic_partially_vocalized")
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output paths must be different.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    started_at = datetime.now(timezone.utc)
    statistics = PartialStatistics()

    with args.input.open("r", encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("Input TSV has no header.")
        if args.vocalized_column not in reader.fieldnames:
            raise KeyError(
                f"Column {args.vocalized_column!r} not found. Available: {reader.fieldnames}"
            )
        fieldnames = list(reader.fieldnames)
        if args.output_column not in fieldnames:
            fieldnames.append(args.output_column)
        writer = csv.DictWriter(destination, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for row in tqdm(reader, desc="Creating partial vocalization", unit="row"):
            if args.limit is not None and statistics.examined >= args.limit:
                break
            statistics.examined += 1
            vocalized = (row.get(args.vocalized_column) or "").strip()
            if not vocalized:
                statistics.empty += 1
                row[args.output_column] = ""
            else:
                row[args.output_column] = remove_word_final_endings(vocalized)
                statistics.converted += 1
            writer.writerow(row)

    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "statistics": asdict(statistics),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Completed: %s", asdict(statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
