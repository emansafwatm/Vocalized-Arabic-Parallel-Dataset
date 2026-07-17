#!/usr/bin/env python3
"""Filter Arabic-English sentence pairs using multilingual sentence similarity.

The script streams a Hugging Face dataset, removes exact duplicate pairs, computes
row-wise cosine similarity in batches, and writes accepted/rejected TSV files plus
machine-readable run metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

LOGGER = logging.getLogger("parallel_filter")
DEFAULT_DATASET = "ymoslem/UN-Arabic-English-Filtered"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TSV_FIELDS = ["source_index", "text_en", "text_ar", "similarity", "decision_reason"]


@dataclass
class RunStatistics:
    """Counters collected during one filtering run."""

    examined: int = 0
    accepted: int = 0
    rejected: int = 0
    duplicates: int = 0
    invalid: int = 0
    errors: int = 0


def normalize_text(text: object) -> str:
    """Convert a dataset value to normalized single-spacing Unicode text."""

    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def pair_digest(text_en: str, text_ar: str) -> str:
    """Return a stable hash used for exact pair deduplication."""

    payload = f"{text_en}\u241f{text_ar}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SQLiteDeduplicator:
    """Disk-backed exact-pair deduplicator suitable for large corpora."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS seen_pairs (digest TEXT PRIMARY KEY)"
        )
        self.connection.commit()

    def is_new(self, digest: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO seen_pairs(digest) VALUES (?)", (digest,)
        )
        return cursor.rowcount == 1

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "SQLiteDeduplicator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()


def iter_batches(
    rows: Iterable[tuple[int, str, str]], batch_size: int
) -> Iterator[list[tuple[int, str, str]]]:
    """Yield fixed-size batches without loading the entire dataset into memory."""

    batch: list[tuple[int, str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def rowwise_cosine_similarity(
    model: SentenceTransformer,
    english_sentences: Sequence[str],
    arabic_sentences: Sequence[str],
    *,
    encode_batch_size: int,
) -> np.ndarray:
    """Encode both languages and compute aligned cosine similarities."""

    english_embeddings = model.encode(
        list(english_sentences),
        batch_size=encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    arabic_embeddings = model.encode(
        list(arabic_sentences),
        batch_size=encode_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.einsum("ij,ij->i", english_embeddings, arabic_embeddings)


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_metadata(
    output_path: Path,
    *,
    arguments: argparse.Namespace,
    statistics: RunStatistics,
    started_at: datetime,
) -> None:
    metadata = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "statistics": asdict(statistics),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "datasets": package_version("datasets"),
            "sentence-transformers": package_version("sentence-transformers"),
            "numpy": package_version("numpy"),
        },
    }
    output_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter Arabic-English sentence pairs using multilingual SBERT."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--english-column", default="text_en")
    parser.add_argument("--arabic-column", default="text_ar")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--min-score", type=float, default=0.70)
    parser.add_argument("--max-score", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Examples: cpu, cuda, cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download/materialize the split instead of using streaming mode.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Do not remove exact duplicate English-Arabic pairs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files in the output directory.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be between 0 and 1.")
    if not 0.0 <= args.max_score <= 1.0:
        raise ValueError("--max-score must be between 0 and 1.")
    if args.min_score > args.max_score:
        raise ValueError("--min-score cannot exceed --max-score.")
    if args.batch_size < 1 or args.encode_batch_size < 1:
        raise ValueError("Batch sizes must be positive integers.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer.")


def prepare_output_paths(output_dir: Path, overwrite: bool) -> Mapping[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "accepted": output_dir / "accepted_pairs.tsv",
        "rejected": output_dir / "rejected_pairs.tsv",
        "errors": output_dir / "processing_errors.jsonl",
        "metadata": output_dir / "filtering_summary.json",
        "dedup": output_dir / "deduplication.sqlite3",
    }
    protected = list(paths.values())
    existing = [path for path in protected if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Output files already exist: {names}. Use --overwrite to replace them."
        )
    if overwrite:
        for path in paths.values():
            if path.exists():
                path.unlink()
    return paths


def main() -> int:
    args = build_parser().parse_args()
    validate_arguments(args)
    output_paths = prepare_output_paths(args.output_dir, args.overwrite)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    started_at = datetime.now(timezone.utc)
    statistics = RunStatistics()

    LOGGER.info("Loading dataset %s (split=%s)", args.dataset_name, args.split)
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        revision=args.dataset_revision,
        streaming=not args.no_streaming,
    )
    LOGGER.info("Loading sentence-embedding model %s", args.model_name)
    model = SentenceTransformer(args.model_name, device=args.device)

    accepted_handle = output_paths["accepted"].open("w", encoding="utf-8", newline="")
    rejected_handle = output_paths["rejected"].open("w", encoding="utf-8", newline="")
    error_handle = output_paths["errors"].open("w", encoding="utf-8")
    accepted_writer = csv.DictWriter(accepted_handle, fieldnames=TSV_FIELDS, delimiter="\t")
    rejected_writer = csv.DictWriter(rejected_handle, fieldnames=TSV_FIELDS, delimiter="\t")
    accepted_writer.writeheader()
    rejected_writer.writeheader()

    deduplicator_context = (
        SQLiteDeduplicator(output_paths["dedup"])
        if not args.keep_duplicates
        else None
    )

    try:
        def valid_rows() -> Iterator[tuple[int, str, str]]:
            for source_index, example in enumerate(dataset):
                if args.limit is not None and statistics.examined >= args.limit:
                    break
                statistics.examined += 1
                text_en = normalize_text(example.get(args.english_column))
                text_ar = normalize_text(example.get(args.arabic_column))

                if not text_en or not text_ar:
                    statistics.invalid += 1
                    statistics.rejected += 1
                    rejected_writer.writerow(
                        {
                            "source_index": source_index,
                            "text_en": text_en,
                            "text_ar": text_ar,
                            "similarity": "",
                            "decision_reason": "empty_text",
                        }
                    )
                    continue

                if deduplicator_context is not None:
                    digest = pair_digest(text_en, text_ar)
                    if not deduplicator_context.is_new(digest):
                        statistics.duplicates += 1
                        statistics.rejected += 1
                        rejected_writer.writerow(
                            {
                                "source_index": source_index,
                                "text_en": text_en,
                                "text_ar": text_ar,
                                "similarity": "",
                                "decision_reason": "exact_duplicate",
                            }
                        )
                        continue
                yield source_index, text_en, text_ar

        progress = tqdm(desc="Sentence pairs processed", unit="pair")
        for batch in iter_batches(valid_rows(), args.batch_size):
            indices = [row[0] for row in batch]
            english = [row[1] for row in batch]
            arabic = [row[2] for row in batch]
            try:
                scores = rowwise_cosine_similarity(
                    model,
                    english,
                    arabic,
                    encode_batch_size=args.encode_batch_size,
                )
            except Exception as exc:
                statistics.errors += len(batch)
                LOGGER.exception("Embedding failed for a batch beginning at source row %s", indices[0])
                for source_index, text_en, text_ar in batch:
                    error_handle.write(
                        json.dumps(
                            {
                                "source_index": source_index,
                                "text_en": text_en,
                                "text_ar": text_ar,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                progress.update(len(batch))
                continue

            for (source_index, text_en, text_ar), score in zip(batch, scores, strict=True):
                similarity = float(score)
                row = {
                    "source_index": source_index,
                    "text_en": text_en,
                    "text_ar": text_ar,
                    "similarity": f"{similarity:.6f}",
                    "decision_reason": "within_similarity_range",
                }
                if args.min_score <= similarity <= args.max_score:
                    statistics.accepted += 1
                    accepted_writer.writerow(row)
                else:
                    statistics.rejected += 1
                    row["decision_reason"] = (
                        "below_min_score"
                        if similarity < args.min_score
                        else "above_max_score"
                    )
                    rejected_writer.writerow(row)

            if deduplicator_context is not None:
                deduplicator_context.commit()
            accepted_handle.flush()
            rejected_handle.flush()
            error_handle.flush()
            progress.update(len(batch))
        progress.close()
    finally:
        if deduplicator_context is not None:
            deduplicator_context.close()
        accepted_handle.close()
        rejected_handle.close()
        error_handle.close()
        write_metadata(
            output_paths["metadata"],
            arguments=args,
            statistics=statistics,
            started_at=started_at,
        )

    LOGGER.info("Completed: %s", asdict(statistics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
