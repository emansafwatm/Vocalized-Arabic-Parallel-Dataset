# Supervisor Review Guide

This repository exposes the complete preprocessing logic used to construct the Vocalized Arabic-English Parallel Corpus. The public code is separated into three auditable stages:

1. **Semantic filtering**: streams the source Arabic-English corpus, removes exact duplicate pairs, computes multilingual Sentence-BERT cosine similarity in batches, and retains pairs inside a configurable score interval.
2. **Full vocalization**: applies the Mishkal Arabic vocalization system to the Arabic side and records failures explicitly.
3. **Partial vocalization**: derives a second Arabic representation by removing word-final inflectional marks while retaining internal diacritics and shadda.

## Review priorities

- Thresholds, model identifiers, source dataset revision, and package versions are written to JSON summaries.
- Generated data is not committed by default.
- Exceptions are logged; no rows are silently discarded.
- TSV output preserves sentence alignment and avoids Python-list serialization.
- Unit tests cover the transformation used for partial vocalization.

## Reproducibility note

For exact reproduction of a published corpus count, provide the specific Hugging Face dataset revision used in the experiment through `--dataset-revision`. Source datasets and software packages can change after an experiment is completed.

## Methodological limitation

Sentence-embedding similarity is an automatic filtering signal rather than a substitute for bilingual human validation. Mishkal is a rule-based vocalizer, so its outputs can contain lexical, morphological, or syntactic errors. These limitations should be stated in any paper or dataset card that uses the generated corpus.
