Vocalized Arabic-English Parallel Dataset

[![Tests](https://github.com/emansafwatm/Vocalized-Arabic-Parallel-Dataset/actions/workflows/tests.yml/badge.svg)](https://github.com/emansafwatm/Vocalized-Arabic-Parallel-Dataset/actions/workflows/tests.yml)

A reproducible preprocessing pipeline for constructing an Arabic-English parallel corpus with **nonvocalized**, **fully vocalized**, and **partially vocalized** Arabic variants.

The current corpus contains **584,284 aligned sentence pairs** selected through semantic filtering. The code in this repository documents the filtering and Arabic vocalization workflow so that the methodology can be inspected, reproduced, and extended.

## Dataset variants

Each retained English sentence is aligned with three Arabic representations:

- **Nonvocalized Arabic**: standard Arabic text without diacritics.
- **Fully vocalized Arabic**: Arabic text vocalized with Mishkal.
- **Partially vocalized Arabic**: internal diacritics are retained, while word-final inflectional marks are removed. Terminal shadda is preserved.

## Source data and processing

The pipeline starts from [`ymoslem/UN-Arabic-English-Filtered`](https://huggingface.co/datasets/ymoslem/UN-Arabic-English-Filtered), an Arabic-English dataset derived from MultiUN and UN Parallel Corpus resources. The source dataset card identifies `text_en` and `text_ar` as the aligned columns.

Semantic similarity is calculated with [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). The default public script retains sentence pairs with cosine similarity from **0.70 through 0.99**, inclusive. Thresholds are command-line parameters and are recorded in the run summary.

Arabic vocalization is generated with [`mishkal`](https://github.com/linuxscout/mishkal), a rule-based Arabic vocalization system.

## Repository structure

```text
.
├── notebooks/
│   ├── Filtering.ipynb
│   ├── Mishkal.ipynb
│   └── Tashkeel.ipynb
├── scripts/
│   ├── filter_parallel_corpus.py
│   ├── vocalize_with_mishkal.py
│   └── create_partial_vocalization.py
├── tests/
│   └── test_arabic_diacritics.py
├── docs/
│   └── SUPERVISOR_REVIEW.md
├── data/sample/sample_pairs.tsv
├── requirements.txt
└── CITATION.cff
```

The notebooks provide readable demonstrations. The scripts contain the reusable and auditable implementation.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Filter the parallel corpus

A small validation run:

```bash
python scripts/filter_parallel_corpus.py \
  --limit 1000 \
  --output-dir data/processed/filter_demo \
  --overwrite
```

Full run:

```bash
python scripts/filter_parallel_corpus.py \
  --dataset-name ymoslem/UN-Arabic-English-Filtered \
  --split train \
  --min-score 0.70 \
  --max-score 0.99 \
  --batch-size 128 \
  --encode-batch-size 64 \
  --output-dir data/processed/filter_full \
  --overwrite
```

For exact experimental reproducibility, also pass the source revision used in the original run:

```bash
--dataset-revision <HUGGING_FACE_COMMIT_HASH>
```

Outputs:

- `accepted_pairs.tsv`
- `rejected_pairs.tsv`
- `processing_errors.jsonl`
- `filtering_summary.json`
- `deduplication.sqlite3`

The filter uses streaming mode by default so the complete source split is not loaded into a pandas DataFrame. Exact duplicate detection is disk-backed, and embedding inference is batched.

## 2. Generate fully vocalized Arabic

```bash
python scripts/vocalize_with_mishkal.py \
  --input data/processed/filter_full/accepted_pairs.tsv \
  --output data/processed/vocalized_pairs.tsv \
  --overwrite
```

The output preserves all original columns and adds `arabic_vocalized`. Failures are written to a separate JSONL file instead of being silently ignored.

## 3. Generate partially vocalized Arabic

```bash
python scripts/create_partial_vocalization.py \
  --input data/processed/vocalized_pairs.tsv \
  --output data/processed/final_parallel_corpus.tsv \
  --overwrite
```

The output adds `arabic_partially_vocalized`.

Example:

```text
Fully vocalized:    الطَّالِبُ مُجْتَهِدٌ
Partially vocalized: الطَّالِب مُجْتَهِد
```

## Testing

```bash
pytest -q
```

The tests verify that the partial-vocalization step removes final vowel/tanween marks, preserves internal diacritics, preserves terminal shadda, and does not alter non-Arabic tokens.

## Reproducibility and quality controls

- UTF-8 TSV is used to preserve sentence alignment safely.
- Existing outputs are not overwritten unless `--overwrite` is supplied.
- Empty rows, duplicates, rejected scores, and processing failures are counted separately.
- Filtering parameters, timestamps, package versions, and runtime environment are saved in JSON summaries.
- The filtering stage does not silently skip failed batches.
- The repository excludes generated corpus files by default to avoid unintentionally publishing restricted or very large data.

## Limitations

Semantic similarity is an automatic quality-control signal and does not guarantee that every retained pair is a perfect translation. Mishkal is a rule-based vocalization system and may produce lexical, morphological, or syntactic errors. Human evaluation is recommended for benchmark subsets and high-stakes downstream use.

The exact number of retained pairs may change when the source dataset revision, sentence-transformer version, preprocessing rules, or score thresholds change.

## Data access

The complete processed corpus is available upon reasonable research request. Contact: `emansafwatm@gmail.com`.

Do not commit or redistribute the full corpus until the licensing and redistribution conditions of the source datasets have been verified.

## Licensing

- The repository currently includes a `CC0-1.0` license for the original repository materials.
- The source dataset is listed as `CC BY 4.0` on its Hugging Face dataset card.
- Mishkal is distributed under the GNU GPL.
- The processed corpus may also be subject to source-corpus terms. Verify redistribution rights before making the full data public.

## Citation

If you use this dataset or preprocessing pipeline, please cite the associated paper:

```bibtex
@inproceedings{khater2025vocalized,
  title     = {Vocalized Arabic-English Parallel Corpus: A High-Quality Resource for Machine Translation and Diacritization},
  author    = {Khater, Eman and Elemam, Mohamed and Aborezka, Mohamed and Mahar, Khaled},
  booktitle = {2025 35th International Conference on Computer Theory and Applications (ICCTA)},
  pages     = {359--364},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICCTA68914.2025.11520015}
}
```

Paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11520015)

GitHub can also generate citation formats automatically from the repository's `CITATION.cff` file.

Please additionally cite the source corpus, Sentence-BERT model, multilingual MiniLM model, and Mishkal when applicable to the specific experiment.
