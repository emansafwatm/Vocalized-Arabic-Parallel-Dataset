# Public Release Checklist

## Required before publication

- [ ] Confirm the exact Hugging Face dataset revision used to produce the reported 584,284 sentence pairs.
- [ ] Run a small end-to-end validation using `--limit 1000`.
- [ ] Run `pytest -q` and confirm all tests pass.
- [ ] Inspect a bilingual sample of accepted and rejected pairs.
- [ ] Inspect a representative sample of Mishkal outputs for vocalization errors.
- [ ] Confirm that no generated corpus files, local paths, credentials, or notebook outputs are committed.
- [ ] Resolve the repository licensing statement before distributing the full processed corpus.

## Licensing issue to resolve

The current GitHub repository displays a `CC0-1.0` repository license, while the previous README described the corpus using a different Creative Commons license. The source Hugging Face dataset is marked `CC BY 4.0`, and Mishkal is GPL-licensed. These layers should be described separately:

1. license for this repository's original code and documentation;
2. license and attribution requirements for the source corpus;
3. permitted access and redistribution terms for the processed corpus;
4. third-party software licenses.

Do not make the complete processed corpus directly downloadable until this has been checked.

## Recommended GitHub description

> Reproducible scripts for semantic filtering and full/partial vocalization of an Arabic-English parallel corpus.

## Recommended topics

`arabic-nlp`, `machine-translation`, `parallel-corpus`, `arabic-diacritization`, `sentence-transformers`, `mishkal`, `dataset-preprocessing`

## Suggested commit sequence

From a local clone of the repository, copy the files from this update package and run:

```bash
git checkout -b public-code-release
git add README.md requirements.txt .gitignore CITATION.cff scripts notebooks tests docs data/sample
git status
git diff --cached
git commit -m "Add reproducible corpus filtering and vocalization pipeline"
git push -u origin public-code-release
```

Review the branch on GitHub, then merge it into `main` after verifying rendered notebooks and links.
