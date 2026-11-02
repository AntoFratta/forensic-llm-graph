# Forensic LLM Graph

Experimental master's thesis project on Italian forensic entity recognition.
The current implementation is the first pass: rules and an Italian spaCy model.
Selective LLM processing and perplexity will be added in later stages.

## Environment

Use Python 3.10–3.13. The baseline uses spaCy 3.8.11 and `it_core_news_lg` 3.8.0.
Both are installed by `requirements.txt`, including the model's fixed download
URL and checksum. No model files are stored in this repository.

```bash
python -m pip install -r requirements.txt
```

The extractor runs on CPU. The GPU can be used for later language-model stages.
The installation has not yet been verified in a live Colab session.

## Input

Place the extracted REDIT `data-tsv/` directory and the four ArchivioAntimafia
PDFs in `data/raw/`, then run:

```bash
python scripts/prepare_data.py
```

This creates three local files in `data/processed/`:

- `documents.json`: complete document texts and source metadata for extraction.
- `redit_annotations.json`: original token-level annotation fields, stored separately.
- `preparation_report.json`: document/page counts, extraction version and warnings.

REDIT text is reconstructed from token offsets, filling gaps with spaces. Its
source labels and relation fields are preserved without conversion to a new gold
standard. PDF text is extracted as stored, including existing OCR errors, and
pages are joined with two newlines. Page boundaries are recorded in the joined
text. No dates, names or spelling are corrected. Empty or low-text pages are
reported; this does not detect all extraction problems. OCR is not performed.

The input to the extractor, `documents.json`, has this structure:

```json
[
  {
    "id": "example-01",
    "text": "Mario Rossi vive a Roma. Email: nome@example.org.",
    "document": "example.txt",
    "page": 1
  }
]
```

Only `id` and `text` are required. Identifiers must be unique strings. Text is
not normalized. Optional source fields are copied to the results; existing input
annotations are not copied or used by the extractor. Local data and outputs are
excluded from Git. The development material under `research/` is also excluded
and will not be available after cloning the repository on Colab.

## Run the first pass

```bash
python scripts/run_pipeline.py --config configs/baseline.json
```

Paths in the configuration are relative to the repository root. Absolute paths,
including mounted Google Drive paths, are also accepted. The model is loaded once
and reused across documents. A missing or mismatched model stops the run.

For a rules-only check, without installing spaCy:

```bash
python scripts/run_pipeline.py --config configs/baseline.json --rules-only
```

Use `--overwrite` only when intentionally replacing previous results. Otherwise
use `--output` to keep each run separately, without editing the configuration:

```bash
python scripts/run_pipeline.py --output outputs/02_address_review_spacy_lg.json
```

The effective output path is recorded in the run metadata.

## Google Colab

Run these cells in a fresh session after pushing the code to GitHub:

```python
!git clone https://github.com/AntoFratta/forensic-llm-graph.git
%cd /content/forensic-llm-graph
%pip install -r requirements.txt
```

Restart the runtime if requested by the installation, then return to the
repository with `%cd /content/forensic-llm-graph`. The raw documents must also be
present under `data/raw/`. Automatic downloading from the original sources is
not implemented yet. Do not upload credentials into the repository.

```python
!python scripts/prepare_data.py
!python scripts/run_pipeline.py --config configs/baseline.json
```

The setup installs the model in the Colab environment. It does not need to be
downloaded again for every extraction in the same session. Download the output
or use an output path on mounted Drive to retain it after the runtime ends.
For a recorded experiment, check out the intended Git commit before execution.

## Results

`outputs/baseline.json` contains run metadata and per-document candidates. Each
candidate includes its text, label, source, start/end character offsets and any
review reasons. Start is inclusive and end is exclusive, relative to the document:
`text[start:end]` equals the returned mention.

Labels are `PERSON`, `COMPANY`, `ROLE`, `LOCATION`, `EMAIL`, `PHONE`, `TAX_CODE`,
`VAT_NUMBER`, `BANK_ACCOUNT`, and `TIME`. Bank identifiers have a subtype when
recognized. Original dates are retained without interpreting day/month order.

Review flags are heuristic signals, not confidence scores. Unflagged candidates
can still be wrong. Names with no proper-noun POS tag, generic place terms,
legislative abbreviations and possible form fields receive additional review
flags. POS signals are used only when the NLP pipeline provides those tags.
An `ORG` prediction without a recognized legal suffix is retained
as a tentative `COMPANY` candidate with a review flag. A word such as `società`
alone is not sufficient to remove that flag. Administrative prefixes such as
`Comune di` are excluded from geographic spans and the adjustment is flagged.
Generic place words cannot be added as city names when merging addresses.
NLP names fully contained in an address are stored in its `absorbed_entities`
field rather than counted as separate mentions. Other overlaps are retained and
flagged. Digit-count
checks do not validate checksums or establish that an account exists.

The run records the input hash, code revision, uncommitted-change status, model
and library versions, mode, candidate counts and elapsed times. These are
development outputs, not final LLM decisions or precision/recall/F1 scores.
Timing covers the full selected spaCy pipeline, not only its NER component.
Generated JSON files use LF line endings on both Windows and Colab to avoid
platform-dependent input hashes caused by serialization alone.
