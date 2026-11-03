# Forensic LLM Graph

Experimental master's thesis project on Italian forensic entity recognition.
The first pass combines rules and an Italian spaCy model. The second-pass runner
adds Minerva perplexity and selective Qwen review. Its first real GPU execution
is a technical smoke test; task accuracy has not yet been established.

## Environment

Use Python 3.10–3.13. The baseline uses spaCy 3.8.11 and `it_core_news_lg` 3.8.0.
Both are installed by `requirements.txt`, including the model's fixed download
URL and checksum. No model files are stored in this repository.
The same requirements file includes the libraries for Minerva and Qwen; their
weights are downloaded when the second-pass runner loads them.

```bash
python -m pip install -r requirements.txt
```

The extractor runs on CPU. The GPU can be used for later language-model stages.
The initial 130-document baseline has run on Colab with Python 3.13 and these
versions. The expanded corpus uses the same model and dependencies.

## Input

Download the configured sources and prepare their texts:

```bash
python scripts/download_data.py
python scripts/prepare_data.py
```

`configs/data_sources.json` records URLs, file hashes and the synthetic selection.
The downloader checks existing files before reuse, restores missing REDIT TSVs,
and ignores macOS archive metadata. Changed files require explicit `--overwrite`.
The synthetic subset is reconstructed from the first 2,000 records of a pinned
source revision, retaining at most five Italian banking examples per template.
It reproduces the same 155 records without downloading the entire 1.3 GB source.
Hash mismatches stop execution rather than silently changing the corpus.

The prepared collection contains 126 REDIT documents, seven ArchivioAntimafia
PDFs (198 pages), 10,363 DICE articles and 155 synthetic diagnostic texts:
10,651 documents overall. DICE contributes 10,395 source records before excluding
21 empty texts and 11 exact text duplicates. This is a working corpus, not an
independently annotated final test set.

This creates four local files in `data/processed/`:

- `documents.json`: complete document texts and source metadata for extraction.
- `redit_annotations.json`: original token-level annotation fields, stored separately.
- `synthetic_annotations.json`: original synthetic entities and a banking-only mapping,
  when the optional subset is present; these require review before evaluation.
- `preparation_report.json`: document/page counts, extraction version and warnings.

REDIT text is reconstructed from token offsets, filling gaps with spaces. Its
source labels and relation fields are preserved without conversion to a new gold
standard. PDF text is extracted as stored, including existing OCR errors, and
pages are joined with two newlines. Page boundaries are recorded in the joined
text. No dates, names or spelling are corrected. Empty or low-text pages are
reported; this does not detect all extraction problems. OCR is not performed.

When present, the DICE CSV is decoded as UTF-8 or Windows-1252. Empty article
texts and exact text duplicates are excluded, with their identifiers and reasons
recorded in the preparation report. Whitespace normalization is used only for
duplicate comparison. The retained article text is not changed; source URL,
publisher crime category and dates remain metadata, not reference annotations.
All retained articles are included, without selection based on model predictions.
Near-duplicate articles and related events still require grouping before a final
development/test split. If the CSV is absent, its omission is reported explicitly.

The optional `data/raw/rizzo_banking_subset.json` contains an explicitly synthetic
Italian diagnostic collection with its dataset revision and selection procedure.
Its original texts, source line identifiers and template groups are preserved.
No source annotations enter the extractor input. Synthetic results must be
reported separately; template groups must remain together in any development/test
split. The current subset is a bounded, template-balanced sample, not a random
sample or proof of performance on real forensic documents. Missing optional data
are explicitly recorded; the pipeline does not download them automatically.

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
Texts are processed through `nlp.pipe` with `spacy_batch_size` set to eight in the
configuration. Results retain input order and original offsets. The same spaCy
components remain enabled as in the earlier sequential runs, including POS tags.
Batching is an execution change; its effect on elapsed time must be measured.

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

Open [the first-pass notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/first_pass_colab.ipynb)
after pushing the changes to GitHub. It contains the complete ordered setup,
download, preparation, extraction and inspection cells. Use a CPU runtime for
this stage. No perplexity or LLM component runs yet.

The equivalent commands, after cloning the repository and installing requirements,
are:

```python
!python scripts/download_data.py
!python scripts/prepare_data.py
!python scripts/run_pipeline.py --output outputs/05_batched_form_boundaries_spacy_lg.json
```

The setup installs the model in the Colab environment. It does not need to be
downloaded again for every extraction in the same session. Download the output
manually from the Colab file panel to retain it after the runtime ends.
For a recorded experiment, check out the intended Git commit before execution.
If the session is still active after a short local change, push the code and run
`git pull --ff-only` before rerunning the necessary cells. A new session requires
the complete setup again. Preserve the output before disconnecting for longer
local analysis or code changes.

## Checks

```bash
python -m unittest discover -s tests -v
```

These checks run offline and require no spaCy model. They cover reuse of matching
files, protection of changed data, checksum failure, selective ZIP extraction and
deterministic synthetic selection. Temporary test files are created under the
ignored `outputs/` directory and removed afterwards.
Additional checks cover personal-field boundaries, address prefixes, original
dates, preprocessed NLP input and batch output integrity. Model stand-ins test
the integration logic; recognition quality and real batch execution require the
installed model on Colab.

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
Common form terms predicted as names are flagged even when their POS tag is
PROPN. This vocabulary is an explicit development heuristic and may also flag
legitimate names; candidates are not discarded solely on that basis.
The context `sottoscritto/a ... nato/a` or an M/F field can suggest a person span.
When its boundaries agree with an NLP span, apart from an M/F suffix, the
form-based PERSON candidate takes precedence over a conflicting name prediction.
The change is flagged and the original prediction is retained in
`alternative_predictions`. Original spaCy labels are recorded as `nlp_label`.
Adjacent `Nome` and `Cognome` fields produce separate PERSON values, excluding
their labels. Conflicting NLP spans are replaced only when the surrounding text
contains field labels, separators or an M/F suffix; unexplained words remain
flagged for review. Residence prefixes such as `residente in` are excluded from
LOCATION spans and the boundary adjustment is recorded.
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
The effective batch size and enabled components are recorded in each run.
The `by_source` field reports counts separately for each collection, including
the synthetic diagnostics. Progress is printed every 100 documents.
Generated JSON files use LF line endings on both Windows and Colab to avoid
platform-dependent input hashes caused by serialization alone.

## Second pass

Install the shared `requirements.txt` in a T4 Colab runtime. It includes both
stages and reuses the installed CUDA PyTorch when compatible. Minerva's gated
repository requires acceptance on Hugging Face; provide a permitted read token
through the Colab Secret named `HF_TOKEN`, not a notebook cell or Git file.

Open `notebooks/second_pass_colab.ipynb` from GitHub and upload the saved run 05
JSON through the Colab file panel. The notebook reuses its complete texts and
candidates rather than downloading raw data and rerunning spaCy. It selects two
passages per source automatically for the first technical test:

```bash
python scripts/run_hybrid.py --input /content/05_batched_form_boundaries_spacy_lg.json --smoke-test
```

`configs/hybrid.json` pins both model revisions. Minerva-350M scores original
passage text, excluding padding. It is released before Qwen3-8B loads with NF4
weights and FP16 computation. Qwen's thinking mode is disabled. Model loading,
GPU memory, Linux peak process RAM, token counts and generation times are saved.
No training or fine-tuning is performed.

Passages have disjoint 1,200-character cores and 200 characters of surrounding
context. The code preserves every character and assigns entities by their start
offset. Qwen returns exact strings and occurrence indices; Python resolves the
original offsets. Only a fully valid, non-truncated response can replace the
core's original candidates. Failed responses retain the first-pass candidates
and remain visible in the output. Syntactic validity and exact text matching
do not establish semantic correctness. Original first-pass files stay intact.

The smoke test sends all eight sampled passages to Qwen to test execution across
collections. It also records what the diagnostic routing policy would select;
it is not a quality evaluation or a selective-processing benchmark. Only the
represented documents appear in this output, with their complete original text
and the remaining first-pass candidates. It does not claim the full corpus has
received LLM review.

The runner can prepare passages offline, without model dependencies:

```bash
python scripts/run_hybrid.py --prepare-only --smoke-test --output outputs/hybrid_preview.json
```

For later development analysis, `--score-only` calculates perplexity without
loading Qwen. An extended review requires a fixed `perplexity_threshold` chosen
on development data. The sample's diagnostic 80th percentile is available only
in smoke mode. Routing combines review signals, possible omissions and high
perplexity, capped by an explicit fraction and passage budget. Current limits
are provisional engineering settings, not validated confidence thresholds.

Completed scores and responses are flushed to a companion `.checkpoints.jsonl`
file. If a cell is interrupted, rerun the same command with `--resume`. Recovery
requires matching input bytes, code, model settings, scope and inference-library
versions. An existing final result is protected; choose a new output or use
`--overwrite` intentionally. Save the checkpoint manually before deleting the
runtime if recovery in another session is needed. No automatic download cell
is included.

The broader account-context rules (`sul conto`, `numero di conto`, `conto postale`)
take effect in newly generated first-pass runs. Reusing run 05 deliberately
preserves that run's original predictions for this model-loading check.
