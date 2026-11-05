# Forensic LLM Graph

Master's thesis project on named entity recognition (NER) in Italian forensic and legal texts. The current implementation extracts entity mentions with rules and spaCy, then uses a language model to review selected passages. Relation extraction and graph construction are planned for a later stage.

## How it works

1. **Rules and spaCy.** Rules identify structured mentions such as dates, tax codes and bank coordinates. The Italian spaCy model `it_core_news_lg` detects names and places. The extractor records character offsets in the original text and flags candidates with uncertain boundaries, overlaps or ambiguous context.
2. **Perplexity and LLM review.** Minerva-350M scores passages for perplexity. The router combines this signal with first-pass review flags and possible omissions to prioritize passages for Qwen3-8B. The pipeline validates the returned labels and spans against the original text, then merges them with the first-pass candidates. If the review fails validation, the original candidates are retained.

The ten labels are `PERSON`, `COMPANY`, `ROLE`, `LOCATION`, `EMAIL`, `PHONE`, `TAX_CODE`, `VAT_NUMBER`, `BANK_ACCOUNT` and `TIME`. Dates are kept as written, and bank coordinates can include IBANs and other account numbers.

## Corpus

| Collection | Contents | Documents |
| --- | --- | ---: |
| REDIT | Administrative forms with fictitious personal values | 126 |
| ArchivioAntimafia | Italian judicial and investigative PDFs | 7 |
| DICE | Italian crime news articles | 10,363 |
| Rizzo synthetic subset | Generated Italian banking examples | 155 |
| **Total** | | **10,651** |

The downloader retrieves the source files listed in [`configs/data_sources.json`](configs/data_sources.json). Preparation reconstructs text from the REDIT token offsets, extracts the text already embedded in PDFs, and filters empty or duplicate DICE articles. Raw data, generated outputs and reviewed reference files are kept outside Git.

## Run on Google Colab

1. Open the **[rules and spaCy notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/rules_and_spacy_ner_colab.ipynb)** in a CPU runtime. Run the setup, data preparation and extraction cells. Download `outputs/baseline.json` from the Files panel.
2. Open the **[perplexity and LLM review notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/perplexity_llm_review_colab.ipynb)** in a T4 GPU runtime. Upload `baseline.json` to `/content/`. Enable the `HF_TOKEN` Colab Secret with a Hugging Face read token that has access to `sapienzanlp/Minerva-350M-base-v1.0`.
3. In the second notebook, set `run_corpus = True` to process the corpus and download `outputs/hybrid_corpus.json` when it finishes. The reference evaluation sections are optional and require a matching, human-reviewed reference JSON.

The notebooks clone the repository and install the dependencies. Model files are downloaded into the runtime. The recorded GPU pass took about one hour on a T4 with cached weights. Allow additional time for the initial model downloads.

### Command-line equivalents

Run these commands from the repository root with Python 3.10–3.13.

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/run_pipeline.py --output outputs/baseline.json
python scripts/run_hybrid.py --input outputs/baseline.json --output outputs/hybrid_corpus.json
```

The first pass runs on CPU. The hybrid command requires CUDA and access to Minerva through `HF_TOKEN`. Both stages write JSON with run metadata, original document text and entity spans. The hybrid output also records passage scores, routing decisions and review status.

### Evaluate saved results

Use the optional evaluation sections in the second notebook with the first-pass JSON, saved hybrid predictions and matching annotations. Scoring saved predictions runs on CPU without a Hugging Face token.

<details>
<summary>Example reference JSON</summary>

This example assumes the first-pass result contains document `example-01` from source `example` with the exact text shown below. Use the passage ID and boundaries written by `python scripts/run_hybrid.py --prepare-only --output outputs/passage_preview.json`. Offsets are zero-based, document-relative character positions with exclusive end offsets. Compute `text_sha256` over the passage text encoded as UTF-8.

```json
{
  "schema_version": 1,
  "split": "test",
  "split_frozen": true,
  "core_chars": 800,
  "context_chars": 160,
  "units": [
    {
      "id": "example-01@0:18",
      "reference_id": "example-01",
      "document_id": "example-01",
      "source": "example",
      "split": "test",
      "annotation_status": "reviewed",
      "human_review": true,
      "start": 0,
      "core_start": 0,
      "core_end": 18,
      "end": 18,
      "text": "Mario vive a Roma.",
      "text_sha256": "7e726d5cdd334db8313f40e3fcb883b6ff5bd9a5d78a2bb3ea45313a010c50af",
      "entities": [
        {"start": 0, "end": 5, "label": "PERSON", "text": "Mario"},
        {"start": 13, "end": 17, "label": "LOCATION", "text": "Roma"}
      ]
    }
  ]
}
```

</details>

## Experimental results

### Held-out reference

The controlled comparison used 12 human-reviewed excerpts with 137 annotated mentions. A separate set of 12 excerpts was used for development. A prediction counts as correct only if its start, end and label match the reference. Micro precision, recall and F1 pool the entity counts across all 12 test excerpts and all labels.

| Method | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Rules and spaCy | 63.70% | 62.77% | 63.24% |
| Selective Qwen review with perplexity | 64.44% | 63.50% | 63.97% |
| Selective Qwen review without perplexity | 62.14% | 63.50% | 62.82% |

Both hybrid variants had the same review budget and selected three excerpts each. With perplexity, F1 was 0.73 percentage points higher than the baseline on this sample. The small reference limits how far this result can be generalized.

### Full-corpus execution

Minerva scored all **28,581 passages** from the **10,651 documents**. The review budget selected **100 passages** for Qwen, 31 from ArchivioAntimafia and 69 from DICE. Qwen returned 81 complete, 18 partial and one invalid response after validation.

The global selection did not include any of the 12 annotated test excerpts, so their precision, recall and F1 remained at the baseline values in the table. The accuracy of the globally selected passages was outside the coverage of the annotated reference.

The reported comparisons use a saved first-pass result from commit [`59ddcfe`](https://github.com/AntoFratta/forensic-llm-graph/commit/59ddcfe77dfd4f03beb3801e8088d049a07c9e0c) and the hybrid method from commit [`b20a42b`](https://github.com/AntoFratta/forensic-llm-graph/commit/b20a42ba6620948c0f881bdcc2181aabb232e93b). Reproducing the recorded evaluation requires the saved first-pass JSON and reviewed reference. Running the notebooks from the current checkout generates new predictions.

## Repository layout

| Path | Contents |
| --- | --- |
| [`configs/`](configs/) | Data sources, model revisions and routing settings |
| [`scripts/`](scripts/) | Data preparation, extraction and evaluation entry points |
| [`src/`](src/) | Entity rules, passage construction, model loading, validation and scoring |
| [`notebooks/`](notebooks/) | Colab workflows |
| [`tests/`](tests/) | Automated tests |

Run the tests with `python -m unittest discover -s tests`.
