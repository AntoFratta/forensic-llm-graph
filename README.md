# Forensic LLM Graph

This master's thesis project studies named entity recognition (NER) in Italian forensic and legal texts. It tests whether a language model can improve the results of rules and spaCy by reviewing selected passages, and whether perplexity helps choose those passages. A second stage uses the saved NER results to build a graph of people, companies and documents with links back to the original text.

## How it works

### Prepare the documents

The input sources use different formats. The download script retrieves the source files and the preparation script converts them into `data/processed/documents.json`. Each record contains the original or reconstructed text, a document ID and source metadata. Any source annotations are stored separately from the text passed to the extractor. [Download script](scripts/download_data.py) · [Preparation script](scripts/prepare_data.py)

### Extract candidates with rules and spaCy

The first pass processes every document. Rules look for patterns and cues such as the form of an email address or the words that introduce a phone number. The Italian spaCy model uses context to identify people, places and organisations. The extractor maps the model's predictions to the project labels, merges identical predictions and flags candidates with issues such as overlapping spans or an ambiguous organisation label.

The first pass saves the text, candidates and their character offsets in `outputs/baseline.json`. The second pass reads this file to select passages for review. [First-pass script](scripts/run_pipeline.py) · [Extraction rules and spaCy mapping](src/deterministic.py) · [First-pass settings](configs/baseline.json)

The ten labels are `PERSON`, `COMPANY`, `ROLE`, `LOCATION`, `EMAIL`, `PHONE`, `TAX_CODE`, `VAT_NUMBER`, `BANK_ACCOUNT` and `TIME`. Dates are kept as written, and bank coordinates can include IBANs and other account numbers.

### Select passages for review

The second pass splits each document into consecutive pieces of about 800 characters and adds up to 160 characters of surrounding context on each side. This gives the models context near a boundary. A mention is assigned to the piece where it starts so that neighbouring passages can share context while each occurrence is counted once.

Minerva assigns a probability to each successive token, such as a word or part of a word, given the text before it. Perplexity summarises these probabilities across the passage. The higher the perplexity, the less predictable the passage was for Minerva.

The pipeline considers perplexity alongside first-pass flags and possible omissions. An overlap between two proposed entities can give a passage higher priority than high perplexity alone. Scores at or above the threshold of about 42.77 raise a passage's priority, and perplexity also breaks ties. The thresholds tried during development selected the same passages, so the 80th percentile of the development scores was used as the fixed threshold for the test.

The review budget is 20% of **all passages being considered**, rounded up and capped at 100. The system fills that budget with the highest-priority passages that have a reason for review. For example, with 1,000 passages the budget is 100. If only 80 are flagged for review, Qwen receives 80. [Passage construction and selection](src/passages.py) · [Hybrid settings](configs/hybrid.json)

### Review the selected text with Qwen

Qwen receives the original passage and instructions about the allowed categories. It proposes its own mentions before the program compares them with the first-pass candidates. Each proposal includes the mention, its label and a short quotation that helps locate the correct occurrence. The program checks the response format, label and matching text before applying any changes.

For example, the first pass might propose `direttore` as `ROLE` and `direttore Mario Rossi` as `PERSON`. Qwen can propose `direttore` and `Mario Rossi` as separate mentions, allowing the program to correct the flagged name boundary.

A response that passes validation can replace flagged candidates while preserving unflagged candidates that Qwen omits. A partially valid response can add verified mentions while retaining the first-pass proposals. For passages with an unusable response or outside the review selection, the program keeps the first-pass candidates. The second pass saves the combined mentions, selection decisions and review status in `outputs/hybrid_corpus.json`. [Second-pass script](scripts/run_hybrid.py) · [Model execution](src/language_models.py) · [Response validation and merging](src/llm_filter.py)

## Models

| Model | Role | Official page |
| --- | --- | --- |
| `it_core_news_lg` 3.8.0 | Italian spaCy entity recognition | [spaCy model release](https://github.com/explosion/spacy-models/releases/tag/it_core_news_lg-3.8.0) |
| `sapienzanlp/Minerva-350M-base-v1.0` | Passage perplexity | [Hugging Face model card](https://huggingface.co/sapienzanlp/Minerva-350M-base-v1.0) |
| `Qwen/Qwen3-8B` | Entity review on selected passages | [Hugging Face model card](https://huggingface.co/Qwen/Qwen3-8B) |

Downloading the Minerva weights requires accepting the access conditions on Hugging Face.

## Corpus

| Collection | Contents | Documents | Original source |
| --- | --- | ---: | --- |
| REDIT | Administrative forms with fictitious personal values | 126 | [Project page](https://github.com/dhfbk/redit) and [TSV archive](https://github.com/dhfbk/redit/blob/main/data-tsv.zip) |
| ArchivioAntimafia | Italian judicial and investigative PDFs | 7 | [Archive website](https://www.archivioantimafia.org/) and individual PDFs below |
| DICE | Italian crime news articles | 10,363 | [Dataset description](https://github.com/federicarollo/Italian-Crime-News), [original CSV](https://github.com/federicarollo/Italian-Crime-News/blob/main/italian_crime_news.csv) and [annotated article files](https://github.com/federicarollo/Italian-Crime-News/tree/main/annotation%20csv) |
| Rizzo synthetic subset | Generated Italian banking examples | 155 | [Hugging Face dataset card](https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset). The downloader builds the subset from `synthetic/synthetic_pii_it_200k.jsonl`. |
| **Total** | | **10,651** | |

<details>
<summary>Read the seven ArchivioAntimafia PDFs used in the corpus</summary>

- [Andreotti–Pecorelli case, Court of Cassation ruling](https://www.archivioantimafia.org/sentenze2/andreotti/andreotti_pecorelli_cassazione.pdf)
- [Giuseppe Ayala testimony](https://www.archivioantimafia.org/verbali/ayala_deposizione_falcone.pdf)
- [Gaspare Giudice case, parliamentary debate](https://www.archivioantimafia.org/sentenze2/giudice/giudice_dibattito_in_aula.pdf)
- [Gaspare Giudice case, committee report](https://www.archivioantimafia.org/sentenze2/giudice/giudice_relazione_giunta.pdf)
- [Berlusconi, Dell’Utri and Confalonieri wiretap transcript](https://www.archivioantimafia.org/altri_documenti/intercettazione_berlusconi_dellutri_confalonieri.pdf)
- [Mafia and public contracts, request for dismissal](https://www.archivioantimafia.org/sentenze2/mafia_appalti_archiviazione.pdf)
- [Vincenzo Scarantino interrogation record, 24 June 1994](https://www.archivioantimafia.org/altri_documenti/scarantino/scarantino_240694.pdf)

</details>

The REDIT TSVs are distributed in a ZIP archive. The full DICE CSV exceeds GitHub’s preview limit and is available through the raw file link on its page.

The downloader retrieves the source files listed in [`configs/data_sources.json`](configs/data_sources.json). Preparation reconstructs text from the REDIT token offsets, extracts the text already embedded in PDFs, and filters empty or duplicate DICE articles. Raw data, generated outputs and reviewed reference files are kept outside Git.

## Run on Google Colab

1. Open the **[rules and spaCy notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/rules_and_spacy_ner_colab.ipynb)** in a CPU runtime. Run the setup, data preparation and extraction cells. Download `outputs/baseline.json` from the Files panel.
2. Open the **[perplexity and LLM review notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/perplexity_llm_review_colab.ipynb)** in a T4 GPU runtime. Upload `baseline.json` to `/content/`. Enable the `HF_TOKEN` Colab Secret with a Hugging Face read token that has access to `sapienzanlp/Minerva-350M-base-v1.0`.
3. In the second notebook, set `run_corpus = True` to process the corpus and download `outputs/hybrid_corpus.json` when it finishes. The reference evaluation sections are optional and require a matching, human-reviewed reference JSON.

The notebooks clone the repository, install the dependencies and download the model files into the runtime. Allow additional time for the initial model downloads.

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

Use the optional evaluation sections in the second notebook with the first-pass JSON, saved hybrid predictions and matching annotations. Scoring saved predictions runs on CPU without a Hugging Face token. To generate predictions for a new annotated comparison, set `run_reference_evaluation = True` and `generate_reference_predictions = True` in the second notebook on a T4 runtime. The notebook evaluates the same passages under both selection strategies.

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

### Held-out test

The evaluation uses a reference set of passages whose entity categories and boundaries were reviewed manually. During development, 12 excerpts with 90 mentions were used to refine the method. The final comparison used a separate set of 12 excerpts with 137 annotated mentions. The test gives each source three excerpts, while about 97% of the full corpus documents come from DICE. The three judicial excerpts came from one case.

Three methods were compared on the same 12 test excerpts.

1. **Rules and spaCy.** The saved first-pass predictions provide the baseline.
2. **Selective Qwen review without perplexity.** The selection uses first-pass flags and possible omissions.
3. **Selective Qwen review with perplexity.** The selection also uses perplexity to raise priorities and break ties.

Each hybrid method selected three excerpts under the same budget. They selected two excerpts in common and shared the Qwen responses for those excerpts, so Qwen reviewed four distinct excerpts in total. The highest perplexity in the test was 37.92, below the threshold of 42.77. Perplexity changed which excerpts were selected by breaking ties.

A prediction counts as correct only when its start, end and label match the reference. For example, `Mario` as `PERSON` is a boundary error when the reference is `Mario Rossi`. **Precision** is the share of predictions that are correct. **Recall** is the share of annotated mentions that were found. **F1** balances precision and recall. The reported scores pool the entity counts across all test excerpts and labels before computing the metrics. [Evaluation script](scripts/evaluate.py) · [Exact-match scoring](src/evaluation.py)

| Method | Correct predictions | Extra predictions | Missed mentions | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rules and spaCy | 86 | 49 | 51 | 63.70% | 62.77% | 63.24% |
| Selective Qwen review with perplexity | 87 | 48 | 50 | 64.44% | 63.50% | 63.97% |
| Selective Qwen review without perplexity | 87 | 53 | 50 | 62.14% | 63.50% | 62.82% |

Compared with the baseline, the hybrid method with perplexity found one more correct mention and produced one fewer extra prediction. This raised F1 by **0.73 percentage points**. The two hybrid methods found the same number of correct mentions, with five fewer extra predictions when perplexity was used. The result is a preliminary comparison. A broader claim about improvement would require more annotated excerpts from different cases.

### Full-corpus execution

The full run checked how the pipeline operates when passages from the entire corpus compete for the same review budget. Minerva scored all **28,581 passages** from **10,651 documents**. Qwen reviewed the 100 highest-priority passages, about 0.35% of the total. Of these, 31 were from ArchivioAntimafia and 69 from DICE. After validation, 81 reviews were fully usable, 18 were partially usable and one was unusable. The second pass took about one hour on a Tesla T4 with model files already cached, including about 51 minutes of Qwen generation.

The global selection did not include any of the 12 annotated test excerpts, so their precision, recall and F1 remained at the baseline values in the table. The full run demonstrates execution at corpus scale. The quality comparison comes from the separately annotated test excerpts, and the accuracy of the 100 globally selected passages remains unmeasured.

The reported comparisons use a saved first-pass result from commit [`59ddcfe`](https://github.com/AntoFratta/forensic-llm-graph/commit/59ddcfe77dfd4f03beb3801e8088d049a07c9e0c) and the hybrid method from commit [`b20a42b`](https://github.com/AntoFratta/forensic-llm-graph/commit/b20a42ba6620948c0f881bdcc2181aabb232e93b). Recalculating the exact scores from those runs requires the saved first-pass and hybrid predictions and the reviewed reference. Running the notebooks from the current checkout generates new predictions.

## Build and evaluate the relation graph

The graph has document nodes and subject nodes for people and companies. It records where a subject is mentioned, where two subjects occur in the same passage, and where the text states a directed relationship. The explicit extractor covers selected expressions of employment, roles and legal representation. Every edge stores evidence offsets in the input document. Subject names are grouped within each document by normalised spelling and label. [Graph construction](src/relation_graph.py) · [Relation patterns](src/explicit_relations.py)

Build the rules-based graph from saved NER results. Use an output name that identifies the run.

```bash
python scripts/build_relation_graph.py --input outputs/hybrid_corpus.json --output outputs/relation_graph_rules_RUN_ID.json
```

To add Qwen proposals, prepare a review plan. Supplying `--reference` limits the plan to eligible excerpts from a saved evaluation sample. The plan uses the excerpt text and offsets. For a corpus run, omit `--reference` and set the passage budget with `--max-passages`.

```bash
python scripts/review_relation_graph.py prepare --input outputs/hybrid_corpus.json --reference outputs/relation_reference_RUN_ID.json --output outputs/relation_plan_RUN_ID.json
```

Upload the plan to the [relation review notebook](notebooks/qwen_relation_review_colab.ipynb) on a Colab T4 runtime. Download the review JSON and use its generated filename in the build command. The builder verifies that the review belongs to the same NER input and resolves the proposed quotations against the original text.

```bash
python scripts/build_relation_graph.py --input outputs/hybrid_corpus.json --relation-reviews outputs/relation_reviews_RUN_ID.json --output outputs/relation_graph_qwen_RUN_ID.json
```

Score each graph against the same reviewed reference. The scorer counts a relation as correct when the direction, endpoint labels, and all three spans match exactly.

```bash
python scripts/evaluate_relation_graph.py --graph outputs/relation_graph_rules_RUN_ID.json --reference outputs/relation_reference_RUN_ID.json --output outputs/relation_evaluation_rules_RUN_ID.json
python scripts/evaluate_relation_graph.py --graph outputs/relation_graph_qwen_RUN_ID.json --reference outputs/relation_reference_RUN_ID.json --output outputs/relation_evaluation_qwen_RUN_ID.json
```

## Repository layout

| Path | Contents |
| --- | --- |
| [`configs/`](configs/) | Data sources, model revisions and routing settings |
| [`scripts/`](scripts/) | Data preparation, extraction and evaluation entry points |
| [`src/`](src/) | Entity rules, passage construction, model loading, validation and scoring |
| [`notebooks/`](notebooks/) | Colab workflows |
| [`tests/`](tests/) | Automated tests |

Run the tests with `python -m unittest discover -s tests`.
