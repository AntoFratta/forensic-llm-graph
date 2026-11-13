# Forensic LLM Graph

A master's thesis project on named entity recognition (NER) and relation extraction in Italian forensic and legal texts. The pipeline extracts entity mentions, reviews selected passages with language models and builds a graph whose edges can be traced to the original text.

[Open the end-to-end notebook in Google Colab](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/forensic_llm_graph_colab.ipynb)

## Quick start

1. [Prepare the corpus](#prepare-the-corpus) locally to generate `data/processed/documents.json`.
2. [Get access to Minerva](#models-and-access) and add your Hugging Face token to Colab.
3. [Run the notebook on a T4 GPU](#end-to-end-on-google-colab). Upload only `documents.json`, then download the ZIP containing the NER predictions and relation graph.
4. [Read the outputs](#read-the-generated-outputs) to locate an edge's evidence in the original document.

## How the pipeline works

1. **Extract entities.** Rules and Italian spaCy process every document. The ten labels are `PERSON`, `COMPANY`, `ROLE`, `LOCATION`, `EMAIL`, `PHONE`, `TAX_CODE`, `VAT_NUMBER`, `BANK_ACCOUNT` and `TIME`. Banking mentions include IBANs and other account numbers.
2. **Select and review NER passages.** Minerva measures how surprising each passage is to the model using perplexity. The passages have cores of about 800 characters and up to 160 characters of context on either side. Selection combines perplexity with signals such as overlapping candidates and possible omissions. The review budget is 20% of passages, rounded up and capped at 100. Qwen reviews the highest-priority eligible passages within that budget. The pipeline checks proposed labels and spans against the text before merging them with the first pass.
3. **Build the relation graph.** Rules extract explicit relations from the saved NER output. A separate selection of up to 100 passages with relation cues goes to Qwen. The selection reserves half the slots for eligible judicial passages when available, prioritising coverage across cases. The validator checks that the proposed endpoints and direction are supported by the text. [Passage selection](src/passages.py) · [Relation validation](src/relation_review.py)

The graph contains document nodes and subject nodes for people and businesses. `MENTIONED_IN` links a subject to its document. `CO_OCCURS_WITH` records shared context within a passage. `EXPLICIT_RELATION` records a directed employment, role or legal representation relation. Each edge carries zero-based character offsets with exclusive end positions. The same subject can have separate nodes in different documents because grouping is local to each document. `COMPANY` includes commercial operators whose legal form is unspecified. Public bodies and associations fall outside this category. [Graph construction](src/relation_graph.py) · [Relation patterns](src/explicit_relations.py)

## Corpus and sources

| Collection | Contents | Documents | Source |
| --- | --- | ---: | --- |
| DICE | Italian crime news articles | 10,363 | [Dataset repository](https://github.com/federicarollo/Italian-Crime-News) · [CSV](https://github.com/federicarollo/Italian-Crime-News/blob/main/italian_crime_news.csv) |
| REDIT | Administrative forms with fictitious personal values | 126 | [Project repository](https://github.com/dhfbk/redit) · [TSV archive](https://raw.githubusercontent.com/dhfbk/redit/main/data-tsv.zip) |
| ArchivioAntimafia | Judicial and investigative documents | 7 | [Archive](https://www.archivioantimafia.org/) · [Individual PDFs](#judicial-and-investigative-pdfs) |
| Corte di Cassazione | Public copies of criminal judgments from three cases | 3 | [13016/2024](https://www.ilcaso.it/giurisprudenza/archivio/31062.pdf) · [32560/2025](https://www.studiolegalenoto.it/wp-content/uploads/2025/10/cassazione-penale-sentenza-32560-2025.pdf) · [30039/2025](https://generazione231.it/wp-content/uploads/2025/09/Cassazione-30039.pdf) |
| Rizzo synthetic subset | Generated Italian banking examples selected from `synthetic/synthetic_pii_it_200k.jsonl` | 155 | [Dataset card](https://huggingface.co/datasets/rizzoaiacademy/rizzo-pii-it-dataset) |
| **Total** | | **10,654** | |

### Judicial and investigative PDFs

<details>
<summary>ArchivioAntimafia documents used in the corpus</summary>

- [Andreotti–Pecorelli case, Court of Cassation ruling](https://www.archivioantimafia.org/sentenze2/andreotti/andreotti_pecorelli_cassazione.pdf)
- [Giuseppe Ayala testimony](https://www.archivioantimafia.org/verbali/ayala_deposizione_falcone.pdf)
- [Gaspare Giudice case, parliamentary debate](https://www.archivioantimafia.org/sentenze2/giudice/giudice_dibattito_in_aula.pdf)
- [Gaspare Giudice case, committee report](https://www.archivioantimafia.org/sentenze2/giudice/giudice_relazione_giunta.pdf)
- [Berlusconi, Dell’Utri and Confalonieri wiretap transcript](https://www.archivioantimafia.org/altri_documenti/intercettazione_berlusconi_dellutri_confalonieri.pdf)
- [Mafia and public contracts, request for dismissal](https://www.archivioantimafia.org/sentenze2/mafia_appalti_archiviazione.pdf)
- [Vincenzo Scarantino interrogation record, 24 June 1994](https://www.archivioantimafia.org/altri_documenti/scarantino/scarantino_240694.pdf)

</details>

### Prepare the corpus

Use Git and Python 3.10–3.13. Run corpus preparation on CPU from a local clone.

```bash
git clone https://github.com/AntoFratta/forensic-llm-graph.git
cd forensic-llm-graph
python -m pip install pypdf==6.10.0
python scripts/download_data.py
python scripts/prepare_data.py
```

The downloader creates `data/raw/`, checks the source files against their recorded SHA-256 hashes and reconstructs the synthetic banking subset from a pinned dataset revision. Preparation reconstructs REDIT text from token offsets, extracts embedded PDF text with page offsets and filters empty and duplicate DICE articles. It writes `data/processed/documents.json`, a preparation report and separate files for source annotations. Existing OCR errors and redactions remain in the extracted text. [Source manifest with download URLs and checksums](configs/data_sources.json)

## Models and access

| Model | Use | Source |
| --- | --- | --- |
| spaCy `it_core_news_lg` 3.8.0 | First-pass Italian NER | [Model release](https://github.com/explosion/spacy-models/releases/tag/it_core_news_lg-3.8.0) |
| `sapienzanlp/Minerva-350M-base-v1.0` | Passage perplexity | [Model card and access page](https://huggingface.co/sapienzanlp/Minerva-350M-base-v1.0) |
| `Qwen/Qwen3-8B` | Entity and relation review with 4-bit NF4 weights | [Model card](https://huggingface.co/Qwen/Qwen3-8B) |

The configuration pins the Hugging Face model revisions. [Baseline settings](configs/baseline.json) · [Hybrid settings](configs/hybrid.json) · [Dependencies](requirements.txt)

1. Sign in to Hugging Face and accept the access conditions on the model page. [Minerva access](https://huggingface.co/sapienzanlp/Minerva-350M-base-v1.0)
2. From the same account, create a token with **Read** permissions. [Create an access token](https://huggingface.co/settings/tokens) · [Token documentation](https://huggingface.co/docs/hub/security-tokens)
3. In Colab, open **Secrets** using the key icon. Add a secret named `HF_TOKEN`, paste the token as its value and enable **Notebook access**. [Colab Secrets instructions](https://codelabs.developers.google.com/onramp/instructions?hl=en)

## Run the project

Start with `data/processed/documents.json` from [Prepare the corpus](#prepare-the-corpus). Each run creates its own predictions, relation reviews and graph. The following examples use `01` as the run prefix.

### End-to-end on Google Colab

1. Open the notebook, select **Runtime → Change runtime type → T4 GPU** and connect. [Colab notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/forensic_llm_graph_colab.ipynb)
2. Enable the `HF_TOKEN` secret as described in [Models and access](#models-and-access). Upload your generated `data/processed/documents.json` through the **Files** panel. The notebook reads it from `/content/documents.json`.
3. In the setup cell, change `run_number` to `"01"` to follow the filenames below. Select **Runtime → Run all**. The notebook clones the code, prints the Git revision, installs its dependencies and runs both NER passes, the relation review and graph construction.
4. Download `/content/01_run.zip` from the **Files** panel. Extract its contents into `outputs/` in your local clone, creating the folder if needed.

If the runtime still holds its files after an interruption, rerun with the same number. Completed stages are reused and the hybrid NER stage can resume from its checkpoint. An interrupted relation review restarts from the saved plan. Use a fresh prefix for a new run.

### Read the generated outputs

The filenames below are for run `01`. They share the prefix selected in the notebook.

| File | Contents |
| --- | --- |
| `01_ner_baseline.json` | Document text and entity candidates from rules and spaCy |
| `01_ner_corpus.json` | Merged NER predictions, passage scores and review decisions |
| `01_relation_plan.json` | Passages selected for relation review |
| `01_relation_reviews.json` | Saved Qwen responses and validation statuses |
| `01_relation_graph.json` | Nodes and edges with text evidence |

Open `01_relation_graph.json` to inspect the `nodes` and `edges` arrays. Each edge's `source` and `target` are node IDs. To read a supporting passage, take an entry from the edge's `evidence` array, find its `document_id` in the `documents` array of `01_ner_corpus.json`, then read `text[start:end]` using the evidence offsets. Explicit edges also record offsets for both mentions and the predicate.

### Build a graph from saved predictions on CPU

Run these commands from the repository root after generating or extracting the files above. The first command uses graph rules on the saved NER predictions. The second uses those same predictions and the Qwen relation responses from that run. Both commands run on CPU without model weights or a Hugging Face token. Use the Git revision printed in the Colab setup output, or a revision with the same graph code. Choose unused output filenames.

```bash
python scripts/graph.py build --input outputs/01_ner_corpus.json --output outputs/01_relation_graph_rules.json
python scripts/graph.py build --input outputs/01_ner_corpus.json --relation-reviews outputs/01_relation_reviews.json --output outputs/01_relation_graph_rebuilt.json
```

<details>
<summary>Run the pipeline from the command line</summary>

After preparing the corpus, install the dependencies from the repository root and set `HF_TOKEN` in your shell environment. `run_hybrid.py` and `graph.py review` require a CUDA GPU. Entity candidate extraction, relation planning and graph building run on CPU. Change the prefix if you already used `01` in Colab.

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --output outputs/01_ner_baseline.json
python scripts/run_hybrid.py --input outputs/01_ner_baseline.json --output outputs/01_ner_corpus.json
python scripts/graph.py prepare --input outputs/01_ner_corpus.json --output outputs/01_relation_plan.json
python scripts/graph.py review --plan outputs/01_relation_plan.json --output outputs/01_relation_reviews.json
python scripts/graph.py build --input outputs/01_ner_corpus.json --relation-reviews outputs/01_relation_reviews.json --output outputs/01_relation_graph.json
```

</details>

### Evaluate your run

Provide your own human-reviewed reference JSON for the same documents and text offsets. For NER, `evaluate.py` can score the baseline directly. For a controlled hybrid comparison, use `run_hybrid.py --reference` with `--compare-routing` to generate predictions for that reference, then pass them to `evaluate.py` with `--hybrid`. For relations, use `graph.py evaluate` with the graph and a reviewed relation reference tied to its NER input. The scorers run on CPU. Use `--help` on either scoring command for its full arguments. [NER reference and scoring code](src/evaluation.py) · [Relation scoring code](scripts/graph.py)

## Experimental results

**Controlled NER comparison.** The methods were compared on the same 12 reviewed excerpts with 137 annotated mentions. Precision, recall and micro F1 require exact boundaries and labels. Each hybrid variant selected three excerpts under the same budget.

| Method | Precision | Recall | Micro F1 |
| --- | ---: | ---: | ---: |
| Rules and spaCy | 63.70% | 62.77% | 63.24% |
| Selective Qwen review without perplexity | 62.14% | 63.50% | 62.82% |
| Selective Qwen review with perplexity | 64.44% | 63.50% | 63.97% |

**Full-corpus NER, run `05`.** Minerva scored 28,907 passages. Qwen returned 81 fully usable responses, 18 partially usable responses and one unusable response across 100 passages. The hybrid pass changed predictions in 29 documents. None of the 12 reviewed test excerpts was selected at corpus scale, and both the baseline and the hybrid output scored 63.24% micro F1 on those excerpts.

**Relation graph, experiment `06`.** The experiment revalidated the saved Qwen relation responses from run `05`. The resulting graph contains 10,654 document nodes, 38,081 subject nodes, 54,808 co-occurrence edges and 117 explicit edges. The rules-only variant produced 106 explicit edges on the same NER input. Inspection of the 11 added edges found textual support for each under the project's annotation criteria. On a 20-excerpt diagnostic sample, both variants matched 2 of 8 annotated relations exactly with zero false positives. The six missed relations were outside the passages selected for Qwen. Overall graph precision remains unmeasured.

News articles make up about 97% of the corpus, and the annotated samples cover few judicial cases. These figures summarise recorded experiments. Their prediction files, reviewed references and detailed reports are held in the local research workspace and are excluded from Git. [Historical NER revision](https://github.com/AntoFratta/forensic-llm-graph/commit/b20a42ba6620948c0f881bdcc2181aabb232e93b)

## Repository layout

| Path | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | End-to-end Colab notebook |
| [`scripts/`](scripts/) | Corpus download and preparation, both NER passes, NER evaluation and the graph CLI |
| [`src/`](src/) | Extraction rules, passage selection, model execution, validation and graph construction |
| [`configs/`](configs/) | Source manifest, model revisions and experiment settings |
| [`tests/`](tests/) | Automated tests. Run with `python -m unittest discover -s tests` |

The scripts create `data/raw/`, `data/processed/` and `outputs/` as needed. These directories are excluded from Git.
