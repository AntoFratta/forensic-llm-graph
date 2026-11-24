# Forensic LLM Graph

This project helps readers reconstruct the context of Italian forensic and legal documents. It uses named entity recognition (NER) to locate people, companies, identifiers and other mentions in the text. It then builds a graph that shows the relationships and events described within each document and connects documents when there is evidence of a shared case, event or identity. Readers can follow the links back to the text or source metadata that supports them.

[Open the end-to-end notebook in Google Colab](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/forensic_llm_graph_colab.ipynb)

## Quick start

1. [Prepare the corpus](#prepare-the-corpus) locally to generate `data/processed/documents.json`.
2. [Get access to Minerva](#models-and-access) and add your Hugging Face token to Colab.
3. [Run the notebook on a T4 GPU](#end-to-end-on-google-colab). Upload `documents.json` to `/content/`, then download the ZIP containing the NER predictions and relation graph.
4. [Read the outputs](#read-the-generated-outputs) to locate an edge's evidence in the original document.

## How the pipeline works

1. **Extract entities.** Rules and Italian spaCy process every document. The ten labels are `PERSON`, `COMPANY`, `ROLE`, `LOCATION`, `EMAIL`, `PHONE`, `TAX_CODE`, `VAT_NUMBER`, `BANK_ACCOUNT` and `TIME`. Banking mentions include IBANs and other account numbers.
2. **Select and review NER passages.** Minerva measures how surprising each passage is to the model using perplexity. Passages have cores of about 800 characters and normally up to 160 characters of context on either side. Context is extended when needed to keep a first-pass candidate intact. Selection combines perplexity with signals such as overlapping candidates and possible omissions. The review budget is 20% of passages, rounded up and capped at 100. Qwen reviews the highest-priority eligible passages within that budget. The pipeline checks proposed labels and spans against the text before merging them with the first pass.
3. **Build the relation graph.** The pipeline applies text-grounded rules to all documents and sends up to 100 selected passages to Qwen by default. Selection covers available contextual relation and event families, then rotates through the sources. Each passage can contain up to four anchored candidates, prioritising wording or attachments that the rules alone would omit. Qwen accepts or rejects each candidate with a constrained binary response. Saved decisions identify the predicate and argument spans, which are checked again before graph construction. For roles, employment and legal representation, Qwen can also propose endpoints that the validator checks against the original wording. [Relation review](src/relation_review.py) · [Contextual facts](src/context_facts.py)

The graph creates a subgraph for every input document. `MENTIONED_IN` links a subject to the document where it appears. `CO_OCCURS_WITH` records that two subjects appear in the same passage. `EXPLICIT_RELATION` represents appointments, employment and legal representation. `CONTEXT_RELATION` covers directed links such as a person to their place of residence or an account holder to a bank account. A `FACT` node represents an event or a statement and connects the participants to their roles, with place, time or speaker when the text supports them. Evidence records the character offsets of the source text and, for contextual claims, whether the text states, attributes, hypothesises or negates them. [Graph construction](src/relation_graph.py)

Shared case, event and identity nodes connect the local subgraphs. For example, the parliamentary debate and committee report on Gaspare Giudice are connected through a shared case node. Links use source metadata, matching jurisdiction and docket references, reviewed references in the graph configuration, or an explicitly attributed shared identifier. Repeated titles and some repeated names are recorded in `link_candidates` for manual inspection. Documents with fictitious data are linked within their own group. [Cross-document links](src/document_links.py) · [Reviewed graph references](configs/graph_links.json)

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
3. In Colab, open **Secrets** using the key icon. Add a secret named `HF_TOKEN`, paste the token as its value and enable **Notebook access**. [Google's Colab Secrets example](https://colab.research.google.com/github/googlecolab/colabtools/blob/main/notebooks/Gemini_and_Stable_Diffusion.ipynb)

## Run the project

Start with `data/processed/documents.json` from [Prepare the corpus](#prepare-the-corpus), or use a hybrid NER JSON from a previous run. The examples use `01` as the run prefix. The repository contains the code and source manifest. Data and prediction files are generated locally or in Colab.

### End-to-end on Google Colab

1. Open the notebook, select **Runtime → Change runtime type → T4 GPU** and connect. [Colab notebook](https://colab.research.google.com/github/AntoFratta/forensic-llm-graph/blob/main/notebooks/forensic_llm_graph_colab.ipynb)
2. Upload the input to `/content/` through the **Files** panel and choose the matching mode in the setup cell.
   - **Full pipeline.** Upload `documents.json`, keep `reuse_saved_ner = False` and enable `HF_TOKEN`. [Models and access](#models-and-access)
   - **Graph only.** Upload a saved `*_ner_corpus.json`, set `reuse_saved_ner = True` and set `saved_ner_name` to the uploaded filename. This mode starts from the saved entity predictions and runs passage selection, Qwen relation review and graph construction. It still uses a GPU for Qwen. The Hugging Face token is optional because Minerva is skipped.
3. Use `run_number = "01"` for the filenames below, or select a new two-digit prefix. `relation_budget = 100` limits the passages sent to Qwen for relation review. Select **Runtime → Run all**. The notebook clones the code, installs the dependencies and prints `Pipeline revision` for the checkout it uses.
4. Download `/content/01_run.zip` from the **Files** panel. Extract its contents into `outputs/` in your local clone, creating the folder if needed. In graph-only mode, keep the NER file you uploaded so you can inspect the source of each edge.

If the runtime still holds its files after an interruption, rerun with the same input, run number and options. Completed stages are reused, and both hybrid NER and relation review can resume from their checkpoints. By default, Colab stops if the first five passages yield no supported Qwen proposal and saves its responses in `01_relation_reviews.partial.json`. Set `initial_review_check = 0` when an empty initial sample is expected. Use a fresh prefix for a new run.

### Read the generated outputs

The filenames below are for run `01`. They share the prefix selected in the notebook.

| File | Contents |
| --- | --- |
| `01_ner_baseline.json` | Document text and entity candidates from rules and spaCy. Input to the NER review stage. |
| `01_ner_corpus.json` | Document text, merged NER predictions, passage scores and review decisions. Input to relation planning and graph construction, and the source text used for inspecting edges. |
| `01_relation_plan.json` | Selected passages and their candidates for Qwen relation review. |
| `01_relation_reviews.json` | Saved Qwen responses and validation results used to augment the graph. |
| `01_relation_graph.json` | Final nodes, edges, supporting evidence and candidate document pairs. |

In graph-only mode, the ZIP contains the last three files in the table. Keep the uploaded NER JSON with them.

Open `01_relation_graph.json` to inspect the `nodes`, `edges` and `link_candidates` arrays. Each edge's `source` and `target` are node IDs. To read an edge's evidence, use `inspect` with the same NER JSON that produced the graph. The command exports the selected document or shared node, its linked nodes and the `quote` corresponding to each recorded text span. Case links based on metadata identify that metadata in the evidence. For manual lookup, match `document_id` to the NER output and read `text[start:end]`, where `start` is zero-based and `end` is exclusive. The `--edge-type` option can be repeated to narrow the export.

```bash
python scripts/graph.py inspect --graph outputs/01_relation_graph.json --input outputs/01_ner_corpus.json --node dice:1893315 --output outputs/01_document_view.json
```

In the example above, the exported subgraph for article `1893315` contains a directed `difensore` edge from Guido Magnisi to Vasco Rossi, with the wording from the article in its evidence. A `CO_OCCURS_WITH` edge for the same pair records that both appear in the same passage.

### Build a graph from saved predictions on CPU

Run these commands from the repository root after generating or extracting the files above. The first command uses graph rules on the saved NER predictions. The second uses those same predictions and the Qwen relation responses from that run. Both commands run on CPU without model weights or a Hugging Face token. Use the Git revision printed in the Colab setup output, or a revision with the same graph code. Choose unused output filenames.

```bash
python scripts/graph.py build --context --input outputs/01_ner_corpus.json --output outputs/01_relation_graph_rules.json
python scripts/graph.py build --context --input outputs/01_ner_corpus.json --relation-reviews outputs/01_relation_reviews.json --output outputs/01_relation_graph_rebuilt.json
```

Add `--revalidate-reviews` and choose a new output filename to apply updated graph validation to existing responses. Contextual decisions are matched against their saved predicate and argument offsets. Use the original revision to replay older contextual decisions recorded only by position in a candidate list.

<details>
<summary>Run the pipeline from the command line</summary>

After preparing the corpus, install the dependencies from the repository root and set `HF_TOKEN` in your shell environment. `run_hybrid.py` and `graph.py review` require a CUDA GPU. Entity candidate extraction, relation planning and graph building run on CPU. Change the prefix if you already used `01` in Colab.

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --output outputs/01_ner_baseline.json
python scripts/run_hybrid.py --input outputs/01_ner_baseline.json --output outputs/01_ner_corpus.json
python scripts/graph.py prepare --context --input outputs/01_ner_corpus.json --output outputs/01_relation_plan.json
python scripts/graph.py review --plan outputs/01_relation_plan.json --output outputs/01_relation_reviews.json
python scripts/graph.py build --context --input outputs/01_ner_corpus.json --relation-reviews outputs/01_relation_reviews.json --output outputs/01_relation_graph.json
```

</details>

### Evaluate your run

Provide your own human-reviewed reference JSON for the same documents and text offsets. For NER, `evaluate.py` can score the baseline directly. For a controlled hybrid comparison, use `run_hybrid.py --reference` with `--compare-routing` to generate predictions for that reference, then pass them to `evaluate.py` with `--hybrid`. For relations, use `graph.py evaluate` with the graph and a reviewed relation reference tied to its NER input. The scorers run on CPU. Use `--help` on either scoring command for its full arguments. [NER reference and scoring code](src/evaluation.py) · [Relation scoring code](scripts/graph.py)

Use `graph.py score-context` for the expanded graph. The reference records the SHA-256 of the saved NER in `input_sha256`. In `units`, each reviewed excerpt has an `id`, `document_id`, `start`, `end` and a `facts` array. Each fact records `kind`, `predicate_start`, `predicate_end`, `assertion` and `arguments`. The argument keys are role names, and each value records `label`, `start` and `end`. Exact matching checks the predicate and assertion along with the subject and object of a directed relation or all arguments of a fact node. Include excerpts without target facts to check for extractions made in error.

In `document_pairs`, each row specifies `documents` (two IDs), `kind` (`case`, `event` or `identity`) and a boolean `linked`. Include both related documents and similar but distinct cases. Score reviewed configuration entries as checks of the configured links. Use separate examples to measure generalization.

```bash
python scripts/graph.py score-context --graph outputs/01_relation_graph.json --reference reference.json --output outputs/01_context_results.json
```

## Experimental results

### Named entity recognition

**Controlled NER comparison.** The test covers 50 complete documents (26 DICE articles, 20 ReDIt forms and four synthetic texts), partitioned into 88 passages. The reference contains 890 model-annotated mentions. The comparison scores raw spaCy, rules plus spaCy, and selective Qwen review with and without perplexity. Each Qwen strategy has a budget of 30 passages. All four methods are evaluated on the entire reference using exact boundaries and labels. Results are pending.

Generate the raw spaCy predictions on the test documents with `run_pipeline.py --spacy-only`. Pass this output to `evaluate.py --spacy-only` alongside the baseline and the hybrid comparison. For this model-annotated reference, both the hybrid runner and the scorer take `--allow-model-reference`.

**Full-corpus NER, run `05`.** Minerva scored 28,907 passages. Qwen returned 81 fully usable responses, 18 partially usable responses and one unusable response across 100 passages. The hybrid pass changed predictions in 29 documents.

**NER review on selected passages.** A separate diagnostic sample of 12 passages from DICE and ArchivioAntimafia was drawn from those selected for Qwen in the full-corpus run. On that sample, micro F1 rose from 29.03% before review to 40.00% after review.

### Final relation graph

The graph comparison uses the same full-corpus NER predictions for both variants. Rules processed all 10,654 documents in both variants. In run `10`, Qwen reviewed 100 of 2,596 eligible passages across the five collections. The review stage recorded 71 valid, eight partial and 21 invalid responses. The final graph, run `11`, was built on CPU by revalidating those saved responses with the updated validator. [Qwen review revision](https://github.com/AntoFratta/forensic-llm-graph/commit/fe4e595671914339bf1a7d0411233d477486770a) · [Final validation revision](https://github.com/AntoFratta/forensic-llm-graph/commit/1c783236159fd6902cebee563944cb87116fbdf6)

| Graph component | Rules only | Final graph | Added by Qwen |
| --- | ---: | ---: | ---: |
| Document nodes | 10,654 | 10,654 | 0 |
| Contextual relation edges | 408 | 432 | 24 |
| Fact nodes | 632 | 640 | 8 |
| Role, employment and legal representation edges | 106 | 109 | 3 |

Qwen contributed 27 additional relation edges and eight fact nodes. It also confirmed 37 occurrences already extracted by the rules. The final graph has **127,399 nodes** and **520,812 edges**, including 116,100 mention edges and 402,555 co-occurrence edges. Three case nodes account for 24 document-to-case links. One shared event links two documents, and one shared identity connects subject nodes from two documents. A separate list records 304 candidate document pairs for inspection.

**Diagnostic evaluation.** Both graph variants recovered the three annotated contextual facts, linked the five positive document pairs and kept the three negative pairs separate. The reference includes links specified in the graph configuration, so these results describe behaviour on the checked examples. News articles make up about 97% of the corpus, and the annotated samples cover few judicial cases.

The predictions, annotations and detailed results for the thesis runs are kept in the local research workspace. The public repository provides the code and source manifest for generating a new corpus and graph. An exact CPU rebuild of the recorded final graph requires the saved NER predictions from run `05`, the relation responses from run `10` and the final validation revision with `--revalidate-reviews`. [NER evaluation revision](https://github.com/AntoFratta/forensic-llm-graph/commit/b20a42ba6620948c0f881bdcc2181aabb232e93b)

## Repository layout

| Path | Contents |
| --- | --- |
| [`notebooks/`](notebooks/) | End-to-end Colab notebook |
| [`scripts/`](scripts/) | Corpus download and preparation, both NER passes, NER evaluation and the graph CLI |
| [`src/`](src/) | Extraction rules, passage selection, model execution, validation and graph construction |
| [`configs/`](configs/) | Source manifest, model revisions and experiment settings |
| [`tests/`](tests/) | Automated checks for corpus preparation, entity extraction, Qwen response validation and graph construction, including links between documents. Run with `python -m unittest discover -s tests` |

The scripts create `data/raw/`, `data/processed/` and `outputs/` as needed. These directories are excluded from Git.
